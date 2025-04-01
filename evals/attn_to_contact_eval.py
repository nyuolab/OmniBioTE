import argparse
import json
import random
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from model_transparent import OmniBioTA 


# =====================
# Helper Classes/Functions
# =====================
class TokenizerWrapper:
    """
    A thin wrapper around the SentencePiece tokenizer that applies:
        - an offset to each token ID
        - skipping of banned tokens
    """
    def __init__(self, sp, tokenizer_offset, banned_tokens):
        self.sp = sp
        self.tokenizer_offset = tokenizer_offset
        self.banned_tokens = banned_tokens

    def EncodeAsIds(self, x):
        tokens = self.sp.EncodeAsIds(x)
        return [t + self.tokenizer_offset for t in tokens if t not in self.banned_tokens]

    def DecodeIds(self, x):
        if not isinstance(x, list):
            x = [x]
        return self.sp.DecodeIds([t - self.tokenizer_offset for t in x])

    def Decode(self, x):
        return self.DecodeIds(x)


def prepare_sample(peptide_sequence, nucleotide_sequence, nuc_tokenizer, prot_tokenizer):
    if "U" in nucleotide_sequence:
        # Convert RNA to DNA by replacing 'U' with 'T'
        nucleotide_sequence = nuc_tokenizer.EncodeAsIds("<RNA>" + nucleotide_sequence.replace("U", "T") + "<EOS>")
    else:
        nucleotide_sequence = nuc_tokenizer.EncodeAsIds("<DNA>" + nucleotide_sequence + "<EOS>")

    peptide_sequence = "<protein>" + peptide_sequence + "<EOS>"
    peptide_sequence = prot_tokenizer.EncodeAsIds(peptide_sequence)

    return peptide_sequence + nucleotide_sequence


def recurse_load(m):
    """
    Recursively find the original module if it is wrapped by a
    flash-attention-style '_orig_mod'.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m


class ContactPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(ContactPredictor, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.conv4(x)
        
        return x.max(dim=-1)[0]

# =====================
# Main Script
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Train a contact predictor on peptide-nucleotide distances."
    )

    parser.add_argument(
        "--model_fn",
        type=str,
        help="Path to the OmniBioTA model checkpoint (.pt) file."
    )
    parser.add_argument(
        "--nuc_tokenizer_model",
        type=str,
        default="../tokenizers/uniref-2k.model",
        help="Path to the nucleotide SentencePiece tokenizer model."
    )
    parser.add_argument(
        "--prot_tokenizer_model",
        type=str,
        default="../tokenizers/uniref-2k.model",
        help="Path to the protein SentencePiece tokenizer model."
    )
    parser.add_argument(
        "--distance_data",
        type=str,
        default="../datasets/peptide-nucleotide-distances.json",
        help="Path to the peptide-nucleotide-distances JSON file."
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        help="Suffix for the output JSON file (e.g. 'test_run')."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of training steps (outer loop)."
    )
    parser.add_argument(
        "--num_accum_steps",
        type=int,
        default=256,
        help="Number of gradient accumulation steps per training step."
    )
    parser.add_argument(
        "--contact_threshold",
        type=int,
        default=8,
        help="Threshold distance to consider a contact."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Fraction for training split (not used in the classical sense here, but for reference)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )

    args = parser.parse_args()

    print(f"Model checkpoint: {args.model_fn}")
    print(f"Nucleotide tokenizer model: {args.nuc_tokenizer_model}")
    print(f"Protein tokenizer model: {args.prot_tokenizer_model}")
    print(f"Distance data: {args.distance_data}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Number of training steps: {args.num_steps}")
    print(f"Number of gradient accumulation steps: {args.num_accum_steps}")

    # -------------------------
    # Setup Device and Dtype
    # -------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # -------------------------
    # Seed and Randomness
    # -------------------------
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # -------------------------
    # Load tokenizers
    # -------------------------
    nuc_sp = spm.SentencePieceProcessor(args.nuc_tokenizer_model)
    prot_sp = spm.SentencePieceProcessor(args.prot_tokenizer_model)

    nuc_tokenizer = TokenizerWrapper(nuc_sp, 0, [1, 2, 2037])
    prot_tokenizer = TokenizerWrapper(prot_sp, 0, [1, 2, 2044])

    # -------------------------
    # Load dataset
    # -------------------------
    with open(args.distance_data) as f:
        distance_data = json.load(f)

    # Shuffle data once so each fold sees a different subset
    random.shuffle(distance_data)

    # -------------------------
    # Load base model
    # -------------------------
    base_model = torch.load(args.model_fn, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)
    model_sd = base_model.state_dict()
    config = base_model.config

    model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=dtype)
    model.eval()

    # -------------------------
    # Training Loop
    # -------------------------
    all_losses = []
    all_F1s = []

    for fold in range(args.folds):
        contact_predictor = ContactPredictor(config.n_layer ** 2).to(device)
        optimizer = torch.optim.AdamW(contact_predictor.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=args.num_steps
        )

        F1s = []
        losses = []

        # Split for folds (10-fold cross-validation)
        train_set = [item for i, item in enumerate(distance_data) if i % args.folds != fold]
        val_set = [item for i, item in enumerate(distance_data) if i % args.folds == fold]

        pbar = tqdm(range(args.num_steps), total=args.num_steps, desc=f"Fold {fold+1}/{args.folds}")

        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(args.num_accum_steps):
                sample = random.choice(train_set)

                peptide_sequence = sample["peptide_sequences"][list(sample["peptide_sequences"].keys())[0]]
                nucleotide_sequence = sample["nucleotide_sequences"][list(sample["nucleotide_sequences"].keys())[0]]

                # Skip too-short sequences
                if len(nucleotide_sequence) < 4 or len(peptide_sequence) < 4:
                    continue

                with torch.no_grad():
                    X = [prepare_sample(peptide_sequence, nucleotide_sequence, nuc_tokenizer, prot_tokenizer)]
                    X_tensor = torch.tensor(X).to(device)[:, :1024]

                    _, attn = model(X_tensor)
                    for i in range(len(attn)):
                        attn[i] = F.softmax(attn[i], dim=-1)

                    attn = torch.concat(attn).float()
                    attn = attn.reshape(-1, attn.shape[-1], attn.shape[-1])

                # Identify <EOS> for the protein sequence
                indices = [i for i, x in enumerate(X[0]) if x == 3]
                protein_tokens = [
                    prot_tokenizer.Decode([x]) for x in X[0][1:indices[0]]
                ]

                ground_truth_tensor = torch.tensor(sample["closest_nucleotides"], device=device)
                ground_truth = (ground_truth_tensor <= args.contact_threshold).float()

                # Each "protein token" can expand to multiple sub-tokens
                Y = []
                ptr = 0
                for i_tok in range(len(protein_tokens)):
                    # Mark as contact if *any* sub-token in that range is within threshold
                    Y.append(torch.any(ground_truth[ptr:ptr + len(protein_tokens[i_tok])]))
                    ptr += len(protein_tokens[i_tok])

                Y = torch.tensor(Y, device=device).float()[:1023]

                logits = contact_predictor(attn).squeeze()
                logits = logits[1:indices[0]]

                if logits.shape != Y.shape:
                    continue  # skip malformed samples

                loss = F.binary_cross_entropy_with_logits(logits, Y) / args.num_accum_steps
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            losses.append(total_loss)

            # Validation check every N steps (here, 16)
            if step % 16 == 0 and len(val_set) > 0:
                all_preds = []
                all_ground_truths = []

                for i_val, val_sample in enumerate(val_set):
                    peptide_sequence = val_sample["peptide_sequences"][list(val_sample["peptide_sequences"].keys())[0]]
                    nucleotide_sequence = val_sample["nucleotide_sequences"][list(val_sample["nucleotide_sequences"].keys())[0]]

                    if len(nucleotide_sequence) < 4 or len(peptide_sequence) < 4:
                        continue

                    with torch.no_grad():
                        X = [prepare_sample(peptide_sequence, nucleotide_sequence, nuc_tokenizer, prot_tokenizer)]
                        X_tensor = torch.tensor(X).to(device)[:, :1024]

                        _, attn = model(X_tensor)
                        attn = torch.concat(attn).float()
                        for i in range(len(attn)):
                            attn[i] = F.softmax(attn[i], dim=-1)
                        attn = attn.reshape(-1, attn.shape[-1], attn.shape[-1])

                        indices = [i for i, x in enumerate(X[0]) if x == 3]
                        protein_tokens = [
                            prot_tokenizer.Decode([x]) for x in X[0][1:indices[0]]
                        ]
                        ground_truth_np = (np.asarray(val_sample["closest_nucleotides"]) <= args.contact_threshold)

                        probs = contact_predictor(attn).squeeze().sigmoid()
                        probs = probs[1:indices[0]].cpu().tolist()

                    pred = []
                    for j in range(len(probs)):
                        pred.extend([probs[j]] * len(protein_tokens[j]))

                    ground_truth_np = ground_truth_np[:len(pred)]

                    all_preds.extend(pred)
                    all_ground_truths.extend(ground_truth_np)

                all_preds = np.asarray(all_preds)
                all_preds_bin = (all_preds > 0.5).astype(int)
                F1 = f1_score(all_ground_truths, all_preds_bin)
                F1s.append(F1)

                pbar.set_description(f"(Fold {fold+1}/{args.folds}) Step [{step}/{args.num_steps}] F1: {F1:.4f}")

        all_losses.append(losses)
        all_F1s.append(F1s)
    
    # calculate mean of the final F1 scores
    mean_F1 = np.mean([F1s[-1] for F1s in all_F1s])
    std_F1 = np.std([F1s[-1] for F1s in all_F1s])
    
    print(f"Mean F1: {mean_F1:.4f} ± {std_F1/np.sqrt(args.folds):.4f}")

    # -------------------------
    # Write out results
    # -------------------------
    output_filename = f"attn_contact_preds_{args.output_suffix}.json"
    with open(output_filename, "w") as f:
        json.dump([all_F1s, all_losses], f, indent=2)

    print(f"Training complete. Results saved to: {output_filename}")


if __name__ == "__main__":
    main()