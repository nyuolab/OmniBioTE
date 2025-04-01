import argparse
import json
import os
import random
import sys
import copy
import re

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr

# Insert training path
sys.path.insert(0, "../training")
from model import OmniBioTA
from loader import EOS_TOKEN, PAD_TOKEN

# ----------------------------- DEVICE/DTYPE SETUP -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# ----------------------------- TOKENIZER (SINGLE-CHAR) -----------------------------
class Tokenizer:
    """
    A simple single-character + <token> tokenizer used for DNA or peptides.
    """
    def __init__(self, tokenizer_type):
        assert tokenizer_type in ["DNA", "peptide"], "Invalid tokenizer type"
        if tokenizer_type == "peptide":
            # Maps each character or token to a vocabulary ID
            self.vocab = {
                'A': 23, 'R': 24, 'N': 25, 'D': 26, 'C': 27, 'Q': 28, 'E': 29,
                'G': 30, 'H': 31, 'I': 32, 'L': 33, 'K': 34, 'M': 35, 'F': 36,
                'P': 37, 'S': 38, 'T': 39, 'W': 40, 'Y': 41, 'V': 42, 'B': 43,
                'Z': 44, 'X': 45, 'U': 46, 'O': 47,
                '<protein>': 48,
                '<EOS>': EOS_TOKEN,
            }
        else:  # DNA
            self.vocab = {
                'A': 4, 'T': 5, 'C': 6, 'G': 7, 'N': 8,
                '<DNA>': 9, '<mRNA>': 10, '<RNA>': 11, '<rRNA>': 12, '<tRNA>': 13,
                '<cRNA>': 14, '<ss-RNA>': 15, '<ss-DNA>': 16, '<ds-mRNA>': 17,
                '<ds-rRNA>': 18, '<ds-RNA>': 19, '<ms-DNA>': 20, '<ms-RNA>': 21,
                '<ds-cRNA>': 22,
                '<EOS>': EOS_TOKEN,
            }

        # We'll use a regex to capture <some_token> or single chars:
        self._pattern = re.compile(r'<[^>]*>|.')

    def Encode(self, sequence):
        """
        Convert 'sequence' into a list of IDs, where each recognized token 
        is mapped to an ID from 'self.vocab'. Unknown => 0.
        """
        tokens = self._pattern.findall(sequence)
        return [self.vocab.get(tok, 0) for tok in tokens]

    def EncodeAsIds(self, sequence):
        return self.Encode(sequence)

    def Decode(self, token_ids):
        """
        Convert a list of IDs back into tokens (only used for debugging).
        """
        # Inverse lookup:
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(t, "<UNK>") for t in token_ids)

# ----------------------------- UTILITY FUNCTIONS -----------------------------
def set_seed(seed: int = 0):
    """
    Set all relevant random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def recurse_load(m):
    """
    Recursively find the underlying module if wrapped by _orig_mod (flash-attention).
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m

def prepare_nucleotide_string(nucleotide_sequences, nuc_type, use_strandedness_token, single_strand):
    """
    Same logic as the golden code. We add <ss-RNA>, <ds-RNA>, etc. if use_strandedness_token.
    Then we append <EOS> after each strand. If single_strand is True, only use the first strand.
    """
    if nuc_type == "RNA":
        if len(nucleotide_sequences) == 2:
            prefix = "<ds-RNA>" if use_strandedness_token else "<RNA>"
        else:
            prefix = "<ss-RNA>" if use_strandedness_token else "<RNA>"
    elif nuc_type == "DNA":
        if len(nucleotide_sequences) == 2:
            prefix = "<ds-DNA>" if use_strandedness_token else "<DNA>"
        else:
            prefix = "<ss-DNA>" if use_strandedness_token else "<DNA>"
    else:
        raise ValueError(f"Unknown nucleotide type: {nuc_type}")

    out_str = f"{prefix}{nucleotide_sequences[0]}<EOS>"
    if len(nucleotide_sequences) == 2 and not single_strand:
        out_str += f"{prefix}{nucleotide_sequences[1]}<EOS>"
    return out_str

def prepare_sample(nuc_tokenizer, prot_tokenizer, peptide_sequence, nucleotide_sequence):
    """
    Encodes <protein> + pep_seq + <EOS> + nuc_seq.
    """
    pep_encoded = prot_tokenizer.EncodeAsIds(f"<protein>{peptide_sequence}<EOS>")
    nuc_encoded = nuc_tokenizer.EncodeAsIds(nucleotide_sequence)
    return pep_encoded + nuc_encoded

def create_sample(
    nuc_tokenizer,
    prot_tokenizer,
    nucleotide_sequences,
    peptides,
    G0s,
    idx=None
):
    """
    Picks a random index (or uses the provided 'idx'), encodes the single 'wild' sequence,
    returns token IDs plus the G0 value for training.
    """
    if idx is None:
        idx = np.random.randint(0, len(nucleotide_sequences))
    x_tokens = prepare_sample(
        nuc_tokenizer,
        prot_tokenizer,
        peptides[idx],
        nucleotide_sequences[idx]
    )
    return [x_tokens], [G0s[idx]]

def evaluate_g0_predictions(
    model,
    head,
    test_data,
    nuc_tokenizer,
    prot_tokenizer,
    G0_mean,
    G0_std,
    use_strandedness_token=False,
    single_strand=False
):
    """
    Evaluates predicted G0 on 'wild' sequences only. 
    (No difference-based evaluation.)
    Returns arrays of predictions and ground truths.
    """
    model.eval()
    head.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for entry in test_data:
            nuc_type = entry["sequence_type"]
            if nuc_type not in ["RNA", "DNA"]:
                continue
            if entry["wild_G0"] == 0:
                continue

            pep_seq = entry["peptide_sequence"]
            # Build the string for the nucleotides
            nuc_str = prepare_nucleotide_string(
                entry["wild_nucleotide_sequence"],
                nuc_type,
                use_strandedness_token,
                single_strand
            )
            # Encode
            tokens = prepare_sample(nuc_tokenizer, prot_tokenizer, pep_seq, nuc_str)
            X = torch.tensor([tokens], device=device, dtype=torch.long)[:, :2048]

            emb = model(X, return_embeddings=True)[:, 0]
            # Scale factor from golden code
            scale_factor = 1024 / model.transformer.wte.weight.shape[-1]
            raw_pred = head(emb).item()
            # Denormalize
            pred_g0 = raw_pred * scale_factor * G0_std + G0_mean

            all_preds.append(pred_g0)
            all_gts.append(entry["wild_G0"])

    return all_preds, all_gts

# ----------------------------- MAIN SCRIPT -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-character OmniBioTA, golden-code style crossval (wild-only).")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pretrained model .pt")
    parser.add_argument("--save_dir", type=str, default="char_outputs", help="Where to save crossval results.")

    parser.add_argument("--num_accumulation_steps", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--num_splits", type=int, default=10, help="Number of crossval splits")
    parser.add_argument("--start_at", type=int, default=0, help="Start at this fold (for resuming).")

    parser.add_argument("--single_strand", action="store_true", help="Use only first strand if 2-stranded input.")
    parser.add_argument("--use_strandedness_token", action="store_true", help="Prefix <ds-RNA>, <ss-DNA>, etc.")
    parser.add_argument("--memory_efficient", action="store_true", help="Set memory_efficient in config, if available.")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Using args: {args}")
    set_seed(0)

    # ----------------- Load the single-char tokenizers -----------------
    dna_tokenizer = Tokenizer("DNA")
    pep_tokenizer = Tokenizer("peptide")

    # ----------------- Load the model and config -----------------
    loaded = torch.load(args.model_dir, map_location="cpu")
    loaded = recurse_load(loaded).to(device="cpu", dtype=dtype)

    config = loaded.config
    config.memory_efficient = args.memory_efficient
    sd = loaded.state_dict()

    base_model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    base_model.load_state_dict(sd)
    base_model.to(device=device, dtype=dtype)
    base_model.eval()

    print(f"Model loaded. #Params={base_model.get_num_params()/1e6:.2f}M")

    # ----------------- Load crossval + train_only data (wild only) -----------------
    crossval_data = []
    with open("../datasets/pronab_crossval_noddg.jsonl", "r") as f:
        for line in f:
            crossval_data.append(json.loads(line))

    train_only_data = []
    with open("../datasets/pronab_crossval_train_only_noddg.jsonl", "r") as f:
        for line in f:
            train_only_data.append(json.loads(line))

    # Shuffle
    random.shuffle(crossval_data)
    random.shuffle(train_only_data)

    # Group crossval by peptide seq
    grouped_sequences = {}
    for entry in crossval_data:
        pep_seq = entry["peptide_sequence"]
        if pep_seq not in grouped_sequences:
            grouped_sequences[pep_seq] = []
        grouped_sequences[pep_seq].append(entry)

    # ---------- Build the training arrays from train_only ----------
    # (like in golden code, we only do random sampling from these + crossval groups)
    nuc_train_all = []
    pep_train_all = []
    G0_train_all = []
    for entry in train_only_data:
        if entry["sequence_type"] not in ["RNA", "DNA"]:
            continue
        if entry["wild_G0"] == 0:
            continue
        nuc_str = prepare_nucleotide_string(
            entry["wild_nucleotide_sequence"],
            entry["sequence_type"],
            args.use_strandedness_token,
            args.single_strand
        )
        nuc_train_all.append(nuc_str)
        pep_train_all.append(entry["peptide_sequence"])
        G0_train_all.append(entry["wild_G0"])

    # ----------------- CROSS VALIDATION -----------------
    pep_keys = list(grouped_sequences.keys())

    for fold in range(args.num_splits):
        if fold < args.start_at:
            continue
        print(f"\n=== FOLD {fold} of {args.num_splits} ===")
        nuc_train = []
        pep_train = []
        G0_train = []
        test_data = []

        # Partition crossval_data by pep_seq
        for i, key in enumerate(pep_keys):
            for entry in grouped_sequences[key]:
                if i % args.num_splits == fold:
                    # test
                    test_data.append(entry)
                else:
                    # train
                    if entry["sequence_type"] not in ["RNA", "DNA"]:
                        continue
                    if entry["wild_G0"] == 0:
                        continue
                    nuc_str = prepare_nucleotide_string(
                        entry["wild_nucleotide_sequence"],
                        entry["sequence_type"],
                        args.use_strandedness_token,
                        args.single_strand
                    )
                    nuc_train.append(nuc_str)
                    pep_train.append(entry["peptide_sequence"])
                    G0_train.append(entry["wild_G0"])

        # Add the train_only data
        nuc_train.extend(nuc_train_all)
        pep_train.extend(pep_train_all)
        G0_train.extend(G0_train_all)

        # Stats
        G0_mean = np.mean(G0_train) if len(G0_train) > 0 else 0.0
        G0_std = np.std(G0_train) if len(G0_train) > 0 else 1.0
        if G0_std < 1e-9:
            G0_std = 1.0

        # Re-init a fresh copy of base model for each fold
        model_fold = OmniBioTA(config).to(device="cpu", dtype=dtype)
        model_fold.load_state_dict(sd)
        model_fold.to(device=device, dtype=dtype)
        model_fold.eval()

        # Create a new linear head
        d_model = model_fold.transformer.wte.weight.shape[-1]
        head = torch.nn.Linear(d_model, 1).to(device=device, dtype=dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

        # Create the optimizer with wte vs. non-wte vs. head param groups
        param_groups = [
            {
                "params": [p for n, p in model_fold.named_parameters() if "wte" in n],
                "lr": args.embed_lr
            },
            {
                "params": [p for n, p in model_fold.named_parameters() if "wte" not in n],
                "lr": args.lr
            },
            {
                "params": head.parameters(),
                "lr": args.head_lr
            },
        ]
        # total steps
        num_steps = int(args.num_epochs * len(nuc_train) / args.num_accumulation_steps)
        if num_steps < 1:
            num_steps = 1

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.embed_lr, args.lr, args.head_lr],
            total_steps=num_steps,
            pct_start=0.05
        )

        # --------------- TRAIN (random sampling on wild sequences) ---------------
        print(f"[Fold {fold}] #Train samples={len(nuc_train)}, #Steps={num_steps}")
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(args.num_accumulation_steps):
                X_list, g0_list = create_sample(dna_tokenizer, pep_tokenizer, nuc_train, pep_train, G0_train)
                # standardize
                y_val = (np.array(g0_list) - G0_mean) / (G0_std + 1e-9)

                X = torch.tensor(X_list, device=device, dtype=torch.long)[:, :2048]
                emb = model_fold(X, return_embeddings=True)[:, 0]
                # scale factor
                scale_factor = 1024 / d_model
                pred = head(emb) * scale_factor

                y_tensor = torch.tensor(y_val, device=device, dtype=dtype)
                loss = ((pred - y_tensor) ** 2).mean() / args.num_accumulation_steps
                loss.backward()

                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            pbar.set_description(f"(Fold {fold}) Step {step}, Loss={total_loss:.4f}")

        pbar.close()

        # --------------- EVAL on test set ---------------
        print(f"[Fold {fold}] #Test samples={len(test_data)} => evaluating G0 predictions.")
        all_preds, all_gts = evaluate_g0_predictions(
            model_fold, head, test_data, dna_tokenizer, pep_tokenizer, G0_mean, G0_std,
            use_strandedness_token=args.use_strandedness_token,
            single_strand=args.single_strand
        )
        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)

        if len(all_preds) > 0:
            pcc = pearsonr(all_gts, all_preds)[0]
            mae = np.mean(np.abs(all_gts - all_preds))
        else:
            pcc, mae = 0.0, 0.0

        print(f"[Fold {fold}] G0 PCC={pcc:.4f}, MAE={mae:.4f}")

        # Save fold results
        out_json = {
            "fold": fold,
            "predictions": all_preds.tolist(),
            "ground_truths": all_gts.tolist(),
            "pcc": pcc,
            "MAE": mae
        }
        with open(os.path.join(args.save_dir, f"fold{fold}.jsonl"), "w") as f_out:
            json.dump(out_json, f_out)
            f_out.write("\n")

        torch.save(model_fold.cpu(), os.path.join(args.save_dir, f"model_fold{fold}.pt"))
        torch.save(head.cpu(), os.path.join(args.save_dir, f"head_fold{fold}.pt"))

        # cleanup
        del model_fold, head, optimizer, scheduler
        torch.cuda.empty_cache()

    print("All folds complete!")

if __name__ == "__main__":
    main()
