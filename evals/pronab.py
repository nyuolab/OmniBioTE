import argparse
import json
import random
import sys
import os

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from scipy.stats import pearsonr

# For loading model definitions, etc.
sys.path.insert(0, "../training")
from model import OmniBioTA

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# ----------------------------- UTILITY FUNCTIONS -----------------------------
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

def prepare_nucleotide_string(nucleotide_sequences, nuc_type, use_strandedness_token, single_strand):
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
    
    nucleotide_str = f"{prefix}{nucleotide_sequences[0]}<EOS>"

    if len(nucleotide_sequences) == 2 and not single_strand:
        nucleotide_str += f"{prefix}{nucleotide_sequences[1]}<EOS>"

    return nucleotide_str

def prepare_sample(nuc_sp, prot_sp, peptide_sequence, nucleotide_sequence):
    """
    Tokenizes peptide and nucleotide sequences, removing any banned tokens.

    Args:
        sp (sentencepiece.SentencePieceProcessor): Tokenizer.
        peptide_sequence (str): Raw peptide sequence (will prepend <protein> and append <EOS>).
        nucleotide_sequence (str): Raw nucleotide sequence (already has <RNA>/<DNA> and <EOS>).
        banned_tokens (list): Token IDs to remove entirely from sequences.
        mutate (bool): (Currently unused) whether we do any special mutation logic.

    Returns:
        list[int]: The concatenated peptide + nucleotide token IDs.
    """
    # Encode Nucleotide
    nuc_encoded = nuc_sp.EncodeAsIds(nucleotide_sequence)
    # Encode Peptide
    pep_encoded = prot_sp.EncodeAsIds(f"<protein>{peptide_sequence}<EOS>")

    # Concatenate
    return pep_encoded + nuc_encoded

def create_sample(
    nuc_sp,
    prot_sp,
    nucleotide_sequences,
    peptides,
    G0s,
    index=None
):
    """
    Creates a single training sample by randomly selecting or
    indexing into the dataset and preparing tokens.

    Args:
        sp (sentencepiece.SentencePieceProcessor): Tokenizer.
        nucleotide_sequences (list[str]): Nucleotide sequences.
        peptides (list[str]): Peptide sequences.
        Kds (list[float]): Kd values.
        G0s (list[float]): G0 values.
        banned_tokens (list): Banned tokens.
        mutate (bool): Whether we apply mutation logic (unused).
        index (int or None): If None, picks random index, else a specific one.

    Returns:
        Tuple[list[list[int]], list[float], list[float]]:
          - A batch of tokenized sequences (shape [1, seq_len])
          - Corresponding Kd values
          - Corresponding G0 values
    """
    idx = np.random.randint(0, len(nucleotide_sequences)) if index is None else index
    x_tokens = prepare_sample(
        nuc_sp,
        prot_sp,
        peptides[idx],
        nucleotide_sequences[idx],
    )
    return [x_tokens], [G0s[idx]]

def evaluate_dG_predictions(
    model,
    head,
    test_data,
    nuc_tokenizer,
    protein_tokenizer,
    G0_mean,
    G0_std,
    use_strandedness_token=False,
    single_strand=False
):
    """
    For each entry in test_data, encode wild and mutant sequences, predict G0
    with the current model/head, and compute:
      - difference in predicted G0
      - difference in ground-truth G0
    Then returns:
      - difference arrays (predictions, ground_truths)
      - full G0 arrays (predictions, ground_truths)
    """
    model.eval()
    diffs_pred, diffs_gt = [], []
    all_g0_pred, all_g0_gt = [], []

    for entry in test_data:
        pep_seq = entry["peptide_sequence"]

        # Prepare wild
        wild_tokens = prepare_sample(nuc_tokenizer, protein_tokenizer, pep_seq, prepare_nucleotide_string(entry["wild_nucleotide_sequence"], entry["sequence_type"],
                                                                                                          use_strandedness_token, single_strand))
        X_wild = torch.tensor([wild_tokens], device=device, dtype=torch.long)[:, :1024]

        # Forward pass
        emb_wild = model(X_wild, return_embeddings=True)[:, 0]

        # Predicted G0
        pred_wild = head(emb_wild).item() * G0_std * (1024 / model.transformer.wte.weight.shape[-1]) + G0_mean

        gt_wild_g0 = entry["wild_G0"]

        all_g0_pred.append(pred_wild)
        all_g0_gt.append(gt_wild_g0)

    return all_g0_pred, all_g0_gt

# ----------------------------- MAIN SCRIPT LOGIC -----------------------------
def recurse_load(m):
    """
    Recursively find the original module if it is wrapped by a
    flash-attention-style `_orig_mod`.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m


def main():
    parser = argparse.ArgumentParser(description="Finetune and evaluate OmniBioTA model on pronab/mutation data.")
    parser.add_argument("--model_fn", type=str, help="Path to the pretrained model file (.pt).")
    parser.add_argument("--nucleotide_tokenizer", type=str, required=True,
                        help="Path to the nucleotide tokenizer .model file (SentencePiece).")
    parser.add_argument("--protein_tokenizer", type=str, required=True,
                        help="Path to the protein tokenizer .model file (SentencePiece).")
    
    parser.add_argument("--num_accumulation_steps", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--nuc_banned_token", type=int, default=2037)
    parser.add_argument("--prot_banned_token", type=int, default=2044)
    parser.add_argument("--prot_offset", type=int, default=2048)
    parser.add_argument("--single_strand", action="store_true", help="Only input one strand of the nucleotide sequence if double stranded")
    parser.add_argument("--use_strandedness_token", action="store_true", help="Encorporate the strandedness into the tokenization")
    parser.add_argument("--num_splits", type=int, default=10, help="Number of cross-validation splits.")
    parser.add_argument("--memory_efficient", action="store_true", help="If True, use a memory-efficient version of the model.")

    parser.add_argument("--save_dir", type=str, help="Suffix to append to output files.")
    parser.add_argument("--start_at", type=int, default=0, help="Start at a specific fold.")
    args = parser.parse_args()

    model_fn = args.model_fn
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Using args: {args}")

    # Set seeds
    set_seed(0)

    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.nucleotide_tokenizer)
    nuc_tokenizer = TokenizerWrapper(sp, 0, [1, 2, args.nuc_banned_token])

    sp = spm.SentencePieceProcessor()
    sp.Load(args.protein_tokenizer)
    prot_tokenizer = TokenizerWrapper(sp, args.prot_offset, [1, 2, args.prot_banned_token])

    crossval_data = []
    with open("../datasets/pronab_crossval_noddg.jsonl", "r") as f:
        for line in f:
            crossval_data.append(json.loads(line))
    
    train_only = []
    with open("../datasets/pronab_crossval_train_only_noddg.jsonl", "r") as f:
        for line in f:
            train_only.append(json.loads(line))
    
    # shuffle data
    random.seed(0)
    random.shuffle(crossval_data)
    random.shuffle(train_only)

    # Group by peptide sequence
    grouped_sequences = {}
    for entry in crossval_data:
        pep_seq = entry["peptide_sequence"]
        if pep_seq not in grouped_sequences:
            grouped_sequences[pep_seq] = []
        grouped_sequences[pep_seq].append(entry)

    for fold in range(args.num_splits):
        if fold < args.start_at:
            continue

        # Load model
        base_model = torch.load(model_fn, map_location="cpu")
        base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)
        model_sd = base_model.state_dict()
        config = base_model.config
        config.memory_efficient = args.memory_efficient
        model = OmniBioTA(config).to(device="cpu", dtype=dtype)
        model.load_state_dict(model_sd)
        model.to(device=device, dtype=dtype)
        model.eval()

        del base_model

        print(f"Loaded model from {model_fn}, #Params: {model.get_num_params() / 1e6:.2f}M")

        # Create and initialize head
        head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(device).to(dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        
        # generate train and test sets
        nuc_train = []
        pep_train = []
        G0_train = []

        test_set = []
        for i, key in enumerate(grouped_sequences):
            for entry in grouped_sequences[key]:
                nucleotide_sequences = entry["wild_nucleotide_sequence"]
                nuc_type = entry["sequence_type"]

                if nuc_type not in ["RNA", "DNA"]:
                    continue

                if entry["wild_G0"] == 0:
                    continue
                
                nucleotide_str = prepare_nucleotide_string(nucleotide_sequences, nuc_type, args.use_strandedness_token, args.single_strand)

                if i % args.num_splits == fold:
                    test_set.append(entry)
                else:
                    nuc_train.append(nucleotide_str)
                    pep_train.append(entry["peptide_sequence"])
                    G0_train.append(entry["wild_G0"])
        
        for entry in train_only:
            nucleotide_sequences = entry["wild_nucleotide_sequence"]
            nuc_type = entry["sequence_type"]

            if nuc_type not in ["RNA", "DNA"]:
                continue

            if entry["wild_G0"] == 0:
                continue
            
            nucleotide_str = prepare_nucleotide_string(nucleotide_sequences, nuc_type, args.use_strandedness_token, args.single_strand)

            nuc_train.append(nucleotide_str)
            pep_train.append(entry["peptide_sequence"])
            G0_train.append(entry["wild_G0"])
        
        G0_mean = np.mean(G0_train)
        G0_std = np.std(G0_train)

        # Setup training hyperparams
        num_accumulation_steps = args.num_accumulation_steps
        num_epochs = args.num_epochs
        lr = args.lr
        embed_lr = args.embed_lr
        head_lr = args.head_lr

        # Optimizer
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "wte" in n], "lr": embed_lr},
            {"params": [p for n, p in model.named_parameters() if "wte" not in n], "lr": lr},
            {"params": head.parameters(), "lr": head_lr}
        ]
        num_steps = int(num_epochs * len(nuc_train) / num_accumulation_steps)
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[embed_lr, lr, head_lr],
            total_steps=num_steps,
            pct_start=0.05
        )

        # ---------------------- Train on pronab data ----------------------
        print("Starting pronab pre-training...")
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(num_accumulation_steps):
                X_list, g0_list = create_sample(
                    nuc_tokenizer,
                    prot_tokenizer,
                    nuc_train,
                    pep_train,
                    G0_train,
                )
                # Choose target
                target_vals = np.array(g0_list)
                target_vals = (target_vals - G0_mean) / G0_std

                X_tensor = torch.tensor(X_list, device=device, dtype=torch.long)[:, :1024]
                y_tensor = torch.tensor(target_vals, device=device, dtype=dtype)

                emb = model(X_tensor, return_embeddings=True)[:, 0]
                pred = head(emb) * (1024 / model.transformer.wte.weight.shape[-1])
                loss = ((pred - y_tensor) ** 2).mean() / num_accumulation_steps
                loss.backward()

                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Step: {step}, Loss: {total_loss:.4f}")

        pbar.close()
        #torch.save(model, f"pronab_all_ft_{output_suffix}.pt")

        # ---------------------- Evaluate on cross-validation set ----------------------
        print("Evaluating on cross-validation set...")
        with torch.no_grad():
            model.eval()
            head.eval()

            all_g0_pred, all_g0_gt = evaluate_dG_predictions(
                model, head, test_set, nuc_tokenizer, prot_tokenizer, G0_mean, G0_std,
                args.use_strandedness_token, args.single_strand
            )
            
            all_g0_pred = np.asarray(all_g0_pred)
            all_g0_gt = np.asarray(all_g0_gt)

            pcc_g0 = pearsonr(all_g0_gt, all_g0_pred)[0]
            mae_g0 = np.mean(np.abs(all_g0_gt - all_g0_pred))

            print(f"G0 PCC = {pcc_g0:.4f}, G0 MAE = {mae_g0:.4f}")

            # Log results
            out_json = {
                "dG_ground_truths": all_g0_gt.tolist(),
                "dG_predictions": all_g0_pred.tolist(),
                "dG_pcc": pcc_g0,
                "dG_MAE": mae_g0
            }
            with open(os.path.join(args.save_dir, f"fold{fold}.jsonl"), "w") as f:
                json.dump(out_json, f)
                f.write("\n")
            
            torch.save(model.cpu(), os.path.join(args.save_dir, f"model_fold{fold}.pt"))
            torch.save(head.cpu(), os.path.join(args.save_dir, f"head_fold{fold}.pt"))
            np.save(os.path.join(args.save_dir, f"fold_stats{fold}.npy"), np.array([pcc_g0, mae_g0]))

if __name__ == "__main__":
    main()
