import argparse
import json
import os
import random
import sys
import copy

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from scipy.stats import pearsonr

# For loading model definitions, etc.
sys.path.insert(0, "../training")
from model import OmniBioTA

# ----------------------------- DEVICE/DTYPE SETUP -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# ----------------------------- TOKENIZER WRAPPER -----------------------------
class TokenizerWrapper:
    """
    A thin wrapper around the SentencePiece tokenizer that applies:
        - an offset to each token ID
        - skipping of banned tokens
    """
    def __init__(self, sp_processor, tokenizer_offset, banned_tokens):
        self.sp = sp_processor
        self.tokenizer_offset = tokenizer_offset
        self.banned_tokens = set(banned_tokens)

    def EncodeAsIds(self, text: str):
        tokens = self.sp.EncodeAsIds(text)
        # Offset each ID, and skip any banned tokens
        final = []
        for t in tokens:
            if t not in self.banned_tokens:
                final.append(t + self.tokenizer_offset)
        return final

    def DecodeIds(self, ids):
        if not isinstance(ids, list):
            ids = [ids]
        # Remove offset
        real_ids = [t - self.tokenizer_offset for t in ids]
        return self.sp.DecodeIds(real_ids)

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
    Recursively find the original module if it is wrapped by a
    flash-attention-style `_orig_mod`.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m

def prepare_nucleotide_string(nucleotide_sequences, nuc_type, use_strandedness_token, single_strand):
    """
    Mimics the "golden code" approach:
      - Possibly prepends <ds-RNA>, <ss-RNA>, etc. if use_strandedness_token
      - Appends <EOS> after each strand
      - If single_strand, only keep the first strand
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

    # Always include the first strand
    nuc_str = f"{prefix}{nucleotide_sequences[0]}<EOS>"

    # If there is a second strand and we are not single-stranded, append it
    if len(nucleotide_sequences) == 2 and not single_strand:
        nuc_str += f"{prefix}{nucleotide_sequences[1]}<EOS>"

    return nuc_str

def prepare_sample(
    nuc_tokenizer, 
    prot_tokenizer, 
    peptide_sequence, 
    nucleotide_sequence
):
    """
    Encode the peptide as <protein>PEP<EOS>, and encode the nucleotide string
    (which presumably includes <RNA>/<DNA> and <EOS>).
    Returns: (prot_tokens, nuc_tokens)
    """
    # Protein tokens
    prot_tokens = prot_tokenizer.EncodeAsIds(f"<protein>{peptide_sequence}<EOS>")
    # Nucleotide tokens
    nuc_tokens = nuc_tokenizer.EncodeAsIds(nucleotide_sequence)

    return prot_tokens, nuc_tokens

def create_sample(
    nuc_tokenizer,
    prot_tokenizer,
    nucleotide_sequences,
    peptides,
    G0s,
    index=None
):
    """
    Pick a random index (or use the given one), then create a single training sample
    for the 'wild' sequence. Return the token IDs plus the target G0 value.

    Returns:
      prot_tokens_batch, nuc_tokens_batch, G0_batch
        each shape ~ [1, seq_len]
    """
    idx = np.random.randint(0, len(nucleotide_sequences)) if index is None else index
    # Encode
    prot_tokens, nuc_tokens = prepare_sample(
        nuc_tokenizer,
        prot_tokenizer,
        peptides[idx],
        nucleotide_sequences[idx]
    )
    g0_val = G0s[idx]

    return [prot_tokens], [nuc_tokens], [g0_val]

def evaluate_g0_predictions(
    prot_model,
    nuc_model,
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
    For each entry in test_data, we only do the 'wild' sequence G₀ prediction.
    (No mutated sequences or differences.)
    
    Returns:
      all_preds, all_gts
    """
    prot_model.eval()
    nuc_model.eval()
    head.eval()

    all_g0_pred = []
    all_g0_gt = []

    with torch.no_grad():
        for entry in test_data:
            pep_seq = entry["peptide_sequence"]
            nuc_type = entry["sequence_type"]
            # skip unknown type or zero G0 if desired
            if nuc_type not in ["RNA", "DNA"]:
                continue
            if entry["wild_G0"] == 0:
                continue

            # Prepare the input strings
            nuc_str = prepare_nucleotide_string(
                entry["wild_nucleotide_sequence"],
                nuc_type,
                use_strandedness_token,
                single_strand
            )

            # Encode
            prot_tokens, nuc_tokens = prepare_sample(prot_tokenizer, nuc_tokenizer, pep_seq, nuc_str)
            prot_tokens = torch.tensor([prot_tokens], device=device, dtype=torch.long)[:, :1024]
            nuc_tokens = torch.tensor([nuc_tokens], device=device, dtype=torch.long)[:, :1024]

            # Forward pass
            prot_emb = prot_model(prot_tokens, return_embeddings=True)[:, 0]  # [batch, d_prot]
            nuc_emb = nuc_model(nuc_tokens, return_embeddings=True)[:, 0]     # [batch, d_nuc]

            concat_emb = torch.cat([prot_emb, nuc_emb], dim=1)  # [batch, d_prot + d_nuc]
            # As in the golden code, we do a scaling factor for the dimension if desired:
            d_total = (prot_model.transformer.wte.weight.shape[-1] 
                       + nuc_model.transformer.wte.weight.shape[-1])
            # In the golden code we used (1024 / model_dim). We'll do the same approach:
            scale_factor = 1024 / d_total

            raw_pred = head(concat_emb).item()  # 1D
            pred_g0 = raw_pred * scale_factor * G0_std + G0_mean

            all_g0_pred.append(pred_g0)
            all_g0_gt.append(entry["wild_G0"])

    return all_g0_pred, all_g0_gt

# ----------------------------- MAIN SCRIPT -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-model OmniBioTA, adapted to match the 'golden code' style for cross-validation (wild-only).")

    parser.add_argument("--nuc_model_fn", type=str, required=True, help="Path to the pretrained nucleotide model file (.pt).")
    parser.add_argument("--protein_model_fn", type=str, required=True, help="Path to the pretrained protein model file (.pt).")
    parser.add_argument("--nucleotide_tokenizer", type=str, required=True, help="Path to the nucleotide tokenizer .model file.")
    parser.add_argument("--protein_tokenizer", type=str, required=True, help="Path to the protein tokenizer .model file.")

    # Model/training parameters
    parser.add_argument("--nuc_banned_token", type=int, default=2037)
    parser.add_argument("--prot_banned_token", type=int, default=2044)
    parser.add_argument("--prot_offset", type=int, default=2048)
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory_efficient config if available.")

    # Strandedness
    parser.add_argument("--single_strand", action="store_true", help="If True, only uses the first strand if double stranded.")
    parser.add_argument("--use_strandedness_token", action="store_true", help="If True, prefix sequences with <ds-XXX> or <ss-XXX> tokens.")

    # Training hyperparameters
    parser.add_argument("--num_accumulation_steps", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--num_splits", type=int, default=10, help="Number of cross-validation splits.")

    parser.add_argument("--save_dir", type=str, default="two_model_outputs", help="Where to save results and checkpoints.")
    args = parser.parse_args()

    # Make sure the save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Using args: {args}")
    set_seed(0)

    # ----------------- Load the tokenizers -----------------
    sp_nuc = spm.SentencePieceProcessor()
    sp_nuc.Load(args.nucleotide_tokenizer)
    nuc_tokenizer = TokenizerWrapper(sp_nuc, tokenizer_offset=0, banned_tokens=[1, 2, args.nuc_banned_token])

    sp_prot = spm.SentencePieceProcessor()
    sp_prot.Load(args.protein_tokenizer)
    prot_tokenizer = TokenizerWrapper(sp_prot, tokenizer_offset=args.prot_offset, banned_tokens=[1, 2, args.prot_banned_token])

    # ----------------- Load the two models -----------------
    # Nucleotide model
    chk_nuc = torch.load(args.nuc_model_fn, map_location="cpu")
    chk_nuc = recurse_load(chk_nuc).to(device="cpu", dtype=dtype)
    config_nuc = chk_nuc.config
    config_nuc.memory_efficient = args.memory_efficient
    sd_nuc = chk_nuc.state_dict()
    del chk_nuc

    nuc_model = OmniBioTA(config_nuc).to(device="cpu", dtype=dtype)
    nuc_model.load_state_dict(sd_nuc)
    nuc_model.to(device=device, dtype=dtype)
    nuc_model.eval()

    # Protein model
    chk_prot = torch.load(args.protein_model_fn, map_location="cpu")
    chk_prot = recurse_load(chk_prot).to(device="cpu", dtype=dtype)
    config_prot = chk_prot.config
    config_prot.memory_efficient = args.memory_efficient
    sd_prot = chk_prot.state_dict()
    del chk_prot

    protein_model = OmniBioTA(config_prot).to(device="cpu", dtype=dtype)
    protein_model.load_state_dict(sd_prot)
    protein_model.to(device=device, dtype=dtype)
    protein_model.eval()

    # Print param counts
    print(f"[Loaded NUC model] #Params: {nuc_model.get_num_params()/1e6:.2f}M")
    print(f"[Loaded PROT model] #Params: {protein_model.get_num_params()/1e6:.2f}M")

    # ----------------- Load cross-validation data & train-only data -----------------
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

    # ----------------- CROSS-VALIDATION -----------------
    pep_keys = list(grouped_sequences.keys())

    for fold in range(args.num_splits):
        print(f"\n=== FOLD {fold} of {args.num_splits} ===")

        # Build train/test sets
        nuc_train = []
        pep_train = []
        G0_train = []
        test_data = []

        for i, key in enumerate(pep_keys):
            # If i % num_splits == fold => test, else train
            for entry in grouped_sequences[key]:
                if i % args.num_splits == fold:
                    test_data.append(entry)
                else:
                    # only use valid data
                    if entry["sequence_type"] not in ["RNA", "DNA"]:
                        continue
                    if entry["wild_G0"] == 0:
                        continue
                    # prepare the text for training (the actual tokenization is done later)
                    nuc_str = prepare_nucleotide_string(
                        entry["wild_nucleotide_sequence"],
                        entry["sequence_type"],
                        args.use_strandedness_token,
                        args.single_strand
                    )
                    nuc_train.append(nuc_str)
                    pep_train.append(entry["peptide_sequence"])
                    G0_train.append(entry["wild_G0"])

        # Also add the train_only data
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
            nuc_train.append(nuc_str)
            pep_train.append(entry["peptide_sequence"])
            G0_train.append(entry["wild_G0"])

        # Compute mean/std
        G0_mean = np.mean(G0_train) if len(G0_train) > 0 else 0.0
        G0_std = np.std(G0_train) if len(G0_train) > 0 else 1.0
        if G0_std < 1e-9:
            G0_std = 1.0

        # Re-load the base models from CPU checkpoint so each fold starts from the same base
        base_nuc = OmniBioTA(config_nuc).to(device="cpu", dtype=dtype)
        base_nuc.load_state_dict(sd_nuc)
        base_nuc.to(device=device, dtype=dtype)
        base_nuc.eval()

        base_prot = OmniBioTA(config_prot).to(device="cpu", dtype=dtype)
        base_prot.load_state_dict(sd_prot)
        base_prot.to(device=device, dtype=dtype)
        base_prot.eval()

        # Create a new linear head
        d_prot = base_prot.transformer.wte.weight.shape[-1]
        d_nuc = base_nuc.transformer.wte.weight.shape[-1]
        head = torch.nn.Linear(d_prot + d_nuc, 1).to(device=device, dtype=dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

        # Build optimizer
        # We separate wte params for embed_lr, everything else for lr, plus the head
        # for head_lr
        param_groups = [
            # protein model
            {
                "params": [p for n, p in base_prot.named_parameters() if "wte" in n],
                "lr": args.embed_lr
            },
            {
                "params": [p for n, p in base_prot.named_parameters() if "wte" not in n],
                "lr": args.lr
            },
            # nuc model
            {
                "params": [p for n, p in base_nuc.named_parameters() if "wte" in n],
                "lr": args.embed_lr
            },
            {
                "params": [p for n, p in base_nuc.named_parameters() if "wte" not in n],
                "lr": args.lr
            },
            # head
            {
                "params": head.parameters(),
                "lr": args.head_lr
            }
        ]

        num_steps = int(args.num_epochs * len(nuc_train) / args.num_accumulation_steps)
        if num_steps < 1:  # guard
            num_steps = 1

        optimizer = torch.optim.AdamW(param_groups)
        # For OneCycleLR, we pass a list for max_lr that matches the param_groups order:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.embed_lr, args.lr, args.embed_lr, args.lr, args.head_lr],
            total_steps=num_steps,
            pct_start=0.05
        )

        # ----------------- TRAIN (wild-seq G₀) -----------------
        print(f"Fold {fold}: Starting training with {len(nuc_train)} samples, {num_steps} steps.")
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(args.num_accumulation_steps):
                prot_batch, nuc_batch, g0_batch = create_sample(
                    nuc_tokenizer,
                    prot_tokenizer,
                    nuc_train,
                    pep_train,
                    G0_train
                )
                # g0_batch => standardize
                y_arr = (np.array(g0_batch) - G0_mean) / (G0_std + 1e-9)

                X_prot = torch.tensor(prot_batch, device=device, dtype=torch.long)[:, :1024]
                X_nuc = torch.tensor(nuc_batch, device=device, dtype=torch.long)[:, :1024]
                y_tensor = torch.tensor(y_arr, device=device, dtype=dtype)

                emb_prot = base_prot(X_prot, return_embeddings=True)[:, 0]  # [B, d_prot]
                emb_nuc = base_nuc(X_nuc, return_embeddings=True)[:, 0]    # [B, d_nuc]
                concat_emb = torch.cat([emb_prot, emb_nuc], dim=1)

                # scale factor if you want, or do it inside the loss. Let's match golden code:
                d_total = (base_prot.transformer.wte.weight.shape[-1]
                           + base_nuc.transformer.wte.weight.shape[-1])
                scale_factor = 1024 / d_total

                pred = head(concat_emb) * scale_factor
                loss = ((pred - y_tensor) ** 2).mean() / args.num_accumulation_steps
                loss.backward()

                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            pbar.set_description(f"[Fold {fold}] Step {step}, Loss={total_loss:.4f}")

        pbar.close()

        # ----------------- EVALUATE -----------------
        print(f"[Fold {fold}] Evaluating on test set (#entries={len(test_data)})...")
        all_pred, all_gt = evaluate_g0_predictions(
            prot_model=base_prot,
            nuc_model=base_nuc,
            head=head,
            test_data=test_data,
            nuc_tokenizer=nuc_tokenizer,
            prot_tokenizer=prot_tokenizer,
            G0_mean=G0_mean,
            G0_std=G0_std,
            use_strandedness_token=args.use_strandedness_token,
            single_strand=args.single_strand
        )
        all_pred = np.array(all_pred)
        all_gt = np.array(all_gt)

        if len(all_pred) > 0 and len(all_gt) > 0:
            pcc = pearsonr(all_gt, all_pred)[0]
            mae = np.mean(np.abs(all_gt - all_pred))
        else:
            pcc, mae = 0.0, 0.0

        print(f"[Fold {fold}] G0 PCC={pcc:.4f}, MAE={mae:.4f}")

        # Save the results
        out_json = {
            "fold": fold,
            "num_train": len(nuc_train),
            "num_test": len(test_data),
            "G0_predictions": all_pred.tolist(),
            "G0_ground_truths": all_gt.tolist(),
            "g0_pcc": pcc,
            "g0_mae": mae,
        }
        with open(os.path.join(args.save_dir, f"fold{fold}.jsonl"), "w") as f_out:
            json.dump(out_json, f_out)
            f_out.write("\n")

        # Optionally save the model
        torch.save(base_prot.cpu(), os.path.join(args.save_dir, f"prot_model_fold{fold}.pt"))
        torch.save(base_nuc.cpu(), os.path.join(args.save_dir, f"nuc_model_fold{fold}.pt"))
        torch.save(head.cpu(), os.path.join(args.save_dir, f"head_fold{fold}.pt"))

        # Cleanup
        del base_prot, base_nuc, head, optimizer, scheduler
        torch.cuda.empty_cache()

    print("All folds complete!")


if __name__ == "__main__":
    main()
