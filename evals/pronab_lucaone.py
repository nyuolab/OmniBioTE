import argparse
import json
import random
import os

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr

from transformers import AutoModel

# ----------------------------- DEVICE/DTYPE SETUP -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# ----------------------------- ALPHABET (TOKENIZER) -----------------------------
import itertools
from typing import Sequence, List

class Alphabet(object):
    def __init__(
            self,
            standard_toks: Sequence[str] = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*'],
            prepend_toks: Sequence[str] = ['[PAD]', '[UNK]'],
            append_toks: Sequence[str] = ['[CLS]', '[SEP]', '[MASK]'],
            prepend_bos: bool = True,
            append_eos: bool = True
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.append_toks)
        self.all_toks.extend(self.standard_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["[UNK]"]
        self.padding_idx = self.get_idx("[PAD]")
        self.pad_token_id = self.padding_idx
        self.cls_idx = self.get_idx("[CLS]")
        self.mask_idx = self.get_idx("[MASK]")
        self.eos_idx = self.get_idx("[SEP]")
        self.all_special_tokens = prepend_toks + append_toks
        self.all_special_token_idx_list = [self.tok_to_idx[v] for v in self.all_special_tokens]
        self.unique_no_split_tokens = self.all_toks
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_predefined(cls, name: str):
        if name.lower() == "prot":
            standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']
        elif name.lower() == "gene":
            standard_toks = ['1', '2', '3', '4', '5', '.', '-', '*']
        elif name.lower() in ["gene_prot", "prot_gene"]:
            standard_toks = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']
        else:
            raise Exception("Not support tokenizer name: %s" % name)

        prepend_toks = ['[PAD]', '[UNK]']
        append_toks = ['[CLS]', '[SEP]', '[MASK]']
        prepend_bos = True
        append_eos = True

        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        A simple rule-based tokenizer: 
        Splits only on known special tokens or keeps them as separate tokens.
        """
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i > 0:
                    # re-insert the token
                    result.append(tok)
                if sub_text:
                    result.append(sub_text)
            return result

        tokenized_text = [text]
        # First, split on each special token if it appears
        for special in self.unique_no_split_tokens:
            pieces = []
            for t in tokenized_text:
                if t == special:
                    pieces.append(t)
                else:
                    pieces.extend(split_on_token(special, t))
            tokenized_text = pieces

        # Finally, flatten everything by whitespace if needed
        final_tokens = []
        for t in tokenized_text:
            if t not in self.unique_no_split_tokens:
                final_tokens.extend(t.split())
            else:
                final_tokens.append(t)
        return final_tokens

    def EncodeAsIds(self, text: str):
        tokens = self.tokenize(text)
        return [self.get_idx(tok) for tok in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        tokens = [self.get_tok(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.all_special_tokens]
        return "".join(tokens)

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

def prepare_nucleotide_string(nucleotide_sequences, seq_type):
    """
    This is a minimal stand-in for the "golden" code's approach.
    We just separate the two strands (if any) with [SEP].
    """
    if len(nucleotide_sequences) == 2:
        return f"{nucleotide_sequences[0]}[SEP]{nucleotide_sequences[1]}[SEP]"
    else:
        return f"{nucleotide_sequences[0]}[SEP]"

def prepare_sample(nuc_tokenizer, prot_tokenizer, peptide_sequence, nucleotide_sequence):
    """
    Encode [CLS]peptide[SEP] plus the gene tokens, as in your old script.
    Returns the final token IDs and the number of gene vs. protein tokens.
    """
    # Protein side:
    pep_encoded = prot_tokenizer.EncodeAsIds(f"[CLS]{peptide_sequence}[SEP]")
    # Nucleotide side:
    nuc_encoded = nuc_tokenizer.EncodeAsIds(nucleotide_sequence)

    # Combine
    combined = pep_encoded + nuc_encoded
    return combined, len(nuc_encoded), len(pep_encoded)

def create_sample(
    nuc_tokenizer,
    prot_tokenizer,
    nucleotide_sequences,
    peptides,
    G0s,
    index=None
):
    """
    Randomly pick an index (unless specified),
    then prepare the sample and the G0 target.
    """
    idx = np.random.randint(0, len(nucleotide_sequences)) if index is None else index
    x_tokens, nuc_len, pep_len = prepare_sample(
        nuc_tokenizer,
        prot_tokenizer,
        peptides[idx],
        nucleotide_sequences[idx],
    )
    return [x_tokens], [G0s[idx]], nuc_len, pep_len

def evaluate_dG_predictions(
    model,
    head,
    test_data,
    nuc_tokenizer,
    prot_tokenizer,
    G0_mean,
    G0_std,
    max_length=1280
):
    """
    Replicates the golden code logic for evaluation:
    For each entry in test_data, we only have a "wild" sequence,
    so we predict G0. Then we store the predicted G0 and ground truth.
    """
    model.eval()
    head.eval()

    all_g0_pred, all_g0_gt = [], []

    with torch.no_grad():
        for entry in test_data:
            pep_seq = entry["peptide_sequence"]
            seq_type = entry["sequence_type"]

            # Prepare the tokens
            nuc_str = prepare_nucleotide_string(entry["wild_nucleotide_sequence"], seq_type)
            tokens, nuc_len, pep_len = prepare_sample(nuc_tokenizer, prot_tokenizer, pep_seq, nuc_str)

            X_wild = torch.tensor([tokens], device=device, dtype=torch.long)[:, :max_length]
            token_type_ids = torch.tensor([[1] * pep_len + [0] * nuc_len], device=device, dtype=torch.long)[:, :max_length]

            # forward pass
            outputs = model(input_ids=X_wild, token_type_ids=token_type_ids)
            if hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0]  # [batch, hidden_dim]
            else:
                emb = outputs.hidden_states[:, 0]

            # final G0
            pred = head(emb).item() * G0_std + G0_mean

            all_g0_pred.append(pred)
            all_g0_gt.append(entry["wild_G0"])

    return all_g0_pred, all_g0_gt

# ----------------------------- MAIN SCRIPT -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Adapted code to match golden script style for LucaOne model.")
    parser.add_argument("--num_accumulation_steps", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--num_splits", type=int, default=10, help="Number of cross-validation splits.")
    parser.add_argument("--save_dir", type=str, default="outputs_luca", help="Directory to save results.")

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Using args: {args}")

    # Set seeds
    set_seed(0)

    # --------------------------- LOAD TOKENIZERS ---------------------------
    nuc_tokenizer = Alphabet.from_predefined("gene_prot")
    prot_tokenizer = nuc_tokenizer

    # --------------------------- LOAD MODEL ---------------------------
    # LucaOne from huggingface
    model = AutoModel.from_pretrained("Yuanfei/LucaOne", trust_remote_code=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    # --------------------------- LOAD DATA ---------------------------
    # 1) crossval_data
    crossval_data = []
    with open("../datasets/pronab_crossval_noddg.jsonl", "r") as f:
        for line in f:
            crossval_data.append(json.loads(line))

    # 2) train_only
    train_only = []
    with open("../datasets/pronab_crossval_train_only_noddg.jsonl", "r") as f:
        for line in f:
            train_only.append(json.loads(line))

    # shuffle data
    random.seed(0)
    random.shuffle(crossval_data)
    random.shuffle(train_only)

    # Group the crossval data by peptide sequence (like golden code)
    grouped_sequences = {}
    for entry in crossval_data:
        pep_seq = entry["peptide_sequence"]
        if pep_seq not in grouped_sequences:
            grouped_sequences[pep_seq] = []
        grouped_sequences[pep_seq].append(entry)

    # We'll do a 10-fold cross-validation on these groups
    pep_keys = list(grouped_sequences.keys())

    # --------------------- CROSS VALIDATION ---------------------
    for fold in range(args.num_splits):
        print(f"\n=== FOLD {fold} / {args.num_splits} ===")

        # For each fold, we create train/test sets:
        nuc_train = []
        pep_train = []
        G0_train = []
        test_set = []

        for i, key in enumerate(pep_keys):
            for entry in grouped_sequences[key]:
                # If i % args.num_splits == fold => test, else train
                if i % args.num_splits == fold:
                    test_set.append(entry)
                else:
                    # We'll gather the data for training
                    nuc_str = prepare_nucleotide_string(
                        entry["wild_nucleotide_sequence"], entry["sequence_type"]
                    )
                    nuc_train.append(nuc_str)
                    pep_train.append(entry["peptide_sequence"])
                    G0_train.append(entry["wild_G0"])

        # Also include the "train_only" data:
        for entry in train_only:
            nuc_str = prepare_nucleotide_string(entry["wild_nucleotide_sequence"], entry["sequence_type"])
            nuc_train.append(nuc_str)
            pep_train.append(entry["peptide_sequence"])
            G0_train.append(entry["wild_G0"])

        # Basic stats:
        G0_mean = np.mean(G0_train)
        G0_std = np.std(G0_train)

        # ------------------ RE-INIT MODEL & HEAD FOR THIS FOLD ------------------
        # You said your best practice is to keep the same base model weights
        # and re-init the final head. If you truly want a "fresh" base each time,
        # re-load from_pretrained. We'll do that to match the "golden code" approach
        # which re-loads the .pt each fold.

        base_model = AutoModel.from_pretrained("Yuanfei/LucaOne", trust_remote_code=True)
        base_model.to(device=device, dtype=dtype)
        base_model.eval()

        # Create a new linear head
        # Check hidden dimension from LucaOne. In your old code it was 2560.
        # If your model's hidden size is 2560, then do:
        head = torch.nn.Linear(2560, 1).to(device, dtype=dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

        # Setup training hyperparams
        num_accumulation_steps = args.num_accumulation_steps
        num_epochs = args.num_epochs
        lr = args.lr
        embed_lr = args.embed_lr
        head_lr = args.head_lr

        # Set up optimizer
        # If you want to separate "embedding" parameters from the rest, filter them by name:
        param_groups = [
            {
                "params": [p for n, p in base_model.named_parameters() if "embed" in n],
                "lr": embed_lr,
            },
            {
                "params": [p for n, p in base_model.named_parameters() if "embed" not in n],
                "lr": lr,
            },
            {
                "params": head.parameters(),
                "lr": head_lr,
            },
        ]

        # Number of steps is (epochs * data_size) / accumulation
        num_steps = int(num_epochs * len(nuc_train) / num_accumulation_steps)
        if num_steps < 1:
            num_steps = 1

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[embed_lr, lr, head_lr],
            total_steps=num_steps,
            pct_start=0.05
        )

        # ---------------------- TRAIN (PRONAB PRE-TRAIN) ----------------------
        print("Starting fold training...")
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(num_accumulation_steps):
                X_list, g0_list, nuc_len, pep_len = create_sample(
                    nuc_tokenizer,
                    prot_tokenizer,
                    nuc_train,
                    pep_train,
                    G0_train,
                )
                target_vals = (np.array(g0_list) - G0_mean) / (G0_std + 1e-9)

                X_tensor = torch.tensor(X_list, device=device, dtype=torch.long)[:, :1280]
                token_type_ids = torch.tensor(
                    [[1] * pep_len + [0] * nuc_len], device=device, dtype=torch.long
                )[:, :1280]
                y_tensor = torch.tensor(target_vals, device=device, dtype=dtype)

                outputs = base_model(input_ids=X_tensor, token_type_ids=token_type_ids)
                # Again, confirm the correct final hidden representation:
                if hasattr(outputs, "last_hidden_state"):
                    emb = outputs.last_hidden_state[:, 0]  # [batch, hidden_dim]
                else:
                    emb = outputs.hidden_states[:, 0]

                pred = head(emb)
                loss = ((pred - y_tensor) ** 2).mean() / num_accumulation_steps
                loss.backward()

                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Step: {step}, Loss: {total_loss:.4f}")

        pbar.close()

        # ---------------------- EVALUATE ON TEST SET ----------------------
        if len(test_set) > 0:
            print("Evaluating on fold test set...")
            all_g0_pred, all_g0_gt = evaluate_dG_predictions(
                base_model, head, test_set, nuc_tokenizer, prot_tokenizer, G0_mean, G0_std
            )
            all_g0_pred = np.asarray(all_g0_pred)
            all_g0_gt = np.asarray(all_g0_gt)

            pcc_g0 = pearsonr(all_g0_gt, all_g0_pred)[0]
            mae_g0 = np.mean(np.abs(all_g0_gt - all_g0_pred))
            print(f"(Fold {fold}) G0 PCC = {pcc_g0:.4f}, G0 MAE = {mae_g0:.4f}")

            # Save fold results
            out_json = {
                "fold": fold,
                "dG_ground_truths": all_g0_gt.tolist(),
                "dG_predictions": all_g0_pred.tolist(),
                "dG_pcc": pcc_g0,
                "dG_MAE": mae_g0
            }
            with open(os.path.join(args.save_dir, f"fold{fold}.jsonl"), "w") as f:
                json.dump(out_json, f)
                f.write("\n")

            np.save(os.path.join(args.save_dir, f"fold_stats{fold}.npy"), np.array([pcc_g0, mae_g0]))

        # optionally save the model and head
        torch.save(base_model.cpu(), os.path.join(args.save_dir, f"model_fold{fold}.pt"))
        torch.save(head.cpu(), os.path.join(args.save_dir, f"head_fold{fold}.pt"))
        # Clean up
        del base_model
        torch.cuda.empty_cache()

    print("All folds complete!")


if __name__ == "__main__":
    main()
