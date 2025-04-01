import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Adjust sys.path to import custom modules
sys.path.insert(0, '../training/')
sys.path.insert(0, '../evals/')
from loader import EOS_TOKEN, PAD_TOKEN
from model import OmniBioTA

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

# Global constants and settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
TOKENIZER_FN = "/gpfs/data/oermannlab/users/chens59/OmniBioTA_home/tokenizers/bpe/uniref-2k.model"
DATASET_PATH = "../datasets/peptide-nucleotide-distances.json"

def set_seeds(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_distance_data(filepath):
    """Load the JSON dataset."""
    with open(filepath) as f:
        return json.load(f)


def extract_sequences(distance_data, threshold=8):
    """
    Extract peptide sequences, nucleotide sequences, and label arrays.
    Each entry uses the first key from its dictionaries.
    """
    peptides, nucleotides, labels = [], [], []
    for entry in distance_data:
        peptide = list(entry["peptide_sequences"].values())[0]
        nucleotide = list(entry["nucleotide_sequences"].values())[0]
        peptides.append(peptide)
        nucleotides.append(nucleotide)
        # Convert closest_nucleotides into a boolean array where distances <= 8 are True
        label = np.float32(np.asarray(entry["closest_nucleotides"]) <= threshold)
        labels.append(label)
    return peptides, nucleotides, labels


def clean_sequences(peptides, nucleotides, labels):
    """
    Remove entries where the length of the peptide sequence
    does not match the length of the label array.
    """
    cleaned_peptides, cleaned_nucleotides, cleaned_labels = [], [], []
    num_removed = 0
    for pep, nuc, lab in zip(peptides, nucleotides, labels):
        if len(pep) != len(lab):
            num_removed += 1
            continue
        cleaned_peptides.append(pep)
        cleaned_nucleotides.append(nuc)
        cleaned_labels.append(lab)
    print(f"Removed {num_removed} sequences")
    return cleaned_peptides, cleaned_nucleotides, cleaned_labels


def process_sequence(peptide_sequence, nucleotide_sequence, distance_data, sp):
    """
    Tokenize the peptide and nucleotide sequences and compute labels for each token.
    Returns:
      - tokenized: the concatenated token list with special tokens added.
      - token_labels: list of maximum distance values per token in the peptide.
      - protein_len: length of the tokenized peptide portion.
    """
    tokenized = sp.EncodeAsIds(peptide_sequence)
    token_lens = [len(sp.DecodeIds([t])) for t in tokenized]
    token_labels = []

    ptr = 0
    for length in token_lens:
        assert length > 0, "Token length must be greater than 0."
        label = np.max(distance_data[ptr:ptr + length])
        ptr += length
        token_labels.append(label)

    protein_len = len(tokenized)
    # Build the final tokenized sequence:
    # 18 as start token, then peptide tokens, EOS_TOKEN, token 4 as a separator,
    # followed by nucleotide tokens and final EOS_TOKEN.
    nucleotide_tokens = sp.EncodeAsIds(nucleotide_sequence)
    tokenized = [18] + tokenized + [EOS_TOKEN, 4] + nucleotide_tokens + [EOS_TOKEN]

    return tokenized, token_labels, protein_len


def unprocess_sequence(tokenized, token_labels, sp):
    """
    Reconstruct the original sequence and repeat labels for each character.
    This function repeats each label by the length of the token decoded string.
    """
    sequence = sp.DecodeIds(tokenized)
    labels = []
    for i, token in enumerate(tokenized):
        token_str = sp.DecodeIds([token])
        labels.extend([token_labels[i]] * len(token_str))
    assert len(sequence) == len(labels), f"{len(sequence)} != {len(labels)}"
    return sequence, labels


def group_by_peptide(peptides, nucleotides, labels):
    """
    Group the data by peptide sequence for per-peptide processing.
    Returns a dictionary mapping peptide sequences to lists of (peptide, nucleotide, label) tuples.
    """
    peptide_data = {}
    for pep, nuc, lab in zip(peptides, nucleotides, labels):
        if pep not in peptide_data:
            peptide_data[pep] = []
        peptide_data[pep].append((pep, nuc, lab))
    return peptide_data


def split_train_test(peptide_data, fold, sp):
    """
    For a given fold, split the data into training and test sets.
    Data for each peptide is shuffled, and one in every 10 examples (by index mod 10)
    is assigned to the test set.
    """
    X_train, Y_train, X_train_protein_lens = [], [], []
    X_test, Y_test, X_test_protein_lens = [], [], []
    for peptide, data_list in peptide_data.items():
        random.shuffle(data_list)
        for i, (pep, nuc, lab) in enumerate(data_list):
            tokenized, token_labels, protein_len = process_sequence(pep, nuc, lab, sp)
            if len(tokenized) > 1024:
                continue
            if i % 10 == fold:
                X_test.append(tokenized)
                Y_test.append(lab)
                X_test_protein_lens.append(protein_len)
            else:
                X_train.append(tokenized)
                Y_train.append(token_labels)
                X_train_protein_lens.append(protein_len)
    return X_train, Y_train, X_train_protein_lens, X_test, Y_test, X_test_protein_lens


def train_fold(model_fn, fold, X_train, Y_train, X_train_protein_lens,
               sp, num_epochs=32, num_accumulation_steps=256,
               lr=5e-5, embed_lr=1e-3, head_lr=1e-2):
    """
    Load the model and head, and train for one fold.
    Uses gradient accumulation and a OneCycleLR scheduler.
    """
    model = torch.load(model_fn, map_location="cpu").to(DTYPE).to(DEVICE)
    model.train()
    head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(DEVICE).to(DTYPE)

    param_groups = [
        {"params": [p for name, p in model.named_parameters() if "wte" in name], "lr": embed_lr},
        {"params": [p for name, p in model.named_parameters() if "wte" not in name], "lr": lr},
        {"params": head.parameters(), "lr": head_lr}
    ]

    num_steps = int(num_epochs * len(X_train) / num_accumulation_steps)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, head_lr],
                                                      total_steps=num_steps, pct_start=0.05)

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(num_accumulation_steps):
            idx = np.random.randint(0, len(X_train))
            x = torch.tensor(X_train[idx], device=DEVICE, dtype=torch.long).unsqueeze(0)
            y = torch.tensor(Y_train[idx], device=DEVICE, dtype=DTYPE)
            embeddings = model(x, return_embeddings=True)
            y_pred = torch.sigmoid(head(embeddings) * 1024 / model.transformer.wte.weight.shape[-1])[:, 1:X_train_protein_lens[idx] + 1]
            loss = F.binary_cross_entropy(y_pred.flatten(), y)
            loss /= num_accumulation_steps
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"Loss: {total_loss:.4f}")
    return model, head


def evaluate_fold(model, head, X_test, Y_test, X_test_protein_lens, sp):
    """
    Evaluate the trained model on the test set and compute the ROC AUC.
    """
    model.eval()
    all_preds = []
    all_truths = []
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], device=DEVICE, dtype=torch.long).unsqueeze(0)
        y_true = Y_test[i]
        with torch.no_grad():
            embeddings = model(x, return_embeddings=True)
            y_pred = torch.sigmoid(head(embeddings) * 1024 / model.transformer.wte.weight.shape[-1])[:, 1:X_test_protein_lens[i] + 1]
            # Process predictions to align with original token lengths
            _, y_pred_processed = unprocess_sequence(X_test[i][1:X_test_protein_lens[i] + 1],
                                                     y_pred.flatten().cpu().float().numpy(),
                                                     sp)
        all_truths.extend(y_true.tolist())
        all_preds.extend(y_pred_processed)
    fpr, tpr, _ = roc_curve(all_truths, all_preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc, all_preds, all_truths


def main():
    if len(sys.argv) < 8:
        print("Usage: python script.py <model_fn> <name_suffix> <lr> <embed_lr> <head_lr> <tokenizer_offset> <distance_threshold>")
        sys.exit(1)
    
    model_fn = sys.argv[1]
    name_suffix = sys.argv[2]
    lr = float(sys.argv[3])
    embed_lr = float(sys.argv[4])
    head_lr = float(sys.argv[5])
    tokenizer_offset = int(sys.argv[6])
    threshold = int(sys.argv[7])

    # Set seeds and load tokenizer/data
    set_seeds(0)
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_FN)
    sp = TokenizerWrapper(sp, tokenizer_offset, [0, 1, 2044])
    distance_data = load_distance_data(DATASET_PATH)
    peptides, nucleotides, labels = extract_sequences(distance_data, threshold=threshold)
    peptides, nucleotides, labels = clean_sequences(peptides, nucleotides, labels)
    assert len(peptides) == len(nucleotides) == len(labels)

    # Group data by peptide for splitting
    peptide_data = group_by_peptide(peptides, nucleotides, labels)

    # Hyperparameters
    num_epochs = 32
    num_accumulation_steps = 256

    print(f"Training {model_fn} with {num_epochs} epochs, {num_accumulation_steps} accumulation steps, "
          f"lr={lr}, embed_lr={embed_lr}, head_lr={head_lr}, with output suffix {name_suffix}")

    # Run 10-fold training and evaluation
    for fold in range(10):
        (X_train, Y_train, X_train_protein_lens,
         X_test, Y_test, X_test_protein_lens) = split_train_test(peptide_data, fold, sp)

        model, head = train_fold(model_fn, fold, X_train, Y_train, X_train_protein_lens, sp,
                                 num_epochs, num_accumulation_steps, lr, embed_lr, head_lr)
        roc_auc, all_preds, all_truths = evaluate_fold(model, head, X_test, Y_test, X_test_protein_lens, sp)
        print(f"Fold {fold + 1} ROC AUC: {roc_auc}")

        # Save evaluation results for this fold
        with open(f"pdb_contact_eval_{name_suffix}.jsonl", "a") as f:
            json_line = json.dumps({
                                    "fold": fold, 
                                    "preds": [float(pred) for pred in all_preds],
                                    "truths": [float(truth) for truth in all_truths]
                                })
            f.write(json_line + "\n")


if __name__ == "__main__":
    main()