import sys
sys.path.insert(0, '../training/')
sys.path.insert(0, '../evals/')
from loader import EOS_TOKEN, PAD_TOKEN
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from model import OmniBioTA
import torch.nn.functional as F
import json
import random
from sklearn.metrics import roc_curve, auc

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

tokenizer_fn = "/gpfs/home/chens59/OmniBioTA/tokenizers/bpe/mixed_bpe.model"
model_fn = sys.argv[1]
name_suffix = sys.argv[2]
banned_tokens = [65533]

# seed random number generator
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

sp = spm.SentencePieceProcessor(model_file=tokenizer_fn)

# load dataset
with open("../datasets/peptide-nucleotide-distances.json") as f:
    distance_data = json.load(f)

peptide_sequences = [distance_data[i]["peptide_sequences"][list(distance_data[i]["peptide_sequences"].keys())[0]] for i in range(len(distance_data))]
nucleotide_sequences = [distance_data[i]["nucleotide_sequences"][list(distance_data[i]["nucleotide_sequences"].keys())[0]] for i in range(len(distance_data))]
labels = [np.float32(np.asarray(distance_data[i]["closest_nucleotides"]) <= 8) for i in range(len(distance_data))]

assert len(peptide_sequences) == len(labels) == len(nucleotide_sequences)

num_removed = 0
for i in range(len(peptide_sequences) - 1, -1, -1):
    if len(peptide_sequences[i]) != len(labels[i]):
        del peptide_sequences[i]
        del labels[i]
        del nucleotide_sequences[i]
        num_removed += 1

print(f"Removed {num_removed} sequences")

def process_sequence(peptide_sequence, nucleotide_sequence, distance_data):
    tokenized = sp.encode(peptide_sequence)
    tokenized = [t for t in tokenized if t not in banned_tokens]
    token_lens = [len(sp.decode([t])) for t in tokenized]
    token_labels = []

    ptr = 0
    for i in range(len(token_lens)):
        assert token_lens[i] > 0
        label = np.max(distance_data[ptr:ptr + token_lens[i]])
        ptr += token_lens[i]
        token_labels.append(label)

    assert len(token_labels) == len(tokenized), f"{len(token_labels)} != {len(tokenized)}"

    protein_len = len(tokenized)

    tokenized = [18] + tokenized + [EOS_TOKEN, 4] + [t for t in sp.encode(nucleotide_sequence) if t not in banned_tokens] + [EOS_TOKEN]
    
    return tokenized, token_labels, protein_len

def unprocess_sequence(tokenized, token_labels):
    sequence = sp.decode(tokenized)
    labels = []

    for i in range(len(tokenized)):
        labels += [token_labels[i]] * len(sp.decode([tokenized[i]]))

    assert len(sequence) == len(labels), f"{len(sequence)} != {len(labels)}"
    return sequence, labels

# group data by peptide
peptide_data = {}
for i in range(len(peptide_sequences)):
    peptide = peptide_sequences[i]
    if peptide not in peptide_data:
        peptide_data[peptide] = []
    peptide_data[peptide].append((peptide_sequences[i], nucleotide_sequences[i], labels[i]))

num_accumulation_steps = 256
num_epochs = 32

lr = 5e-5
embed_lr = 1e-3
head_lr = 1e-2

print(f"Training {model_fn} with {num_epochs} epochs, {num_accumulation_steps} accumulation steps, lr={lr}, embed_lr={embed_lr}, head_lr={head_lr}, with output suffix {name_suffix}")

for fold in range(10):
    X_train = []
    X_train_protein_lens = []
    Y_train = []

    X_test = []
    X_test_protein_lens = []
    Y_test = []

    for peptide in peptide_data:
        data = peptide_data[peptide]
        random.shuffle(data)

        for i in range(len(data)):
            tokenized, token_labels, protein_len = process_sequence(data[i][0], data[i][1], data[i][2])
            if len(tokenized) > 1024:
                continue
            if i % 10 == fold:
                X_test.append(tokenized)
                Y_test.append(data[i][2])
                X_test_protein_lens.append(protein_len)
            else:
                X_train.append(tokenized)
                Y_train.append(token_labels)
                X_train_protein_lens.append(protein_len)

    model = torch.load(model_fn, map_location="cpu").to(dtype).to(device)
    model.eval()

    head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(device).to(dtype)

    param_groups = [
        {"params": [p for name, p in model.named_parameters() if "wte" in name], "lr": embed_lr},
        {"params": [p for name, p in model.named_parameters() if "wte" not in name], "lr": lr},
        {"params": head.parameters(), "lr": head_lr}
    ]

    num_steps = int(num_epochs*len(X_train)/num_accumulation_steps)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, head_lr], total_steps=num_steps, pct_start=0.05)

    pbar = tqdm(range(num_steps))
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0.0
        for _ in range(0, num_accumulation_steps):
            idx = np.random.randint(0, len(X_train))
            x = torch.tensor(X_train[idx], device=device, dtype=torch.long).unsqueeze(0)
            y = torch.tensor(Y_train[idx], device=device, dtype=dtype)

            x = model(x, return_embeddings=True)
            y_pred = torch.sigmoid(head(x))[:, 1:X_train_protein_lens[idx] + 1] 

            loss = F.binary_cross_entropy(y_pred.flatten(), y)

            loss /= num_accumulation_steps
            loss.backward()

            total_loss += loss.item()
        
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}")

    model.eval()

    all_preds = []
    all_truths = []

    for i in range(0, len(X_test)):
        x = torch.tensor(X_test[i], device=device, dtype=torch.long).unsqueeze(0)
        y = Y_test[i]

        with torch.no_grad():
            x = model(x, return_embeddings=True)
            y_pred = torch.sigmoid(head(x))[:, 1:X_test_protein_lens[i] + 1]

            _, y_pred = unprocess_sequence(X_test[i][1:X_test_protein_lens[i]+1], y_pred.flatten().cpu().float().numpy())
        
        all_truths += y.tolist()
        all_preds += y_pred

    fpr, tpr, _ = roc_curve(all_truths, all_preds)
    roc_auc = auc(fpr, tpr)
    print(f"Fold {fold+1} ROC AUC: {roc_auc}")

    with open(f"pdb_contact_eval_{name_suffix}.jsonl", "a") as f:
        all_preds = np.asarray(all_preds).tolist()
        all_truths = np.asarray(all_truths).tolist()
        f.write(json.dumps({"fold": fold, "preds": all_preds, "truths": all_truths}) + "\n")