import sys
sys.path.insert(0, '../training')
from loader import EOS_TOKEN, PAD_TOKEN
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from model import OmniBioTA, OmniBioTAConfig
from mup import set_base_shapes, MuAdamW
import json
from scipy.stats import pearsonr
import copy
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging = True
dtype = torch.bfloat16

tokenizer_fn = sys.argv[1]
model_fn = sys.argv[2]
output_suffix = sys.argv[3]
banned_tokens = [65533]

# seed random number generators
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# seed torch cuda
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def process_dataset(dataset, noise_floor=1e-15):
    nucleotide_sequences = []
    peptides = []

    Kd = []
    G0 = []

    for key in dataset:
        for item in dataset[key]["binding data"]:
            if item[2] == 0 or item[3] == 0:
                continue
            nucleotide_sequence = item[0]
            if item[1] == "RNA":
                nucleotide_sequence = "<RNA>" + nucleotide_sequence + "<EOS>"
            else:
                nucleotide_sequence = "<DNA>" + nucleotide_sequence + "<EOS>"
            
            nucleotide_sequences.append(nucleotide_sequence)
            peptides.append(dataset[key]['Sequence']) # we exclude the tags here because they are added in prepare_sample after mutation

            Kd.append(np.log10(item[2] + noise_floor*np.random.uniform() + noise_floor))
            G0.append(item[3])

    return nucleotide_sequences, peptides, Kd, G0

def prepare_sample(peptide_sequence, nucleotide_sequence, mutate=False):
    nucleotide_sequence = sp.EncodeAsIds(nucleotide_sequence)
    
    peptide_sequence = "<protein>" + peptide_sequence + "<EOS>"

    peptide_sequence = sp.EncodeAsIds(peptide_sequence)

    while True:
        contains_banned_token = False
        for banned_token in banned_tokens:
            if banned_token in nucleotide_sequence:
                nucleotide_sequence.remove(banned_token)
                contains_banned_token = True
            elif banned_token in peptide_sequence:
                peptide_sequence.remove(banned_token)
                contains_banned_token = True
        if not contains_banned_token:
            break
    
    return peptide_sequence + nucleotide_sequence

def create_sample(nucleotide_sequences, peptides, Kds, G0s, mutate=False, index=None):
    idx = np.random.randint(0, len(nucleotide_sequences)) if index is None else index

    return [prepare_sample(peptides[idx], nucleotide_sequences[idx], mutate=mutate)], [Kds[idx]], [G0s[idx]]

train_target = "G0" # "Kd" or "G0"
num_accumulation_steps = 256
num_epochs = 32

lr = 1e-4
embed_lr = 1e-3
head_lr = 1e-2

sp = spm.SentencePieceProcessor()
sp.Load(tokenizer_fn)

model = torch.load(model_fn, map_location="cpu").to(dtype).to(device)
model.eval()

print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

with open(f'../datasets/pronab_no_mutations.json', 'r') as f:
    train_set = json.load(f)

nuc_train, peptide_train, Kd_train, G0_train = process_dataset(train_set)

############ DECONTAMINATE TRAIN SET ############
test_dataset = []
with open('../datasets/mutation_data.jsonl', 'r') as f:
    for line in f:
        test_dataset.append(json.loads(line))

sequences = [test_dataset[i]["peptide_sequence"] for i in range(len(test_dataset))]

# group by peptide sequence
grouped_sequences = {}
for i in range(len(test_dataset)):
    if test_dataset[i]["peptide_sequence"] not in grouped_sequences:
        grouped_sequences[test_dataset[i]["peptide_sequence"]] = []
    grouped_sequences[test_dataset[i]["peptide_sequence"]].append(test_dataset[i])

original = len(nuc_train)
deleted = 0
for i in range(len(nuc_train)-1, -1, -1):
    if peptide_train[i] in grouped_sequences:
        del nuc_train[i]
        del peptide_train[i]
        del Kd_train[i]
        del G0_train[i]
        deleted += 1
print(f"Deleted {deleted} entries from training set out of {original}")
##############################################

Kd_mean = np.mean(Kd_train)
Kd_std = np.std(Kd_train)

G0_mean = np.mean(G0_train)
G0_std = np.std(G0_train)

head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(device).to(dtype)
head.weight.data = torch.zeros_like(head.weight.data)
head.bias.data = torch.zeros_like(head.bias.data)

param_groups = [
    {"params": [p for name, p in model.named_parameters() if "wte" in name], "lr": embed_lr},
    {"params": [p for name, p in model.named_parameters() if "wte" not in name], "lr": lr},
    {"params": head.parameters(), "lr": head_lr}
]

num_steps = int(num_epochs*len(nuc_train)/num_accumulation_steps)
optimizer = torch.optim.AdamW(param_groups)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, head_lr], total_steps=num_steps, pct_start=0.05)

last_test_pcc = 0.0
MAE = 0.0
pbar = tqdm(range(num_steps))
accs = []
print("Starting pre-training")
for step in pbar:
    optimizer.zero_grad()
    total_loss = 0.0
    for _ in range(num_accumulation_steps):
        X, Kd, G0 = create_sample(nuc_train, peptide_train, Kd_train, G0_train, mutate=True)
        target = Kd if train_target == "Kd" else G0
        target = np.asarray(target)
        target = (target - Kd_mean) / Kd_std if train_target == "Kd" else (target - G0_mean) / G0_std

        X = torch.tensor(X, device=device, dtype=torch.long)
        X = X[:, :1024]
        target = torch.tensor(target, device=device, dtype=dtype)

        embeddings = model(X, return_embeddings=True)[:, 0]
        output = head(embeddings)

        loss = ((output - target)**2).mean() / num_accumulation_steps
        loss.backward()

        total_loss += loss.item()
    
    optimizer.step()
    scheduler.step()

    pbar.set_description(f"Loss: {total_loss:.4f}")

torch.save(model, f"pronab_all_ft_{output_suffix}.pt")

test_dataset = []
with open('../datasets/mutation_data.jsonl', 'r') as f:
    for line in f:
        test_dataset.append(json.loads(line))

sequences = [test_dataset[i]["peptide_sequence"] for i in range(len(test_dataset))]

random.shuffle(sequences)

# group by peptide sequence
grouped_sequences = {}
for i in range(len(test_dataset)):
    if test_dataset[i]["peptide_sequence"] not in grouped_sequences:
        grouped_sequences[test_dataset[i]["peptide_sequence"]] = []
    grouped_sequences[test_dataset[i]["peptide_sequence"]].append(test_dataset[i])

model.cpu()
original_model = copy.deepcopy(model)
best_head = copy.deepcopy(head)

last_test_pcc = 0.0
MAE = 0.0

last_dG_pcc = 0.0
dG_MAE = 0.0

original_model = original_model.cpu()

num_accumulation_steps = 256
num_epochs = 256









print("Evaluating pre-trained model")
test_set = []
for i, key in enumerate(grouped_sequences.keys()):
    test_set += grouped_sequences[key]

model = copy.deepcopy(original_model)
model.to(device).to(dtype)
model.train()

wild_head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(device).to(dtype)

wild_head.weight.data = copy.deepcopy(best_head.weight.data)
wild_head.bias.data = copy.deepcopy(best_head.bias.data)

with torch.no_grad():
    model.eval()
    ground_truths = []
    predictions = []

    dG_predictions = []
    dG_ground_truths = []

    for i in range(0, len(test_set)):
        X_wild = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["wild_nucleotide_sequence"], mutate=False)
        X_mutated = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["mutated_nucleotide_sequence"], mutate=False)

        X_wild = torch.tensor([X_wild], device=device, dtype=torch.long)[:, :1024]
        X_mutated = torch.tensor([X_mutated], device=device, dtype=torch.long)[:, :1024]
        
        G0_wild = wild_head(model(X_wild, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean
        G0_mutated = wild_head(model(X_mutated, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean

        dG_predictions.extend([G0_wild, G0_mutated])
        dG_ground_truths.extend([test_set[i]["wild_G0"], test_set[i]["mutant_G0"]])

        difference = G0_mutated - G0_wild
        ground_truth_difference = test_set[i]["mutant_G0"] - test_set[i]["wild_G0"]

        ground_truths.append(ground_truth_difference)
        predictions.append(difference)

        pbar.set_description(f"(Testing {i}/{len(test_set)}) Loss: {total_loss:.4f}, test pcc: {last_test_pcc:.4f}, test MAE: {MAE:.4f}, dG pcc: {last_dG_pcc:.4f}, dG MAE: {dG_MAE:.4f}")

    ground_truths = np.asarray(ground_truths)
    predictions = np.asarray(predictions)

    dG_ground_truths = np.asarray(dG_ground_truths)
    dG_predictions = np.asarray(dG_predictions)

    accs = np.asarray(accs)

    last_test_pcc = pearsonr(ground_truths, predictions)[0]
    MAE = np.abs(ground_truths - predictions).mean()

    last_dG_pcc = pearsonr(dG_ground_truths, dG_predictions)[0]
    dG_MAE = np.abs(dG_ground_truths - dG_predictions).mean()

    with open(f"pronab-mutant-dual_{output_suffix}.jsonl", "a") as f:
        json.dump({"ground_truths": ground_truths.tolist(), "predictions": predictions.tolist(), 
                    "dG_ground_truths": dG_ground_truths.tolist(), "dG_predictions": dG_predictions.tolist(),
                    "pcc": last_test_pcc, "MAE": MAE,
                    "dG_pcc": last_dG_pcc, "dG_MAE": dG_MAE}, f)
        f.write("\n")












print("Starting cross-val")
for split in range(10):
    train_set = []
    test_set = []
    for i, key in enumerate(grouped_sequences.keys()):
        if i % 10 == split:
            test_set += grouped_sequences[key]
        train_set += grouped_sequences[key]

    model = copy.deepcopy(original_model)
    model.to(device).to(dtype)
    model.train()

    wild_head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], 1).to(device).to(dtype)

    wild_head.weight.data = copy.deepcopy(best_head.weight.data)
    wild_head.bias.data = copy.deepcopy(best_head.bias.data)

    # zero out wild_head parameters
    #wild_head.weight.data = torch.zeros_like(wild_head.weight.data)
    #wild_head.bias.data = torch.zeros_like(wild_head.bias.data)

    param_groups = [
        {"params": [p for name, p in model.named_parameters() if "wte" in name], "lr": embed_lr},
        {"params": [p for name, p in model.named_parameters() if "wte" not in name], "lr": lr},
        {"params": wild_head.parameters(), "lr": head_lr},
    ]

    num_steps = int(num_epochs*len(train_set)/num_accumulation_steps)
    optimizer = torch.optim.AdamW(param_groups)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, head_lr], total_steps=num_steps, pct_start=0.05)

    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(num_accumulation_steps):
            i = np.random.randint(0, len(train_set))
            X_wild = prepare_sample(train_set[i]["peptide_sequence"], train_set[i]["wild_nucleotide_sequence"], mutate=False)
            X_mutated = prepare_sample(train_set[i]["peptide_sequence"], train_set[i]["mutated_nucleotide_sequence"], mutate=False)

            X_wild = torch.tensor([X_wild], device=device, dtype=torch.long)[:, :1024]
            X_mutated = torch.tensor([X_mutated], device=device, dtype=torch.long)[:, :1024]
            
            G0_wild = wild_head(model(X_wild, return_embeddings=True)[:, 0]).reshape(-1) * G0_std + G0_mean
            G0_mutated = wild_head(model(X_mutated, return_embeddings=True)[:, 0]).reshape(-1) * G0_std + G0_mean

            difference = G0_mutated - G0_wild
            ground_truth_difference = train_set[i]["mutant_G0"] - train_set[i]["wild_G0"]

            loss = ((difference - ground_truth_difference)**2).mean() / num_accumulation_steps
            dg_loss = ((G0_wild - train_set[i]["wild_G0"])**2).mean() / num_accumulation_steps + ((G0_mutated - train_set[i]["mutant_G0"])**2).mean() / num_accumulation_steps

            loss += dg_loss

            loss.backward()

            total_loss += loss.item()
        
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}, last pcc: {last_test_pcc:.4f}, MAE: {MAE:.4f}, dG pcc: {last_dG_pcc:.4f}, dG MAE: {dG_MAE:.4f}")
        
        if step % 100 == 0:
            with torch.no_grad():
                model.eval()
                ground_truths = []
                predictions = []

                dG_predictions = []
                dG_ground_truths = []

                for i in range(0, len(test_set)):
                    X_wild = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["wild_nucleotide_sequence"], mutate=False)
                    X_mutated = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["mutated_nucleotide_sequence"], mutate=False)

                    X_wild = torch.tensor([X_wild], device=device, dtype=torch.long)[:, :1024]
                    X_mutated = torch.tensor([X_mutated], device=device, dtype=torch.long)[:, :1024]
                    
                    G0_wild = wild_head(model(X_wild, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean
                    G0_mutated = wild_head(model(X_mutated, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean

                    dG_predictions.extend([G0_wild, G0_mutated])
                    dG_ground_truths.extend([test_set[i]["wild_G0"], test_set[i]["mutant_G0"]])

                    difference = G0_mutated - G0_wild
                    ground_truth_difference = test_set[i]["mutant_G0"] - test_set[i]["wild_G0"]

                    ground_truths.append(ground_truth_difference)
                    predictions.append(difference)

                    pbar.set_description(f"(Testing {i}/{len(test_set)}) Loss: {total_loss:.4f}, test pcc: {last_test_pcc:.4f}, test MAE: {MAE:.4f}")

                ground_truths = np.asarray(ground_truths)
                predictions = np.asarray(predictions)
                accs = np.asarray(accs)

                dG_ground_truths = np.asarray(dG_ground_truths)
                dG_predictions = np.asarray(dG_predictions)

                last_test_pcc = pearsonr(ground_truths, predictions)[0]
                MAE = np.abs(ground_truths - predictions).mean()

                last_dG_pcc = pearsonr(dG_ground_truths, dG_predictions)[0]
                dG_MAE = np.abs(dG_ground_truths - dG_predictions).mean()

                model.train()
    
    with torch.no_grad():
        model.eval()
        ground_truths = []
        predictions = []

        dG_predictions = []
        dG_ground_truths = []

        for i in range(0, len(test_set)):
            X_wild = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["wild_nucleotide_sequence"], mutate=False)
            X_mutated = prepare_sample(test_set[i]["peptide_sequence"], test_set[i]["mutated_nucleotide_sequence"], mutate=False)

            X_wild = torch.tensor([X_wild], device=device, dtype=torch.long)[:, :1024]
            X_mutated = torch.tensor([X_mutated], device=device, dtype=torch.long)[:, :1024]
            
            G0_wild = wild_head(model(X_wild, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean
            G0_mutated = wild_head(model(X_mutated, return_embeddings=True)[:, 0]).reshape(-1).item() * G0_std + G0_mean

            dG_predictions.extend([G0_wild, G0_mutated])
            dG_ground_truths.extend([test_set[i]["wild_G0"], test_set[i]["mutant_G0"]])

            difference = G0_mutated - G0_wild
            ground_truth_difference = test_set[i]["mutant_G0"] - test_set[i]["wild_G0"]

            ground_truths.append(ground_truth_difference)
            predictions.append(difference)

            pbar.set_description(f"(Testing {i}/{len(test_set)}) Loss: {total_loss:.4f}, test pcc: {last_test_pcc:.4f}, test MAE: {MAE:.4f}, dG pcc: {last_dG_pcc:.4f}, dG MAE: {dG_MAE:.4f}")

        ground_truths = np.asarray(ground_truths)
        predictions = np.asarray(predictions)

        dG_ground_truths = np.asarray(dG_ground_truths)
        dG_predictions = np.asarray(dG_predictions)

        accs = np.asarray(accs)

        last_test_pcc = pearsonr(ground_truths, predictions)[0]
        MAE = np.abs(ground_truths - predictions).mean()

        last_dG_pcc = pearsonr(dG_ground_truths, dG_predictions)[0]
        dG_MAE = np.abs(dG_ground_truths - dG_predictions).mean()

        with open(f"pronab-mutant-dual_{output_suffix}.jsonl", "a") as f:
            json.dump({"ground_truths": ground_truths.tolist(), "predictions": predictions.tolist(), 
                       "dG_ground_truths": dG_ground_truths.tolist(), "dG_predictions": dG_predictions.tolist(),
                       "pcc": last_test_pcc, "MAE": MAE,
                       "dG_pcc": last_dG_pcc, "dG_MAE": dG_MAE}, f)
            f.write("\n")
        
    # delete everything and flush cuda memory
    del wild_head
    del model
    del optimizer
    del scheduler
    del param_groups
    torch.cuda.empty_cache()