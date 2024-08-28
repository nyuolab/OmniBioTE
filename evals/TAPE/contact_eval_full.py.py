import json
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '../training/')
sys.path.insert(0, '../evals/')
from loader import EOS_TOKEN, PAD_TOKEN
import torch
import sentencepiece as spm
from model import OmniBioTA, OmniBioTAConfig
import torch.nn.functional as F
import pickle
import wandb
from copy import deepcopy
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# file paths
tokenizer_fn = sys.argv[1]
model_fn = sys.argv[2]
banned_token = int(sys.argv[3])
wandb_prefix = sys.argv[4]
data_dir = "/home/sully/OmniBioTA/datasets/TAPE/data"#"/gpfs/home/chens59/OmniBioTA/datasets/TAPE/data"

print(f"Tokenizer: {tokenizer_fn}")
print(f"Model: {model_fn}")
print(f"Banned token: {banned_token}")
print(f"Wandb prefix: {wandb_prefix}")

logging = True
num_accumulation_steps = 128
num_epochs = 128
head_dim = 128
num_resnet_blocks = 8
num_tests = 256 # how many times to evaluate the model on the val set throughout training

lr = 1e-4
embed_lr = 1e-4
head_lr = 1e-3
contact_pred_lr = 1e-3

def load_data(split, base_dir):
    with open(os.path.join(base_dir, f"proteinnet/proteinnet_{split}.json"), "r") as f:
        data = json.load(f)

    sequences = []
    distance_matrices = []
    masks = []
    contact_maps = []

    # these masks are used to mask the objectives for the different ranges
    # CASP targets are divided into short, medium, and long range contacts, and we care only about medium and long range contacts
    # medium is 12-23 residues apart, long is 24 and greater residues apart
    medium_range_mask = []
    long_range_mask = []

    for item in tqdm(data):
        sequences.append(item["primary"])
        
        tertiary = np.asarray(item["tertiary"]) # positions of the amino acids in 3D space
        distance_matrix = np.zeros((len(tertiary), len(tertiary)))
        distance_matrix = np.linalg.norm(tertiary[:, np.newaxis, :] - tertiary[np.newaxis, :, :], axis=2)
        distance_matrices.append(distance_matrix)

        mask = np.float32(item["valid_mask"]).reshape(-1, 1)
        masks.append(mask @ mask.T)

        index_distance = np.abs(np.arange(len(tertiary)) - np.arange(len(tertiary))[:, np.newaxis])
        medium_range_mask.append(np.logical_and(index_distance >= 12, index_distance <= 23))
        long_range_mask.append(index_distance >= 24)
    
        contact_map = distance_matrix < 8
        contact_maps.append(contact_map)

    return sequences, contact_maps, masks, medium_range_mask, long_range_mask

train_sequences, train_contact_maps, train_masks, _, _ = load_data("train", data_dir)
val_sequences, val_contact_maps, val_masks, val_medium_range_masks, val_long_range_masks = load_data("valid", data_dir)
test_sequences, test_contact_maps, test_masks, test_medium_range_masks, test_long_range_masks = load_data("test", data_dir)

sp = spm.SentencePieceProcessor()
sp.Load(tokenizer_fn)

def process_sample(sequence, contact_map, mask):
    '''
    Tokenizes a protein sequence and converts the contact map to the tokenized space.
    This is a relatively convoluted process because the contact map is a matrix of distances between amino acids,
    and the tokenized space is a sequence of tokens. The contact map is converted to the tokenized space by taking the maximum distance
    between amino acids that are represented by the same token.

    Because the contact map is also masked, we need to keep track of the mask in the tokenized space as well.
    If a token contains even a single unmaksed amino acid, the token is considered unmasked, so that we can generate predictions for it.

    Lastly, we have the non_short_range mask, which is used to determine which amino acids are not short range contacts (i.e., we need to predict them).
    If even a single amino acid in a token is not a short range contact, the token is considered not a short range contact.
    '''
    index_distance = np.abs(np.arange(len(contact_map)) - np.arange(len(contact_map))[:, np.newaxis])
    non_short_range = index_distance >= 12

    tokenized = sp.encode("<protein>" + sequence + "<EOS>")
    tokenized = [t for t in tokenized if t != banned_token]

    token_lens = [len(sp.decode([t])) for t in tokenized]
    
    masked_contact_map = contact_map * mask

    tokenized_contact_map = np.zeros((len(tokenized)-2, len(tokenized)-2))
    tokenized_mask = np.zeros((len(tokenized)-2, len(tokenized)-2))
    tokenized_non_short_range = np.zeros((len(tokenized)-2, len(tokenized)-2))

    idx_i = 0
    for i, token_len_x in enumerate(token_lens[1:-1]): # ignore the <protein> and <EOS> tokens
        idx_j = 0
        for j, token_len_y in enumerate(token_lens[1:-1]):
            tokenized_contact_map[i, j] = np.max(masked_contact_map[idx_i: idx_i + token_len_x, idx_j:idx_j + token_len_y])
            tokenized_mask[i, j] = np.max(mask[idx_i:idx_i + token_len_x, idx_j:idx_j + token_len_y])
            tokenized_non_short_range[i, j] = np.max(non_short_range[idx_i:idx_i + token_len_x, idx_j:idx_j + token_len_y])
            
            idx_j += token_len_y
            
        idx_i += token_len_x

    return tokenized, tokenized_contact_map, tokenized_mask, tokenized_non_short_range

def inverse_process_sample(tokenized, pred_contact_map):
    '''
    This function takes the tokenized sequence and the predicted contact map and converts it back to the original space.
    This is considerably simpler than the forward process.
    '''
    token_lens = [len(sp.decode([t])) for t in tokenized[1:-1]]
    contact_map = np.zeros((np.sum(token_lens), np.sum(token_lens)))

    idx_i = 0
    for i, token_len_x in enumerate(token_lens[1:-1]):
        idx_j = 0
        for j, token_len_y in enumerate(token_lens[1:-1]):
            contact_map[idx_i:idx_i + token_len_x, idx_j:idx_j + token_len_y] = pred_contact_map[i, j]
            idx_j += token_len_y
        idx_i += token_len_x

    return contact_map

if os.path.exists(os.path.join(data_dir, "proteinnet/processed_train_bpe_mixed.pkl")):
    with open(os.path.join(data_dir, "proteinnet/processed_train_bpe_mixed.pkl"), "rb") as f:
        train_sequences_tokenized, train_contact_maps_tokenized, train_masks_tokenized, train_non_short_range = pickle.load(f)
else:
    print("Preprocessing train data...")
    train_sequences_tokenized = []
    train_contact_maps_tokenized = []
    train_masks_tokenized = []
    train_non_short_range = []

    for sequence, contact_map, mask in tqdm(zip(train_sequences, train_contact_maps, train_masks), total=len(train_sequences)):
        tokenized, tokenized_contact_map, tokenized_mask, tokenized_non_short_range = process_sample(sequence, contact_map, mask)
        train_sequences_tokenized.append(tokenized)
        train_contact_maps_tokenized.append(tokenized_contact_map)
        train_masks_tokenized.append(tokenized_mask)
        train_non_short_range.append(tokenized_non_short_range)
    
    with open(os.path.join(data_dir, "proteinnet/processed_train_bpe_mixed.pkl"), "wb") as f:
        pickle.dump((train_sequences_tokenized, train_contact_maps_tokenized, train_masks_tokenized, train_non_short_range), f)

del train_sequences, train_contact_maps, train_masks # free up some memory since we don't really need these anymore

val_sequences_tokenized = []

for sequence, contact_map, mask in tqdm(zip(val_sequences, val_contact_maps, val_masks), total=len(val_sequences)):
    tokenized, tokenized_contact_map, tokenized_mask, _ = process_sample(sequence, contact_map, mask)
    val_sequences_tokenized.append(tokenized)

test_sequences_tokenized = []
for sequence, contact_map, mask in tqdm(zip(test_sequences, test_contact_maps, test_masks), total=len(test_sequences)):
    tokenized, tokenized_contact_map, tokenized_mask, _ = process_sample(sequence, contact_map, mask)
    test_sequences_tokenized.append(tokenized)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = torch.load(model_fn, map_location=device)
model.to(dtype)

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        return out + x
    
class ContactPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, resnet_blocks=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.resnet_blocks = torch.nn.ModuleList([ResNetBlock(out_channels, out_channels) for _ in range(resnet_blocks)])
        self.conv2 = torch.nn.Conv2d(out_channels, 1, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        for block in self.resnet_blocks:
            out = block(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return out

'''
def compute_precision(pred_contact_map, contact_map, mask, range_mask):
    pred_contact_map = pred_contact_map * mask * range_mask
    contact_map = contact_map * mask * range_mask

    if contact_map.sum() == 0:
        return None

    # threshold
    pred_contact_map = pred_contact_map > 0.5

    # compute precision
    true_positives = (pred_contact_map * contact_map).sum()
    false_positives = (pred_contact_map * (1 - contact_map)).sum()

    precision = true_positives / (true_positives + false_positives + 1e-6)

    return precision

def compute_AUPRC(pred_contact_map, contact_map, mask, range_mask):
    pred_contact_map = pred_contact_map * mask * range_mask
    contact_map = contact_map * mask * range_mask

    if contact_map.sum() == 0:
        return None

    # threshold
    pred_contact_map = pred_contact_map > 0.5

    # compute precision
    true_positives = (pred_contact_map * contact_map).sum()
    false_positives = (pred_contact_map * (1 - contact_map)).sum()

    precision = true_positives / (true_positives + false_positives + 1e-6)

    # compute AUPRC
    precision, recall, _ = precision_recall_curve(contact_map.flatten(), pred_contact_map.flatten())
    auprc = auc(recall, precision)

    return auprc

def evaluate_dataset(model, head, contact_predictor, sequences_tokenized, contact_maps, masks, medium_range_masks, long_range_masks):
    long_precisions = []
    medium_precisions = []

    long_auprcs = []
    medium_auprcs = []

    for idx in range(len(sequences_tokenized)):
        sequence = torch.tensor(sequences_tokenized[idx], device=device, dtype=torch.long)

        if sequence.shape[0] <= 5:
            continue

        contact_map = torch.tensor(sequences_tokenized[idx], device=device, dtype=dtype)

        embeddings = model(sequence.unsqueeze(0), return_embeddings=True)[:, 1:-1]
        embeddings = head(embeddings)

        features = torch.cat([embeddings[:, :, None, :].expand(-1, -1, embeddings.shape[1], -1), embeddings[:, None, :, :].expand(-1, embeddings.shape[1], -1, -1)], dim=-1)
        features = features.permute(0, 3, 1, 2)

        pred_contact_map = contact_predictor(features)

        pred_contact_map = inverse_process_sample(sequences_tokenized[idx], pred_contact_map.float().squeeze(0).squeeze(0).detach().cpu().numpy())

        contact_map = contact_maps[idx]
        mask = masks[idx]
        contact_map = contact_map * mask

        precision = compute_precision(pred_contact_map, contact_map, mask, medium_range_masks[idx])
        if precision is not None:
            medium_precisions.append(precision)
            medium_auprcs.append(compute_AUPRC(pred_contact_map, contact_map, mask, medium_range_masks[idx]))

        precision = compute_precision(pred_contact_map, contact_map, mask, long_range_masks[idx])
        if precision is not None:
            long_precisions.append(precision)
            long_auprcs.append(compute_AUPRC(pred_contact_map, contact_map, mask, long_range_masks[idx]))

    return medium_precisions, long_precisions, medium_auprcs, long_auprcs
'''

def evaluate_dataset(model, head, contact_predictor, sequences_tokenized, contact_maps, masks, medium_range_masks, long_range_masks):
    long_precision = 0
    medium_precision = 0
    long_auprc = 0
    medium_auprc = 0

    medium_ground_truth = []
    medium_predictions = []
    medium_probs = []

    long_ground_truth = []
    long_predictions = []
    long_probs = []

    for idx in range(len(sequences_tokenized)):
        sequence = torch.tensor(sequences_tokenized[idx], device=device, dtype=torch.long)

        if sequence.shape[0] <= 5:
            continue

        embeddings = model(sequence.unsqueeze(0), return_embeddings=True)[:, 1:-1]
        embeddings = head(embeddings)

        features = torch.cat([embeddings[:, :, None, :].expand(-1, -1, embeddings.shape[1], -1), embeddings[:, None, :, :].expand(-1, embeddings.shape[1], -1, -1)], dim=-1)
        features = features.permute(0, 3, 1, 2)

        pred_contact_map = contact_predictor(features)

        pred_contact_map = inverse_process_sample(sequences_tokenized[idx], pred_contact_map.float().squeeze(0).squeeze(0).detach().cpu().numpy())

        contact_map = contact_maps[idx]
        mask = masks[idx]
        medium_range_mask = medium_range_masks[idx]
        long_range_mask = long_range_masks[idx]

        preds = pred_contact_map > 0.5
        probs = pred_contact_map

        medium_ground_truth.extend(contact_map[(mask * medium_range_mask) == 1].flatten())
        medium_predictions.extend(preds[(mask * medium_range_mask) == 1].flatten())
        medium_probs.extend(probs[(mask * medium_range_mask) == 1].flatten())

        long_ground_truth.extend(contact_map[(mask * long_range_mask) == 1].flatten())
        long_predictions.extend(preds[(mask * long_range_mask) == 1].flatten())
        long_probs.extend(probs[(mask * long_range_mask) == 1].flatten())
    
    medium_precision = precision_score(medium_ground_truth, medium_predictions)
    long_precision = precision_score(long_ground_truth, long_predictions)
    
    # compute auprcs using sklearn
    medium_precision_curve, medium_recall_curve, _ = precision_recall_curve(medium_ground_truth, medium_probs)
    long_precision_curve, long_recall_curve, _ = precision_recall_curve(long_ground_truth, long_probs)
    medium_auprc = auc(medium_recall_curve, medium_precision_curve)
    long_auprc = auc(long_recall_curve, long_precision_curve)

    return medium_precision, long_precision, medium_auprc, long_auprc

if logging:
    print("Logging to wandb...")
    name = f"[{wandb_prefix}] LRs=({embed_lr}, {lr}, {head_lr}, {contact_pred_lr}), bs=1, num_accum={num_accumulation_steps}, head_dim={head_dim}, resnet_blocks={num_resnet_blocks}"
    wandb.init(project=f"omnibiota-contact-eval-final", name=name)

head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], head_dim)
head.to(dtype).to(device)

contact_predictor = ContactPredictor(head_dim*2, 64, resnet_blocks=num_resnet_blocks)
contact_predictor.to(dtype).to(device)

param_groups = [
    {"params": [p for name, p in model.named_parameters() if "wte" in name], "lr": embed_lr},
    {"params": [p for name, p in model.named_parameters() if "wte" not in name], "lr": lr},
    {"params": head.parameters(), "lr": head_lr},
    {"params": contact_predictor.parameters(), "lr": contact_pred_lr}
]

num_steps = int(num_epochs*len(train_sequences_tokenized)/num_accumulation_steps)
optimizer = torch.optim.AdamW(param_groups)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, head_lr, contact_pred_lr], total_steps=num_steps, pct_start=0.05)

best_average_precision = 0

best_model = None
best_head = None
best_contact_predictor = None

pbar = tqdm(range(0, num_steps))
for step in pbar:
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for _ in range(num_accumulation_steps):
        idx = np.random.randint(len(train_sequences_tokenized))
        sequence = torch.tensor(train_sequences_tokenized[idx], device=device, dtype=torch.long)
        if sequence.shape[0] <= 5:
            continue
        contact_map = torch.tensor(train_contact_maps_tokenized[idx], device=device, dtype=dtype)
        mask = torch.tensor(train_masks_tokenized[idx], device=device, dtype=dtype)
        non_short_range = torch.tensor(train_non_short_range[idx], device=device, dtype=dtype)

        embeddings = model(sequence.unsqueeze(0), return_embeddings=True)[:, 1:-1]
        embeddings = head(embeddings)

        features = torch.cat([embeddings[:, :, None, :].expand(-1, -1, embeddings.shape[1], -1), embeddings[:, None, :, :].expand(-1, embeddings.shape[1], -1, -1)], dim=-1)
        features = features.permute(0, 3, 1, 2)

        pred_contact_map = contact_predictor(features)        

        # binary cross entropy loss
        loss = F.binary_cross_entropy(pred_contact_map, contact_map.unsqueeze(0).unsqueeze(0), reduction="none")
        
        loss = loss * mask * non_short_range
        
        if (mask*non_short_range).sum() > 0:
            loss = loss.sum() / (mask*non_short_range).sum()
        
            loss /= num_accumulation_steps
            loss.backward()

            total_loss += loss.item()
    
    if step % (num_steps//num_tests) == 0:
        model.eval()
        head.eval()
        contact_predictor.eval()

        with torch.no_grad():
            medium_precision, long_precision, medium_auprc, long_auprc = evaluate_dataset(model, head, contact_predictor, val_sequences_tokenized, val_contact_maps, val_masks, val_medium_range_masks, val_long_range_masks)

        model.train()
        head.train()
        contact_predictor.train()

        if medium_precision + long_precision > best_average_precision:
            best_average_precision = medium_precision + long_precision
            best_model = deepcopy(model)
            best_head = deepcopy(head)
            best_contact_predictor = deepcopy(contact_predictor)

        if logging:
            wandb.log({"val/medium_precision": medium_precision,
                       "val/long_precision": long_precision,
                       "val/medium_auprc": medium_auprc,
                       "val/long_auprc": long_auprc},
                       step=(step*num_accumulation_steps))
            
    if logging:
        # get current lrs
        log = {"loss": total_loss}
        for i, param_group in enumerate(optimizer.param_groups):
            log[f"lr/{i}"] = param_group["lr"]

        wandb.log(log, step=(step*num_accumulation_steps))

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    pbar.set_description(f"Loss: {total_loss:.4f}")

model.eval()
head.eval()
contact_predictor.eval()

medium_precision, long_precision, medium_auprc, long_auprc = evaluate_dataset(model, head, contact_predictor, test_sequences_tokenized, test_contact_maps, test_masks, test_medium_range_masks, test_long_range_masks)

if logging:
    wandb.log({"test/medium_precision": medium_precision, "test/long_precision": long_precision,
                "test/medium_auprc": medium_auprc, "test/long_auprc": long_auprc
               }, step=(step*num_accumulation_steps))

model = best_model
head = best_head
contact_predictor = best_contact_predictor

medium_precisions, long_precisions, medium_auprc, long_auprc = evaluate_dataset(model, head, contact_predictor, test_sequences_tokenized, test_contact_maps, test_masks, test_medium_range_masks, test_long_range_masks)

if logging:
    wandb.log({"test/best_medium_precision": medium_precision, "test/best_long_precision": long_precision,
                "test/best_medium_auprc": medium_auprc, "test/best_long_auprc": long_auprc
               }, step=(step*num_accumulation_steps))
    wandb.finish()