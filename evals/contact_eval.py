import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '../training/')
sys.path.insert(0, '../evals/')
from loader import EOS_TOKEN, PAD_TOKEN
import torch
import sentencepiece as spm
from model import OmniBioTA
import torch.nn.functional as F
import pickle
import wandb
from copy import deepcopy
from sklearn.metrics import precision_score, precision_recall_curve, auc

# ---------------------- CLASSES & FUNCTIONS ----------------------

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


def load_data(split, base_dir):
    """
    Loads ProteinNet data for the specified split (train/valid/test) and returns:
        - sequences: list of protein sequences
        - contact_maps: list of NxN boolean contact maps
        - masks: list of NxN masking arrays
        - medium_range_mask: boolean mask for medium-range contact positions
        - long_range_mask: boolean mask for long-range contact positions
    """
    file_path = os.path.join(base_dir, f"proteinnet/proteinnet_{split}.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    sequences = []
    distance_matrices = []
    masks = []
    contact_maps = []
    medium_range_mask = []
    long_range_mask = []

    # CASP targets: short range < 12, medium range 12-23, long range >= 24
    for item in tqdm(data, desc=f"Loading {split} data"):
        sequences.append(item["primary"])

        tertiary = np.asarray(item["tertiary"])
        dist_matrix = np.linalg.norm(
            tertiary[:, np.newaxis, :] - tertiary[np.newaxis, :, :],
            axis=2
        )
        distance_matrices.append(dist_matrix)

        mask = np.float32(item["valid_mask"]).reshape(-1, 1)
        masks.append(mask @ mask.T)

        index_distance = np.abs(
            np.arange(len(tertiary)) - np.arange(len(tertiary))[:, np.newaxis]
        )
        medium_range_mask.append(np.logical_and(index_distance >= 12, index_distance <= 23))
        long_range_mask.append(index_distance >= 24)

        contact_map = dist_matrix < 8
        contact_maps.append(contact_map)

    return sequences, contact_maps, masks, medium_range_mask, long_range_mask


def process_sample(sp, sequence, contact_map, mask):
    """
    Tokenizes a protein sequence and converts the contact map to the tokenized space.

    - We build a tokenized sequence for <protein> + sequence + <EOS>.
    - The contact map is mapped to the new token space by taking maximum distances
      among amino acids that collapse into the same token.
    - The mask is similarly collapsed into token space (if a single AA is unmasked,
      the token is unmasked).
    - We also compute a `non_short_range` mask which is True (1) where the residue
      indices are >= 12 apart (i.e., medium or long range).
    """
    index_distance = np.abs(np.arange(len(contact_map)) - np.arange(len(contact_map))[:, np.newaxis])
    non_short_range = index_distance >= 12

    # We add <protein> and <EOS> tokens around the raw sequence before tokenizing
    tokenized = sp.EncodeAsIds("<protein>" + sequence + "<EOS>")
    token_lens = [len(sp.DecodeIds([t])) for t in tokenized]

    masked_contact_map = contact_map * mask

    # We'll ignore <protein> and <EOS> tokens for the contact map space
    # (since they do not correspond to actual residue positions)
    t_len_no_special = len(tokenized) - 2
    tokenized_contact_map = np.zeros((t_len_no_special, t_len_no_special))
    tokenized_mask = np.zeros((t_len_no_special, t_len_no_special))
    tokenized_non_short_range = np.zeros((t_len_no_special, t_len_no_special))

    idx_i = 0
    # ignore the <protein> token at index=0 and <EOS> token at index=-1
    for i, token_len_x in enumerate(token_lens[1:-1]):
        idx_j = 0
        for j, token_len_y in enumerate(token_lens[1:-1]):
            tokenized_contact_map[i, j] = np.max(
                masked_contact_map[idx_i: idx_i + token_len_x,
                                   idx_j: idx_j + token_len_y]
            )
            tokenized_mask[i, j] = np.max(
                mask[idx_i: idx_i + token_len_x,
                     idx_j: idx_j + token_len_y]
            )
            tokenized_non_short_range[i, j] = np.max(
                non_short_range[idx_i: idx_i + token_len_x,
                                idx_j: idx_j + token_len_y]
            )
            idx_j += token_len_y
        idx_i += token_len_x

    return tokenized, tokenized_contact_map, tokenized_mask, tokenized_non_short_range


def inverse_process_sample(sp, tokenized, pred_contact_map):
    """
    Takes the tokenized sequence and the predicted contact map in the tokenized space
    and converts it back to the original (amino-acid) space.
    """
    # ignoring the <protein> token (index=0) and <EOS> token (index=-1)
    token_lens = [len(sp.DecodeIds([t])) for t in tokenized[1:-1]]
    total_len = np.sum(token_lens)
    contact_map = np.zeros((total_len, total_len))

    idx_i = 0
    for i, token_len_x in enumerate(token_lens[1:-1]):
        idx_j = 0
        for j, token_len_y in enumerate(token_lens[1:-1]):
            contact_map[idx_i: idx_i + token_len_x, idx_j: idx_j + token_len_y] = pred_contact_map[i, j]
            idx_j += token_len_y
        idx_i += token_len_x
    return contact_map


class ResNetBlock(torch.nn.Module):
    """
    A simple ResNet block for 2D contact map refinement.
    """
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
    """
    A contact predictor head that applies a few ResNet blocks to the pairwise
    token embeddings (stacked along channels).
    """
    def __init__(self, in_channels, out_channels, resnet_blocks=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.resnet_blocks = torch.nn.ModuleList([
            ResNetBlock(out_channels, out_channels) for _ in range(resnet_blocks)
        ])
        self.conv2 = torch.nn.Conv2d(out_channels, 1, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        for block in self.resnet_blocks:
            out = block(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return out


def recurse_load(m):
    """
    Recursively find the original module if it is wrapped by a
    flash-attention-style `_orig_mod`.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m

def evaluate_dataset(
    model,
    head,
    contact_predictor,
    sp,
    sequences_tokenized,
    contact_maps,
    masks,
    medium_range_masks,
    long_range_masks,
    device
):
    """
    Runs evaluation on the given dataset split (tokenized sequences + contact maps + masks)
    and returns precision and AUPRC metrics for medium- and long-range contacts.
    """
    long_precision, medium_precision = 0, 0
    long_auprc, medium_auprc = 0, 0

    medium_ground_truth, medium_predictions, medium_probs = [], [], []
    long_ground_truth, long_predictions, long_probs = [], [], []

    model.eval()
    head.eval()
    contact_predictor.eval()

    for idx in range(len(sequences_tokenized)):
        seq_tokens = torch.tensor(sequences_tokenized[idx], device=device, dtype=torch.long)

        # skip trivially small sequences
        if seq_tokens.shape[0] <= 5:
            continue

        with torch.no_grad():
            embeddings = model(seq_tokens.unsqueeze(0), return_embeddings=True)[:, 1:-1]
            embeddings = head(embeddings)

            # Broadcast embeddings to a pairwise grid for the contact predictor
            features = torch.cat([
                embeddings[:, :, None, :].expand(-1, -1, embeddings.shape[1], -1),
                embeddings[:, None, :, :].expand(-1, embeddings.shape[1], -1, -1),
            ], dim=-1)
            features = features.permute(0, 3, 1, 2)

            seq_tokens = seq_tokens.cpu().tolist()
            pred_contact_map = contact_predictor(features)
            # Convert back to the original (amino-acid) space
            pred_contact_map = inverse_process_sample(
                sp, seq_tokens, pred_contact_map.float().squeeze(0).squeeze(0).cpu().numpy()
            )

        gt_contact_map = contact_maps[idx]
        mask = masks[idx]
        medium_mask = medium_range_masks[idx]
        long_mask = long_range_masks[idx]

        preds = pred_contact_map > 0.5
        probs = pred_contact_map

        # Medium range metrics
        medium_region = (mask * medium_mask) == 1
        medium_ground_truth.extend(gt_contact_map[medium_region].flatten())
        medium_predictions.extend(preds[medium_region].flatten())
        medium_probs.extend(probs[medium_region].flatten())

        # Long range metrics
        long_region = (mask * long_mask) == 1
        long_ground_truth.extend(gt_contact_map[long_region].flatten())
        long_predictions.extend(preds[long_region].flatten())
        long_probs.extend(probs[long_region].flatten())

    # Compute precision
    medium_precision = precision_score(medium_ground_truth, medium_predictions) if medium_ground_truth else 0
    long_precision = precision_score(long_ground_truth, long_predictions) if long_ground_truth else 0

    # Compute AUPRC
    if medium_ground_truth:
        m_prec_curve, m_recall_curve, _ = precision_recall_curve(medium_ground_truth, medium_probs)
        medium_auprc = auc(m_recall_curve, m_prec_curve)
    if long_ground_truth:
        l_prec_curve, l_recall_curve, _ = precision_recall_curve(long_ground_truth, long_probs)
        long_auprc = auc(l_recall_curve, l_prec_curve)

    return medium_precision, long_precision, medium_auprc, long_auprc


# ---------------------- MAIN ----------------------

def main():
    parser = argparse.ArgumentParser(description="Train a protein contact predictor with an OmniBioTA model.")
    parser.add_argument("--tokenizer_fn", type=str, help="Path to the tokenizer file.")
    parser.add_argument("--model_fn", type=str, help="Path to the saved model file.")
    parser.add_argument("--banned_token", default=2044, type=int, help="Banned token ID.")
    parser.add_argument("--tokenizer_offset", type=int, help="Offset to be added to token IDs.")
    parser.add_argument("--wandb_prefix", type=str, help="Prefix for wandb logging run.")
    parser.add_argument("--data_dir", type=str, default="/gpfs/home/chens59/OmniBioTA/datasets/TAPE/data",
                        help="Path to the dataset directory.")
    parser.add_argument("--num_accumulation_steps", type=int, default=128, help="Number of gradient accumulation steps.")
    parser.add_argument("--num_epochs", type=int, default=128, help="Number of training epochs.")
    parser.add_argument("--head_dim", type=int, default=128, help="Dimensionality of the head linear projection.")
    parser.add_argument("--num_resnet_blocks", type=int, default=8, help="Number of ResNet blocks in ContactPredictor.")
    parser.add_argument("--num_tests", type=int, default=256, help="How many times to evaluate on the validation set.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the main model (excluding embedding).")
    parser.add_argument("--embed_lr", type=float, default=1e-4, help="Learning rate for the embedding layer.")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate for the linear head.")
    parser.add_argument("--contact_pred_lr", type=float, default=1e-3, help="Learning rate for the contact predictor.")
    parser.add_argument("--logging", action="store_true", help="Whether to enable wandb logging.")
    args = parser.parse_args()

    # Print arguments for clarity
    print(f"Tokenizer: {args.tokenizer_fn}")
    print(f"Model: {args.model_fn}")
    print(f"Banned token: {args.banned_token}")
    print(f"Tokenizer offset: {args.tokenizer_offset}")
    print(f"Wandb prefix: {args.wandb_prefix}")

    # ---------------------- LOAD DATA ----------------------
    train_sequences, train_contact_maps, train_masks, _, _ = load_data("train", args.data_dir)
    val_sequences, val_contact_maps, val_masks, val_medium_masks, val_long_masks = load_data("valid", args.data_dir)
    test_sequences, test_contact_maps, test_masks, test_medium_masks, test_long_masks = load_data("test", args.data_dir)

    # ---------------------- TOKENIZER ----------------------
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_fn)
    sp = TokenizerWrapper(sp, args.tokenizer_offset, [1, 2, args.banned_token])

    # ---------------------- PREPROCESS TRAIN ----------------------
    print("Preprocessing train data...")
    train_sequences_tokenized = []
    train_contact_maps_tokenized = []
    train_masks_tokenized = []
    train_non_short_range = []

    for sequence, c_map, msk in tqdm(zip(train_sequences, train_contact_maps, train_masks),
                                     total=len(train_sequences), desc="Processing train"):
        tokenized, tokenized_cmap, tokenized_mask, tokenized_non_sr = process_sample(sp, sequence, c_map, msk)
        train_sequences_tokenized.append(tokenized)
        train_contact_maps_tokenized.append(tokenized_cmap)
        train_masks_tokenized.append(tokenized_mask)
        train_non_short_range.append(tokenized_non_sr)

    # Free memory
    del train_sequences, train_contact_maps, train_masks

    # ---------------------- PREPROCESS VAL ----------------------
    val_sequences_tokenized = []
    for sequence, c_map, msk in tqdm(zip(val_sequences, val_contact_maps, val_masks),
                                     total=len(val_sequences), desc="Processing val"):
        tok, tok_cmap, tok_mask, _ = process_sample(sp, sequence, c_map, msk)
        val_sequences_tokenized.append(tok)

    # ---------------------- PREPROCESS TEST ----------------------
    test_sequences_tokenized = []
    for sequence, c_map, msk in tqdm(zip(test_sequences, test_contact_maps, test_masks),
                                     total=len(test_sequences), desc="Processing test"):
        tok, tok_cmap, tok_mask, _ = process_sample(sp, sequence, c_map, msk)
        test_sequences_tokenized.append(tok)

    # ---------------------- SETUP MODEL ----------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Load base model
    base_model = torch.load(args.model_fn, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)
    model_sd = base_model.state_dict()
    config = base_model.config
    model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=dtype)

    # Create linear head and contact predictor
    head = torch.nn.Linear(model.transformer.wte.weight.shape[-1], args.head_dim).to(device=device, dtype=dtype)
    contact_predictor = ContactPredictor(
        in_channels=args.head_dim * 2,
        out_channels=64,
        resnet_blocks=args.num_resnet_blocks
    ).to(device=device, dtype=dtype)

    # Setup parameters and optimizers
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "wte" in n], "lr": args.embed_lr},
        {"params": [p for n, p in model.named_parameters() if "wte" not in n], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
        {"params": contact_predictor.parameters(), "lr": args.contact_pred_lr},
    ]

    num_steps = int(args.num_epochs * len(train_sequences_tokenized) / args.num_accumulation_steps)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.embed_lr, args.lr, args.head_lr, args.contact_pred_lr],
        total_steps=num_steps,
        pct_start=0.05
    )

    # ---------------------- LOGGING ----------------------
    logging = args.logging
    if logging:
        print("Logging to wandb...")
        run_name = (
            f"[{args.wandb_prefix}] "
            f"LRs=({args.embed_lr}, {args.lr}, {args.head_lr}, {args.contact_pred_lr}), "
            f"bs=1, num_accum={args.num_accumulation_steps}, head_dim={args.head_dim}, "
            f"resnet_blocks={args.num_resnet_blocks}"
        )
        wandb.init(project="OmniBioTA-v2 Contact Eval", name=run_name)

    # ---------------------- TRAIN LOOP ----------------------
    best_average_precision = 0
    best_model = None
    best_head = None
    best_contact_predictor = None

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        model.train()
        head.train()
        contact_predictor.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for _ in range(args.num_accumulation_steps):
            idx = np.random.randint(len(train_sequences_tokenized))

            sequence = torch.tensor(train_sequences_tokenized[idx], device=device, dtype=torch.long)
            if sequence.shape[0] <= 5:
                continue

            gt_contact_map = torch.tensor(train_contact_maps_tokenized[idx], device=device, dtype=dtype)
            msk = torch.tensor(train_masks_tokenized[idx], device=device, dtype=dtype)
            non_short_range = torch.tensor(train_non_short_range[idx], device=device, dtype=dtype)

            embeddings = model(sequence.unsqueeze(0), return_embeddings=True)[:, 1:-1]
            embeddings = head(embeddings)

            # Prepare pairwise features
            features = torch.cat([
                embeddings[:, :, None, :].expand(-1, -1, embeddings.shape[1], -1),
                embeddings[:, None, :, :].expand(-1, embeddings.shape[1], -1, -1),
            ], dim=-1).permute(0, 3, 1, 2)

            pred_contact_map = contact_predictor(features)

            loss = F.binary_cross_entropy(
                pred_contact_map,
                gt_contact_map.unsqueeze(0).unsqueeze(0),
                reduction="none"
            )
            loss = loss * msk * non_short_range

            denom = (msk * non_short_range).sum()
            if denom > 0:
                loss = loss.sum() / denom
                loss = loss / args.num_accumulation_steps
                loss.backward()
                total_loss += loss.item()

        # Evaluate periodically
        if step % max(1, (num_steps // args.num_tests)) == 0:
            with torch.no_grad():
                m_prec, l_prec, m_auprc, l_auprc = evaluate_dataset(
                    model,
                    head,
                    contact_predictor,
                    sp,
                    val_sequences_tokenized,
                    val_contact_maps,
                    val_masks,
                    val_medium_masks,
                    val_long_masks,
                    device
                )

            # Save the best model if it improves
            if (m_prec + l_prec) > best_average_precision:
                best_average_precision = m_prec + l_prec
                best_model = deepcopy(model)
                best_head = deepcopy(head)
                best_contact_predictor = deepcopy(contact_predictor)

            if logging:
                wandb.log({
                    "val/medium_precision": m_prec,
                    "val/long_precision": l_prec,
                    "val/medium_auprc": m_auprc,
                    "val/long_auprc": l_auprc
                }, step=(step * args.num_accumulation_steps))

        # Logging to wandb
        if logging:
            log_dict = {"loss": total_loss}
            for i, param_group in enumerate(optimizer.param_groups):
                log_dict[f"lr/{i}"] = param_group["lr"]
            wandb.log(log_dict, step=(step * args.num_accumulation_steps))

        # Gradient clipping + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}")

    # ---------------------- FINAL EVALUATION ----------------------
    # Evaluate final model on test
    m_prec, l_prec, m_auprc, l_auprc = evaluate_dataset(
        model,
        head,
        contact_predictor,
        sp,
        test_sequences_tokenized,
        test_contact_maps,
        test_masks,
        test_medium_masks,
        test_long_masks,
        device
    )

    if logging:
        wandb.log({
            "test/medium_precision": m_prec,
            "test/long_precision": l_prec,
            "test/medium_auprc": m_auprc,
            "test/long_auprc": l_auprc
        }, step=(step * args.num_accumulation_steps))

    # Evaluate best model on test
    if best_model is not None:
        model = best_model
        head = best_head
        contact_predictor = best_contact_predictor
        m_prec, l_prec, m_auprc, l_auprc = evaluate_dataset(
            model,
            head,
            contact_predictor,
            sp,
            test_sequences_tokenized,
            test_contact_maps,
            test_masks,
            test_medium_masks,
            test_long_masks,
            device
        )
        if logging:
            wandb.log({
                "test/best_medium_precision": m_prec,
                "test/best_long_precision": l_prec,
                "test/best_medium_auprc": m_auprc,
                "test/best_long_auprc": l_auprc
            }, step=(step * args.num_accumulation_steps))
            wandb.finish()


if __name__ == "__main__":
    main()