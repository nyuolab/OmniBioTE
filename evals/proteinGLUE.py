import os
import argparse
import pickle
import re
import copy

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import pandas as pd
from loader import EOS_TOKEN, PAD_TOKEN, MASK_TOKEN
from model import OmniBioTA

########################################################################
# Global Device Setup
########################################################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

########################################################################
# Data Cleaning / Loading
########################################################################
def get_cleaned_evals(eval_dir, force=False):
    """
    Cleans and prepares CSV-based datasets into a dictionary-based format.
    If a processed pickle file exists, it will be loaded unless 'force' is True.

    Args:
        eval_dir (str): Path to the directory containing raw CSVs.
        force (bool): If True, force re-creation of the processed file.

    Returns:
        dict: A dictionary containing cleaned datasets.
    """
    dfs = []
    fns = []

    if not os.path.exists(eval_dir):
        print(f"Directory {eval_dir} does not exist.")
        return

    if os.path.isfile("../datasets/ProteinGLUE_processed.pkl") and not force:
        with open("../datasets/ProteinGLUE_processed.pkl", "rb") as f:
            datasets = pickle.load(f)
        return datasets

    for fn in os.listdir(eval_dir):
        if fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(eval_dir, fn))

            # Clean brackets, quotes, newlines
            for col in df.columns:
                df[col] = df[col].apply(lambda x: re.sub(r"[\[\]\'b\n]", "", str(x)))
            dfs.append(df)
            fns.append(fn)

    names = [fn[:-4] for fn in fns]  # Remove '.csv' to get dataset name

    datasets = {}
    for name, df in zip(names, dfs):
        datasets[name] = {}
        datasets[name]["sequences"] = []
        label_columns = [
            col for col in df.columns.tolist() if col != "sequence"
        ]
        error_indices = []

        for label_column in label_columns:
            raw_labels = df[label_column].tolist()
            num_errors = 0
            labels = []

            for i, raw_label in enumerate(raw_labels):
                # Some labels have "..." in them
                if "..." in raw_label.split():
                    num_errors += 1
                    error_indices.append(i)
                    continue
                try:
                    labels.append([float(item) for item in raw_label.split()])
                except ValueError:
                    # If conversion to float fails, mark as error
                    num_errors += 1
                    error_indices.append(i)
                    continue

            datasets[name][label_column] = labels
            print(f"Dataset: {name}, Label: {label_column}, Num Errors: {num_errors}")

        # Add sequences to the dataset
        raw_sequences = df["sequence"].tolist()
        for i, seq in enumerate(raw_sequences):
            if i not in error_indices:
                datasets[name]["sequences"].append(seq)

        # Verify dataset consistency
        for label in datasets[name].keys():
            if label == "sequences":
                continue

            # Check label count matches sequence count
            if len(datasets[name][label]) != len(datasets[name]["sequences"]):
                print(
                    f"ERROR {name} {label} "
                    f"{len(datasets[name][label])} "
                    f"{len(datasets[name]['sequences'])}"
                )
                continue

            # Check label/sequence lengths match on each entry
            for i in range(len(datasets[name][label])):
                if len(datasets[name][label][i]) != len(datasets[name]["sequences"][i]):
                    print(
                        f"ERROR {name} {label} {i} "
                        f"{len(datasets[name][label][i])} "
                        f"{len(datasets[name]['sequences'][i])}"
                    )
            print()
    return datasets

def load_task(task, datasets):
    """
    Given a task identifier, retrieve train/val/test splits from the datasets dictionary.

    Args:
        task (str): The name of the task (e.g., "SS3", "ASA", etc.).
        datasets (dict): The preprocessed datasets dictionary.

    Returns:
        tuple: (train_sequences, train_inference, val_sequences, val_inference, test_sequences, test_inference)
    """
    if task == "SS3":
        train_sequences = datasets["ss_training"]["sequences"]
        train_inference = datasets["ss_training"]["ss3"]
        val_sequences = datasets["ss_validation"]["sequences"]
        val_inference = datasets["ss_validation"]["ss3"]
        test_sequences = datasets["ss_test"]["sequences"]
        test_inference = datasets["ss_test"]["ss3"]
    elif task == "SS8":
        train_sequences = datasets["ss_training"]["sequences"]
        train_inference = datasets["ss_training"]["ss8"]
        val_sequences = datasets["ss_validation"]["sequences"]
        val_inference = datasets["ss_validation"]["ss8"]
        test_sequences = datasets["ss_test"]["sequences"]
        test_inference = datasets["ss_test"]["ss8"]
    elif task == "CB513SS8":
        train_sequences = None
        train_inference = None
        val_sequences = None
        val_inference = None
        test_sequences = datasets["ss_cb513_test"]["sequences"]
        test_inference = datasets["ss_cb513_test"]["ss8"]
    elif task == "CB513SS3":
        train_sequences = None
        train_inference = None
        val_sequences = None
        val_inference = None
        test_sequences = datasets["ss_cb513_test"]["sequences"]
        test_inference = datasets["ss_cb513_test"]["ss3"]
    elif task == "BUR":
        train_sequences = datasets["asabu_training"]["sequences"]
        train_inference = datasets["asabu_training"]["buried"]
        val_sequences = datasets["asabu_validation"]["sequences"]
        val_inference = datasets["asabu_validation"]["buried"]
        test_sequences = datasets["asabu_test"]["sequences"]
        test_inference = datasets["asabu_test"]["buried"]
    elif task == "ASA":
        train_sequences = datasets["asabu_training"]["sequences"]
        train_inference = datasets["asabu_training"]["solvent_accessibility"]
        val_sequences = datasets["asabu_validation"]["sequences"]
        val_inference = datasets["asabu_validation"]["solvent_accessibility"]
        test_sequences = datasets["asabu_test"]["sequences"]
        test_inference = datasets["asabu_test"]["solvent_accessibility"]
    elif task == "PPI":
        train_sequences = datasets["ppi_hetro_homo_training"]["sequences"]
        train_inference = datasets["ppi_hetro_homo_training"]["interface"]
        val_sequences = datasets["ppi_hetro_homo_validation"]["sequences"]
        val_inference = datasets["ppi_hetro_homo_validation"]["interface"]
        test_sequences = datasets["ppi_hetro_homo_test"]["sequences"]
        test_inference = datasets["ppi_hetro_homo_test"]["interface"]
    elif task == "EPI":
        train_sequences = datasets["Epitope_anti_training_1"]["sequences"]
        train_inference = datasets["Epitope_anti_training_1"]["interface"]
        val_sequences = datasets["Epitope_anti_validation_1"]["sequences"]
        val_inference = datasets["Epitope_anti_validation_1"]["interface"]
        test_sequences = datasets["Epitope_anti_test_1"]["sequences"]
        test_inference = datasets["Epitope_anti_test_1"]["interface"]
    elif task == "HPR":
        train_sequences = datasets["HPrank_training"]["sequences"]
        train_inference = datasets["HPrank_training"]["hydrophobic_patch"]
        val_sequences = datasets["HPrank_validation"]["sequences"]
        val_inference = datasets["HPrank_validation"]["hydrophobic_patch"]
        test_sequences = datasets["HPrank_test"]["sequences"]
        test_inference = datasets["HPrank_test"]["hydrophobic_patch"]
    else:
        raise ValueError(f"Unknown task {task}")

    return train_sequences, train_inference, val_sequences, val_inference, test_sequences, test_inference


def create_normalizer(train_list_of_lists):
    """
    Creates a normalizer function (min-max) from training data.

    Args:
        train_list_of_lists (list): A list of lists (floats) from training.

    Returns:
        function: A normalizer function that can be applied to other data.
    """
    flattened_train = np.hstack(train_list_of_lists)
    min_val, max_val = np.min(flattened_train), np.max(flattened_train)

    def normalizer(list_of_lists):
        flattened = np.hstack(list_of_lists)
        norm_flattened = (flattened - min_val) / (max_val - min_val)

        # Reshape into original nested structure
        normalized_list_of_lists = []
        start = 0
        for sublist in list_of_lists:
            end = start + len(sublist)
            normalized_list_of_lists.append(norm_flattened[start:end].tolist())
            start = end
        return normalized_list_of_lists

    return normalizer


########################################################################
# Tokenization / Formatting
########################################################################
def format_sequence_and_label(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence with SentencePiece and aligns tokens with
    label mode (most frequent integer in chunk).

    Returns:
        (tokenized_sequence, label_modes, token_lens, sequence_length)
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)

    token_lens = [len(sp.DecodeIds([token])) for token in tokenized_sequence]
    label_modes = []
    ptr = 0

    for i, length in enumerate(token_lens):
        try:
            if tokenized_sequence[i] - sp.tokenizer_offset == 0:  # unknown token
                length = 1
            label_chunk = label[ptr : ptr + length]
            label_modes.append(np.bincount(label_chunk).argmax())
        except Exception as e:
            print(f"Error in {sequence}, {label}, {tokenized_sequence}, {token_lens}")
            raise e
        ptr += length

    sequence_length = len(sequence)
    return tokenized_sequence, label_modes, token_lens, sequence_length


def format_sequence_and_value(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence with SentencePiece and aligns tokens with
    label value means (average float in chunk).

    Returns:
        (tokenized_sequence, label_means, token_lens, sequence_length)
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)

    token_lens = [len(sp.DecodeIds([token])) for token in tokenized_sequence]
    label_means = []
    ptr = 0

    for i, length in enumerate(token_lens):
        if tokenized_sequence[i] - sp.tokenizer_offset == 0:  # unknown token
            length = 1
        label_chunk = label[ptr : ptr + length]
        label_means.append(np.mean(label_chunk))
        ptr += length

    sequence_length = len(sequence)
    return tokenized_sequence, label_means, token_lens, sequence_length


def process_data(sp, sequences, target, format_func, prefix, banned_tokens=None):
    """
    Applies SentencePiece tokenization/label formatting and adds an EOS token.

    Args:
        sp (SentencePieceProcessor): Tokenizer instance.
        sequences (list): List of raw sequences.
        target (list): List of raw label structures.
        format_func (callable): Function to format each sequence/labels.
        prefix (list): Prefix tokens to prepend (e.g., <protein>).
        banned_tokens (list): Tokens to be excluded.

    Returns:
        (list_of_tokenized_sequences, list_of_formatted_targets)
    """
    if banned_tokens is None:
        banned_tokens = []

    sequences_tokenized = []
    targets = []

    for i in tqdm(range(len(sequences))):
        tokenized_sequence, target_values, _, _ = format_func(
            sequences[i], target[i], sp, banned_tokens
        )
        sequences_tokenized.append(tokenized_sequence)
        targets.append(target_values)

    # Append prefix and EOS_TOKEN
    sequences_tokenized = [prefix + item + [EOS_TOKEN] for item in sequences_tokenized]
    return sequences_tokenized, targets


def get_training_sets(task, dataset, sp, format_func, prefix, banned_tokens):
    """
    Helper that loads (train, val, test) from 'load_task' and
    processes data via 'process_data' for tokenization.
    Applies normalizer if needed (ASA, HPR).

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
        or (None, None, None, None, X_test, y_test) if CB513 tasks.
    """
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = load_task(
        task, dataset
    )

    # Some tasks only have test set (CB513)
    if task not in ["CB513SS3", "CB513SS8"]:
        X_train, y_train = process_data(
            sp, X_train_raw, y_train_raw, format_func, prefix, banned_tokens
        )
        X_val, y_val = process_data(
            sp, X_val_raw, y_val_raw, format_func, prefix, banned_tokens
        )
    else:
        X_train, y_train = None, None
        X_val, y_val = None, None

    X_test, y_test = process_data(
        sp, X_test_raw, y_test_raw, format_func, prefix, banned_tokens
    )

    # Normalize certain tasks
    if task in ["ASA", "HPR"]:
        normalizer = create_normalizer(y_train_raw)
        y_train = normalizer(y_train)
        y_val = normalizer(y_val)
        y_test = normalizer(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


########################################################################
# Model Finetuning
########################################################################
@torch.jit.script
def block_attn(attn_mask: torch.Tensor, start: int, end: int, batch_idx: int) -> int:
    """
    Creates zeros in 'attn_mask' to allow attentions in the block [start, end].
    """
    attn_mask[batch_idx, start:end, start:end] = 0
    return 0


@torch.jit.script
def create_attention_mask(
    attn_mask: torch.Tensor, input_ids: torch.Tensor, EOS_TOKEN: int = 3, padding: bool = False
) -> torch.Tensor:
    """
    Creates an attention mask that splits attention blocks at each EOS token.
    """
    if not padding:
        temp = torch.ones(input_ids.size(0), input_ids.size(1) + 1, device=input_ids.device, dtype=input_ids.dtype)
        temp[:, :-1] = input_ids
        temp[:, -1] = EOS_TOKEN
        input_ids = temp

    EOS_positions = (input_ids == EOS_TOKEN).nonzero()
    attn_mask.fill_(-1e9)
    prev_index = 0
    prev_batch_idx = 0

    for i in range(len(EOS_positions)):
        if EOS_positions[i][0] == prev_batch_idx:
            block_attn(attn_mask, prev_index, EOS_positions[i][1] + 1, prev_batch_idx)
            prev_index = EOS_positions[i][1] + 1
        else:
            prev_batch_idx = EOS_positions[i][0]
            prev_index = 0
            block_attn(attn_mask, prev_index, EOS_positions[i][1] + 1, prev_batch_idx)

    # if no EOS found for a batch row
    for i in range(input_ids.size(0)):
        if not torch.any(EOS_positions[:, 0] == i):
            attn_mask[i, :, :] = 0

    return attn_mask


def pad_attn(attn_mask, x):
    """
    Masks out attention for PAD tokens.
    """
    pad_locations = (x == PAD_TOKEN).nonzero()
    for i in range(len(pad_locations)):
        attn_mask[pad_locations[i][0], pad_locations[i][1] + 1 :, :] = -1e9
        attn_mask[pad_locations[i][0], :, pad_locations[i][1] + 1 :] = -1e9
    return attn_mask


def finetune_on_task(
    task,
    model,
    dataset,
    sp,
    banned_tokens,
    format_func,
    metric,
    loss_str,
    device,
    dtype=torch.bfloat16,
    num_epochs=4,
    batch_size=1,
    num_accumulation_steps=8,
    lr=1e-4,
    embed_lr=1e-2,
    test_freq=100,
):
    """
    Finetunes the model on a given ProteinGLUE task.
    Returns the finetuned base model and the classification/regression head.
    """
    prefix = sp.EncodeAsIds("<protein>")
    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(
        task, dataset, sp, format_func, prefix, banned_tokens
    )

    base_model = copy.deepcopy(model)
    base_model.train()

    # Determine output dimension for the linear head
    if loss_str == "mse":
        output_dim = 1
    else:
        # classification => # classes
        max_label = max(item for sublist in Y_train for item in sublist)
        output_dim = max_label + 1

    head = torch.nn.Linear(
        base_model.transformer.wte.weight.shape[-1], output_dim
    ).to(device).to(dtype)

    # Parameter groups for different LR
    param_groups = [
        {"params": [p for n, p in base_model.named_parameters() if "wte" in n], "lr": embed_lr},
        {"params": [p for n, p in base_model.named_parameters() if "wte" not in n], "lr": lr},
        {"params": head.parameters(), "lr": 1e-2},
    ]

    # Prepare optimizer/scheduler
    num_steps = int(num_epochs * len(X_train) / (batch_size * num_accumulation_steps))
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[embed_lr, lr, 1e-2],
        total_steps=num_steps,
        pct_start=0.05
    )

    if loss_str == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_str == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss function")

    if metric == "PCC":
        metric_fn = lambda y, pred: pearsonr(y, pred)[0]
    elif metric == "ACC":
        metric_fn = accuracy_score
    elif metric == "AUC":
        metric_fn = roc_auc_score
    else:
        raise ValueError(f"Unknown metric {metric}")

    loss_hist = []
    val_scores = []
    last_val_score = 0
    best_val_score = float("-inf")
    best_model_sd = copy.deepcopy(base_model.state_dict())
    best_head_sd = copy.deepcopy(head.state_dict())

    pbar = tqdm(range(num_steps), desc="Finetuning")
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        base_model.train()
        head.train()

        # Gradient accumulation
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = np.array(X_train, dtype=object)[indices]
            Y_batch = np.array(Y_train, dtype=object)[indices]

            # Clip input if too long
            X_batch = [x[:1024] for x in X_batch]
            Y_batch = [y[:1023] for y in Y_batch]

            # Loop over each sample in batch
            for x, y in zip(X_batch, Y_batch):
                x_tensor = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                embeddings = base_model(x_tensor, return_embeddings=True)
                embeddings = embeddings[:, 1 : len(y) + 1]  # shift by 1 for prefix
                outputs = head(embeddings).squeeze(0)

                if loss_str == "mse":
                    y_tensor = torch.tensor(y, device=device, dtype=dtype).unsqueeze(1)
                else:
                    y_tensor = torch.tensor(y, device=device, dtype=torch.long)

                # Normalize by (accumulation_steps * batch_size)
                loss = loss_fn(outputs, y_tensor) / (num_accumulation_steps * batch_size)
                loss.backward()
                total_loss += loss.item()

        optimizer.step()
        scheduler.step()
        loss_hist.append(total_loss)
        pbar.set_description(f"Loss: {total_loss:.4f}, Val Score: {last_val_score:.4f}")

        # Quick validation check
        if step % max(1, num_steps // test_freq) == 0:
            base_model.eval()
            head.eval()
            with torch.no_grad():
                ground_truths = []
                predictions = []

                for i in range(len(X_val)):
                    x_val = X_val[i][:1024]
                    y_val = Y_val[i][:1023]

                    x_tensor = torch.tensor(x_val, device=device, dtype=torch.long).unsqueeze(0)
                    embeddings = base_model(x_tensor, return_embeddings=True)
                    embeddings = embeddings[:, 1 : len(y_val) + 1]
                    output = head(embeddings).squeeze(0)

                    y_val_tensor = torch.tensor(y_val, device=device, dtype=dtype)
                    ground_truths.extend(y_val_tensor.cpu().tolist())

                    if loss_str == "mse":
                        predictions.extend(output.reshape(-1).cpu().tolist())
                    else:
                        predictions.extend(output.argmax(dim=-1).cpu().tolist())

                ground_truths = np.asarray(ground_truths)
                predictions = np.asarray(predictions)
                last_val_score = metric_fn(ground_truths, predictions)
                val_scores.append(last_val_score)

                # Keep track of best model
                if last_val_score > best_val_score:
                    best_val_score = last_val_score
                    best_model_sd = copy.deepcopy(base_model.state_dict())
                    best_head_sd = copy.deepcopy(head.state_dict())

                pbar.set_description(
                    f"(Val) Step {step}/{num_steps} Loss: {total_loss:.4f}, "
                    f"Val Score: {last_val_score:.4f}"
                )

    # Load best state
    base_model.load_state_dict(best_model_sd)
    head.load_state_dict(best_head_sd)
    return base_model, head

def test_model(task, model, head, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype=torch.bfloat16):
    """
    Evaluates the finetuned model on the test set of the specified task.
    Prints and returns the metric performance.
    """
    model.eval()
    head.eval()

    prefix = sp.EncodeAsIds("<protein>")
    _, _, _, _, X_test, Y_test = get_training_sets(
        task, dataset, sp, format_func, prefix, banned_tokens
    )

    if metric == "PCC":
        metric_fn = lambda y, pred: pearsonr(y, pred)[0]
    elif metric == "ACC":
        metric_fn = accuracy_score
    elif metric == "AUC":
        metric_fn = roc_auc_score
    else:
        raise ValueError(f"Unknown metric {metric}")

    predictions = []
    ground_truths = []
    with torch.no_grad():
        for i in range(len(X_test)):
            x_test = X_test[i][:1024]
            y_test = Y_test[i][:1023]

            x_tensor = torch.tensor(x_test, device=device, dtype=torch.long).unsqueeze(0)
            embeddings = model(x_tensor, return_embeddings=True)
            embeddings = embeddings[:, 1 : len(y_test) + 1]
            output = head(embeddings).squeeze(0)

            y_tensor = torch.tensor(y_test, device=device, dtype=dtype)
            ground_truths.extend(y_tensor.cpu().tolist())

            if loss_str == "mse":
                predictions.extend(output.reshape(-1).cpu().tolist())
            else:
                predictions.extend(output.argmax(dim=-1).cpu().tolist())

    ground_truths = np.asarray(ground_truths)
    predictions = np.asarray(predictions)
    performance = metric_fn(ground_truths, predictions)

    print(f"{task} -> {metric}: {performance}")
    return performance

def recurse_load(m):
    """
    Recursively find the original module if it is wrapped by a
    flash-attention-style `_orig_mod`.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m

########################################################################
# Main Experiment Runner
########################################################################
def run_experiment(
    sp_dir,
    tokenizer_offset,
    model_dir,
    banned_token=[2044],
    extra_pretrain=False,
    pretraining_epochs=4,
    pretraining_num_accum_steps=4,
    batch_size=32,
    pretraining_lr=1e-3,
    finetuning_lr=2e-4,
    output_suffix="",
):
    """
    Main entry point for running the experiment with specified parameters.
    """
    print(f"Loading tokenizer from {sp_dir}...")
    print(f"Loading model from {model_dir}...")
    print(f"Using banned token(s): {banned_token}")
    print(f"Pretraining config => epochs: {pretraining_epochs}, accumulation steps: {pretraining_num_accum_steps}, lr: {pretraining_lr}")
    print(f"Finetuning lr: {finetuning_lr}")
    print(f"Output suffix: {output_suffix}")

    tasks = ["SS3", "SS8", "ASA", "HPR", "PPI", "BUR", "EPI"]
    dtype = torch.bfloat16

    banned_tokens = banned_token if isinstance(banned_token, list) else [banned_token]

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_dir)
    sp = TokenizerWrapper(sp, tokenizer_offset, [1, 2] + banned_tokens)

    # Load your pretrained model
    base_model = torch.load(model_dir, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)
    model_sd = base_model.state_dict()
    config = base_model.config
    config.memory_efficient = True
    model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=dtype)
    model.eval()

    print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

    # Load / Clean dataset
    dataset = get_cleaned_evals("../datasets/ProteinGLUE", force=False)

    # Optional extra pretraining on sequences from all tasks
    if extra_pretrain:
        all_sequences = []
        for task in tasks:
            prefix = sp.EncodeAsIds("<protein>")
            X_train = get_training_sets(
                task, dataset, sp, format_sequence_and_label, prefix, banned_tokens
            )[0]
            if X_train is not None:
                all_sequences += X_train

        loss_hist = []
        num_steps = int(pretraining_epochs * len(all_sequences) / (batch_size * pretraining_num_accum_steps))
        optimizer = torch.optim.AdamW(model.parameters(), lr=pretraining_lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps
        )
        pbar = tqdm(range(num_steps), desc="Extra Pretraining")

        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(pretraining_num_accum_steps):
                indices = np.random.choice(len(all_sequences), batch_size, replace=False)
                max_len = max(len(all_sequences[i]) for i in indices)
                x_batch = torch.ones((batch_size, max_len), dtype=torch.long, device=device) * PAD_TOKEN

                for i, idx in enumerate(indices):
                    seq_len = len(all_sequences[idx])
                    x_batch[i, :seq_len] = torch.tensor(all_sequences[idx], dtype=torch.long, device=device)

                # Create an attention mask
                attn_mask = torch.zeros((x_batch.shape[0], x_batch.shape[1], x_batch.shape[1]), device=device)
                attn_mask = pad_attn(attn_mask, x_batch)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, model.transformer.h[0].attn.n_head, -1, -1).to(dtype)

                # 15% random masking
                token_mask = torch.rand(x_batch.shape, device=device) < 0.15
                masked_tokens = x_batch.clone()
                masked_tokens[token_mask] = MASK_TOKEN

                out = model(masked_tokens, attn_mask=attn_mask)

                # Compute cross-entropy ignoring PAD
                loss = torch.nn.functional.cross_entropy(
                    out.view(-1, out.shape[-1]),
                    x_batch.view(-1),
                    ignore_index=PAD_TOKEN,
                    reduction="sum",
                )
                denom = (x_batch != PAD_TOKEN).sum()
                loss = loss / denom / pretraining_num_accum_steps

                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            loss_hist.append(total_loss)
            pbar.set_description(f"Pretrain Loss: {total_loss:.4f}")

    ########################################################################
    # Finetune on each task and evaluate
    ########################################################################
    results_ft = {}
    done_skipping = False
    for task in tasks:
        if task in "BUR":
            done_skipping = True
        if not done_skipping:
            continue
        if task in ["SS3", "SS8", "BUR"]:
            metric = "ACC"
            loss_str = "cross_entropy"
            format_func = format_sequence_and_label
        elif task in ["ASA", "HPR"]:
            metric = "PCC"
            loss_str = "mse"
            format_func = format_sequence_and_value
        elif task in ["PPI", "EPI"]:
            metric = "AUC"
            loss_str = "cross_entropy"
            format_func = format_sequence_and_label
        else:
            raise ValueError(f"Unknown task {task}")

        ft_epochs = 16 if task in ["EPI", "PPI"] else 64
        ft_batch_size = 32
        ft_accum_steps = 1
        embed_lr = 1e-2

        print("\n" + "-" * 60)
        print(f"Evaluating task {task} with {ft_epochs} epochs...")

        base_model, head = finetune_on_task(
            task=task,
            model=model,
            dataset=dataset,
            sp=sp,
            banned_tokens=banned_tokens,
            format_func=format_func,
            metric=metric,
            loss_str=loss_str,
            device=device,
            dtype=dtype,
            batch_size=ft_batch_size,
            num_accumulation_steps=ft_accum_steps,
            num_epochs=ft_epochs,
            lr=finetuning_lr,
            embed_lr=embed_lr,
            test_freq=100,
        )

        result = test_model(task, base_model, head, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype)
        results_ft[task] = result

        # Evaluate CB513 if relevant
        if task == "SS3":
            result_cb513 = test_model(
                "CB513SS3", base_model, head, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype
            )
            results_ft["CB513SS3"] = result_cb513

        if task == "SS8":
            result_cb513 = test_model(
                "CB513SS8", base_model, head, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype
            )
            results_ft["CB513SS8"] = result_cb513

    # Save final results
    with open(f"ProteinGLUE_{output_suffix}_results.csv", "w") as f:
        for tsk, res in results_ft.items():
            f.write(f"{tsk},{res}\n")


########################################################################
# Argparse Entry Point
########################################################################
def parse_args():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="ProteinGLUE experiment runner.")
    parser.add_argument("--sp_dir", type=str, required=True, help="Path to the SentencePiece model.")
    parser.add_argument("--tokenizer_offset", type=int, default=2048, help="Offset to be added to token IDs.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the PyTorch model file.")
    parser.add_argument(
        "--banned_token",
        type=int,
        default=2044,
    )
    parser.add_argument(
        "--extra_pretrain",
        action="store_true",
        help="Whether to do extra pretraining on all sequences.",
    )
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=4,
        help="Number of extra pretraining epochs.",
    )
    parser.add_argument(
        "--pretraining_num_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for extra pretraining.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extra pretraining.")
    parser.add_argument("--pretraining_lr", type=float, default=1e-3, help="Learning rate for extra pretraining.")
    parser.add_argument("--finetuning_lr", type=float, default=2e-4, help="Learning rate for finetuning tasks.")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix for the output file name (results CSV).",
    )
    return parser.parse_args()


def main():
    """
    Main function to parse arguments and run the experiment.
    """
    args = parse_args()
    run_experiment(
        sp_dir=args.sp_dir,
        tokenizer_offset=args.tokenizer_offset,
        model_dir=args.model_dir,
        banned_token=args.banned_token,
        extra_pretrain=args.extra_pretrain,
        pretraining_epochs=args.pretraining_epochs,
        pretraining_num_accum_steps=args.pretraining_num_accum_steps,
        batch_size=args.batch_size,
        pretraining_lr=args.pretraining_lr,
        finetuning_lr=args.finetuning_lr,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()