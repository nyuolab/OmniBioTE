import os
import sys
import argparse
import pickle
import re
import copy

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import pandas as pd
from transformers import AutoModel

########################################################################
# Global Device Setup
########################################################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    @classmethod
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def EncodeAsIds(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
    
    def DecodeAsIds(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """
        Convert a sequence of token IDs back into a string.
        
        Args:
            token_ids (Sequence[int]): List (or sequence) of token ids.
            skip_special_tokens (bool): Whether or not to filter out special tokens.
        
        Returns:
            str: The decoded string.
        """
        # Convert IDs back to tokens.
        tokens = [self.get_tok(i) for i in token_ids]
        
        # Optionally remove any special tokens (e.g. [PAD], [UNK], [CLS], [SEP], [MASK]).
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.all_special_tokens]
        
        # Heuristic for detokenization:
        # If any token looks like a special token (starts with '[' and ends with ']'),
        # join all tokens with a space. Otherwise (for instance, if all tokens are single characters),
        # join them with no delimiter.
        if any(token.startswith('[') and token.endswith(']') for token in tokens):
            decoded = "".join(tokens)
        else:
            decoded = "".join(tokens)
            
        return decoded

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
def format_sequence_and_label(sequence, label, sp):
    """
    Tokenizes a sequence with SentencePiece and aligns tokens with
    label mode (most frequent integer in chunk).

    Returns:
        (tokenized_sequence, label_modes, token_lens, sequence_length)
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)

    token_lens = [1 for token in tokenized_sequence]
    label_modes = []

    '''
    ptr = 0

    for i, length in enumerate(token_lens):
        if tokenized_sequence[i] == 0:  # unknown token
            length = 1
        label_chunk = label[ptr : ptr + length]
        label_modes.append(np.bincount(label_chunk).argmax())
        ptr += length
    '''
    label_modes = label

    sequence_length = len(sequence)
    return tokenized_sequence, label_modes, token_lens, sequence_length


def format_sequence_and_value(sequence, label, sp):
    """
    Tokenizes a sequence with SentencePiece and aligns tokens with
    label value means (average float in chunk).

    Returns:
        (tokenized_sequence, label_means, token_lens, sequence_length)
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)

    token_lens = [1 for token in tokenized_sequence]
    label_means = []

    '''
    ptr = 0

    for i, length in enumerate(token_lens):
        if tokenized_sequence[i] == 0:  # unknown token
            length = 1
        label_chunk = label[ptr : ptr + length]
        label_means.append(np.mean(label_chunk))
        ptr += length
    '''
    label_means = label

    sequence_length = len(sequence)
    return tokenized_sequence, label_means, token_lens, sequence_length


def process_data(sp, sequences, target, format_func, prefix):
    """
    Applies SentencePiece tokenization/label formatting and adds an EOS token.

    Args:
        sp (SentencePieceProcessor): Tokenizer instance.
        sequences (list): List of raw sequences.
        target (list): List of raw label structures.
        format_func (callable): Function to format each sequence/labels.
        prefix (list): Prefix tokens to prepend (e.g., <protein>).

    Returns:
        (list_of_tokenized_sequences, list_of_formatted_targets)
    """
    sequences_tokenized = []
    targets = []

    for i in tqdm(range(len(sequences))):
        tokenized_sequence, target_values, _, _ = format_func(
            sequences[i], target[i], sp
        )
        sequences_tokenized.append(tokenized_sequence)
        targets.append(target_values)

    # Append prefix and EOS_TOKEN
    sequences_tokenized = [prefix + item + sp.EncodeAsIds("[SEP]") for item in sequences_tokenized]
    return sequences_tokenized, targets


def get_training_sets(task, dataset, sp, format_func, prefix):
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
            sp, X_train_raw, y_train_raw, format_func, prefix
        )
        X_val, y_val = process_data(
            sp, X_val_raw, y_val_raw, format_func, prefix
        )
    else:
        X_train, y_train = None, None
        X_val, y_val = None, None

    X_test, y_test = process_data(
        sp, X_test_raw, y_test_raw, format_func, prefix
    )

    # Normalize certain tasks
    if task in ["ASA", "HPR"]:
        normalizer = create_normalizer(y_train_raw)
        y_train = normalizer(y_train)
        y_val = normalizer(y_val)
        y_test = normalizer(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def finetune_on_task(
    task,
    model,
    dataset,
    sp,
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
    prefix = sp.EncodeAsIds("[CLS]")
    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(
        task, dataset, sp, format_func, prefix
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
        2560, int(output_dim)
    ).to(device).to(dtype)

    # Parameter groups for different LR
    param_groups = [
        {"params": [p for n, p in base_model.named_parameters() if "embed" in n], "lr": embed_lr},
        {"params": [p for n, p in base_model.named_parameters() if "embed" not in n], "lr": lr},
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
            X_batch = [x[:1280] for x in X_batch]
            Y_batch = [y[:1279] for y in Y_batch]

            # Loop over each sample in batch
            for x, y in zip(X_batch, Y_batch):
                x_tensor = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
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
                    x_val = X_val[i][:1280]
                    y_val = Y_val[i][:1279]

                    x_tensor = torch.tensor(x_val, device=device, dtype=torch.long).unsqueeze(0)
                    embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
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

def test_model(task, model, head, dataset, sp, format_func, metric, loss_str, device, dtype=torch.bfloat16):
    """
    Evaluates the finetuned model on the test set of the specified task.
    Prints and returns the metric performance.
    """
    model.eval()
    head.eval()

    prefix = sp.EncodeAsIds("[CLS]")
    _, _, _, _, X_test, Y_test = get_training_sets(
        task, dataset, sp, format_func, prefix
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
            x_test = X_test[i][:1280]
            y_test = Y_test[i][:1279]

            x_tensor = torch.tensor(x_test, device=device, dtype=torch.long).unsqueeze(0)
            embeddings = model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
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
    print(f"Pretraining config => epochs: {pretraining_epochs}, accumulation steps: {pretraining_num_accum_steps}, lr: {pretraining_lr}")
    print(f"Finetuning lr: {finetuning_lr}")
    print(f"Output suffix: {output_suffix}")

    tasks = ["SS3", "SS8", "ASA", "HPR", "PPI", "BUR", "EPI"]
    dtype = torch.bfloat16

    # Create tokenizer    
    sp = Alphabet.from_predefined("gene_prot")

    # Load model
    model = AutoModel.from_pretrained("Yuanfei/LucaOne", trust_remote_code=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    # Load / Clean dataset
    dataset = get_cleaned_evals("../datasets/ProteinGLUE", force=False)

    ########################################################################
    # Finetune on each task and evaluate
    ########################################################################
    results_ft = {}
    for task in tasks:
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

        result = test_model(task, base_model, head, dataset, sp, format_func, metric, loss_str, device, dtype)
        results_ft[task] = result

        # Evaluate CB513 if relevant
        if task == "SS3":
            result_cb513 = test_model(
                "CB513SS3", base_model, head, dataset, sp, format_func, metric, loss_str, device, dtype
            )
            results_ft["CB513SS3"] = result_cb513

        if task == "SS8":
            result_cb513 = test_model(
                "CB513SS8", base_model, head, dataset, sp, format_func, metric, loss_str, device, dtype
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