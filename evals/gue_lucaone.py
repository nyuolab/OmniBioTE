import os
import copy
import fire
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score
import itertools
from typing import Sequence, List
from transformers import AutoModel

device = "cuda"

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
    
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
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

def gene_seq_replace(seq):
    '''
    Nucleic acid: gene replace: A->1, U/T->2, C->3, G->4, N->5
    :param seq:
    :return:
    '''
    new_seq = ""
    for ch in seq:
        if ch in ["A", "a"]:
            new_seq += "1"
        elif ch in ["T", "U", "t", "u"]:
            new_seq += "2"
        elif ch in ["C", "c"]:
            new_seq += "3"
        elif ch in ["G", "g"]:
            new_seq += "4"
        else: # unknown
            new_seq += "5"
    return new_seq

def load_task(task_dir: str):
    """
    Loads train/dev/test CSV files from a specified directory.

    Each line in the CSV file is assumed to be "sequence,label".

    Args:
        task_dir (str): The path to the directory containing 'train.csv', 'dev.csv', and 'test.csv'.

    Returns:
        Tuple of lists: (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    def _load_csv(file_path):
        X, Y = [], []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # skip header
                seq, label = line.strip().split(",")
                X.append(gene_seq_replace(seq))
                Y.append(label)
        return X, Y

    train_file = os.path.join(task_dir, "train.csv")
    val_file = os.path.join(task_dir, "dev.csv")
    test_file = os.path.join(task_dir, "test.csv")

    X_train, Y_train = _load_csv(train_file)
    X_val, Y_val = _load_csv(val_file)
    X_test, Y_test = _load_csv(test_file)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def finetune_on_task(
    task: str,
    sp: spm.SentencePieceProcessor,
    device: str,
    dtype=torch.bfloat16,
    num_epochs: int = 4,
    batch_size: int = 4,
    num_accumulation_steps: int = 8,
    lr: float = 1e-4,
    embed_lr: float = 1e-2,
    head_lr: float = 1e-2,
    test_freq: int = 100,
):
    """
    Finetunes the given model on a single classification task.

    The data is loaded from CSV files in 'task' directory. A linear head is added
    on top of the model embeddings (from base_model) and trained end-to-end.

    Args:
        task (str): Path to the task directory (containing train.csv, dev.csv, test.csv).
        model (torch.nn.Module): The pretrained model to finetune.
        sp (spm.SentencePieceProcessor): The SentencePiece tokenizer.
        banned_tokens (list): List of tokens to exclude from any input.
        device (str): Device to use for training ('cuda:0' or 'cpu').
        dtype: Torch dtype (default: torch.bfloat16).
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        num_accumulation_steps (int): Accumulation steps for gradient update.
        lr (float): Learning rate for the transformer layers.
        embed_lr (float): Learning rate for the embedding layer.
        test_freq (int): Frequency of validation checks.
        tokenizer_offset (int): Offset to add to each token ID.

    Returns:
        Tuple[float, float]: The final MCC and F1 on the test set after training,
                             followed by loading the best checkpoint and re-evaluating.
    """
    # Copy the model so as not to modify the original reference
    #base_model = copy.deepcopy(model)
    base_model = AutoModel.from_pretrained("Yuanfei/LucaOne", trust_remote_code=True)
    base_model.to(device=device, dtype=dtype)
    base_model.train()

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_task(task)
    Y_train = np.array([int(y) for y in Y_train])
    Y_val = np.array([int(y) for y in Y_val])
    Y_test = np.array([int(y) for y in Y_test])

    # Create a linear head, dimension = number of classes
    n_classes = max(Y_train) + 1
    head = torch.nn.Linear(2560, n_classes).to(device).to(dtype)

    # Setup parameter groups
    param_groups = [
        {
            "params": [p for name, p in base_model.named_parameters() if "embed" in name],
            "lr": embed_lr
        },
        {
            "params": [p for name, p in base_model.named_parameters() if "embed" not in name],
            "lr": lr
        },
        {
            "params": head.parameters(),
            "lr": head_lr
        }
    ]

    num_steps = 30000#int(num_epochs * len(X_train) / (batch_size * num_accumulation_steps))
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[embed_lr, lr, head_lr],
        total_steps=num_steps,
        pct_start=0.05
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Tracking best val MCC
    val_mcc = 0
    val_f1 = 0
    best_val_mcc = 0
    best_model_and_head = None
    total_loss = 0

    pbar = tqdm(range(num_steps))
    for step in pbar:
        # Periodically evaluate on validation
        if step % max(1, (num_steps // test_freq)) == 0:
            base_model.eval()
            head.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(X_val), batch_size):
                    pbar.set_description(
                        f"(Testing {i}/{len(X_val)}) Loss: {total_loss:.4f}, val_mcc: {val_mcc*100:.2f}, val_f1: {val_f1*100:.2f}"
                    )
                    y_batch = torch.tensor(Y_val[i : i + batch_size], dtype=torch.long, device=device)
                    x_subset = X_val[i : i + batch_size]

                    # Tokenize each sample
                    lens = []
                    x_tokenized = []
                    
                    for sequence in x_subset:
                        tokenized = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
                        x_tokenized.append(tokenized)
                        lens.append(len(tokenized))
                    
                    max_len = max(lens)
                    x_padded = torch.full((len(x_tokenized), max_len), fill_value=sp.pad_token_id, dtype=torch.long, device=device)
                    for i, seq in enumerate(x_tokenized):
                        x_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

                    x_single = torch.tensor(x_padded, dtype=torch.long, device=device)
                    embeddings = base_model(input_ids=x_single, token_type_ids=torch.zeros_like(x_single)).hidden_states[:, 0]
                    y_pred = head(embeddings)
                    preds.extend(y_pred.argmax(dim=-1).cpu().numpy().tolist())

                val_mcc = matthews_corrcoef(Y_val, preds)
                val_f1 = f1_score(Y_val, preds, average="weighted")

                # Save best checkpoint
                if val_mcc > best_val_mcc:
                    best_val_mcc = val_mcc
                    best_model_and_head = (
                        copy.deepcopy(base_model.state_dict()),
                        copy.deepcopy(head.state_dict())
                    )

            base_model.train()
            head.train()

        # Training / forward-backward over 'num_accumulation_steps' mini-batches
        base_model.train()
        head.train()
        total_loss = 0
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            y_batch = torch.tensor(Y_train[indices], dtype=torch.long, device=device)

            lens = []
            x_tokenized = []
            for idx in indices:
                tokenized = sp.EncodeAsIds("[CLS]" + X_train[idx] + "[SEP]")
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))

            max_len = max(lens)
            x_padded = torch.full((len(x_tokenized), max_len), fill_value=sp.pad_token_id, dtype=torch.long, device=device)
            for i, seq in enumerate(x_tokenized):
                x_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

            optimizer.zero_grad()
            embeddings = base_model(input_ids=x_padded, token_type_ids=torch.zeros_like(x_padded)).hidden_states[:, 0]

            y_pred = head(embeddings)
            loss = loss_fn(y_pred, y_batch) / num_accumulation_steps
            loss.backward()

            total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}, val_mcc: {val_mcc*100:.2f}, val_f1: {val_f1*100:.2f}")

    pbar.close()

    # Evaluate final model on test set
    base_model.eval()
    head.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), batch_size):
            y_batch = torch.tensor(Y_test[i : i + batch_size], dtype=torch.long, device=device)
            x_subset = X_test[i : i + batch_size]

            lens = []
            x_tokenized = []
            for sequence in x_subset:
                tokenized = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))
            
            max_len = max(lens)
            x_padded = torch.full((len(x_tokenized), max_len), fill_value=sp.pad_token_id, dtype=torch.long, device=device)
            for i, seq in enumerate(x_tokenized):
                x_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

            embeddings = base_model(input_ids=x_padded, token_type_ids=torch.zeros_like(x_padded)).hidden_states[:, 0]

            y_pred = head(embeddings)
            preds.extend(y_pred.argmax(dim=-1).cpu().numpy().tolist())

        test_mcc = matthews_corrcoef(Y_test, preds)
        test_f1 = f1_score(Y_test, preds, average="weighted")
        print(f"Test MCC: {test_mcc*100:.2f}, Test F1: {test_f1*100:.2f}")

    # Load the best model (by val_mcc) and re-evaluate on test
    if best_model_and_head is not None:
        base_model.load_state_dict(best_model_and_head[0])
        head.load_state_dict(best_model_and_head[1])

        base_model.eval()
        head.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_test), batch_size):
                y_batch = torch.tensor(Y_test[i : i + batch_size], dtype=torch.long, device=device)
                x_subset = X_test[i : i + batch_size]

                lens = []
                x_tokenized = []
                for sequence in x_subset:
                    tokenized = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
                    x_tokenized.append(tokenized)
                    lens.append(len(tokenized))
                
                max_len = max(lens)
                x_padded = torch.full((len(x_tokenized), max_len), fill_value=sp.pad_token_id, dtype=torch.long, device=device)
                for i, seq in enumerate(x_tokenized):
                    x_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

                embeddings = base_model(input_ids=x_padded, token_type_ids=torch.zeros_like(x_padded)).hidden_states[:, 0]
                y_pred = head(embeddings)
                preds.extend(y_pred.argmax(dim=-1).cpu().numpy().tolist())

            test_mcc = matthews_corrcoef(Y_test, preds)
            test_f1 = f1_score(Y_test, preds, average="weighted")
            print(f"(Best model) Test MCC: {test_mcc*100:.2f}, Test F1: {test_f1*100:.2f}")
    
    del base_model, optimizer
    torch.cuda.empty_cache()

    return test_mcc, test_f1

def main(
    num_accum_steps: int = 1,
    batch_size: int = 32,
    lr: float = 0.00015625,
    embed_lr: float = 0.00015625,
    head_lr: float = 1e-2,
    output_suffix: str = ""
):
    """
    Main entry point. Loads tokenizer and model, then iterates over GUE tasks to finetune/evaluate.

    Args:
        sp_dir (str): Path to the SentencePiece tokenizer model.
        model_dir (str): Path to the pretrained model checkpoint.
        tokenizer_suffix (str): Tokenizer key to select the correct banned token from genbank_banned_tokens.
        tokenizer_offset (int): ID offset applied to tokens (default=0).
        num_accum_steps (int): Gradient accumulation steps (default=4).
        batch_size (int): Batch size (default=32).
        lr (float): Finetuning learning rate (default=1e-3).
        embed_lr (float): Embedding layer learning rate (default=1e-4).
        output_suffix (str): Suffix appended to output CSV filename.
    """
    print(f"Finetuning with lr={lr}, embed_lr={embed_lr}, head_lr={head_lr}, batch_size={batch_size}, num_accum_steps={num_accum_steps}")
    print(f"Saving results with suffix: {output_suffix}")

    base_dir = "../datasets/GUE"
    all_subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    tasks = [
        os.path.join(all_subdirs[i], subdir)
        for i in range(len(all_subdirs))
        for subdir in os.listdir(all_subdirs[i])
    ]
    dtype = torch.bfloat16

    sp = Alphabet.from_predefined("gene_prot")

    # Finetune on tasks
    results_ft = {}
    for task in tasks:
        if "EMP" in task:
            epochs = 32
        elif "mouse" in task:
            epochs = 100
            continue
        elif "covid" in task:
            epochs = 32
            continue
        elif "tata" in task:
            epochs = 32
            continue
        elif "notata" in task:
            epochs = 32
            continue
        elif "all" in task:
            epochs = 32
            continue
        elif "splice" in task:
            epochs = 32
        elif "tf" in task:
            epochs = 32
        else:
            raise ValueError(f"Unknown task type in path: {task}")

        print("---------------------------------------------------------------")
        print(f"Evaluating task '{task}', training for {epochs} epochs...")
        mcc, f1_ = finetune_on_task(
            task,
            sp,
            device=device,
            dtype=dtype,
            batch_size=batch_size if "covid" not in task else 8,
            num_accumulation_steps=num_accum_steps if "covid" not in task else 4,
            num_epochs=epochs,
            lr=lr,
            embed_lr=embed_lr,
            head_lr=head_lr,
            test_freq=100,
        )
        results_ft[task] = {"mcc": mcc, "f1": f1_}
        print("---------------------------------------------------------------")

    # Save results
    out_file = f"GUE_results_{output_suffix}.csv"
    with open(out_file, "w") as f:
        f.write("Task,MCC,F1\n")
        for t, metrics in results_ft.items():
            f.write(f"{t},{metrics['mcc']},{metrics['f1']}\n")

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)