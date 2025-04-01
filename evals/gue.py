import os
import copy
import fire
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score
from model import OmniBioTA
from loader import EOS_TOKEN, PAD_TOKEN

device = "cuda"

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

def pad_attn(attn_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    In-place modifies 'attn_mask' to mask out positions after any PAD_TOKEN in 'x'.
    This effectively prevents the model from attending to padded positions.

    Args:
        attn_mask (torch.Tensor): Attention mask of shape (batch, seq_len, seq_len).
        x (torch.Tensor): Tokenized input of shape (batch, seq_len).

    Returns:
        torch.Tensor: The modified attention mask.
    """
    pad_locations = (x == PAD_TOKEN).nonzero()
    for i in range(len(pad_locations)):
        b_idx, pad_pos = pad_locations[i][0].item(), pad_locations[i][1].item()
        # Mask out positions after pad position (both in row and column)
        attn_mask[b_idx, pad_pos + 1 :, :] = -1e38
        attn_mask[b_idx, :, pad_pos + 1 :] = -1e38
    return attn_mask

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
                X.append(seq)
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
    model: torch.nn.Module,
    sp: spm.SentencePieceProcessor,
    device: str,
    dtype=torch.bfloat16,
    batch_size: int = 4,
    num_accumulation_steps: int = 8,
    lr: float = 1e-4,
    embed_lr: float = 1e-2,
    head_lr: float = 1e-2,
    test_freq: int = 100,
    head_multiplier: float = None,
    embedding_type: str = "first",
    wd=True
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
    base_model = copy.deepcopy(model)
    base_model.train()

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_task(task)
    Y_train = np.array([int(y) for y in Y_train])
    Y_val = np.array([int(y) for y in Y_val])
    Y_test = np.array([int(y) for y in Y_test])

    # Create a linear head, dimension = number of classes
    n_classes = max(Y_train) + 1
    head = torch.nn.Linear(base_model.transformer.wte.weight.shape[-1], n_classes).to(device).to(dtype)
    out_scale = 1024 / base_model.transformer.wte.weight.shape[-1]
    #out_scale = 1.0

    # Setup parameter groups
    param_groups = [
        {
            "params": [p for name, p in base_model.named_parameters() if "wte" in name],
            "lr": embed_lr
        },
        {
            "params": [p for name, p in base_model.named_parameters() if "wte" not in name],
            "lr": lr
        },
        {
            "params": head.parameters(),
            "lr": head_lr
        }
    ]

    num_steps = 30000

    if wd:
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)

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
                        tokenized = sp.EncodeAsIds("<DNA>" + sequence) + [EOS_TOKEN]
                        x_tokenized.append(tokenized)
                        lens.append(len(tokenized))

                    if batch_size != 1:
                        max_len = max(lens)
                        x_padded = torch.full((len(x_subset), max_len), PAD_TOKEN, dtype=torch.long, device=device)
                        for j, tokens in enumerate(x_tokenized):
                            x_padded[j, : lens[j]] = torch.tensor(tokens, dtype=torch.long, device=device)

                        attn_mask = torch.zeros((x_padded.shape[0], x_padded.shape[1], x_padded.shape[1]), device=device)
                        attn_mask = pad_attn(attn_mask, x_padded)
                        attn_mask = attn_mask.unsqueeze(1).expand(
                            -1,
                            base_model.transformer.h[0].attn.n_head,
                            -1,
                            -1
                        ).to(dtype)

                        embeddings = base_model(x_padded, attn_mask=attn_mask, return_embeddings=True)
                        if embedding_type == "first":
                            embeddings = embeddings[:, 0]
                        elif embedding_type == "mean":
                            embeddings = embeddings.mean(dim=1)
                        elif embedding_type == "max":
                            embeddings = embeddings.max(dim=1)[0]
                        else:
                            raise ValueError(f"Unknown embedding type: {embedding_type}")

                        if head_multiplier is not None:
                            embeddings *= head_multiplier
                        y_pred = head(embeddings) * out_scale
                        preds += y_pred.argmax(dim=-1).cpu().numpy().tolist()
                    else:
                        x_single = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                        embeddings = base_model(x_single, return_embeddings=True)
                        if embedding_type == "first":
                            embeddings = embeddings[:, 0]
                        elif embedding_type == "mean":
                            embeddings = embeddings.mean(dim=1)
                        elif embedding_type == "max":
                            embeddings = embeddings.max(dim=1)[0]
                        else:
                            raise ValueError(f"Unknown embedding type: {embedding_type}")
                        if head_multiplier is not None:
                            embeddings *= head_multiplier
                        y_pred = head(embeddings) * out_scale
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
                tokenized = sp.EncodeAsIds("<DNA>" + X_train[idx]) + [EOS_TOKEN]
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))

            if batch_size != 1:
                max_len = max(lens)
                x_padded = torch.full((batch_size, max_len), PAD_TOKEN, dtype=torch.long, device=device)
                for j, tokens in enumerate(x_tokenized):
                    x_padded[j, : lens[j]] = torch.tensor(tokens, dtype=torch.long, device=device)

                attn_mask = torch.zeros((x_padded.shape[0], x_padded.shape[1], x_padded.shape[1]), device=device)
                attn_mask = pad_attn(attn_mask, x_padded)
                attn_mask = attn_mask.unsqueeze(1).expand(
                    -1,
                    base_model.transformer.h[0].attn.n_head,
                    -1,
                    -1
                ).to(dtype)
            else:
                x_padded = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                attn_mask = None  # Not needed for single-sample batch

            optimizer.zero_grad()
            if batch_size != 1:
                embeddings = base_model(x_padded, attn_mask=attn_mask, return_embeddings=True)
                if embedding_type == "first":
                            embeddings = embeddings[:, 0]
                elif embedding_type == "mean":
                    embeddings = embeddings.mean(dim=1)
                elif embedding_type == "max":
                    embeddings = embeddings.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_type}")
            else:
                embeddings = base_model(x_padded, return_embeddings=True)
                if embedding_type == "first":
                    embeddings = embeddings[:, 0]
                elif embedding_type == "mean":
                    embeddings = embeddings.mean(dim=1)
                elif embedding_type == "max":
                    embeddings = embeddings.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_type}")
            
            if head_multiplier is not None:
                embeddings *= head_multiplier
            y_pred = head(embeddings) * out_scale
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
                tokenized = sp.EncodeAsIds("<DNA>" + sequence) + [EOS_TOKEN]
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))

            if batch_size != 1:
                max_len = max(lens)
                x_padded = torch.full((len(x_subset), max_len), PAD_TOKEN, dtype=torch.long, device=device)
                for j, tokens in enumerate(x_tokenized):
                    x_padded[j, : lens[j]] = torch.tensor(tokens, dtype=torch.long, device=device)

                attn_mask = torch.zeros((x_padded.shape[0], x_padded.shape[1], x_padded.shape[1]), device=device)
                attn_mask = pad_attn(attn_mask, x_padded)
                attn_mask = attn_mask.unsqueeze(1).expand(
                    -1,
                    base_model.transformer.h[0].attn.n_head,
                    -1,
                    -1
                ).to(dtype)

                embeddings = base_model(x_padded, attn_mask=attn_mask, return_embeddings=True)
                if embedding_type == "first":
                    embeddings = embeddings[:, 0]
                elif embedding_type == "mean":
                    embeddings = embeddings.mean(dim=1)
                elif embedding_type == "max":
                    embeddings = embeddings.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_type}")
            else:
                x_single = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                embeddings = base_model(x_single, return_embeddings=True)
                if embedding_type == "first":
                    embeddings = embeddings[:, 0]
                elif embedding_type == "mean":
                    embeddings = embeddings.mean(dim=1)
                elif embedding_type == "max":
                    embeddings = embeddings.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_type}")
            
            if head_multiplier is not None:
                embeddings *= head_multiplier
            y_pred = head(embeddings) * out_scale
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
                    tokenized = sp.EncodeAsIds("<DNA>" + sequence) + [EOS_TOKEN]
                    x_tokenized.append(tokenized)
                    lens.append(len(tokenized))

                if batch_size != 1:
                    max_len = max(lens)
                    x_padded = torch.full((len(x_subset), max_len), PAD_TOKEN, dtype=torch.long, device=device)
                    for j, tokens in enumerate(x_tokenized):
                        x_padded[j, : lens[j]] = torch.tensor(tokens, dtype=torch.long, device=device)

                    attn_mask = torch.zeros((x_padded.shape[0], x_padded.shape[1], x_padded.shape[1]), device=device)
                    attn_mask = pad_attn(attn_mask, x_padded)
                    attn_mask = attn_mask.unsqueeze(1).expand(
                        -1,
                        base_model.transformer.h[0].attn.n_head,
                        -1,
                        -1
                    ).to(dtype)
                    embeddings = base_model(x_padded, attn_mask=attn_mask, return_embeddings=True)
                    if embedding_type == "first":
                        embeddings = embeddings[:, 0]
                    elif embedding_type == "mean":
                        embeddings = embeddings.mean(dim=1)
                    elif embedding_type == "max":
                        embeddings = embeddings.max(dim=1)[0]
                    else:
                        raise ValueError(f"Unknown embedding type: {embedding_type}")
                else:
                    x_single = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                    embeddings = base_model(x_single, return_embeddings=True)
                    if embedding_type == "first":
                        embeddings = embeddings[:, 0]
                    elif embedding_type == "mean":
                        embeddings = embeddings.mean(dim=1)
                    elif embedding_type == "max":
                        embeddings = embeddings.max(dim=1)[0]
                    else:
                        raise ValueError(f"Unknown embedding type: {embedding_type}")
                
                if head_multiplier is not None:
                    embeddings *= head_multiplier
                y_pred = head(embeddings) * out_scale
                preds.extend(y_pred.argmax(dim=-1).cpu().numpy().tolist())

            test_mcc = matthews_corrcoef(Y_test, preds)
            test_f1 = f1_score(Y_test, preds, average="weighted")
            print(f"(Best model) Test MCC: {test_mcc*100:.2f}, Test F1: {test_f1*100:.2f}")

    return test_mcc, test_f1

def main(
    sp_dir: str,
    model_dir: str,
    tokenizer_suffix: str,
    tokenizer_offset: int = 0,
    num_accum_steps: int = 1,
    batch_size: int = 32,
    lr: float = 0.00015625,
    embed_lr: float = 0.00015625,
    head_lr: float = 1e-2,
    output_suffix: str = "",
    head_multiplier: float = None,
    memory_efficient: bool = False,
    embedding_type: str = "first",
    iter_cap: int = None,
    wd: bool = True,
    start_at = None,
):
    print(f"Loading tokenizer from {sp_dir}...")
    print(f"Loading model from {model_dir}...")

    genbank_banned_tokens = {
        "2k": 2037
    }
    assert (
        tokenizer_suffix in genbank_banned_tokens
    ), f"Unknown tokenizer suffix {tokenizer_suffix}. Allowed: {list(genbank_banned_tokens.keys())}"

    banned_token = genbank_banned_tokens[tokenizer_suffix]

    print(f"Using tokenizer offset {tokenizer_offset}")
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

    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_dir)
    sp = TokenizerWrapper(sp, tokenizer_offset, [1, 2, banned_token])

    def recurse_load(m):
        """
        Recursively find the original module if it is wrapped by a
        flash-attention-style `_orig_mod`.
        """
        if hasattr(m, "_orig_mod"):
            return recurse_load(m._orig_mod)
        return m

    # Load pretrained model
    base_model = torch.load(model_dir, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)
    model_sd = base_model.state_dict()
    config = base_model.config
    if memory_efficient:
        config.memory_efficient = True
    model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

    # Finetune on tasks
    results_ft = {}
    for task in tasks:
        if start_at is not None and start_at not in task:
            continue

        start_at = None

        print("---------------------------------------------------------------")
        print(f"Evaluating task '{task}'")
        mcc, f1_ = finetune_on_task(
            task,
            model,
            sp,
            banned_tokens=[1, 2, banned_token],
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            num_accumulation_steps=num_accum_steps,
            lr=lr,
            embed_lr=embed_lr,
            head_lr=head_lr,
            test_freq=100,
            tokenizer_offset=tokenizer_offset,
            head_multiplier=head_multiplier,
            embedding_type=embedding_type,
            iter_cap=iter_cap,
            wd=wd
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