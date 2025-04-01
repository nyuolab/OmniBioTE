from torch.utils.data import Dataset
import os
import random
import numpy as np
import torch
import gzip
from threading import Thread
from queue import Queue

UNKNOWN_TOKEN = 0
EOS_TOKEN = 3
MASK_TOKEN = 2
PAD_TOKEN = 1

def line_reader(filenames, tokenizer, banned_tokens, offset):
    '''
    Generator function that yields a single sequence at a time from the entire list of files.
    It preloads one file ahead to improve efficiency.
    '''
    file_queue = Queue(maxsize=4)  # Queue to hold preloaded files
    max_chunk_size = 4096

    def file_loader():
        while True:
            for filename in filenames:
                try:
                    with gzip.open(filename, 'r') as f:
                        data = f.read().decode('utf-8')
                    file_queue.put((filename, data))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    file_queue.put((filename, ""))  # Put empty data to avoid blocking

    # Start the loader thread
    loader_thread = Thread(target=file_loader)
    loader_thread.start()

    while True:
        item = file_queue.get()
        
        if item is None:
            continue  # Skip empty blocks

        filename, block = item

        sub_blocks = block.split("<EOS>")

        # Shuffle sub_blocks
        sub_block_order = np.arange(len(sub_blocks))
        np.random.shuffle(sub_block_order)

        for idx in sub_block_order:
            sub_block = sub_blocks[idx]
            if len(sub_block) > 0:
                sub_block = sub_block.strip() + "<EOS>"

                # pick a random chunk of max_chunk_size tokens
                if len(sub_block) > max_chunk_size:
                    start = np.random.randint(0, len(sub_block) - max_chunk_size)
                    end = start + max_chunk_size

                    tokenized = tokenizer.Encode(sub_block[start:end])
                else:
                    tokenized = tokenizer.Encode(sub_block)

                tokenized = [t + offset for t in tokenized if t not in banned_tokens]

                if len(tokenized) > 0:
                    yield np.int32(tokenized)

    # Wait for the loader thread to finish
    loader_thread.join()


def get_sequence(reader, ctx_len, USE_PADDING=False):
    '''
    This function pulls lines from the reader until the sequence is ctx_len long or shorter, then pads the sequence (if padding is enabled), finally yielding it
    '''
    sequence = [] # the current sequence
    seq_len = 0 # the length of the current sequence

    while True:
        line = next(reader)

        seq_len = len(sequence)

        # if the sequence is full, yield it
        if seq_len == ctx_len:
            yield sequence
            sequence = []
            seq_len = 0

            continue

        # if adding this line would make the sequence too long, either pad it or truncate it
        if seq_len + len(line) > ctx_len:
            if USE_PADDING: # if padding is enabled, pad the sequence if we can't fit the whole line in
                if seq_len == 0:
                    # if the sequence is empty, we don't want to return an empty sequence, so we skip this line
                    continue

                # if the sequence is not empty, pad it and yield it
                sequence.extend([PAD_TOKEN] * (ctx_len - seq_len))
            else:
                # if padding is not enabled, truncate the line and yield the sequence
                sequence.extend(line[:ctx_len - seq_len])

            yield sequence
            sequence = []
            seq_len = 0

            continue

        # otherwise, add the line to the sequence
        # in the case that the sequence was yielded, this line will be the first line of the next sequence
        sequence.extend(line)
        seq_len = len(sequence)

        if seq_len > ctx_len:
            raise ValueError("Unreachable code reached")

class OmniDataset(Dataset):
    def __init__(
        self,
        directories,
        ctx_len,
        tokenizers,
    ):
        """
        Custom Dataset for loading OmniBioTE data

        Parameters:
        - directories: list of dictionaries containign three keys: "path", "type", "fraction"
        - batch_size: batch size for DataLoader
        """
        self.directories = directories
        self.ctx_len = ctx_len

        for i in range(0, len(directories)):
            directories[i]["files"] = [os.path.join(directories[i]["path"], f) for f in os.listdir(directories[i]["path"]) if os.path.isfile(os.path.join(directories[i]["path"], f))]
        
        for directory in directories:
            random.shuffle(directory["files"])
        
        self.distribution = [directory["fraction"] for directory in directories]
        assert sum(self.distribution) == 1, "Fractions must sum to 1"

        readers = [line_reader(directory["files"], tokenizers[directory["type"]], directory["banned_tokens"], directory["offset"]) for directory in directories]
        self.sequence_generators = [get_sequence(reader, self.ctx_len, USE_PADDING=False) for reader in readers]

    def __len__(self):
        return int(1e12) # return a large number to allow for infinite iterations

    def __getitem__(self, _):
        sequence_gen = random.choices(self.sequence_generators, self.distribution)[0]
            
        return torch.tensor(next(sequence_gen), dtype=torch.long)