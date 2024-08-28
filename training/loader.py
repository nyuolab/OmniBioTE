import numpy as np
import torch

EOS_TOKEN = 3
MASK_TOKEN = 2
PAD_TOKEN = 1

def data_loader_parallel(batch_queue, batch_generator, device):
    '''
    A function to load data in a separate thread to speed up training.

    Args:
        batch_queue: a queue to store the batches in
        batch_generator: a generator that yields batches
        device: the device to move the batches to
    '''
    while True:
        try:
            data = next(batch_generator)
            data = data.to(device)
            batch_queue.put(data)
        except StopIteration:
            break

def line_reader(filenames, banned_tokens):
    '''
    Generator function that yields a single sequence at a time from entire list of files
    '''
    while True:
        # shuffle filenames
        np.random.shuffle(filenames)

        chunk_size = 10 # number of files to load at a time. Each file takes ~100  MB of memory, so this should be tuned to the available memory
        # chunk files into groups of chunk_size
        chunked_filenames = np.split(filenames, np.arange(chunk_size, len(filenames), chunk_size))

        for name in chunked_filenames:
            block = []
            for filename in name:
                block.append(np.load(filename))
            
            block = np.concatenate(block)
            eos_indices = np.where(block == EOS_TOKEN)[0]
            sub_blocks = np.split(block, eos_indices + 1)
            
            # Create an array of indices and shuffle it
            sub_block_order = np.arange(len(sub_blocks))
            np.random.shuffle(sub_block_order)
            
            for idx in sub_block_order:
                sub_block = sub_blocks[idx]
                if len(sub_block) > 0:
                    # Use NumPy's vectorized operations for filtering
                    if len(banned_tokens) == 1:
                        mask = sub_block != banned_tokens[0]
                    else:
                        mask = ~np.isin(sub_block, banned_tokens)
                    sub_block = sub_block[mask]
                    yield np.int32(sub_block)
"""
def get_sequence(reader, ctx_len, USE_PADDING=False):
    '''
    This function pulls lines from the reader until the sequence is ctx_len long or shorter, then pads the sequence (if padding is enabled), finally yielding it
    '''
    sequence = [] # the current sequence
    seq_len = 0 # the length of the current sequence
    leftover = None

    while True:
        if leftover is not None: # if there was a leftover line from the previous iteration, use it
            line = leftover
            leftover = None
        else: # otherwise, get the next line from the reader
            line = next(reader)
        
        # if the line is longer than ctx_len, truncate it and save the leftover
        if USE_PADDING:
            if len(line) > ctx_len:
                leftover = line[ctx_len:]
                line = line[:ctx_len]
        else:
            if len(line) + seq_len > ctx_len:
                leftover = line[ctx_len - seq_len:]
                line = line[:ctx_len - seq_len]
        
        # if adding this line would make the sequence too long, pad it and yield it
        if seq_len + len(line) > ctx_len:
            if USE_PADDING:
                # if adding this line would make the sequence too long, pad it and yield it
                sequence.extend([PAD_TOKEN] * (ctx_len - seq_len))
                yield sequence

                # reset the sequence and sequence length
                sequence = []
                seq_len = 0

                continue
            else:
                # raise error because this should never happen with truncation
                raise ValueError("Unreachable code reached")

        
        if seq_len == ctx_len:
            yield sequence

            # reset the sequence and sequence length
            sequence = []
            seq_len = 0

            continue
        
        # otherwise, add the line to the sequence
        # in the case that the sequence was yielded, this line will be the first line of the next sequence
        sequence.extend(line)
        seq_len += len(line)
"""

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

def get_batch(generators, train_ints, return_pt=False, device="cpu"):
    '''
    This function pulls train_ints[0] lines from generators[0], train_ints[1] lines from generators[1], etc.
    '''
    while True:
        batch = []
        for generator, train_int in zip(generators, train_ints):
            for _ in range(train_int):
                batch.append(next(generator))
        
        # shuffle the batch
        np.random.shuffle(batch)

        if return_pt:
            yield torch.tensor(batch, dtype=torch.long, device=device)
        else:
            yield np.asarray(batch)

def get_sequence_multireader(readers, probs, ctx_len):
    '''
    This function pulls lines from the reader until the sequence is ctx_len long or shorter, then pads the sequence, finally yielding it
    '''
    sequence = [] # the current sequence
    seq_len = 0 # the length of the current sequence
    leftover = None

    while True:
        if leftover is not None: # if there was a leftover line from the previous iteration, use it
            line = leftover
            leftover = None
        else: # otherwise, get the next line from the reader
            reader = np.random.choice(readers, p=probs)
            line = next(reader)
        
        # if the line is longer than ctx_len, truncate it and save the leftover
        if len(line) > ctx_len:
            line = line[:ctx_len]
            leftover = line[ctx_len:]
        
        # if adding this line would make the sequence too long, pad it and yield it
        if seq_len + len(line) > ctx_len:
            # if adding this line would make the sequence too long, pad it and yield it
            sequence.extend([PAD_TOKEN] * (ctx_len - seq_len))
            yield sequence

            # reset the sequence and sequence length
            sequence = []
            seq_len = 0
        
        # otherwise, add the line to the sequence
        # in the case that the sequence was yielded, this line will be the first line of the next sequence
        sequence.extend(line)
        seq_len += len(line)

def get_batch_multireader(generator, batch_size, return_pt=False, device="cpu"):
    '''
    This function pulls train_ints[0] lines from generators[0], train_ints[1] lines from generators[1], etc.
    '''
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        
        # shuffle the batch
        np.random.shuffle(batch)

        if return_pt:
            yield torch.tensor(batch, dtype=torch.long, device=device)
        else:
            yield np.asarray(batch)