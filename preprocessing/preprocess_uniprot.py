import gzip
from tqdm import tqdm

def read_large_gz_file(file_path, chunk_size=128*1024**2):
    """
    Generator function to read a large gzip file in chunks.

    Args:
    - file_path (str): Path to the gzip file.
    - chunk_size (int): Size of each chunk to read. Default is set to 128 MB.

    Yields:
    - str: A chunk of data from the file.
    """
    with gzip.open(file_path, 'rt') as f:  # 'rt' mode for text reading
        while True:
            chunk = f.read(chunk_size)  # Read a chunk of the file
            if not chunk:  # Break the loop if chunk is empty
                break
            yield chunk  # Yield the chunk to be processed

def process_chunk(chunk, residual):
    """
    Processes a chunk of text data, splitting it into sequences.

    Args:
    - chunk (str): A chunk of text data.
    - residual (str): Residual data from the previous chunk processing.

    Returns:
    - list: A list of processed sequences.
    - str: Residual data that should be combined with the next chunk.
    """
    chunk = residual + chunk  # Combine residual with the current chunk
    split = chunk.split('>')  # Split the chunk based on the delimiter
    if split[0] == '':
        split = split[1:]  # Remove the first element if it's empty
    
    residual = split[-1]  # Set the last element as the new residual
    split = split[:-1]  # Remove the last element from current processing

    sequences = []
    for s in split:
        seq = ''.join(s.split('\n')[1:])  # Join lines to form a sequence
        sequences.append(seq)
    
    sequences = [s for s in sequences if s != '']  # Remove any empty sequences
    
    return sequences, residual

total_len = 0
file_path = r'E:\uniref100.fasta.gz'

residual = ''  # Initialize residual for processing
chunk_num = 0  # Counter for chunk number
for chunk in tqdm(read_large_gz_file(file_path), total=1441):  # total is 1441 if using chunk_size=128*1024**2
    sequences, residual = process_chunk(chunk, residual)  # Process each chunk

    out_str = "<EOS><protein>".join(sequences)  # Format sequences for output
    out_str = "<protein>" + out_str + "<EOS>"
    
    # Write processed sequences to a gzipped file
    with gzip.open(r'E:\uniref100_processed\uniref100_{}.txt.gz'.format(chunk_num), 'wt') as f:
        f.write(out_str)
    
    chunk_num += 1  # Increment chunk number

# Process last residual data
sequences, residual = process_chunk('', residual)
out_str = "<EOS><protein>".join(sequences)
out_str = "<protein>" + out_str + "<EOS>"
with gzip.open(r'E:\uniref100_processed\uniref100_{}.txt.gz'.format(chunk_num+1), 'wt') as f:
    f.write(out_str)  # Write the final processed data to a file
