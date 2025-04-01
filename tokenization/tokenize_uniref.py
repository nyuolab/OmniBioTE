import os
from tqdm import tqdm
import gzip
import numpy as np
import pickle
import sentencepiece as spm

# base directories for the data
base_dirs = ["D:/uniref100_processed/"]
save_dirs = ["D:/uniref100 tokenized/"]

# if saved dirs don't exist, create them
for save_dir in save_dirs:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# get all .txt.gz files in the base directories
antibody_files = []
genbank_files = []
uniref_files = []
for base_dir in base_dirs:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt.gz"):
                if base_dir == base_dirs[0]:
                    antibody_files.append(os.path.join(root, file))
                elif base_dir == base_dirs[1]:
                    uniref_files.append(os.path.join(root, file))

# load the sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load("m.model")

def tokenize_files(files, save_dir):
    '''
    This function tokenizes the files in the given list and saves them to the given directory, in 50 million token chunks (~100 MB per file).
    '''
    pbar = tqdm(files)
    tokenized_sequence = []
    error_files = []
    num_tokens_processed = 0
    num_files_processed = 0
    for file in pbar:
        pbar.set_description("Tokenizing %s" % file)
        try:
            with gzip.open(file, "rt") as f:
                lines = f.read()
                
                # split lines by <EOS> token to get individual sequences and avoid memory issues
                lines = lines.split("<EOS>")

                # re-add the <EOS> token to each line
                lines = [line + "<EOS>" for line in lines]
        except:
            error_files.append(file)
            continue

        # tokenize each line
        for line in lines:
            tokens = sp.EncodeAsIds(line, add_bos=False, add_eos=False)
            tokenized_sequence += tokens
            num_tokens_processed += len(tokens)
            if num_tokens_processed > 1e9:
                pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**9:.3f} B tokens total)")
            elif num_tokens_processed > 1e6:
                pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**6:.3f} M tokens total)")

            if len(tokenized_sequence) >= 5e7:
                # save the tokenized sequence to a file
                save_fn = os.path.join(save_dir, "tokenized_%d.npy" % num_files_processed)

                # save as uint16 to save space
                np.save(save_fn, np.array(tokenized_sequence, dtype=np.uint16))
                
                # reset the tokenized sequence
                tokenized_sequence = []
                num_files_processed += 1

    # save the last tokenized sequence to a file
    if len(tokenized_sequence) > 0:
        # save the tokenized sequence to a file
        save_fn = os.path.join(save_dir, "tokenized_%d.npy" % num_files_processed)

        # save as uint16 to save space
        np.save(save_fn, np.array(tokenized_sequence, dtype=np.uint16))
    
    return error_files, num_tokens_processed

antibody_errors, antibody_token_count = tokenize_files(antibody_files, save_dirs[0])
uniref_errors, uniref_token_count = tokenize_files(uniref_files, save_dirs[1])

data_to_save = {
    "antibody_errors": antibody_errors,
    "antibody_token_count": antibody_token_count,
    "uniref_errors": uniref_errors,
    "uniref_token_count": uniref_token_count
}

# Specify a file name for saving
filename = "tokenization_data.pkl"

# Save the data using pickle
with open(filename, 'wb') as file:
    pickle.dump(data_to_save, file)