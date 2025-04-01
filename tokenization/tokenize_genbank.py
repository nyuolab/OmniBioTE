import os
from tqdm import tqdm
import gzip
import numpy as np
import pickle
import sentencepiece as spm

# base directories for the data
base_dir = "E:/genbank_sequences_processed_train/"
save_dir = "E:/genbank_sequences_tokenized_train/"

# if saved dir doesn't exist, create them
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get all .txt.gz files in the base directories
genbank_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt.gz"):
            genbank_files.append(os.path.join(root, file))

if os.path.exists(r"E:\genbank_chkpt.pkl"):
    with open(r"E:\genbank_chkpt.pkl", "rb") as f:
        chkpt = pickle.load(f)

# load the sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load("m.model")

def tokenize_files(files, save_dir):
    '''
    This function tokenizes the files in the given list and saves them to the given directory, in 50 million token chunks (~100 MB per file).
    '''
    tokenized_sequence = [] # the current tokenized sequence to be saved to a file
    previously_processed_files = [] # the files that have already been processed
    error_files = [] # the files that caused an error
    num_files_processed = 0 # the number of files that have been processed
    num_tokens_processed = 0 # the number of tokens that have been processed
    processed_files = [] # the names of the numpy files that have been processed

    # if there is a checkpoint, load it
    if os.path.exists(r"E:\genbank_chkpt.pkl"):
        with open(r"E:\genbank_chkpt.pkl", "rb") as f:
            processed_files = pickle.load(f) # the names of the numpy files that have been processed
            temp = []

            for item in processed_files:
                temp.extend(item['output_fns'])
                previously_processed_files.append(item['input_fn'])
            indices = [int(fn.split('_')[-1].split('.')[0]) for fn in temp] # the indices of the numpy files that have been processed
            num_files_processed = max(indices) + 1 # set the starting index to the next numpy file to be created
    
    # check if an error file checkpoint exists
    if os.path.exists(r"E:\genbank_error_files.pkl"):
        with open(r"E:\genbank_error_files.pkl", "rb") as f:
            error_files = pickle.load(f)

    file_list = [file for file in files if file not in previously_processed_files] # the files that have not been processed yet
    pbar = tqdm(file_list)
    print(f"resuming at file {num_files_processed}")
    for file in pbar:
        processed_file_output_names = [] # the names of the numpy files that have been created for this file
        pbar.set_description("Tokenizing %s" % file)

        # try to open the file, otherwise add it to the error files list and continue
        try:
            with gzip.open(file, "rt") as f:
                lines = f.read()
                # split lines by <EOS> token to get individual sequences and avoid memory issues
                lines = lines.split("<EOS>")
                # re-add the <EOS> token to each line
                lines = [line + "<EOS>" for line in lines]
        except:
            error_files.append(file)
            with open("genbank_error_files.pkl", "wb") as f:
                pickle.dump(error_files, f)
            continue

        # tokenize each line
        for line in lines:
            tokens = [] # the tokens for the current line. It's actually reset every iteration, but this is here to make the code more understandable

            ######## LINE IS GREATER THAN 1e5 CHARACTERS ########
            if len(line) > 1e5:
                if num_tokens_processed > 1e9:
                    pbar.set_description(f"[LARGE LINE] Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**9:.3f} B tokens total), {len(processed_file_output_names)} files created for this file")
                elif num_tokens_processed > 1e6:
                    pbar.set_description(f"[LARGE LINE] Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**6:.3f} M tokens total), {len(processed_file_output_names)} files created for this file")
                # tokenize in chunks of 1e5 if the line is longer than 1e5
                tokens = [] # the tokens for the current line
                for i in range(0, len(line), int(1e5)):
                    line_chunk = sp.EncodeAsIds(line[i:i+int(1e5)], add_bos=False, add_eos=False)
                    tokens += line_chunk # add the tokens for this chunk to the tokens list
                    num_tokens_processed += len(line_chunk) # increment the number of tokens processed
                    
                    # it's likely that the line isn't long enough to satifsy this if loop, but it's here just in case
                    if len(tokenized_sequence) + len(tokens) > 5e7: # if adding these tokens would make the sequence too long, save the tokenized sequence to a file with the extra tokens
                        # save the tokenized sequence to a file
                        save_fn = os.path.join(save_dir, "tokenized_%d.npy" % num_files_processed) # the name of the file to save the tokenized sequence to

                        # save as uint16 to save space
                        np.save(save_fn, np.array(tokenized_sequence + tokens, dtype=np.uint16)) # save the tokenized sequence to a file
                        processed_file_output_names.append(save_fn) # add the name of the file to the list of files that have been created for this file

                        # update counters
                        num_files_processed += 1 # increment the number of files processed

                        # reset the tokenized sequence
                        tokenized_sequence = []
                        tokens = []

                        if num_tokens_processed > 1e9:
                            pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**9:.3f} B tokens total), {len(processed_file_output_names)} files created for this file")
                        elif num_tokens_processed > 1e6:
                            pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**6:.3f} M tokens total), {len(processed_file_output_names)} files created for this file")
            ######################################################
            else: # otherwise, tokenize the entire line
                tokens = sp.EncodeAsIds(line, add_bos=False, add_eos=False)
                num_tokens_processed += len(tokens)

            # there is technically a chance at this point that `tokens` is empty, but it's extremely extremely unlikely
            # even if it is empty, it doesn't affect the code below

            tokenized_sequence += tokens # add the tokens to the tokenized sequence
            
            if num_tokens_processed > 1e9:
                pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**9:.3f} B tokens total), {len(processed_file_output_names)} files created for this file")
            elif num_tokens_processed > 1e6:
                pbar.set_description(f"Tokenizing {file.split('/')[-1]} ({num_tokens_processed / 10**6:.3f} M tokens total), {len(processed_file_output_names)} files created for this file")

            if len(tokenized_sequence) >= 5e7: # same save procedure as above
                # save the tokenized sequence to a file
                save_fn = os.path.join(save_dir, "tokenized_%d.npy" % num_files_processed)

                # save as uint16 to save space
                np.save(save_fn, np.array(tokenized_sequence, dtype=np.uint16))
                processed_file_output_names.append(save_fn)
                
                # reset the tokenized sequence
                tokenized_sequence = []
                num_files_processed += 1
            
        processed_files.append({"input_fn": file, "output_fns": processed_file_output_names})

        with open("genbank_chkpt.pkl", "wb") as f:
            pickle.dump(processed_files, f)
        with open("genbank_error_files.pkl", "wb") as f:
            pickle.dump(error_files, f)

    # save the last tokenized sequence to a file
    if len(tokenized_sequence) > 0:
        # save the tokenized sequence to a file
        save_fn = os.path.join(save_dir, "tokenized_%d.npy" % num_files_processed)

        # save as uint16 to save space
        np.save(save_fn, np.array(tokenized_sequence, dtype=np.uint16))
    
    return error_files, num_tokens_processed

genbank_errors, genbank_token_count = tokenize_files(genbank_files, save_dir)

with open("genbank_error_files_final.pkl", "wb") as f:
        pickle.dump(genbank_errors, f)