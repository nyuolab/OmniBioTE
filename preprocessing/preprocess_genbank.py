import gzip
from Bio import SeqIO
import io
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Directories for input and output data
base_dir = 'E:\\Genbank Sequences'
save_dir = 'E:\\Genbank Sequences Processed'

def process_seq_gz(gz_file_path):
    """
    Process a single genbank sequence gzip file.
    
    Args:
    gz_file_path (str): Path to the gzip file to be processed.
    
    Returns:
    dict: A dictionary of counts of different molecule types in the sequence.
    """
    try:
        # Open and read the gzip file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            data = io.StringIO(gz_file.read().decode('utf-8'))
        
        # Initialize variables for sequence string and molecule type counts
        sequence_string = ""
        counts = {}

        # Parse the genbank file and concatenate sequences
        for seq_record in SeqIO.parse(data, "genbank"):
            sequence_string += f"<{seq_record.annotations['molecule_type']}>" + str(seq_record.seq) + f"<EOS>"
            if seq_record.annotations['molecule_type'] not in counts:
                counts[seq_record.annotations['molecule_type']] = 1
            else:
                counts[seq_record.annotations['molecule_type']] += 1

        # Output the processed data to a new file
        output_file = save_dir + '\\' + gz_file_path.split('\\')[-1].replace('.seq.gz', '.txt.gz')
        with gzip.open(output_file, 'wb') as f:
            f.write(sequence_string.encode('utf-8'))

        return counts
    except Exception as e:
        print(f"Error processing {gz_file_path}: {e}")
        return {}

def update_counts(global_counts, new_counts):
    """
    Update the global molecule type counts with new counts from a file.

    Args:
    global_counts (dict): The global counts of molecule types.
    new_counts (dict): Counts from a newly processed file.
    """
    for key, value in new_counts.items():
        if key not in global_counts:
            global_counts[key] = value
        else:
            global_counts[key] += value

def main():
    """
    Main function to process all genbank sequence gzip files in the base directory.
    """
    # Get list of files to process
    gz_files = glob.glob(base_dir + '\\*.seq.gz')
    processed_files = glob.glob(save_dir + '\\*.txt.gz')
    gz_files = [file for file in gz_files if file.replace(base_dir, save_dir).replace('.seq.gz', '.txt.gz') not in processed_files]

    # Initialize a dictionary for sequence counts
    sequence_counts = {}

    # Process files in parallel using a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_seq_gz, gz_file): gz_file for gz_file in gz_files}

        for future in tqdm(as_completed(futures), total=len(gz_files), desc="Processing files"):
            gz_file = futures[future]
            try:
                counts = future.result()
                update_counts(sequence_counts, counts)
            except Exception as e:
                print(f"Error processing {gz_file}: {e}")

    # Print the total counts of sequences processed
    print("Total sequence counts:", sequence_counts)

if __name__ == '__main__':
    main()
