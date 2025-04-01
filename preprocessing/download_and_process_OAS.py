from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import pandas as pd
from io import StringIO
import requests
import time
from multiprocessing import Manager

def download_and_process_file(url, start_time, total_files):
    """
    Downloads a file from a given URL, processes its content, and writes the processed content to a new file.

    This function downloads a compressed CSV file from the provided URL, extracts amino acid sequences,
    formats them, and then compresses and saves the formatted sequences to a new file.

    Args:
    - url (str): The URL to download the CSV file from.
    - start_time (float): The time when the downloading process started, used for logging.
    - total_files (int): Total number of files to be processed, used for logging.

    Returns:
    - str or None: The name of the output file if successful, or None if there was an error.
    """
    try:
        fn = url.split('/')[-1]  # Extract the filename from the URL
        r = requests.get(url, allow_redirects=True)  # Send a GET request to the URL

        if r.status_code == 200:
            csv_content = gzip.decompress(r.content).decode('utf-8')  # Decompress and decode the CSV content
            csv_content = csv_content.split('\n', 1)[1]  # Remove the first line (assuming it's metadata)
            data = pd.read_csv(StringIO(csv_content))  # Read CSV content into a pandas DataFrame
            amino_acid_sequences = data['sequence_alignment_aa'].dropna().tolist()  # Extract amino acid sequences and drop missing values
            sequences_str = "<antibody>" + "<EOS><antibody>".join(amino_acid_sequences) + "<EOS>"  # Format sequences for output
            compressed_content = gzip.compress(sequences_str.encode('utf-8'))  # Compress the formatted string
            output_filename = fn.replace('.csv.gz', '_sequences.txt.gz')  # Create an output filename

            with open(output_filename, 'wb') as f_out:
                f_out.write(compressed_content)  # Write the compressed content to a file

            return output_filename
        else:
            print(f"Failed to download the file: {url}")
            return None
    except Exception as e:
        print(f"Exception in processing {url}: {str(e)}")
        return None

def update_processed_files(future, start_time, total_files):
    """
    Callback function to update and display the progress of file processing.

    This function is called when a file processing task completes. It updates the count of processed files,
    calculates the elapsed time, and estimates the remaining time for processing all files.

    Args:
    - future (concurrent.futures.Future): The future object of the completed task.
    - start_time (float): The time when the entire process started, used for calculating elapsed time.
    - total_files (int): The total number of files to process, used for calculating remaining files and estimated time.
    """
    processed_files.value += 1  # Update the count of processed files
    elapsed_time = time.time() - start_time  # Calculate the elapsed time since the start
    remaining_files = total_files - processed_files.value  # Calculate the number of remaining files
    avg_time_per_file = elapsed_time / processed_files.value  # Calculate the average time taken per file
    estimated_remaining_time = avg_time_per_file * remaining_files  # Estimate the remaining time for all files
    # Print the progress details
    print(
        f"Elapsed Time: {int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}, "
        f"Processed Files: {processed_files.value}, "
        f"Remaining Files: {remaining_files}, Estimated Time Left: "
        f"{int(estimated_remaining_time // 3600):02d}:{int((estimated_remaining_time % 3600) // 60):02d}:{int(estimated_remaining_time % 60):02d}"
    )

if __name__ == '__main__':
    # Main execution block
    with open('bulk_antibody_download.sh', 'r') as f:
        urls = f.readlines()  # Read URLs from a file

    urls = [url.split(' ')[-1].strip() for url in urls if url.startswith('wget')]  # Filter and clean the URLs
    total_files = len(urls)  # Count the total number of files to process

    manager = Manager()  # Use a manager to keep track of the number of processed files across processes
    processed_files = manager.Value('i', 0)  # Initialize the counter for processed files
    start_time = time.time()  # Record the start time

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(download_and_process_file, url, start_time, total_files) for url in urls]
        for future in as_completed(futures):
            future.add_done_callback(lambda f: update_processed_files(f, start_time, total_files))

    print("All files processed.")