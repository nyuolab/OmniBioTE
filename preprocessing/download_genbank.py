import ftplib
import os
import time

def download_seq_files(ftp_address, directory, file_format, max_retries=5, retry_delay=5):
    """
    Downloads files from an FTP server based on the specified file format.

    Args:
    - ftp_address (str): The address of the FTP server.
    - directory (str): The directory on the FTP server to download files from.
    - file_format (str): The file format to filter and download.
    - max_retries (int): Maximum number of retries for downloading a file.
    - retry_delay (int): Delay between retries in seconds.
    """

    def download_file(ftp, file):
        """
        Helper function to download a single file from FTP server.

        Args:
        - ftp (ftplib.FTP): FTP connection object.
        - file (str): Name of the file to download.
        """
        if os.path.exists(file):  # Check if the file exists locally
            local_size = os.path.getsize(file)  # Get local file size
            ftp.voidcmd('TYPE I')  # Switch to binary mode for size check
            remote_size = ftp.size(file)  # Get remote file size

            if local_size == remote_size:
                print(f"{file} has already been completely downloaded.")
                return

            print(f"Downloading {file} from start (resume not supported)...")
            with open(file, 'wb') as local_file:  # Overwrite existing file
                ftp.retrbinary('RETR ' + file, local_file.write)
        else:
            with open(file, 'wb') as local_file:  # Write file if not exists
                ftp.retrbinary('RETR ' + file, local_file.write)

    with ftplib.FTP(ftp_address) as ftp:
        ftp.login()  # Login anonymously to the FTP server
        ftp.cwd(directory)  # Change to the target directory
        files = ftp.nlst()  # List files in the directory
        seq_files = [f for f in files if f.endswith(file_format)]  # Filter files by format

        for file in seq_files:
            retries = 0  # Initialize retry counter
            while retries < max_retries:
                try:
                    download_file(ftp, file)  # Attempt to download file
                    break  # Break out of the retry loop if download succeeds
                except (ftplib.error_temp, EOFError, IOError) as e:
                    print(f"Error downloading {file}: {e}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)  # Delay before retrying
                    retries += 1  # Increment retry counter
                    ftp = ftplib.FTP(ftp_address)  # Reconnect to the FTP server
                    ftp.login()  # Login again after reconnect
                    ftp.cwd(directory)  # Change to the target directory again
            if retries == max_retries:
                print(f"Failed to download {file} after {max_retries} attempts.")

if __name__ == "__main__":
    # FTP server details
    ftp_address = "ftp.ncbi.nih.gov"
    directory = "/genbank"
    file_format = ".seq.gz"

    # Download files
    download_seq_files(ftp_address, directory, file_format)