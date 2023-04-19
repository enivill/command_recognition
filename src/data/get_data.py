# source: https://upengareri.github.io/notes/python/downloading_and_extracting/

# IMPORTING MODULES
import os
import zipfile
import tarfile
import gzip
import shutil
import requests
from tqdm import tqdm

# ARCHIVE EXTENSIONS
ZIP_EXTENSION = ".zip"
TAR_EXTENSION = ".tar"
TAR_GZ_EXTENSION = ".tar.gz"
TGZ_EXTENSION = ".tgz"
GZ_EXTENSION = ".gz"

EMPTY_URL_ERROR = "ERROR: URL should not be empty."
FILENAME_ERROR = "ERROR: Filename should not be empty."
UNKNOWN_FORMAT = "ERROR: Unknown file format. Can't extract."

# TODO test if it works properly
def download_dataset(url: str, target_path: str = "data/external/", keep_download: bool = True,
                     overwrite_download: bool = True):
    """Downloads dataset from a url.
    url: string, a dataset path
    target_path: string, path where data will be downloaded
    keep_download: boolean, keeps the original file after extraction
    overwrite_download: boolean, stops download if dataset already exists
    """
    if url == "" or url is None:
        raise Exception(EMPTY_URL_ERROR)

    filename = get_filename(url)
    file_location = get_file_location(target_path, filename)

    os.makedirs(target_path, exist_ok=True)

    if os.path.exists(file_location) and not overwrite_download:
        print(f"File already exists at {file_location}. Use: 'overwrite_download=True' to \
overwrite download")
        extract_file(target_path, filename)
        return

    print(f"Downloading file from {url} to {file_location}.")
    # Download
    session = requests.Session()
    with open(file_location, 'wb') as f:
        with session.get(url, allow_redirects=True, stream=True) as resp:
            total = int(resp.headers.get('content-length', 0))
            resp.raise_for_status()
            with tqdm(desc=file_location, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in resp.iter_content(chunk_size=1024*1024*10):  # chunk_size in bytes
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)

    print("Finished downloading.")
    print("Extracting the file now ...")
    extract_file(target_path, filename)

    if not keep_download:
        os.remove(file_location)


def extract_file(target_path, filename):
    """Extract file based on file extension
    target_path: string, location where data will be extracted
    filename: string, name of the file along with extension
    """
    if filename == "" or filename is None:
        raise Exception(FILENAME_ERROR)

    file_location = get_file_location(target_path, filename)

    if filename.endswith(ZIP_EXTENSION):
        print("Extracting zip file...")
        zipf = zipfile.ZipFile(file_location, 'r')
        zipf.extractall(target_path)
        zipf.close()
    elif filename.endswith(TAR_EXTENSION) or \
            filename.endswith(TAR_GZ_EXTENSION) or \
            filename.endswith(TGZ_EXTENSION):
        print("Extracting tar file")
        tarf = tarfile.open(file_location, 'r')
        tarf.extractall(target_path)
        tarf.close()
    elif filename.endswith(GZ_EXTENSION):
        print("Extracting gz file")
        out_file = file_location[:-3]
        with open(file_location, "rb") as f_in:
            with open(out_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(UNKNOWN_FORMAT)


def get_filename(url):
    """Extract filename from file url"""
    filename = os.path.basename(url)
    return filename


def get_file_location(target_path, filename):
    """ Concatenate download directory and filename"""
    return target_path + filename
