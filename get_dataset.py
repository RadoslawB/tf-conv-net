import urllib.request
import tarfile
from CONFIG import CONFIG

download_file_path = CONFIG.DATAST_DIR_PATH + CONFIG.DOWNLOADED_FILE_NAME

if __name__ == '__main__':
    urllib.request.urlretrieve(CONFIG.DATASET_URL, download_file_path)

    with tarfile.open(download_file_path, "r:gz") as f:
        f.extractall(CONFIG.DATAST_DIR_PATH)