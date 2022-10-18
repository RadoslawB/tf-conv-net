import urllib.request
import tarfile
from CONFIG import CONFIG

download_file_path = CONFIG.DATAST_DIR_PATH + CONFIG.DOWNLOADED_FILE_NAME

if __name__ == '__main__':
    urllib.request.urlretrieve(CONFIG.DATASET_URL, download_file_path)

    with tarfile.open(download_file_path, "r:gz") as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, CONFIG.DATAST_DIR_PATH)