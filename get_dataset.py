import urllib.request
import gzip
import shutil


url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
file_name = 'download'


...
# Read the first 64 bytes of the file inside the .gz archive located at `url`
with urllib.request.urlopen(url) as response:
    with gzip.GzipFile(fileobj=response) as uncompressed:
        file_header = uncompressed.read(64)