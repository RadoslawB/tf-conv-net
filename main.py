from utils import unpickle
import tarfile
from model import model


data = unpickle('./cifar-10-batches/data_batch_1')

print(data.keys())