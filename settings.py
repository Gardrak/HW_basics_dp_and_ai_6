import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import requests
from tqdm import tqdm


VOCAB_SIZE = 10000
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
MAX_LENGTH = 128
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCE = "https://lib.ru/POEEAST/ARISTOTEL/ritoriki.txt"  