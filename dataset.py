import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BERTDataset(Dataset):
    """BERTDataset class for BERT model.

    Every dataset should subclass it. All subclasses should override `__getitem__`,
    that provides the data and label on a per-sample basis.

    Args:
        data (pd.DataFrame): The data to be used for training, validation, or testing.
        tokenizer (BertTokenizer): The tokenizer to be used for tokenization.
        max_length (int, optional): The max sequence length for input to BERT model. Defaults to 128.
        topk (int, optional): The number of top evidence sentences to be used. Defaults to 5.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        topk: int = 5,
    ):
        """__init__ method for BERTDataset"""
        self.data = data.fillna("")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.topk = topk

    def __len__(self):
        return len(self.data)
