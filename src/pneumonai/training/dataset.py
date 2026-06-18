import pandas as pd
from torch.utils.data import Dataset
from ..preprocessing.transform import preprocess
from ..preprocessing.specification import PreprocessingSpec
from pathlib import Path
import torch

class ChestXrayDataset(Dataset):

    def __init__(self,csv_path : Path, spec : PreprocessingSpec):
        self.df = pd.read_csv(csv_path)
        self.spec = spec
    def __getitem__(self,idx : int):
        row = self.df.iloc[idx]
        proccesed = preprocess(Path(row["image_path"]), self.spec)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return (proccesed, label)
    def __len__(self):
        return self.df.shape[0]