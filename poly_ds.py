from PIL import Image
from torch.utils.data import Dataset

class PolyMNISTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row.label)
