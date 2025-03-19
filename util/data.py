import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Callable, Tuple, Literal

def read_image(image_path: Path | str, out_type: type = "np") -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image) if out_type == "np" else image

def parse_dataset(dataset_root: Path, filename: str = "dataset.csv") -> pd.DataFrame:
    dataset = pd.read_csv(dataset_root / filename)
    dataset.set_index("id", inplace=True)
    dataset["path"] = dataset["path"].apply(lambda x: dataset_root / x)
    return dataset

def dict_collate_fn(batch):
    return {
        "ids": [sample["id"] for sample in batch],
        "images": [sample["image"] for sample in batch]
    }


class ImageDataset(Dataset):
    def __init__(self, dataset_root: Path, transform: Callable | None = None):
        self.dataset_root = Path(dataset_root)

        self.data = pd.read_csv(self.dataset_root / "dataset.csv")
        self.transform = np.array if transform is None else transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        img = read_image(self.dataset_root / sample["path"], out_type="pil")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "id": sample["id"]
        }
    
def make_dataloader(
    dataset_root: Path,
    transform: Callable | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    collate_fn: Callable = dict_collate_fn
) -> Tuple[Dataset, DataLoader]:
    """
    Creates an instance of the `ImageDataset` and a `DataLoader` for that dataset.

    Args:
        dataset_root (Path): The root directory of the dataset.
        transform (Callable | None, optional): The transform to apply to the images. Defaults to `np.array`.
        batch_size (int, optional): The batch size. Defaults to 32.
        num_workers (int, optional): The number of workers. Defaults to 4.

    Returns:
        Tuple[Dataset, DataLoader]: The dataset and the data loader.
    """
    dataset = ImageDataset(dataset_root=dataset_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    return dataset, loader