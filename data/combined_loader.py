from typing import Tuple, List, Optional
import torchvision.transforms as transforms
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image
import os
from PIL import Image


class CombinedDataset(Dataset):
    def __init__(
        self,
        path: str,
        transforms: callable = None,
        fold: Optional[Tuple[int, str]] = (0, None),
        dataset_names: Optional[List[str]] = None,
        leave_out: Optional[List[str]] = None,
        segmentation: bool = False,
    ):
        super().__init__()
        self.path = os.path.dirname(path)
        self.transforms = transforms
        self.fold_no, self.fold_type = fold
        self.data = pd.read_csv(path)
        if dataset_names is not None:
            self.data = self.data[self.data["dataset"].isin(dataset_names)]
        if leave_out is not None:
            self.data = self.data[~self.data["dataset"].isin(leave_out)]
        if self.fold_type:
            assert self.fold_type in ["train", "val", "test"], "Invalid fold type!"
            self.data.drop(
                self.data[self.data[f"fold{self.fold_no}"] != self.fold_type].index,
                inplace=True,
            )
        self.segmentation = segmentation
        self.classes = sorted(self.data["label"].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.data.iloc[idx]["image_path"])
        mask_path = os.path.join(self.path, self.data.iloc[idx]["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        label = self.data.iloc[idx]["label"]

        if self.segmentation:
            return (image, (mask > 0).to(torch.float32), label)
        else:
            return (image, label)


if __name__ == "__main__":
    data = CombinedDataset("C:/Users/stane/OneDrive/Pulpit/git-re-basin-main/matching_mod/data/bus_data/dataset.csv")
    print(len(data))
    pics = []
    num_pics = 4
    for i in range(num_pics):
        for j in range(2):
            pics.append(data[i][j].permute(1, 2, 0))
    plt.figure(figsize=(10, 10))
    plt.subplot(num_pics, 2, 1)
    plt.imshow(pics[0])
    plt.subplot(num_pics, 2, 2)
    plt.imshow(pics[1])
    plt.subplot(num_pics, 2, 3)
    plt.imshow(pics[2])
    plt.subplot(num_pics, 2, 4)
    plt.imshow(pics[3])
    plt.show()

    #### Micro tests- sanity check ####
    train = CombinedDataset("datasets/dataset.csv", fold=(1, "train"))
    print(len(train))

    val = CombinedDataset("datasets/dataset.csv", fold=(1, "val"))
    print(len(val))

    test = CombinedDataset("datasets/dataset.csv", fold=(1, "test"))
    print(len(test))

    assert len(train) + len(val) + len(test) == len(data), "folds don't sum up to whole dataset!"

    busi = CombinedDataset("datasets/dataset.csv", fold=(1, ""), dataset_names=["busi"])
    print("len busi: ", len(busi))

    assert len(busi) == 410, "busi doesn't have the length it has in the csv file!"


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "bus_data", "dataset.csv")

    transform = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor()
    ])

    train_ds = CombinedDataset(data_dir, segmentation=False, transforms=transform, fold=(0, "train"))
    test_ds = CombinedDataset(data_dir, segmentation=False, transforms=transform, fold=(0, "test"))
    val_ds = CombinedDataset(data_dir, segmentation=False, transforms=transform, fold=(0, "val"))

    return train_ds, test_ds, val_ds


def load_data_with_segmentation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "bus_data", "dataset.csv")

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    train_ds = CombinedDataset(data_dir, segmentation=True, transforms=transform, fold=(0, "train"))
    test_ds = CombinedDataset(data_dir, segmentation=True, transforms=transform, fold=(0, "test"))
    val_ds = CombinedDataset(data_dir, segmentation=True, transforms=transform, fold=(0, "val"))
    return train_ds, test_ds, val_ds

