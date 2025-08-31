from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import os
from PIL import Image
import torch



def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "CUB_200_2011", "CUB_200_2011", "images")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    dataset = datasets.ImageFolder(data_dir, transform)

    # === Wczytaj podział train/test ===
    split_path = os.path.join(base_dir, "CUB_200_2011", "CUB_200_2011", "train_test_split.txt")
    train_indices = []
    test_indices = []

    with open(split_path, "r") as f:
        for line in f:
            idx_str, flag_str = line.strip().split()
            idx = int(idx_str) - 1  # bo indeksy w pliku zaczynają się od 1
            flag = int(flag_str)
            if flag == 0:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


class CUBWithSegmentation(Dataset):
    def __init__(self, image_folder, transform=None, target_transform=None, segmentation_folder=None):
        self.image_folder = image_folder
        self.transform = transform
        self.target_transform = target_transform
        self.segmentation_folder = segmentation_folder
        self.samples = image_folder.samples  # list of (image_path, class_index)
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Construct segmentation path
        # Example: replace "images/" with "segmentation/" and keep relative path
        relative_path = os.path.relpath(image_path, self.image_folder.root)
        segmentation_path = os.path.join(self.segmentation_folder, relative_path)
        segmentation_path = os.path.splitext(segmentation_path)[0] + ".png"  # ensure PNG format

        # Load segmentation
        segmentation = Image.open(segmentation_path).convert("L")  # grayscale mask

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            segmentation = self.target_transform(segmentation)

        return image, segmentation, label


def load_data_with_segmentation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "CUB_200_2011", "CUB_200_2011", "images")
    segmentation_dir = os.path.join(base_dir, "CUB_200_2011", "CUB_200_2011", "segmentations")
    split_path = os.path.join(base_dir, "CUB_200_2011", "CUB_200_2011", "train_test_split.txt")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"Segmentation folder not found: {segmentation_dir}")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Train/test split file not found: {split_path}")

    # Normalizacja dla obrazów (standardowa dla ImageNet/ResNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Dla masek segmentacyjnych: tylko ToTensor (bez normalizacji!)
    mask_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    base_image_folder = datasets.ImageFolder(data_dir)

    full_dataset = CUBWithSegmentation(
        image_folder=base_image_folder,
        transform=image_transform,
        target_transform=mask_transform,
        segmentation_folder=segmentation_dir
    )

    # # Wczytaj indeksy train/test
    # train_indices = []
    # test_indices = []
    #
    # with open(split_path, "r") as f:
    #     for line in f:
    #         idx_str, flag_str = line.strip().split()
    #         idx = int(idx_str) - 1  # indeksy w pliku zaczynają się od 1
    #         flag = int(flag_str)
    #         if flag == 0:
    #             train_indices.append(idx)
    #         else:
    #             test_indices.append(idx)

    # train_dataset = Subset(full_dataset, train_indices)
    # test_dataset = Subset(full_dataset, test_indices)
    # test_size = int(0.5*len(test_dataset))
    # val_size = len(test_dataset) - test_size
    # test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])

    generator = torch.Generator().manual_seed(123)
    train_size = int(0.8*len(full_dataset))
    test_size = (len(full_dataset) - train_size)//2
    val_size = len(full_dataset) - test_size - train_size
    train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size], generator=generator)

    return train_dataset, test_dataset, val_dataset
