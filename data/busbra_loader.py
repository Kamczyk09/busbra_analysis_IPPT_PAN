from itertools import count

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms.functional
from torchvision.io import read_image
from torchvision import transforms

D = True
DEPLOY_PATH = "/home/skaminsk/Pulpit/obrazy_med_analiza/"
PATH_TO_IMAGES = DEPLOY_PATH + 'data/BUSBRA/Images/' if D else 'data/BUSBRA/Images/'
PATH_TO_MASKS = DEPLOY_PATH + 'data/BUSBRA/Masks/' if D else 'data/BUSBRA/Masks/'
IMAGES_SUFF = '.png'
MASKS_PREF = 'mask_'

def load_df(path, fold_csv=None, fold_no=1):
    """
    Reads a dataset from a CSV file and optionally processes it into training, validation,
    and test splits for k-fold cross-validation.

    This function loads the input dataset from the specified file path and, if a fold
    CSV file is provided, merges the dataset with cross-validation fold information.
    It can return either a dictionary containing data splits for all folds or
    data splits for a specific fold.

    :param path: Path to the main dataset CSV file.
    :type path: str
    :param fold_csv: Optional path to a CSV file containing fold information for cross-validation.
    :type fold_csv: str, optional
    :param fold_no: Specifies a particular fold to return if fold_csv is provided. Defaults to 1,
        and is ignored if fold_csv is not provided.
    :type fold_no: int, optional
    :return: If no fold CSV is provided, returns the dataset as a Pandas DataFrame. Otherwise,
        returns a dictionary of training, validation, and test datasets for all folds, or
        specific data splits for the requested fold number.
    :rtype: pandas.DataFrame or dict[int, tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]]
    """
    df = pd.read_csv(path)
    if fold_csv is None:
        return df
    else:
        ret_dict = {}
        folds = pd.read_csv(fold_csv)
        df = df.merge(folds, on="ID", suffixes=("", f"_1"))
        for i in range(1, 6):
            df_test = df[df["kFold"] == i]
            df_train = df[df[f"valid_{i}"] == 1]
            df_val = df[df[f"valid_{i}"] == 0]
            ret_dict[i] = df_train, df_val, df_test
        return ret_dict if not fold_no else ret_dict[fold_no]


def easy_data(data=None):
    """
    Prepare and preprocess data for analysis. This function processes a dataset by extracting specific columns,
    encoding categorical variables, constructing file paths for masks and images, and extracting bounding box
    coordinates. Additionally, it validates the dataset by ensuring all "Pathology" entries are correctly classified
    as either "Malignant" or "Benign". The maximum width and height of input images are also calculated.

    :param data: A pandas DataFrame or a string path to a CSV file containing dataset information. If None, it
        defaults to loading a pre-defined dataset file.
    :type data: Optional[Union[pd.DataFrame, str]]
    :return: A tuple containing the following:
        - A modified pandas DataFrame with preprocessed columns.
        - Maximum width across all input images.
        - Maximum height across all input images.
        - Indices of the factorized "Histology" column.
    :rtype: Tuple[pd.DataFrame, int, int, np.ndarray]
    :raises Exception: If there are NaN values in the "Pathology" column, indicating invalid or unclassified entries.
    """
    if data is None:
        data = load_df(DEPLOY_PATH + "data/BUSBRA/bus_data.csv" if D else "data/BUSBRA/bus_data.csv")
    if isinstance(data, str):
        data = load_df(data)
    tmp = data[["ID", "Histology", "Pathology"]].copy()
    tmp["Pathology"] = tmp["Pathology"].apply(lambda x: 1 if x.lower() == "malignant" else 0 if x.lower() == "benign" else np.nan)
    tmp["Histology"], idx = pd.factorize(tmp["Histology"])
    tmp["Mask"] = tmp["ID"].apply(lambda x: PATH_TO_MASKS +  MASKS_PREF + x[4:] + IMAGES_SUFF)
    tmp["Image"] = tmp["ID"].apply(lambda x: PATH_TO_IMAGES + x + IMAGES_SUFF)
    tmp[["x_min", "y_min", 'w', 'h']] = pd.DataFrame(data['BBOX'].apply(lambda x: eval(x)).tolist(), index=data.index)
    max_w, max_h = data["Width"].max(), data["Height"].max()
    if np.isnan(tmp["Pathology"]).sum() > 0:
        raise Exception("NaN in Pathology, something was neither Benign nor Malignant")
    return tmp, max_w, max_h, idx

def geometric_data(data=None):
    """
    Processes and normalizes geometric data from a provided dataset or file path. This function
    handles missing values, drops constant numeric columns, and scales numeric columns to a
    0-1 range. The function ensures no missing values are present at any step after initial
    preprocessing and outputs a clean, normalized dataset.

    :param data: DataFrame or str
        If a string, it should be a path to a CSV file containing the dataset to be loaded. If
        None, a default path to a morphological features dataset is used based on the variable D.
    :return: DataFrame
        A preprocessed and normalized dataset containing only non-constant numeric columns and
        without any missing values.
    """
    if data is None:
        data = DEPLOY_PATH + "data/BUSBRA/morphological_features.csv" if D else "data/BUSBRA/morphological_features.csv"
    if isinstance(data, str):
        data = load_df(data)
    num_rows = len(data)
    data.dropna(inplace=True)

    assert not data.isnull().any().any(), "NaN in data"
    if data.isnull().any().any():
        raise Exception("NaN in data")

    data.dropna(inplace=True)
    if data.isnull().any().any():
        raise Exception("NaN in data")
    numer_cols = [x for x in data.columns if x != 'Filename']
    for col in numer_cols:
        if data[col].min() == data[col].max():
            data.drop(col, axis=1, inplace=True)

    numer_cols = [x for x in data.columns if x != 'Filename']
    data[numer_cols] = (data[numer_cols] - data[numer_cols].min()) / (data[numer_cols].max() - data[numer_cols].min())

    assert not data.isnull().any().any(), "NaN in data after norm"
    print(f"Geometric data loaded. {num_rows - len(data)} rows were removed due to NaNs ({1 - len(data)/num_rows * 100}%)")
    return data


def find_mask_bounds(mask):
        mask_np = mask.squeeze().numpy()
        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, y_max - y_min, x_max - x_min


def cutout(image, mask, x_min, y_min, w, h, margin=40):
    """
    Cuts out a specified region from both the image and corresponding mask based on the
    given bounding box coordinates and optional margin. The function performs slicing
    to extract the specified region, ensuring the indices do not go out of bounds. This
    can be useful for tasks such as cropping regions of interest or preparing data for
    further processing.

    :param image: The image tensor with dimensions (C, H, W) where C is the number
        of channels, H is the height, and W is the width.
    :type image: numpy.ndarray or torch.Tensor

    :param mask: The mask tensor with the same spatial dimensions as the image
        (C, H, W) or (1, H, W).
    :type mask: numpy.ndarray or torch.Tensor

    :param x_min: The minimum x-coordinate of the bounding box.
    :type x_min: int

    :param y_min: The minimum y-coordinate of the bounding box.
    :type y_min: int

    :param w: The width of the bounding box.
    :type w: int

    :param h: The height of the bounding box.
    :type h: int

    :param margin: Optional margin to be added around the bounding box. Default is 40.
    :type margin: int, optional

    :return: A tuple containing the cropped image and mask, with the respective
        extracted regions from the original arrays.
    :rtype: Tuple[numpy.ndarray or torch.Tensor, numpy.ndarray, or torch.Tensor]
    """
    image = image[:, max(0, y_min - margin):min(y_min + w + margin, image.shape[1] -1),
                max(0, x_min- margin):min(x_min + h + margin, image.shape[2] -1)]
    mask = mask[:, max(0, y_min - margin):min(y_min + w + margin, mask.shape[1] -1),
               max(0, x_min - margin):min(x_min + h + margin, mask.shape[2]- 1)]
    return image, mask


class BUSBRADataset(torch.utils.data.Dataset):
    """
    Represents a dataset for BUSBRA imaging data, typically used for image-based pathology analysis.
    Handles preprocessing, merging of additional geometric and morphological data,
    and preparation of image-mask pairs for model training and evaluation.

    This class is a PyTorch Dataset implementation, making it compatible with DataLoader for
    batched training and evaluation. It supports cutout techniques, resizing, and transformations
    on images and masks to adapt to various input formats and requirements.

    :ivar adaptive: Indicates whether adaptive cutout processing is enabled.
    :type adaptive: bool
    :ivar data: The main dataset containing image paths, masks, labels, and other metadata.
    :type data: pandas.DataFrame
    :ivar geom_data: Geometric data merged into the main dataset if provided. Otherwise, None.
    :type geom_data: pandas.DataFrame, optional
    :ivar morph_data: Morphological data merged into the main dataset if provided. Otherwise, None.
    :type morph_data: pandas.DataFrame, optional
    :ivar transform: A callable or transformation pipeline applied to images and masks.
    :type transform: callable, optional
    :ivar max_w: Maximum width to which the images and masks are cropped or resized.
    :type max_w: int
    :ivar max_h: Maximum height to which the images and masks are cropped or resized.
    :type max_h: int
    :ivar cutout: Determines whether cutout regions are extracted from the images and masks.
    :type cutout: bool
    :ivar hist_idx: Optional index for histopathological data association. Default is None.
    :type hist_idx: any, optional
    """
    def __init__(self, data, max_w: int, max_h: int, transform=None, cut=False, adaptive=False, hist_idx=None,
                 geom_data=None, morph_data=None):
        """
        Initializes the instance with the given parameters and processes geometrical and morphological
        data if provided. Handles merging data, managing attributes, and ensuring compatibility between
        datasets.

        :param data: Input data containing information to initialize the class.
        :type data: pandas.DataFrame
        :param max_w: Maximum width to consider for processing.
        :type max_w: int
        :param max_h: Maximum height to consider for processing.
        :type max_h: int
        :param transform: Optional transformations to apply to the data.
        :type transform: callable, optional
        :param cut: Indicates whether to enable cutout processing.
        :type cut: bool, optional
        :param adaptive: Sets whether adaptive processing is enabled.
        :type adaptive: bool, optional
        :param hist_idx: Optional index used for histopathological data processing.
        :type hist_idx: Any, optional
        :param geom_data: Geometric data to merge with the initial dataset. If provided, it manages
            merging and suffix addition to ensure proper integration.
        :type geom_data: pandas.DataFrame, optional
        :param morph_data: Morphological data to merge with the initial dataset. If provided, it
            manages merging and suffix addition to ensure accurate integration.
        :type morph_data: pandas.DataFrame, optional

        :raises AssertionError: If filenames in geometric and morphological data do not match after merging.
        """
        self.adaptive = adaptive
        self.data = data
        self.geom_data = geom_data
        self.morph_data = morph_data
        if self.geom_data is not None:
            print('bus' + geom_data["Filename"].str[4:-4])
            self.geom_data["ID"] = 'bus' + geom_data["Filename"].str[4:-4]
            self.geom_data = self.geom_data.add_suffix("_geom")
            self.geom_data.rename(columns={"ID_geom": "ID"}, inplace=True)
            self.data = pd.merge(self.data, self.geom_data, on="ID", suffixes=("", "_geom"))

            self.geom_data = True
        if self.morph_data is not None:
            print(morph_data["Filename"].str[4:-4])
            self.morph_data["ID"] = 'bus' + morph_data["Filename"].str[4:-4]
            self.morph_data = self.morph_data.add_suffix("_morph")
            self.morph_data.rename(columns={"ID_morph": 'ID'}, inplace=True)
            self.data = pd.merge(self.data, self.morph_data, on="ID", suffixes=("", "_morph"))
            self.morph_data = True

        def diff_series(s1, s2):
            ret = []
            for x, y in zip(s1, s2):
                if x != y:
                    ret.append((x, y))
            return ret

        if self.geom_data is not None and self.morph_data is not None:
            assert self.data['Filename_morph'].equals(self.data[
                                                          'Filename_geom']), f"Filenames in morph and geom data don't match: {diff_series(self.data['Filename_morph'], self.data['Filename_geom'])}"
            self.data.drop(['Filename_morph', 'Filename_geom'], axis=1, inplace=True)



        self.data.dropna(inplace=True)
        self.transform = transform
        self.max_w, self.max_h = max_w, max_h
        self.cutout = cut
        self.classes = sorted(self.data["Pathology"].unique())
        if hist_idx is not None:
            self.hist_idx = hist_idx


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = read_image(self.data.iloc[idx]["Image"]).to(torch.float32)

        if image.shape[0] == 1:  # 1 kanał
            image = image.repeat(3, 1, 1)

        mask = read_image(self.data.iloc[idx]["Mask"]).to(torch.float32)
        image_c = mask_c = geom = morph =  torch.zeros(1)
        if self.cutout:
            if not self.adaptive:
                image_c, mask_c = cutout(image, mask,
                                 self.data.iloc[idx]["x_min"], self.data.iloc[idx]["y_min"],
                                 self.data.iloc[idx]["w"], self.data.iloc[idx]["h"],
                                 margin=160)
            else:
                x_min, y_min, w, h = find_mask_bounds(mask)
                image_c, mask_c = cutout(image, mask, x_min, y_min, w, h, margin=10)
            image_c = torchvision.transforms.functional.resize(image_c, [224, 224])
            mask_c = torchvision.transforms.functional.resize(mask_c, [224, 224])
        elif self.transform is None:
            image = torchvision.transforms.functional.center_crop(image, [self.max_h, self.max_w])
            mask = torchvision.transforms.functional.center_crop(mask, [self.max_h, self.max_w])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).to(torch.float32)
        y_pat = self.data.iloc[idx]["Pathology"]
        y_histo = self.data.iloc[idx]["Histology"]
        if self.geom_data is not None:
            geom = torch.from_numpy(self.data.iloc[idx][[x for x in self.data.columns if (x.endswith('geom') and not x.startswith("Filename"))]].values.astype(np.float32))
        if self.morph_data is not None:
            morph = torch.from_numpy(self.data.iloc[idx][[x for x in self.data.columns if (x.endswith('morph') and not x.startswith("Filename"))]].values.astype(np.float32))
        # return image, mask, geom, morph, image_c, mask_c, y_pat, y_histo  # if only classification on images, then take only image, mask, y_pat
        return image, mask, y_pat

class GrayscaleToRGB:
    def __call__(self, tensor):
        # tensor: [1, H, W] → [3, H, W]
        return tensor.expand(3, -1, -1)

def load_data_with_segmentation():

    data_csv = "data/BUSBRA/bus_data.csv"
    folds_csv = "data/BUSBRA/10-fold-cv.csv"

    train_df, val_df, test_df = load_df(data_csv, fold_csv=folds_csv, fold_no=1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        GrayscaleToRGB(),
        transforms.Normalize(mean=mean, std=std)
    ])

    data, w, h, idx = easy_data(train_df)
    geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
    train_dataset = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)

    data, w, h, idx = easy_data(val_df)
    geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
    val_dataset = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)

    data, w, h, idx = easy_data(test_df)
    geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
    test_dataset = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)

    return train_dataset, test_dataset, val_dataset


if __name__ == "__main__":
    df = load_df("BUSBRA/bus_data.csv")
    data, w, h, idx = easy_data(df)
    print(data.head())
    print(w, h)
    geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
    print(geom.head())
    busbra_dataset = BUSBRADataset(data, w, h, cut=False, hist_idx=idx, geom_data=geom, adaptive=False)
    print(len(busbra_dataset))
    print(busbra_dataset[100][0].shape)
    print(f'\n\n{"#" * 80}\n', busbra_dataset[1][2])
    pics = []
    num_pics = 4
    count = 0
    while count < 8:
        pics = []
        for i in range(num_pics):
            for j in range(2):
                pics.append(busbra_dataset[count + i][j].permute(1, 2, 0))

        for i in range(num_pics * 2):
            plt.subplot(num_pics, 2, i + 1)
            plt.imshow(pics[i])

        plt.show()
        count += 4