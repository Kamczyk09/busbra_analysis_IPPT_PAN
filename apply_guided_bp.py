import torch
import torchvision.models as models
import os
from PIL import Image
from guided_backprop.base import GuidedBackpropBase, return_resnet_features
from guided_backprop.utils import pil_to_tensor, tensor_to_img, denormalize
from models import ResNet18
from torchvision import transforms
from data.busbra_loader import load_data_with_segmentation
import matplotlib.pyplot as plt
import numpy as np
from utils_module import utils
from utils_module.utils import *


def imagenet_sample():
    resnet = models.resnet18(weights='DEFAULT')
    features = return_resnet_features(resnet)

    img_dir = './guided_backprop/imgs'

    pil_imgs = [Image.open(os.path.join(img_dir, img_file)).convert('RGB') for img_file in sorted(os.listdir(img_dir))]
    pil_imgs = [pil_img.resize((224, 224)) for pil_img in pil_imgs] # denormalized images in PIL
    x = torch.cat(([pil_to_tensor(pil_img) for pil_img in pil_imgs]), dim=0)

    guided_bp = GuidedBackpropBase(features)
    gb_out = guided_bp.generate_gradients(x)

    gb_out_imagenet = tensor_to_img(gb_out, normalize_type='imagenet')
    gb_out_maxmin   = tensor_to_img(gb_out, normalize_type='maxmin')
    gb_out_max      = tensor_to_img(gb_out, normalize_type='max')

    pil_vis = [pil_imgs, gb_out_imagenet, gb_out_maxmin, gb_out_max]
    nr, nc  = len(pil_imgs), len(pil_vis)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(2*nc, 2*nr), tight_layout=True)

    for c, vis_imgs in enumerate(pil_vis):
        for r, vis_img in enumerate(vis_imgs):
            axs[r][c].imshow(vis_img)
            axs[r][c].axis('off')

    plt.show()


def plot_gbp_cub(model, image_idx, normalization="imagenet"): #działa tylko z resnetem
    from data.cub200 import load_data_with_segmentation

    features = return_resnet_features(model)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])


    train_ds, test_ds, val_ds = load_data_with_segmentation()
    imgs = [denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD) for i in image_idx]
    imgs_disp = [(denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD).float()).permute(1,2,0) for i in image_idx]
    masks_disp = [test_ds[i][1].permute(1,2,0) for i in image_idx]
    to_pil = transforms.ToPILImage()
    pil_imgs = [to_pil(img) for img in imgs]
    x = torch.cat(([pil_to_tensor(pil_img) for pil_img in pil_imgs]), dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    guided_bp = GuidedBackpropBase(features)
    gb_out = guided_bp.generate_gradients(x)

    # choosing the type of normalization:
    if normalization == "imagenet":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='imagenet')
    elif normalization == "maxmin":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='maxmin')
    elif normalization == "max":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='max')

    binary_masks = [binarize_mask(mask, threshold=0.5) for mask in masks_disp]
    pil_vis = [imgs_disp, masks_disp, binary_masks, gb_out_norm]
    nr, nc  = len(pil_imgs), len(pil_vis)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(2*nc, 2*nr), tight_layout=True)

    for c, vis_imgs in enumerate(pil_vis):
        for r, vis_img in enumerate(vis_imgs):
            axs[r][c].imshow(vis_img)
            axs[r][c].axis('off')

    fig.show()


def save_gbp_cub(model, image_idx, save_path="XAI_numpy/gbp_cub.npz", normalization="imagenet"):
    from data.cub200 import load_data_with_segmentation

    # przygotowanie modelu i ekstraktora cech
    features = return_resnet_features(model)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    masks_bin_list = []
    sal_list = []
    labels_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        # denormalizacja obrazu
        denorm_image = denormalize(image.clone(), IMAGENET_MEAN, IMAGENET_STD)

        # [C, H, W] -> [1, C, H, W]
        x = denorm_image.unsqueeze(0).to(device)

        # guided backprop
        guided_bp = GuidedBackpropBase(features)
        gb_out = guided_bp.generate_gradients(x)

        # wybór normalizacji
        if normalization == "imagenet":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='imagenet')
        elif normalization == "maxmin":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='maxmin')
        elif normalization == "max":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='max')
        else:
            raise ValueError("Unknown normalization type")

        # konwersja saliency mapy na uint8 (0-255)
        sal = gb_out_norm.detach().cpu().numpy()

        # usunięcie batch dimension (1, H, W, C) → (H, W, C)
        if sal.shape[0] == 1:
            sal = sal.squeeze(0)

        # normalizacja min-max (skalowanie do [0,1])
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        # zamiana na uint8 (0–255)
        sal = (sal * 255).astype(np.uint8)

        # maska i binarna maska
        mask = mask.squeeze()
        mask_binarized = binarize_mask_tensor(mask)

        # dodanie do list
        denorm_image_np = denorm_image.permute(1, 2, 0).numpy()
        images_list.append(denorm_image_np)
        masks_list.append(mask.numpy())
        masks_bin_list.append(mask_binarized)
        sal_list.append(sal)
        labels_list.append(label)

    # konwersja list do tablic numpy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    masks_bin_array = np.stack(masks_bin_list)
    sal_array = np.stack(sal_list)
    labels_array = np.array(labels_list)

    # zapis do pliku
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        images=images_array,
        masks=masks_array,
        binarized_masks=masks_bin_array,
        sal=sal_array,
        labels=labels_array
    )

    print(f"Saved in {save_path}")


def plot_gbp_busbra(model, image_idx, normalization="imagenet"):
    from data.busbra_loader import load_data_with_segmentation
    features = return_resnet_features(model)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    train_ds, test_ds, val_ds = load_data_with_segmentation()
    imgs = [denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD) for i in image_idx]
    imgs_disp = [(denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD).float()/255.0).permute(1,2,0) for i in image_idx]
    masks_disp = [test_ds[i][1].permute(1,2,0) for i in image_idx]
    to_pil = transforms.ToPILImage()
    pil_imgs = [to_pil(img) for img in imgs]
    x = torch.cat(([pil_to_tensor(pil_img) for pil_img in pil_imgs]), dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    guided_bp = GuidedBackpropBase(features)
    gb_out = guided_bp.generate_gradients(x)
    print(f"gb_out shape: {gb_out.shape}")

    # choosing the type of normalization:
    if normalization == "imagenet":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='imagenet')
    elif normalization == "maxmin":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='maxmin')
    elif normalization == "max":
        gb_out_norm = tensor_to_img(gb_out, normalize_type='max')

    pil_vis = [imgs_disp, masks_disp, gb_out_norm]
    nr, nc  = len(pil_imgs), len(pil_vis)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(2*nc, 2*nr), tight_layout=True)

    for c, vis_imgs in enumerate(pil_vis):
        for r, vis_img in enumerate(vis_imgs):
            axs[r][c].imshow(vis_img)
            axs[r][c].axis('off')

    # for idx in image_idx:
    #     # Get max positions directly in torch
    #     indices = (gb_out[idx] == gb_out[idx].max()).nonzero(as_tuple=False)
    #     print(indices)
    #     print()


        # coords = []
        # for y, x in indices:  # assuming gb_out[idx] is [H,W]
        #     coords.append((int(x.item()), int(y.item())))
        #
        # img = np.zeros((200, 200), dtype=np.uint8)
        # for x, y in coords:
        #     if 0 <= y < 200 and 0 <= x < 200:  # safety check
        #         img[y, x] = 255


        # axs[i][2].imshow(imgs_disp[i])

    # plt.imshow(gb_out[0])
    # plt.show()
    #
    # print(gb_out[0])
    # print(f"Max val: {gb_out.max().item()}")
    #
    # # finding maximum points
    # indices = np.argwhere(gb_out[0] == gb_out.max().item())
    #
    # coords = []
    # for i in range(len(indices[0])):
    #     temp = (int(indices[1][i]), int(indices[0][i]))
    #     coords.append(temp)
    #
    # print(coords)

    # #plotting those points
    # img = np.zeros((200, 200), dtype=np.uint8)
    # for x, y in coords:
    #     img[y, x] = 255
    # plt.imshow(img, cmap='gray')
    plt.show()

def save_gbp_busbra(model, image_idx, save_path="XAI_numpy/gbp_busbra.npz", normalization="imagenet"):
    from data.busbra_loader import load_data_with_segmentation

    # przygotowanie modelu i ekstraktora cech
    features = return_resnet_features(model)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    sal_list = []
    labels_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        # denormalizacja obrazu
        denorm_image = denormalize(image.clone(), IMAGENET_MEAN, IMAGENET_STD)
        denorm_image_np = denorm_image.permute(1, 2, 0).numpy()

        # normalizacja min-max do 0–255
        denorm_image_np = denorm_image_np - denorm_image_np.min()
        denorm_image_np = denorm_image_np / (denorm_image_np.max() + 1e-8)
        denorm_image_np = (denorm_image_np * 255).astype(np.uint8)

        # [C,H,W] → [1,C,H,W] dla modelu
        x = denorm_image.unsqueeze(0).to(device)

        # guided backprop
        guided_bp = GuidedBackpropBase(features)
        gb_out = guided_bp.generate_gradients(x)

        # wybór normalizacji
        if normalization == "imagenet":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='imagenet')
        elif normalization == "maxmin":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='maxmin')
        elif normalization == "max":
            gb_out_norm = tensor_to_img(gb_out, normalize_type='max')
        else:
            raise ValueError("Unknown normalization type")

        # tensor → numpy
        sal = gb_out_norm
        if sal.shape[0] == 1:
            sal = sal.squeeze(0)
        #
        # # usunięcie batch dim (1,H,W,C) → (H,W,C)
        # if sal.shape[0] == 1:
        #     sal = sal.squeeze(0)
        #
        # # min–max normalizacja do [0,1]
        # sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        #
        # # uint8 (0–255)
        # sal = (sal * 255).astype(np.uint8)

        # maska
        mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]

        # dodanie do list
        images_list.append(denorm_image_np)
        masks_list.append(mask)
        sal_list.append(sal)
        labels_list.append(label)

    # konwersja list do tablic numpy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    sal_array = np.stack(sal_list)
    labels_array = np.array(labels_list)

    # zapis do pliku
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        images=images_array,     # (N,H,W,3)
        masks=masks_array,       # (N,H,W)
        sal=sal_array,           # (N,H,W,3)
        labels=labels_array      # (N,)
    )

    print(f"Saved in {save_path}")



if __name__ == '__main__':
    # model = ResNet18.return_model(200)
    # model.load_state_dict(torch.load('models_checkpoints/resnet18_cub_pretrained.pth'))
    # gbp_cub(model, image_idx=[0,1,2,3], normalization="max")

    model = ResNet18.return_model(2)
    model.load_state_dict(torch.load('models_checkpoints/resnet18_busbra_pretrained.pth'))
    # gbp_busbra(model, image_idx=[0,1,2,3], normalization="max")

    save_gbp_busbra(model, image_idx=[0, 1, 2, 3], normalization="max")
    data = np.load("XAI_numpy/gbp_busbra.npz")
    # print(data["labels"][0])
    # plt.imshow(data["masks"][2])
    sal = data["sal"][1]
    mask1 = data["masks"][1]
    mask2 = create_ring_mask(mask1)
    mask3 = create_sub_mask(mask1)
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(mask1)
    axes[1].imshow(mask2)
    axes[2].imshow(mask3)
    fig.show()

    maximum_points = find_maximum_points(sal)
    print(f"Maximum points: {maximum_points}")

    all_in_mask, max_coords, max_in_mask = max_points_in_mask(sal, mask1)
    print(all_in_mask)
    all_in_mask, max_coords, max_in_mask = max_points_in_mask(sal, mask2)
    print(all_in_mask)
    all_in_mask, max_coords, max_in_mask = max_points_in_mask(sal, mask3)
    print(all_in_mask)



