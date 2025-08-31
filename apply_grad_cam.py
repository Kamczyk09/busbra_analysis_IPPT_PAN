"""
Applying Grad-Cam to BUSBRA and CUB dataset.
"""

from torchvision import transforms
import pytorch_grad_cam
import numpy as np
from models import ResNet18
import torch
from data.busbra_loader import load_data_with_segmentation
import matplotlib.pyplot as plt
import os
from utils_module.utils import binarize_mask_tensor


def denormalize(tensor, mean, std):
    """
    Reverses the normalization step for a tensor image.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def GradCam(model, image, true_label, res=200):
    model.eval()

    # Keep normalization for model input
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # === Reverse normalization for visualization ===
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])

    if isinstance(image, torch.Tensor):
        image_numpy = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    else:
        raise TypeError("Expected image to be a torch.Tensor")

    # Undo normalization and clip to [0,1]
    image_numpy = (image_numpy * std + mean)
    image_numpy = np.clip(image_numpy, 0, 1).astype(np.float32)

    # Drop alpha if present
    if image_numpy.shape[2] == 4:
        image_numpy = image_numpy[:, :, :3]

    target_layers = [model.layer4[-1]]
    targets = [pytorch_grad_cam.ClassifierOutputTarget(true_label)]

    with pytorch_grad_cam.LayerCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = pytorch_grad_cam.show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam

def plot_gradcam_cub(model, image_idx):
    from data.cub200 import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    n_rows = len(image_idx)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, idx in enumerate(image_idx):
        image, mask, label = test_ds[idx]

        visualization, grayscale = GradCam(model, image, true_label=label)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm_image = denormalize(image.clone(), mean, std)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        mask = mask.squeeze()
        mask_binarized = binarize_mask_tensor(mask)

        axes[i, 0].imshow(denorm_image)
        axes[i, 1].imshow(mask)
        axes[i, 2].imshow(mask_binarized)
        axes[i, 3].imshow(denorm_image)
        axes[i, 3].imshow(visualization, alpha=0.5)

    fig.tight_layout()
    fig.show()


def save_gradcam_cub(model, image_idx, save_path="XAI_numpy/gradcam_cub.npz"):
    from data.cub200 import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    masks_bin_list = []
    visualizations_list = []
    labels_list = []

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        visualization, grayscale = GradCam(model, image, true_label=label)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        denorm_image = denormalize(image.clone(), mean, std)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        mask = mask.squeeze()
        mask_binarized = binarize_mask_tensor(mask)

        visualization = np.clip(visualization * 255, 0, 255).astype(np.uint8)  # wizualizacja uint8

        images_list.append(denorm_image)
        masks_list.append(mask)
        masks_bin_list.append(mask_binarized)
        visualizations_list.append(visualization)
        labels_list.append(label)

    # Zamiana list na macierze NumPy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    masks_bin_array = np.stack(masks_bin_list)
    visualizations_array = np.stack(visualizations_list)
    labels_array = np.array(labels_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        images=images_array,
        masks=masks_array,
        binarized_masks=masks_bin_array,
        sal=visualizations_array,
        labels=labels_array
    )

    print(f"Saved in {save_path}")


def plot_gradcam_busbra(model, image_idx):
    from data.busbra_loader import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    n_rows = len(image_idx)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, idx in enumerate(image_idx):
        image, mask, label = test_ds[idx]

        visualization, grayscale = GradCam(model, image, true_label=label)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_disp = denormalize(image, mean, std).float() / 255.0

        axes[i, 0].imshow(image_disp.permute(1,2,0))
        axes[i, 1].imshow(mask.permute(1,2,0))
        axes[i, 2].imshow(image_disp.permute(1,2,0))
        axes[i, 2].imshow(visualization, alpha=0.5)

    fig.tight_layout()
    fig.show()


def save_gradcam_busbra(model, image_idx, save_path="XAI_numpy/gradcam_busbra.npz"):
    from data.busbra_loader import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    visualizations_list = []
    labels_list = []

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        visualization, grayscale = GradCam(model, image, true_label=label)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Denormalizacja obrazu
        denorm_image = denormalize(image, mean, std).float() / 255.0
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        mask = mask.permute(1,2,0).numpy()

        visualization = np.clip(visualization * 255, 0, 255).astype(np.uint8)  # wizualizacja uint8

        images_list.append(denorm_image)
        masks_list.append(mask)
        visualizations_list.append(visualization)
        labels_list.append(label)

    # Zamiana list na macierze NumPy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    visualizations_array = np.stack(visualizations_list)
    labels_array = np.array(labels_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        images=images_array,
        masks=masks_array,
        sal=visualizations_array,
        labels=labels_array
    )

    print(f"Saved in {save_path}")


if __name__ == "__main__":
    # model = ResNet18.return_model(200)
    # model.load_state_dict(torch.load('models_checkpoints/resnet18_cub_pretrained.pth'))
    # ResNet18.evaluate(model)
    # plot_gradcam_cub(model, image_idx=[0, 1, 2, 3])

    # model = ResNet18.return_model(2)
    # model.load_state_dict(torch.load('models_checkpoints/resnet18_busbra_pretrained.pth'))
    # plot_gradcam_busbra(model, image_idx=[0,1,2,3])

    model = ResNet18.return_model(200)
    model.load_state_dict(torch.load('models_checkpoints/resnet18_cub_pretrained.pth'))
    save_gradcam_cub(model, image_idx=[0,1,2,3])

    # model = ResNet18.return_model(2)
    # model.load_state_dict(torch.load('models_checkpoints/resnet18_busbra_pretrained.pth'))
    # save_gradcam_busbra(model, image_idx=[0, 1, 2, 3, 4, 10, 20, 30, 40])

    data = np.load("XAI_numpy/gradcam_cub.npz")
    plt.imshow(data["sal"][0])
    plt.show()

