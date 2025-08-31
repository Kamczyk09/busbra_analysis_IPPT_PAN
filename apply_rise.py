"""
Applying RISE to BUSBRA and CUB dataset.
"""
from rise import *
import matplotlib.pyplot as plt
from models import ResNet18
import torch
from torchvision import transforms
import os
from utils_module.utils import *
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def denormalize(tensor, mean, std):
    """
    Reverses the normalization step for a tensor image.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def rise(model, image, true_label, res=256):

    model = model.to(device)

    klen = 11
    ksig = 5
    kern = evaluation.gkern(klen, ksig).to(device)

    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)

    # Convert CIFAR-10 tensor (3, H, W) to PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))  # Zakładamy, że obraz był w [0,1]
    elif isinstance(image, torch.Tensor):
        image = to_pil_image(image)

    read_tensor = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0)  # (1, 3, H, W)
    ])

    img = read_tensor(image).to(device)
    utils.tensor_imshow(img[0].cpu())

    insertion = evaluation.CausalMetric(model, 'ins', res, substrate_fn=blur)
    deletion = evaluation.CausalMetric(model, 'del', res, substrate_fn=torch.zeros_like)

    explainer = explanations.RISE(model, (res, res))
    path = "rise/masks.npy"

    if os.path.exists(path):
        explainer.load_masks(filepath=path)
    else:
        explainer.generate_masks(N=1000, s=5, p1=0.4)

    sal = explainer(img)[true_label].detach().cpu().numpy()
    print(f"img shape: {img.shape}")
    print(f"explainer(img) shape: {explainer(img).shape}")
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    return sal


def plot_rise_busbra(model, image_idx):
    from data.busbra_loader import load_data_with_segmentation

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model.eval()

    train_ds, test_ds, val_ds = load_data_with_segmentation()

    n_rows = len(image_idx)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, idx in enumerate(image_idx):
        image, mask, label = test_ds[idx]
        image_disp = denormalize(image, mean, std).float() / 255.0
        sal = rise(model, image, true_label=label, res=200)
        axes[i, 0].imshow(image_disp.permute(1,2,0))
        axes[i, 1].imshow(mask.permute(1,2,0))
        axes[i, 2].imshow(image_disp.permute(1,2,0))
        axes[i, 2].imshow(sal, cmap='jet', alpha=0.5)
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            y_pred = torch.argmax(output, dim=1).item()
        # axes[i, 2].set_title(f"y_true: {label} ; y_pred: {y_pred}")

    fig.tight_layout()
    fig.show()


def save_rise_busbra(model, image_idx, save_path="XAI_numpy/rise_busbra.npz"):
    from data.busbra_loader import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    sal_list = []
    labels_list = []

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        sal = rise(model, image, true_label=label, res=200)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        denorm_image = denormalize(image, mean, std).float() / 255.0
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        mask = mask.permute(1,2,0).numpy()

        sal = np.clip(sal * 255, 0, 255).astype(np.uint8)

        images_list.append(denorm_image)
        masks_list.append(mask)
        sal_list.append(sal)
        labels_list.append(label)

    # Zamiana list na macierze NumPy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    sal_array = np.stack(sal_list)
    labels_array = np.array(labels_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        images=images_array,
        masks=masks_array,
        sal=sal_array,
        labels=labels_array
    )

    print(f"Saved in {save_path}")



def plot_rise_cub(model, image_idx):
    from data.cub200 import load_data_with_segmentation
    model.eval()
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    n_rows = len(image_idx)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, idx in enumerate(image_idx):
        image, segmentation, label = test_ds[idx]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        denorm_image = denormalize(image.clone(), mean, std)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        segmentation = segmentation.squeeze()
        segmentation_binarized = binarize_mask_tensor(segmentation)

        sal = rise(model=model, image=denorm_image, true_label=label, res=200)

        axes[i, 0].imshow(denorm_image)
        axes[i, 0].set_title(denorm_image.shape)
        axes[i, 1].imshow(segmentation)
        axes[i, 2].imshow(segmentation_binarized)
        axes[i, 3].imshow(denorm_image)
        axes[i, 3].imshow(sal, cmap='jet', alpha=0.5)

        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            y_pred = torch.argmax(output, dim=1).item()

        # axes[i, 3].set_title(f"Predicted: {y_pred}. True: {label}")

        sal_centre = find_sal_centre(sal)
        # print(f"Sal centre: {sal_centre}")
        # print(f"Is centre in mask?: {metric_1(sal, segmentation)}")
        # print(f"Dice score: {metric_2(sal, segmentation)}")
        # print(f"Precision score: {metric_3(sal, segmentation_binarized)}\n")

        # threshold = np.mean(sal)
        # binary_sal = (sal >= threshold).astype(np.uint8)
        #
        # axes[i, 4].imshow(binary_sal)

    fig.tight_layout()
    fig.show()



def save_rise_cub(model, image_idx, save_path="XAI_numpy/rise_cub.npz"):
    from data.cub200 import load_data_with_segmentation
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    images_list = []
    masks_list = []
    masks_bin_list = []
    sal_list = []
    labels_list = []

    for idx in image_idx:
        image, mask, label = test_ds[idx]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        denorm_image = denormalize(image.clone(), mean, std)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()

        sal = rise(model=model, image=denorm_image, true_label=label, res=200)

        mask = mask.squeeze()
        mask_binarized = binarize_mask_tensor(mask)

        sal = np.clip(sal * 255, 0, 255).astype(np.uint8)

        images_list.append(denorm_image)
        masks_list.append(mask)
        masks_bin_list.append(mask_binarized)
        sal_list.append(sal)
        labels_list.append(label)

    # Zamiana list na macierze NumPy
    images_array = np.stack(images_list)
    masks_array = np.stack(masks_list)
    masks_bin_array = np.stack(masks_bin_list)
    sal_array = np.stack(sal_list)
    labels_array = np.array(labels_list)

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


if __name__ == "__main__":
    from skimage import segmentation, img_as_ubyte

    # model = ResNet18.return_model(2)
    # model.load_state_dict(torch.load("models_checkpoints/resnet18_busbra_pretrained.pth"))
    # plot_rise_busbra(model, image_idx=[10,11,21,31])

    # model = ResNet18.return_model(200)
    # model.load_state_dict(torch.load("models_checkpoints/resnet18_cub_pretrained.pth"))
    # plot_rise_cub(model, range(4))

    # model = ResNet18.return_model(200)
    # model.load_state_dict(torch.load('models_checkpoints/resnet18_cub_pretrained.pth'))
    # save_rise_cub(model, image_idx=[0, 1, 2, 3])

    model = ResNet18.return_model(2)
    model.load_state_dict(torch.load('models_checkpoints/resnet18_busbra_pretrained.pth'))
    save_rise_busbra(model, image_idx=[0, 1, 2])

    data = np.load("XAI_numpy/rise_busbra.npz")
    mask = data["masks"][2]
    # plt.imshow(mask)

    ring_mask = create_ring_mask(mask)
    print(ring_mask)
    # plt.imshow(ring_mask)
    # plt.show()

    sub_mask = create_sub_mask(mask)
    print(sub_mask)
    # plt.imshow(sub_mask)
    # plt.show()




