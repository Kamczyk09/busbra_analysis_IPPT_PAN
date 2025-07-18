import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from data.combined_loader import load_data
import torch.nn as nn
from rise import *
from models import ResNet18
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils_module.utils import *
from utils_module.linear_interpolation import interpolate_softmax
from scipy.stats import spearmanr
import os
from data.busbra_loader import load_data_with_segmentation



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def rise(model, image, true_label, res=200):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    saliency = explainer(img)
    if saliency.shape[0] == 1:
        # model zwraca tylko jeden logit (binarna klasyfikacja)
        sal = saliency[0].detach().cpu().numpy()
    else:
        # klasyfikacja wieloklasowa
        sal = saliency[true_label].detach().cpu().numpy()

    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    return sal


def compare_heatmaps(images_idx, models, add_empty_col=False):

    images = []
    labels = []
    train_ds, test_ds = load_data()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for idx in images_idx:
        images.append(test_ds[idx][0])
        labels.append(test_ds[idx][1])

    denormalized_images = [image * std + mean for image in images]  # tensor [3, H, W]
    np_images = [image.permute(1, 2, 0).numpy() for image in denormalized_images]

    n_rows = len(images_idx)
    n_cols = 1 + len(models) + (1 if add_empty_col else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)  # ensure 2D array when only one row

    for i in range(n_rows):
        axes[i, 0].imshow(np.clip(np_images[i], 0, 1))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Original image")

        for j, model in enumerate(models):
            ax = axes[i, j + 1]
            sal = rise(model=model, image=denormalized_images[i], true_label=labels[i])
            ax.imshow(np.clip(np_images[i], 0, 1))
            ax.imshow(sal, cmap='jet', alpha=0.5)
            ax.axis('off')

            model.eval()
            with torch.no_grad():
                output = model(denormalized_images[i].unsqueeze(0).to(device))
                y_pred = torch.argmax(output, dim=1).item()

            ax.set_title(f"True: {labels[i]}, Pred: {y_pred}")

        # Last column (if enabled) is intentionally left for later use
        if add_empty_col:
            axes[i, -1].axis('off')  # blank, ready for plot later

    return fig, axes


def calc_correlations(images_idx, models):
    images = []
    labels = []
    train_ds, test_ds = load_data()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for idx in images_idx:
        images.append(test_ds[idx][0])
        labels.append(test_ds[idx][1])

    denormalized_images = [image * std + mean for image in images]  # tensor [3, H, W]
    np_images = [image.permute(1, 2, 0).numpy() for image in denormalized_images]

    mean_corrs = []
    for i in range(len(images_idx)):
        sals = []

        for j, model in enumerate(models):
            sal = rise(model=model, image=denormalized_images[i], true_label=labels[i])
            sals.append(sal.flatten())

        corrs = []
        p_vals = []
        for u in range(len(models)-1):
            corr, p_val = spearmanr(sals[u], sals[-1])
            corrs.append(corr)
            p_vals.append(p_val)
        mean_corr = np.mean(corrs)
        mean_corrs.append(mean_corr)

    return np.array(mean_corrs)


def interpolation(model_a, model_b, images_idx, n_alphas, return_array=False):

    model_a.eval()
    model_b.eval()

    alphas = np.linspace(0, 1, n_alphas)
    num_images = len(images_idx)

    # Pusta macierz na korelacje: (len(alphas), num_images)
    corr_matrix = []

    for i, alpha in enumerate(alphas):
        model_c = merge_models(model_a, model_b, alpha=alpha)
        model_c.eval()

        models = [model_a, model_b, model_c]
        correlations = calc_correlations(images_idx, models)  # np.array([corr_img1, corr_img2, ...])
        corr_matrix.append(correlations)
        print(f"Alpha = {i}")

    corr_matrix = np.array(corr_matrix)  # shape: (len(alphas), num_images)

    if return_array:
        return corr_matrix

    fig, ax = plt.subplots()
    for img_idx in range(num_images):
        ax.plot(alphas, corr_matrix[:, img_idx],
                linestyle="--",
                alpha=0.5,
                linewidth=0.5)

    mean_corr_vec = np.mean(corr_matrix, axis=0)
    ax.plot(alphas, mean_corr_vec, linewidth=2, label="Mean")

    ax.set_xlabel("Alpha (Interpolation between models)")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Spearman correlation across interpolation")
    ax.legend()

    writer = SummaryWriter(log_dir="runs/images")
    image_tensor = plot_to_image(fig)
    writer.add_image("rise - interpolation", image_tensor[0])


def create_boxplots(n_images): #comparsion family1 vs family2

    #FAMILY1:
    model_a = ResNet18.return_model(10)
    path_a = "models_checkpoints/resnet18_pretrained.pth"
    model_a.load_state_dict(torch.load(path_a, map_location=torch.device('cpu')))
    model_a.eval()

    model_b = ResNet18.return_model(10)
    path_b = "models_checkpoints/resnet18_pretrained_1.0.pth"
    model_b.load_state_dict(torch.load(path_b, map_location=torch.device('cpu')))
    model_b.eval()

    corr_matrix = interpolation(model_a, model_b, n_alphas=30, images_idx=range(n_images), return_array=True)
    np.save("numpy_arrays/rise_matrix_family1.npy", corr_matrix)

    minimum_vec1 = np.min(corr_matrix, axis=0)
    mean_vec1 = np.mean(corr_matrix, axis=0)

    #FAMILY2:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_parent_weights = ResNet18.train(1, pretrained=True)

    model_a = ResNet18.return_model(10).to(device)
    model_a.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.00005, model=model_a, name="resnet18_T800")

    model_b = ResNet18.return_model(10).to(device)
    model_b.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.0001, model=model_b, name="resnet18_T1000")

    corr_matrix = interpolation(model_a, model_b, n_alphas=30, images_idx=range(n_images), return_array=True)
    np.save("numpy_arrays/rise_matrix_family2.npy", corr_matrix)

    minimum_vec2 = np.min(corr_matrix, axis=0)
    mean_vec2 = np.mean(corr_matrix, axis=0)

    #PLOTTING
    fig, axes = plt.subplots(1,2)
    axes[0].boxplot((minimum_vec1, minimum_vec2))
    axes[1].boxplot((mean_vec1, mean_vec2))

    axes[0].set_title("Minimum Correlation")
    axes[1].set_title("Mean Correlation")

    writer = SummaryWriter(log_dir="runs/images")
    image_tensor = plot_to_image(fig)
    writer.add_image("rise - boxplot", image_tensor[0])

    fig.savefig("rise_boxplot.png")


def get_image(idx):
    train_ds, test_ds = load_data()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = test_ds[idx][0]
    label = test_ds[idx][1]
    denormalized_image = image * std + mean

    return denormalized_image, label


def plot_extremes():
    writer = SummaryWriter(log_dir='runs/rise')

    # FAMILY 1
    model_a = ResNet18.return_model(10)
    path_a = "models_checkpoints/resnet18_pretrained.pth"
    model_a.load_state_dict(torch.load(path_a, map_location=torch.device('cpu')))
    model_a.eval()

    model_b = ResNet18.return_model(10)
    path_b = "models_checkpoints/resnet18_pretrained_1.0.pth"
    model_b.load_state_dict(torch.load(path_b, map_location=torch.device('cpu')))
    model_b.eval()

    path = "numpy_arrays/rise_matrix_family1.npy"

    corr_matrix = np.load(path)
    minimum_vec = np.min(corr_matrix, axis=0)
    mean_vec = np.mean(corr_matrix, axis=0)

    min_indices = np.argsort(minimum_vec)[:30]
    mean_indices = np.argsort(mean_vec)[:30]

    common1 = np.intersect1d(min_indices, mean_indices)[:5]

    fig, axes = compare_heatmaps(images_idx=common1, models=[model_a, model_b], add_empty_col=True)
    n_rows = axes.shape[0]
    last_col = axes.shape[1] - 1

    for i in range(n_rows):
        image, label = get_image(common1[i])
        ax = axes[i, last_col]
        ax.clear()
        interpolate_softmax(model_a, model_b, image, label, ax=ax)
    fig_tensor = plot_to_image(fig)
    writer.add_image('extremes_family1', fig_tensor[0])

    # FAMILY 2
    model_a = ResNet18.return_model(10)
    path_a = "models_checkpoints/resnet18_T800.pth"
    model_a.load_state_dict(torch.load(path_a, map_location=torch.device('cpu')))
    model_a.eval()

    model_b = ResNet18.return_model(10)
    path_b = "models_checkpoints/resnet18_T1000.pth"
    model_b.load_state_dict(torch.load(path_b, map_location=torch.device('cpu')))
    model_b.eval()

    path = "numpy_arrays/rise_matrix_family2.npy"

    corr_matrix = np.load(path)
    minimum_vec = np.min(corr_matrix, axis=0)
    mean_vec = np.mean(corr_matrix, axis=0)

    min_indices = np.argsort(minimum_vec)[:30]
    mean_indices = np.argsort(mean_vec)[:30]

    common2 = np.intersect1d(min_indices, mean_indices)[:5]
    print(common2)

    fig, axes = compare_heatmaps(images_idx=common2, models=[model_a, model_b], add_empty_col=True)
    n_rows = axes.shape[0]
    last_col = axes.shape[1] - 1

    for i in range(n_rows):
        image, label = get_image(common2[i])
        ax = axes[i, last_col]
        ax.clear()
        interpolate_softmax(model_a, model_b, image, label, ax=ax)
    fig_tensor = plot_to_image(fig)
    writer.add_image('extremes_family2', fig_tensor[0])

    writer.close()


def denormalize(tensor, mean, std):
    """
    Reverses the normalization step for a tensor image.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def plot_sal_with_segmentation(model, image_idx):
    writer = SummaryWriter(log_dir='runs/segmentation')

    model.eval()
    train_ds, test_ds, val_ds = load_data_with_segmentation()

    n_rows = len(image_idx)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, idx in enumerate(image_idx):
        image, segmentation, label = test_ds[idx]

        segmentation = segmentation.squeeze(0)
        segmentation_binarized = binarize_mask_tensor(segmentation)
        mean = [0.5]
        std = [0.5]
        denorm_image = denormalize(image.clone(), mean, std)

        sal = rise(model=model, image=denorm_image, true_label=label, res=200)
        print(f"sal: {sal.shape}")

        axes[i, 0].imshow(image.permute(1, 2, 0))
        axes[i, 0].set_title(image.shape)
        axes[i, 1].imshow(segmentation)
        # axes[i, 2].imshow(image.squeeze(0))
        axes[i, 2].imshow(sal, cmap='jet', alpha=0.5)

        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            y_pred = torch.argmax(output, dim=1).item()

        axes[i, 2].set_title(f"Predicted: {y_pred}. True: {label}")

        sal_centre = find_sal_centre(sal)
        print(f"Sal centre: {sal_centre}")
        print(f"Is centre in mask?: {metric_1(sal, segmentation)}")
        print(f"Dice score: {metric_2(sal, segmentation)}")
        print(f"Precision score: {metric_3(sal, segmentation_binarized)}\n")

        threshold = np.mean(sal)
        binary_sal = (sal >= threshold).astype(np.uint8)

        axes[i, 3].imshow(binary_sal)


    fig.tight_layout()
    fig.savefig('rise_segmentation.png')
    fig_tensor = plot_to_image(fig)
    print(fig_tensor.shape)
    writer.add_image('sal', fig_tensor[0])
    writer.close()

    mask_shape = np.load("rise/masks.npy").shape
    print(f"Masks shape: {mask_shape}")

    fig.show()


if __name__ == "__main__":
    # create_boxplots(n_images=5000)
    # plot_extremes()
    model = ResNet18.return_model(1)
    model.load_state_dict(torch.load("models_checkpoints/resnet18_busbra_pretrained.pth"))
    model.eval()
    plot_sal_with_segmentation(model, [0,1,2,3,4])
