from torchvision import transforms
import pytorch_grad_cam
from PIL import Image
from data.cifar10 import load_data
import matplotlib.pyplot as plt
import models.ResNet18 as ResNet18
import torch
import numpy as np
from utils_module.utils import merge_models
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter
from utils_module.utils import *

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#UWAGA ten kod robi grad-cam dla modelu ResNet18 ale dla bloku konwolucyjnego layer2 (wynika to z faktu że cifar10 ma rozdzielczość tylko 32x32)
def GradCam(model, image, true_label, res=32):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])

    input_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    image_numpy = np.array(image.resize((res, res))).astype(np.float32) / 255.0
    if image_numpy.shape[2] == 4:
        image_numpy = image_numpy[:, :, :3]

    target_layers = [model.layer2[-1]]

    targets = [pytorch_grad_cam.ClassifierOutputTarget(true_label)]

    with pytorch_grad_cam.LayerCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = pytorch_grad_cam.show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam


def compare_heatmaps(images_idx, models, calc_corr=False): #funkcja która wyświetla obrazy 4x3: 1 kolumna: 4 obrazy oryginalne, 2 kolumna: 4 heatmapy modelu_a, 3 kolumna: 4 heatmapy modelu_b

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

    fig, axes = plt.subplots(4, 1+len(models), figsize=(20, 20))

    for i in range(4):
        grayscale_cams = []

        axes[i, 0].imshow(np.clip(np_images[i], 0, 1))
        image_pil = Image.fromarray((np_images[i] * 255).astype(np.uint8))

        for j, model in enumerate(models):
            visualization, grayscale_cam = GradCam(model=model, image=image_pil, true_label=labels[i])
            grayscale_cams.append(grayscale_cam.flatten())
            axes[i, j+1].imshow(visualization)

        if calc_corr: #if calculating correlation is on
            corrs = []
            p_vals = []
            for u in range(len(models)-1):
                corr, p_val = spearmanr(grayscale_cams[u], grayscale_cams[-1])
                corrs.append(corr)
                p_vals.append(p_val)
            mean_corr = np.mean(corrs)
            axes[i, -1].set_title(f"Spearman correlation: {np.round(mean_corr, 3)}; p-val: {np.round(np.mean(p_vals), 4)}")

        if i == 0:
            axes[i, 0].set_title("No grad-cam")

    plt.savefig("grad-cam_comparsion.png")
    plt.show()


# FUNCTIONS FOR LAUNCHING:

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
        grayscale_cams = []
        image_pil = Image.fromarray((np_images[i] * 255).astype(np.uint8))

        for j, model in enumerate(models):
            visualization, grayscale_cam = GradCam(model=model, image=image_pil, true_label=labels[i])
            grayscale_cams.append(grayscale_cam.flatten())

        corrs = []
        p_vals = []
        for u in range(len(models)-1):
            corr, p_val = spearmanr(grayscale_cams[u], grayscale_cams[-1])
            corrs.append(corr)
            p_vals.append(p_val)
        mean_corr = np.mean(corrs)
        mean_corrs.append(mean_corr)

    return np.array(mean_corrs)


def compare_resnets(calc_corr=False): # training two models with different initializations
    model_a = ResNet18.return_model(10)
    path_a = "models_checkpoints/resnet18_pretrained.pth"
    model_a.load_state_dict(torch.load(path_a, map_location=torch.device('cpu')))
    model_a.eval()

    model_b = ResNet18.return_model(10)
    path_b = "models_checkpoints/resnet18_pretrained_1.0.pth"
    model_b.load_state_dict(torch.load(path_b, map_location=torch.device('cpu')))
    model_b.eval()

    model_c = merge_models(model_a, model_b)
    model_c.eval()

    images_idx = [11, 110, 112, 1201]
    models = [model_a, model_b, model_c]
    compare_heatmaps(images_idx, models, calc_corr=calc_corr)

    for model in models:
        print(f"Evaluation of {model.__class__.__name__}")
        ResNet18.evaluate(model)


def compare_resnets_1(): #training one model then after some epochs dividing it into two separate models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_parent_weights = ResNet18.train(3, pretrained=True)

    model_a = ResNet18.return_model(10).to(device)
    model_a.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.00005, model=model_a, name="resnet18_T800")

    model_b = ResNet18.return_model(10).to(device)
    model_b.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.0001, model=model_b, name="resnet18_T1000")


    model_c = merge_models(model_a, model_b)
    model_c.eval()

    images_idx = [11, 110, 112, 1201]
    models = [model_a, model_b, model_c]
    compare_heatmaps(images_idx, models)

    for model in models:
        print(f"Evaluation of {model.__class__.__name__}")
        ResNet18.evaluate(model)


def interpolation(model_a, model_b, images_idx, n_alphas, return_array=False):

    model_a.eval()
    model_b.eval()

    alphas = np.linspace(0, 1, n_alphas)
    num_images = len(images_idx)

    # Pusta macierz na korelacje: (len(alphas), num_images)
    corr_matrix = []

    for i,alpha in enumerate(alphas):
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

    plt.savefig("interpolation_corr.png")


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
    np.save("numpy_arrays/grad_cam_matrix_family1.npy", corr_matrix)

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
    np.save("numpy_arrays/grad_cam_matrix_family2.npy", corr_matrix)

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
    writer.add_image("grad_cam - boxplot", image_tensor[0])

    fig.savefig("grad_cam_boxplot.png")


if __name__ == "__main__":
    # interpolation(images_idx = range(0, 50), n_alphas=50)
    # compare_resnets(calc_corr=True)
    create_boxplots(n_images=5000)

