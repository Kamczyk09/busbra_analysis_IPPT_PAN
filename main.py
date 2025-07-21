"""
Ten fragment służy do treningu sieci. Funkcja 'train' przyjmuje parametr 'model',
czyli będzie dotrenowywać istniejąca sieć o nazwie model. Wagi modelu zapisywane są w katalogu models_checkpoints

Funkcja evaluate liczy ACC dla wytrenowanej sieci
"""
from models import ResNet18
from torchvision.models import resnet18

model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = ResNet18.train(nEpochs=30,
                              lr=0.0001,
                              model=model,
                              name="resnet18_busbra_pretrained")
ResNet18.evaluate(name="resnet18_busbra_pretrained")


##########################################################################################
"""
RISE dla okreslonego modelu ze ścieżki.
"""
from rise import *
from data.busbra_loader import load_data_with_segmentation
import matplotlib.pyplot as plt
from models import ResNet18
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor, mean, std):
    """
    Reverses the normalization step for a tensor image.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def rise(model, image, true_label, res=200):
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    read_tensor = transforms.Compose([
        transforms.Resize((res, res)),
        lambda x: torch.unsqueeze(x, 0)  # (1, 3, H, W)
    ])

    img = read_tensor(image).to(device)

    explainer = explanations.RISE(model, (res, res))
    path = "rise/masks.npy"

    if os.path.exists(path):
        explainer.load_masks(filepath=path)
    else:
        explainer.generate_masks(N=1000, s=5, p1=0.4)

    saliency = explainer(img)
    print(f"explainer(img) shape: {saliency.shape}")
    if saliency.shape[0] == 1:
        # binary classification
        sal = saliency[0].detach().cpu().numpy()
    else:
        # multiclass problem
        sal = saliency[true_label].detach().cpu().numpy()

    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    return sal

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


model = ResNet18.return_model(2)
model.load_state_dict(torch.load("models_checkpoints/resnet18_busbra_pretrained_freezed.pth")) #required model's name
model.eval()

train_ds, test_ds, val_ds = load_data_with_segmentation()

fig, axes = plt.subplots(7,6)

for i in range(7):
    image, mask, label = test_ds[84+i]
    image_disp = denormalize(image, mean, std).float() / 255.0
    sal = rise(model, image, true_label=label, res=200)
    axes[i, 0].imshow(image_disp.permute(1,2,0))
    axes[i, 1].imshow(mask.permute(1,2,0))
    axes[i, 2].imshow(sal, cmap='jet', alpha=0.5)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        y_pred = torch.argmax(output, dim=1).item()
    axes[i, 2].set_title(f"y_true: {label} ; y_pred: {y_pred}")

for i in range(7):
    image, mask, label = test_ds[14+i]
    image_disp = denormalize(image, mean, std).float() / 255.0
    sal = rise(model, image, true_label=label, res=200)
    axes[i, 3].imshow(image_disp.permute(1,2,0))
    axes[i, 4].imshow(mask.permute(1,2,0))
    axes[i, 5].imshow(sal, cmap='jet', alpha=0.5)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        y_pred = torch.argmax(output, dim=1).item()
    axes[i, 5].set_title(f"y_true: {label} ; y_pred: {y_pred}")

fig.show()

