
from rise import *
from data.busbra_loader import load_data_with_segmentation
import matplotlib.pyplot as plt
from models import ResNet18
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    klen = 11
    ksig = 5
    kern = evaluation.gkern(klen, ksig).to(device)

    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)

    read_tensor = transforms.Compose([
        transforms.Resize((res, res)),
        lambda x: torch.unsqueeze(x, 0)  # (1, 3, H, W)
    ])

    img = read_tensor(image).to(device)
    # print(f"img shape: {img.shape}")
    # insertion = evaluation.CausalMetric(model, 'ins', res, substrate_fn=blur)
    # deletion = evaluation.CausalMetric(model, 'del', res, substrate_fn=torch.zeros_like)

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
model.load_state_dict(torch.load("models_checkpoints/resnet18_busbra_pretrained_freezed.pth"))
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

#
from models import ResNet18
from torchvision.models import resnet18

model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = ResNet18.train_binary(nEpochs=30, lr=0.0005, model=model, name="resnet18_busbra_pretrained_freezed")
ResNet18.evaluate(name="resnet18_busbra_pretrained_freezed")

# from torchvision import transforms
# from data.busbra_loader import *
# import matplotlib.pyplot as plt
# import os
# print(os.getcwd())
#
# data_csv = "data/BUSBRA/bus_data.csv"
# folds_csv = "data/BUSBRA/10-fold-cv.csv"
#
# train_df, val_df, test_df = load_df(data_csv, fold_csv=folds_csv)
#
# transform = transforms.Compose([
#     transforms.Resize((200, 200))
# ])
#
# data, w, h, idx = easy_data(train_df)
# geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
# train_dataset = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)
#
# data, w, h, idx = easy_data(val_df)
# geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
# val_dataset   = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)
#
# data, w, h, idx = easy_data(test_df)
# geom = geometric_data().astype({col: np.float32 for col in geometric_data().columns[1:]})
# test_dataset  = BUSBRADataset(data, w, h, hist_idx=idx, geom_data=geom, transform=transform)
#
# print(len(train_dataset))
# print(len(val_dataset))
# print(len(test_dataset))
#
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
# val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
# # 5. Przykład użycia
# for batch in train_loader:
#     image, mask,label = batch
#
#     print("Train batch image shape:", image.shape)
#     # plt.imshow(image[1].permute(1, 2, 0), cmap="gray")
#     print(image)
#     # plt.imshow(mask[1].permute(1, 2, 0), cmap="gray")
#     # plt.imshow(mask_c[0], cmap="gray")
#     plt.title("Przykład obrazu z folda 1 (trening)")
#     plt.axis("off")
#     plt.show()
#     break
#
# from data.busbra_loader import load_data_with_segmentation
# import os
# print(os.getcwd())
#
# train_ds, test_ds, val_ds = load_data_with_segmentation()
# image, mask, label = test_ds[0]
# print(image.shape)
# print(mask.shape)
# print(label)