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


def cub_sample():
    model = ResNet18.return_model(2)
    model.load_state_dict(torch.load('models_checkpoints/resnet18_busbra_pretrained.pth'))
    features = return_resnet_features(model)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    range_list = range(10,18)

    train_ds, test_ds, val_ds = load_data_with_segmentation()
    imgs = [denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD) for i in range_list]
    imgs_disp = [(denormalize(test_ds[i][0], IMAGENET_MEAN, IMAGENET_STD).float()/255.0).permute(1,2,0) for i in range_list]
    masks_disp = [test_ds[i][1].permute(1,2,0) for i in range_list]
    to_pil = transforms.ToPILImage()
    pil_imgs = [to_pil(img) for img in imgs]
    x = torch.cat(([pil_to_tensor(pil_img) for pil_img in pil_imgs]), dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    guided_bp = GuidedBackpropBase(features)
    gb_out = guided_bp.generate_gradients(x)

    gb_out_imagenet = tensor_to_img(gb_out, normalize_type='imagenet')
    gb_out_maxmin   = tensor_to_img(gb_out, normalize_type='maxmin')
    gb_out_max      = tensor_to_img(gb_out, normalize_type='max')

    pil_vis = [imgs_disp, masks_disp, gb_out_imagenet, gb_out_maxmin, gb_out_max]
    nr, nc  = len(pil_imgs), len(pil_vis)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(2*nc, 2*nr), tight_layout=True)

    for c, vis_imgs in enumerate(pil_vis):
        for r, vis_img in enumerate(vis_imgs):
            axs[r][c].imshow(vis_img)
            axs[r][c].axis('off')

    plt.show()

# imagenet_sample()
cub_sample()


