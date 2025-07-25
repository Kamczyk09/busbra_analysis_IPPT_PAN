import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def pil_to_tensor(pil_img):
    np_img = np.array(pil_img) / 255.
    np_img = np_img - IMAGENET_MEAN
    np_img = np_img / IMAGENET_STD
    x = torch.tensor(np_img, dtype=torch.float32)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    return x


def tensor_to_img(x, normalize_type):
    x = torch.einsum('nchw->nhwc', x.detach())
    x = x.detach().cpu()
    if normalize_type == 'imagenet':
        x = torch.clip((x * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).int()

    if normalize_type == 'maxmin':
        n, h, w, c = x.shape
        x = x.reshape(n, -1)
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_min = x.min(dim=-1, keepdim=True)[0]
        x = torch.clip(((x - x_min) / (x_max - x_min)) * 255, 0, 255).int()
        x = x.reshape(n, h, w, c)

    if normalize_type == 'max':
        n, h, w, c = x.shape
        x = x.reshape(n, -1)
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = torch.clip((x / x_max) * 255, 0, 255).int()
        x = x.reshape(n, h, w, c)

    return x


def denormalize(tensor, mean, std):
    """
    Reverses the normalization step for a tensor image.
    """
    device = tensor.device
    dtype = tensor.dtype

    mean = torch.tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=dtype, device=device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)  # [1, C, 1, 1]
        std = std.unsqueeze(0)

    return tensor * std + mean