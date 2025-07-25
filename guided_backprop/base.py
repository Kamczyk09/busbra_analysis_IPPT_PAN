import torch
import torch.nn as nn


class GuidedBackpropBase(nn.Module):
    def __init__(self, model_features):
        super().__init__()
        self.encoder = model_features
        self.relu_change_inplace(self.encoder)
        self.relu_set_hook_recursive(self.encoder)

    def relu_change_inplace(self, module):
        for child in module.children():
            if isinstance(child, nn.ReLU):
                child.inplace = False
            else:
                self.relu_change_inplace(child)

    def relu_set_hook_recursive(self, module):
        def relu_backward_hook_function(module, grad_in, grad_out):
            modified_grad_out = torch.clamp(grad_in[0], min=0.)
            return (modified_grad_out,)

        for child in module.children():
            if isinstance(child, nn.ReLU):
                child.register_full_backward_hook(relu_backward_hook_function)
            else:
                self.relu_set_hook_recursive(child)

    def generate_gradients(self, x):
        x = x.clone().detach().requires_grad_(True)
        feature = self.encoder(x)

        n, c, h, w = feature.shape
        feature_flat = feature.reshape(n, -1)
        mask = (feature_flat == feature_flat.max(dim=-1, keepdim=True)[0])
        top1_activation = mask * feature_flat
        top1_activation = top1_activation.reshape(n, c, h, w)

        self.encoder.zero_grad()
        feature.backward(gradient=top1_activation)

        return x.grad.data


def return_resnet_features(resnet):
    features = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )
    return features
