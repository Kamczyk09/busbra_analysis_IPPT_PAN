"""
importy

resnet_cub = ... #już wcześniej wytrenowane
clip_cub = ... #już wcześniej wytrenowane

GRAD-CAM dla resnet_cub
RISE dla resnet_cub
Guided-BP dla resnet_cub

RISE dla clip_cub

funkcja, która przyjmuje jako parametr nazwę pliku npz i plotuje podane indeksy (włącznie z dodatkowymi segmentacjami)
"""
import torch
import numpy as np
from models import CLIP_lora
import apply_rise
import matplotlib.pyplot as plt

#chuj + xd
model = CLIP_lora.return_model(200)
model.load_state_dict(torch.load('models_checkpoints/CLIP_lora_cub_2e_200n.pth'))

apply_rise.save_rise_cub(model, image_idx=[0,1,2,3], save_path="XAI_numpy/cub/rise_CLIP.npz")

data = np.load("XAI_numpy/cub/rise_CLIP.npz")

fig, axes = plt.subplots(1,4)
axes[0].imshow(data['images'][0])
axes[1].imshow(data['masks'][0])
axes[2].imshow(data['binarized_masks'][0])
axes[3].imshow(data['sal'][0])
fig.show()


"""
importy 

for fold in k-folds:
    resnet: trening + eval + saving state (busbra)
    clip: trening + eval + saving state (busbra)
    
for fold in k-folds:
    resnet_busbra = ...
    clip_busbra = ...
    
    GRAD-CAM dla resnet_busbra
    RISE dla resnet_busbra
    Guided-BP dla resnet_busbra
    
    RISE dla clip_busbra
    

"""


