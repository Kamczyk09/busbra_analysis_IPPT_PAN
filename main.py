import torch
import numpy as np
from models import CLIP_lora
import apply_rise
import matplotlib.pyplot as plt

#training CLIP with lora:
CLIP_lora.train(nEpochs=2, name="CLIP_lora_cub_2e_200n")
CLIP_lora.evaluate(name="CLIP_lora_cub_2e_200n")
#


#saving and loading npz file with XAI
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
#


