import torch
import torch.nn as nn
import open_clip
from data.busbra_loader import load_data_with_segmentation
from torchvision import transforms
import torch.nn.functional as F


class CLIP_with_mlp(nn.Module):
    def __init__(self, clip, mlp, preprocess):
        super().__init__()
        self.clip = clip
        self.mlp = mlp
        self.preprocess = preprocess

        # osobna normalizacja dla Tensorów (ta sama co w preprocess)
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            # jeśli to tensor z RISE, to wymuszamy rozmiar i normalizujemy
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = self.normalize(x)
        else:
            # jeśli to PIL / numpy (np. dataset), używamy pełnego preprocess
            x = self.preprocess(x).unsqueeze(0)

        x = self.clip.encode_image(x)
        x = self.mlp(x)
        return x



def merged_model(nOutputNeurons):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k'
    )
    mlp = nn.Sequential(
        nn.Linear(in_features=model.visual.output_dim, out_features=256),  # <-- poprawne dim
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=nOutputNeurons),
    )

    clip_mlp = CLIP_with_mlp(model, mlp, preprocess)
    return clip_mlp, preprocess


if __name__ == '__main__':
    clip_mlp, preprocess = merged_model(nOutputNeurons=10)

    train_ds, test_ds, val_ds = load_data_with_segmentation()

    # używamy preprocess z open_clip
    image = preprocess(train_ds[0][0]).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_mlp(image)

    print(image_features.shape)
