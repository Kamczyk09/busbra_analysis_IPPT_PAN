import torch
from PIL import Image
import open_clip
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from data.busbra_loader import load_data_with_segmentation


class CLIP_with_mlp(nn.Module):
    def __init__(self, clip, mlp):
        super().__init__()
        self.clip = clip
        self.mlp = mlp
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # <-- tu zmiana
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.clip.encode_image(x)
        x = self.mlp(x)
        return x

def merged_model(nOutputNeurons):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    mlp = nn.Sequential(
        nn.Linear(in_features=768, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=nOutputNeurons),
    )

    clip_mlp = CLIP_with_mlp(model, mlp)

    return clip_mlp

if __name__ == '__main__':
    clip_mlp = merged_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    train_ds, test_ds, val_ds = load_data_with_segmentation()
    image = transform(train_ds[0][0]).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_mlp(image)

    print(image_features)
