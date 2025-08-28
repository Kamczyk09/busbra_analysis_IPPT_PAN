import torch
from PIL import Image
import open_clip
from torchvision import transforms
from PIL import Image

from data.busbra_loader import load_data_with_segmentation

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    transforms.ToTensor(),
])

train_ds, test_ds, val_ds = load_data_with_segmentation()
image = transform(train_ds[0]).unsqueeze(0)


model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')



