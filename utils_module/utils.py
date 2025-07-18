import copy
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import torch

def flatten_params(model):
  return model.state_dict()


def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3


def merge_models(model1, model2, alpha=0.5):
  assert type(model1) == type(model2), "Models need to be of same type"
  merged_model = copy.deepcopy(model1)
  state_dict1 = model1.state_dict()
  state_dict2 = model2.state_dict()
  merged_state_dict = {}

  for key in state_dict1:
    merged_state_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]

  merged_model.load_state_dict(merged_state_dict)
  return merged_model

def plot_to_image(figure):
  """Converting matplotlib plot into tensor for TensorBoard."""
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = Image.open(buf)
  image = np.array(image)
  image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
  return image


def find_sal_centre(sal):
  return np.unravel_index(np.argmax(sal), sal.shape)


def metric_1(sal, segmentation):
  # If segmentation has multiple channels, choose the relevant one
  if segmentation.ndim == 3:
    if segmentation.shape[0] > 1:
      segmentation = segmentation[1]  # Adjust this index if needed
    else:
      segmentation = segmentation[0]

  # Confirm it's now 2D
  if segmentation.ndim != 2:
    raise ValueError(f"Expected 2D segmentation mask, got shape {segmentation.shape}")

  # Get saliency center
  y_sal, x_sal = find_sal_centre(sal)
  sal_h, sal_w = sal.shape
  seg_h, seg_w = segmentation.shape

  # Scale coordinates
  y_seg = int(y_sal * seg_h / sal_h)
  x_seg = int(x_sal * seg_w / sal_w)

  # Clamp coordinates
  y_seg = min(max(y_seg, 0), seg_h - 1)
  x_seg = min(max(x_seg, 0), seg_w - 1)

  value = segmentation[y_seg, x_seg]
  return value == 1


def dice_score(map1, map2):
  map1 = (map1 == 1)
  map2 = (map2 == 1)
  intersection = np.logical_and(map1, map2).sum()
  return 2 * intersection / (map1.sum() + map2.sum() + 1e-8)  # unikamy dzielenia przez 0

def precision_score(map1, map2):
  map1 = (map1 == 1)
  map2 = (map2 == 1)
  tp = np.logical_and(map1, map2).sum()
  fp = np.logical_and(~map1, map2).sum()
  return tp / (tp + fp + 1e-8)  # dodajemy epsilon by uniknąć dzielenia przez 0

def metric_2(sal, segmentation, threshold="mean"): #how much mask and saliancy map overlap with dice's score
  if threshold == "mean":
    threshold = np.mean(sal)

  binary_sal = (sal >= threshold).astype(np.uint8)
  binary_segmentation = binarize_mask_tensor(segmentation)

  return dice_score(binary_sal, binary_segmentation)


def metric_3(sal, segmentation, threshold="mean"): #how much mask and saliancy map overlap with precision score
  if threshold == "mean":
    threshold = np.mean(sal)

  binary_sal = (sal >= threshold).astype(np.uint8)
  binary_segmentation = binarize_mask_tensor(segmentation)

  return precision_score(binary_segmentation, binary_sal)


def get_num_output_neurons(dataset):
  base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
  return 1 if len(base_dataset.classes) == 2 else len(base_dataset.classes)


def binarize_mask_tensor(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
  return (mask > threshold).float()

def evaluate_accuracy(model, dataloader, device):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for images, mask, labels in dataloader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)

      if outputs.shape[1] == 1:
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs >= 0.5).long()
      else:
        preds = torch.argmax(outputs, dim=1)

      correct += (preds == labels).sum().item()
      total += labels.size(0)

  accuracy = correct / total
  return accuracy
