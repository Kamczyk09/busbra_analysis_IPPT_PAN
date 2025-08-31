import copy
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
from skimage import segmentation
from skimage import segmentation
from scipy.ndimage import distance_transform_edt
import math

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


def binarize_mask(mask, threshold=0.5):
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


class GrayscaleToRGB:
  def __call__(self, tensor):
    # tensor: [1, H, W] → [3, H, W]
    return tensor.expand(3, -1, -1)


def create_ring_mask(mask):
  """
  Returns a new segmentation mask that is a thickened border of an original mask.
  """
  mask_bin = (mask > 0).astype(bool)

  border = segmentation.find_boundaries(mask_bin, mode='outer')

  perimeter = np.sum(border)

  radius = perimeter / (2 * math.pi)

  thickness = radius/5

  dist_outside = distance_transform_edt(~mask_bin)
  dist_inside = distance_transform_edt(mask_bin)

  border_thick = (dist_outside <= (thickness + 1) / 2) & (dist_inside <= (thickness + 1) / 2)

  return border_thick.astype(np.float32)


def create_sub_mask(mask: np.ndarray) -> np.ndarray:
  """
  Returns a new segmentation mask that is a space beneath an original mask.
  """
  m = mask.astype(bool)

  if m.ndim == 3 and m.shape[2] == 3:
    m = m[..., 0]

  if m.ndim != 2:
    raise ValueError(f"Mask must have shape of (H,W) or (H,W,3), but given {m.shape}")

  h, w = m.shape
  any_true = m.any(axis=0)

  bottom_idx = h - 1 - np.argmax(m[::-1, :], axis=0)
  bottom_idx = np.where(any_true, bottom_idx, h)

  rr = np.arange(h)[:, None]
  out = rr > bottom_idx[None, :]

  return out.astype(np.float32)

def find_maximum_points(array):
  """
  Finds the maximum points of an array.

  Args:
      array (np.ndarray): input array (2D or 3D)

  Returns:
      list of tuples
  """
  max_val = array.max()
  coords = np.argwhere(array == max_val)
  return [tuple(coord) for coord in coords]

def max_points_in_mask(sal, mask):
  """
  Verifies that the maximum points of an array are within a segmentation mask.

  Args:
      sal (np.ndarray): mapa istotności (H,W) lub (H,W,C)
      mask (np.ndarray): binarna maska segmentacji (H,W)

  Returns:
      bool: True jeśli wszystkie maksymalne punkty są w masce, False jeśli którykolwiek nie jest
      list: lista współrzędnych maksymalnych punktów
      list: lista współrzędnych maksymalnych punktów, które są w masce
  """
  # Jeśli mapa jest kolorowa (H,W,C), zredukuj do 2D (np. przez sumę po kanałach)
  if sal.ndim == 3:
    sal_gray = sal.sum(axis=2)
  else:
    sal_gray = sal

  # znajdź współrzędne maksymalnych pikseli
  max_coords = find_maximum_points(sal_gray)

  # sprawdzenie, które maksymalne punkty są w masce
  max_in_mask = [coord for coord in max_coords if mask[coord] > 0]

  all_in_mask = len(max_coords) == len(max_in_mask)

  return all_in_mask, max_coords, max_in_mask