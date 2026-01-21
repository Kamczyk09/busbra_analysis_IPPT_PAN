# BUSBRA Analysis (IPPT PAN)

A research codebase for breast ultrasound analysis and explainable AI (XAI). This repository includes dataset loaders, model definitions (ResNets, CLIP with LoRA, CNNs), training scripts, and several XAI methods (RISE, Grad-CAM, Guided Backpropagation). Example usage is provided in `main.py`.

---

## Highlights
- Implements multiple models for ultrasound image analysis (see `models/`).
- Integrates XAI tools: RISE (`rise/`), Grad-CAM and variants (`pytorch_grad_cam/`), and guided backprop (in `guided_backprop/`).
- Example scripts to generate explanations: `apply_rise.py`, `apply_grad_cam.py`, `apply_guided_bp.py`.

---

## Quick start

1. Create a Python environment (recommended Python 3.8+):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the example that saves and displays a RISE explanation (this is the example provided in `main.py`):

```bash
python main.py
```

---

## Notes:
- `main.py` expects a CLIP LoRA checkpoint at `models_checkpoints/CLIP_lora_cub_2e_200n.pth`. If this checkpoint is not present, either train the CLIP LoRA model (see the `models/CLIP_lora.py` entry points) or adjust `main.py` to point to a different checkpoint.
- Running `main.py` will call `apply_rise.save_rise_cub(...)` and save outputs to `XAI_numpy/cub/rise_CLIP.npz`.

Minimal RISE reproduction (what `main.py` does)

1. Ensure dependencies and a trained model checkpoint are available (see above).
2. Run:

```bash
python main.py
```

3. The example will save an NPZ file. To load and inspect it manually, you can run:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("XAI_numpy/cub/rise_CLIP.npz")
# arrays available: 'images', 'masks', 'binarized_masks', 'sal'

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(data['images'][0])
axes[1].imshow(data['masks'][0])
axes[2].imshow(data['binarized_masks'][0])
axes[3].imshow(data['sal'][0])
plt.show()
```

## Repository layout (important files/folders)
- `main.py` — small example workflow that trains/evaluates CLIP LoRA and saves RISE outputs.
- `apply_rise.py`, `apply_grad_cam.py`, `apply_guided_bp.py` — runner scripts for respective XAI methods.
- `models/` — model definitions and training helper functions (e.g., `ResNet18.py`, `CLIP_lora.py`).
- `data/` — data loaders and BUSBRA dataset artifacts.
  - `data/busbra_loader.py`, `data/combined_loader.py`, and the `BUSBRA/` subfolder with CSVs and images.
- `pytorch_grad_cam/` — implementation of Grad-CAM variants and utilities.
- `rise/` — RISE implementation and helpers.
- `guided_backprop/` — guided backprop implementations.
- `models_checkpoints/` — directory intended for saved model weights (checkpoints).
- `XAI_numpy/` — (generated) outputs from XAI scripts (NPZ files with explanation masks, etc.).
- `training_folds_busbra.py` — example script for cross-validation training on BUSBRA.

---

## How to run training

A training example entry point is `training_folds_busbra.py`. Typical usage:

```bash
python training_folds_busbra.py
```

See the top of the file for configurable parameters (folds, epochs, batch size). Ensure dataset CSVs under `data/BUSBRA/` are present.

---

## XAI utilities

- RISE: `rise/` and `apply_rise.py`. Produces randomized-mask-based saliency maps (saved as NPZ in `XAI_numpy/`).
- Grad-CAM: `pytorch_grad_cam/` and `apply_grad_cam.py`. Multiple CAM variants are implemented.
- Guided Backprop: `guided_backprop/` and `apply_guided_bp.py`.

Tips & troubleshooting

- GPU: If you have a CUDA-capable GPU, ensure PyTorch with CUDA is installed. If `requirements.txt` pins a CPU-only torch, replace/install the appropriate `torch`+`torchvision` build for your CUDA version.
- Missing checkpoints: If `models_checkpoints/` doesn't contain the required `.pth` files, train the models first or copy checkpoints into that folder.
- Paths: The example scripts write outputs to `XAI_numpy/` — create this folder if it doesn't exist or adjust the scripts to a preferred path.

Suggested next steps / improvements
- Add a Dockerfile or environment.yml for reproducible environments.
- Add short unit or smoke tests for data loaders and the XAI runner scripts.
- Provide at least one small sample checkpoint or a download link for pre-trained weights to make the Quickstart smoother.
- Add CLI flags to `main.py` and the `apply_*.py` scripts to control inputs/output locations without editing code.

