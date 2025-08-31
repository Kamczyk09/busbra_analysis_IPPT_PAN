import open_clip
import loralib as lora
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from data.busbra_loader import load_data_with_segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from utils_module.utils import GrayscaleToRGB
import CLIP
from torchmetrics import Accuracy

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    GrayscaleToRGB(),
    transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
])

train_ds, test_ds, val_ds = load_data_with_segmentation(transform=transform)

batch_size = 128
nOutputNeurons = 1
lr = 5e-3
nEpochs = 20
name = "CLIP_lora_busbra_30e_1n"

model = CLIP.merged_model()
transformer_block = model.clip.visual.transformer.resblocks[23]

#Applying LoRA
lora_layer_fc = lora.Linear(in_features=1024, out_features=4096, bias=True)
lora_layer_proj = lora.Linear(in_features=4096, out_features=1024, bias=True)
transformer_block.mlp.c_fc = lora_layer_fc
transformer_block.mlp.c_proj = lora_layer_proj

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.BCEWithLogitsLoss() if nOutputNeurons == 1 else nn.CrossEntropyLoss()

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model.to(device)
model.train()
# lora.mark_only_lora_as_trainable(model.clip)

patience = 8
best_loss = float('inf')
trigger_times = 0
best_model_state = None
train_losses = []
val_losses = []
for epoch in range(nEpochs):
    running_loss = 0.0
    for images, masks, labels in train_loader:
        images = images.to(device)
        if nOutputNeurons == 1:
            labels = labels.float().unsqueeze(1).to(device)  # BCE
        else:
            labels = labels.long().to(device)

        optimizer.zero_grad()
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    epoch_loss_train = running_loss / len(train_loader)
    train_losses.append(epoch_loss_train)

    running_loss = 0.0
    model.eval()
    for images, masks, labels in val_loader:
        images = images.to(device)
        if nOutputNeurons == 1:
            labels = labels.float().unsqueeze(1).to(device)  # BCE
        else:
            labels = labels.long().to(device)

        y_pred = model(images)
        loss = criterion(y_pred, labels)
        running_loss += loss.item()

    epoch_loss_val = running_loss / len(val_loader)
    val_losses.append(epoch_loss_val)

    print(
        f"Epoch [{epoch + 1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss test: {epoch_loss_val:.4f}")

    # Early stopping
    if epoch_loss_val < best_loss:
        best_loss = epoch_loss_val
        trigger_times = 0
        best_model_state = model.state_dict()
    else:
        trigger_times += 1
        print(f"Early stopping trigger count: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print("Early stopping activated.")
            break

model.load_state_dict(best_model_state)

save_dir = "../obrazy_med_analiza/models_checkpoints"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f'{save_dir}/{name}.pth')
print(f"Model saved in {save_dir}/{name}.pth")


#############EVALUATE################
model = CLIP.merged_model()
model.load_state_dict(torch.load(f"../obrazy_med_analiza/models_checkpoints/{name}.pth"))
model.to(device)

if nOutputNeurons == 1:
    accuracy_metric = Accuracy(task="binary").to(device)
else:
    accuracy_metric = Accuracy(task="multiclass", num_classes=nOutputNeurons).to(device)

model.eval()
with torch.no_grad():
    for images, mask, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        if nOutputNeurons == 1:
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
            labels = labels.long()
        else:
            _, preds = torch.max(outputs, 1)

        accuracy_metric.update(preds, labels)

acc = accuracy_metric.compute()
print(f"Accuracy (torchmetrics): {acc.item():.4f}")

#####plotting loss#####
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(train_losses, color='b', label='train')
ax.plot(val_losses, color='r', label='val')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
fig.show()