import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from data.busbra_loader import load_data_with_segmentation

def get_num_output_neurons(dataset):
    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    return 1 if len(base_dataset.classes) == 2 else len(base_dataset.classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(nEpochs, lr=0.005, pretrained=False, model=None, name=None):
    batch_size = 64

    #### LOADING DATA ####
    train_ds, test_ds, val_ds = load_data_with_segmentation()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # nOutputNeurons = 1 if len(train_ds.classes) == 2 else len(train_ds.classes)
    nOutputNeurons = get_num_output_neurons(train_ds)
    print(f"Training on {nOutputNeurons} neurons")

    if model is None:
        model = resnet34(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, nOutputNeurons),
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss() if nOutputNeurons == 1 else nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="runs/ResNet34")

    # EARLY STOPPING PARAMS
    patience = 8
    best_loss = float("inf")
    trigger_times = 0
    best_model_state = None

    for epoch in range(nEpochs):
        model.train()
        running_loss = 0.0

        for images, masks, labels in train_loader:
            images = images.to(device)
            if nOutputNeurons == 1:
                labels = labels.float().unsqueeze(1).to(device)
            else:
                labels = labels.long().to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(device)
                if nOutputNeurons == 1:
                    labels = labels.float().unsqueeze(1).to(device)
                else:
                    labels = labels.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)

        print(f"Epoch {epoch + 1}/{nEpochs}, Train Loss: {avg_loss:.4f}, Test Loss: {val_loss:.4f}")
        # EARLY STOPPING CHECK
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"EarlyStopping trigger: {trigger_times} / {patience}")
            if trigger_times >= patience:
                print("Early stopping activated.")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_dir = "../obrazy_med_analiza/models_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    if name is not None:
        torch.save(model.state_dict(), f'{save_dir}/resnet18_{"pretrained" if pretrained else "raw"}.pth')

    torch.save(model.state_dict(), f'{save_dir}/{name}.pth')

    writer.close()
    return model.state_dict()


def evaluate(model=None, name=None):
    batch_size = 64

    ####LOADING DATA####
    train_ds, test_ds, val_ds = load_data_with_segmentation()
    ####################
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    nOutputNeurons = get_num_output_neurons(train_ds)
    print(f"Testing on {nOutputNeurons} neurons")

    if model is None:
        weights = f"models_checkpoints/{name}.pth"
        model = return_model(nOutputNeurons)
        model.load_state_dict(torch.load(weights))

    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, masks, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if nOutputNeurons == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += preds.eq(labels.float().unsqueeze(1)).sum().item()
            else:
                # Multi-class classification
                _, preds = torch.max(outputs, 1)  # preds shape: [batch_size]
                correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def return_model(nOutputNeurons):
    model = resnet34(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, nOutputNeurons)
    )
    model.to(device)
    return model