import torch
from torchmetrics import Accuracy
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data.busbra_loader import load_data_with_segmentation
import os
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_output_neurons(dataset):
    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    return 1 if len(base_dataset.classes) == 2 else len(base_dataset.classes)

def train(nEpochs, lr=0.0005,
          lambda_l1=None,
          lambda_l2=None,
          pretrained=False,
          model=None,
          name=None):

    assert lambda_l1 is None or lambda_l2 is None, \
        "Only one regularization L1 or L2 may be applied"
    # hyperparameters:
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test, dataset_val = load_data_with_segmentation() # modele muszą być trenowane na tak samo przygotowanych danych
    #############################

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    nOutputNeurons = get_num_output_neurons(dataset_train)
    print(f"Training on {nOutputNeurons} neurons")

    if model is None: #if model wasn't previously trained
        model = models.resnet152(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, nOutputNeurons)
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss() if nOutputNeurons == 1 else nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="runs/logs")

    patience = 8
    best_loss = float('inf')
    trigger_times = 0
    best_model_state = None

    train_losses = []
    test_losses = []
    for epoch in range(nEpochs):
        model.train()
        running_loss = 0.0

        for images, mask, labels in train_loader:
            images = images.to(device)
            if nOutputNeurons == 1:
                labels = labels.float().unsqueeze(1).to(device)  #BCE
            else:
                labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss_train = running_loss / len(train_loader)
        train_losses.append(epoch_loss_train)

        writer.add_scalar("loss/train", epoch_loss_train, global_step=epoch)

        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, mask, labels in val_loader:
                images = images.to(device)
                if nOutputNeurons == 1:
                    labels = labels.float().unsqueeze(1).to(device)  # OK dla BCE
                else:
                    labels = labels.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                if lambda_l1 is not None:
                    l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    loss = loss + lambda_l1 * l1_penalty

                if lambda_l2 is not None:
                    l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                    loss = loss + lambda_l2 * l2_penalty

                running_loss += loss.item()

        epoch_loss_val = running_loss / len(val_loader)
        test_losses.append(epoch_loss_val)

        print(
            f"Epoch [{epoch+1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss test: {epoch_loss_val:.4f}")

        writer.add_scalar("loss/test", epoch_loss_val, global_step=epoch)

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

    if name is not None:
        torch.save(model.state_dict(), f'{save_dir}/resnet101_{"pretrained" if pretrained else "raw"}.pth')

    torch.save(model.state_dict(), f'{save_dir}/{name}.pth')
    print(f"Model saved in {save_dir}/{name}.pth")

    return model.state_dict()


def train_binary(nEpochs, lr=0.0005,
                 pretrained=False,
                 lambda_l1=None,
                 lambda_l2=None,
                 model=None,
                 name=None):

    '''
    Binary classification with two output neurons using Cross Entropy Loss.
    '''

    assert lambda_l1 is None or lambda_l2 is None, \
        "Only one regularization L1 or L2 may be applied"

    # hyperparameters:
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test, dataset_val = load_data_with_segmentation() # modele muszą być trenowane na tak samo przygotowanych danych
    #############################

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    nOutputNeurons = 2
    print(f"Training on {nOutputNeurons} neurons")

    if model is None: #if model wasn't previously trained
        model = models.resnet152(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, nOutputNeurons)
    )
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="runs/logs")

    patience = 5
    best_loss = float('inf')
    trigger_times = 0
    best_model_state = None

    train_losses = []
    test_losses = []
    for epoch in range(nEpochs):
        model.train()
        running_loss = 0.0

        for images, mask, labels in train_loader:
            images = images.to(device)
            if nOutputNeurons == 1:
                labels = labels.float().unsqueeze(1).to(device)  #BCE
            else:
                labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss_train = running_loss / len(train_loader)
        train_losses.append(epoch_loss_train)

        writer.add_scalar("loss/train", epoch_loss_train, global_step=epoch)

        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, mask, labels in val_loader:
                images = images.to(device)
                if nOutputNeurons == 1:
                    labels = labels.float().unsqueeze(1).to(device)  # OK dla BCE
                else:
                    labels = labels.long().to(device)

                outputs = model(images)

                if lambda_l1 is not None:
                    l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    loss = loss + lambda_l1 * l1_penalty

                if lambda_l2 is not None:
                    l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                    loss = loss + lambda_l2 * l2_penalty

                loss = criterion(outputs, labels)
                running_loss += loss.item()

        epoch_loss_val = running_loss / len(val_loader)
        test_losses.append(epoch_loss_val)

        print(
            f"Epoch [{epoch+1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss val: {epoch_loss_val:.4f}")

        writer.add_scalar("loss/val", epoch_loss_val, global_step=epoch)

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

    if name is not None:
        torch.save(model.state_dict(), f'{save_dir}/resnet101_{"pretrained" if pretrained else "raw"}.pth')

    torch.save(model.state_dict(), f'{save_dir}/{name}.pth')
    print(f"Model saved in {save_dir}/{name}.pth")

    return model.state_dict()


def evaluate(model=None, name=None):
    batch_size = 128

    dataset_train, dataset_test, dataset_val = load_data_with_segmentation()
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    nOutputNeurons = get_num_output_neurons(dataset_train)
    print(f"Testing on {nOutputNeurons} neurons")

    if model is None:
        weights = f"models_checkpoints/{name}.pth"
        model = return_model(nOutputNeurons)
        model.load_state_dict(torch.load(weights))

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
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)  # usuń wymiar 1 z preds
                labels = labels.long()  # lub float, ale musi mieć ten sam kształt co preds
            else:
                _, preds = torch.max(outputs, 1)

            accuracy_metric.update(preds, labels)

    acc = accuracy_metric.compute()
    print(f"Accuracy (torchmetrics): {acc.item():.4f}")

    return acc.item()


def evaluate_binary(model=None, name=None):
    '''
    Calculating ACC for models trained with binary classification using two output neurons.
    '''

    batch_size = 128

    dataset_train, dataset_test, dataset_val = load_data_with_segmentation()
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    nOutputNeurons = 2
    print(f"Testing on {nOutputNeurons} neurons")

    if model is None:
        weights = f"models_checkpoints/{name}.pth"
        model = return_model(nOutputNeurons)
        model.load_state_dict(torch.load(weights))

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
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)  # usuń wymiar 1 z preds
                labels = labels.long()  # lub float, ale musi mieć ten sam kształt co preds
            else:
                _, preds = torch.max(outputs, 1)

            accuracy_metric.update(preds, labels)

    acc = accuracy_metric.compute()
    print(f"Accuracy (torchmetrics): {acc.item():.4f}")

    return acc.item()


def return_model(nOutputNeurons):
    model = models.resnet152(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, nOutputNeurons)
    )
    model.to(device)
    return model