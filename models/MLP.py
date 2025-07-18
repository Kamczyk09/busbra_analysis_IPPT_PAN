import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data.cifar10 import load_data
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.layer0 = nn.Linear(3*32*32, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, n_classes)


    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)

        return nn.functional.log_softmax(x)

def return_model(nOutputNeurons):
    model = Model(nOutputNeurons)
    model.to(device)
    return model

def train(nEpochs, pretrained=False):

    # hyperparameters:
    lr = 0.001
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test = load_data() # modele muszą być trenowane na tak samo przygotowanych danych
    #############################

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    nOutputNeurons = 1 if len(dataset_train.classes) == 2 else len(dataset_train.classes)
    print(f"Training on {nOutputNeurons} neurons")

    model = return_model(nOutputNeurons)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss() if nOutputNeurons == 1 else nn.CrossEntropyLoss()

    patience = 5
    best_loss = float('inf')
    trigger_times = 0
    best_model_state = None

    train_losses = []
    test_losses = []
    for epoch in range(nEpochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            if nOutputNeurons == 1:
                labels = labels.float().unsqueeze(1).to(device)  # OK dla BCE
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

        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                if nOutputNeurons == 1:
                    labels = labels.float().unsqueeze(1).to(device)  # OK dla BCE
                else:
                    labels = labels.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        epoch_loss_test = running_loss / len(test_loader)
        test_losses.append(epoch_loss_test)

        print(
            f"Epoch [{epoch+1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss test: {epoch_loss_test:.4f}")

        # Early stopping
        if epoch_loss_test < best_loss:
            best_loss = epoch_loss_test
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            print(f"Early stopping trigger count: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping activated.")
                break

    model.load_state_dict(best_model_state)

    save_dir = "../models_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/mlp_{"pretrained" if pretrained else "raw"}.pth')

    return model.state_dict()


def evaluate(model=None, pretrained=False):
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test = load_data()
    #############################
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    nOutputNeurons = 1 if len(dataset_train.classes) == 2 else len(dataset_train.classes)
    # print(f"Testing on {nOutputNeurons} neurons")

    if model is None:
        weights = f"models_checkpoints/mlp_{'pretrained' if pretrained else 'raw'}.pth"
        model = return_model(nOutputNeurons)
        model.load_state_dict(torch.load(weights))

    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if nOutputNeurons == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += preds.eq(labels.unsqueeze(1)).sum().item()
            else:
                # Multi-class classification
                _, preds = torch.max(outputs, 1)   # preds shape: [batch_size]
                correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy