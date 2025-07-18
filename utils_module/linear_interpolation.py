import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from data.combined_loader import load_data
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate(model):
    batch_size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #####LOADING DATA############
    dataset_train, dataset_test = load_data()
    #############################
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)   # preds shape: [batch_size]
            correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def interpolate_acc(model_a, model_b, n_alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_1 = model_a.state_dict()
    state_dict_2 = model_b.state_dict()

    alphas = np.linspace(0, 1, n_alpha)

    accs = []
    for i, alpha in enumerate(alphas):
        print(f"Alpha: {i+1}")
        interpolated_state_dict = {}

        for key in state_dict_1.keys():
            interpolated_state_dict[key] = ((1 - alpha) * state_dict_1[key].float() + alpha * state_dict_2[key].float())

        model = copy.deepcopy(model_a)
        model.load_state_dict(interpolated_state_dict)
        model.to(device)
        model.eval()
        accuracy = evaluate(model, pretrained=True)

        accs.append(accuracy)

    plt.plot(alphas, accs)
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.savefig('linear_interpolation.png')


def interpolate_softmax(model_a, model_b, image, true_label, n_alpha=30, ax=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_1 = model_a.state_dict()
    state_dict_2 = model_b.state_dict()

    alphas = np.linspace(0, 1, n_alpha)
    probs = []

    for i, alpha in enumerate(alphas):
        print(f"Alpha: {i+1}")
        interpolated_state_dict = {}

        for key in state_dict_1.keys():
            interpolated_state_dict[key] = ((1 - alpha) * state_dict_1[key].float() + alpha * state_dict_2[key].float())

        model = copy.deepcopy(model_a)
        model.load_state_dict(interpolated_state_dict)
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            prob = torch.softmax(output, dim=1)[0, true_label].item()
            probs.append(prob)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(alphas, probs)
    ax.set_xlabel('alpha')
    ax.set_ylabel('class probability')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    return ax


def interpolate_loss(model_a, model_b, n_alpha=30, ax=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_1 = model_a.state_dict()
    state_dict_2 = model_b.state_dict()

    alphas = np.linspace(0, 1, n_alpha)

    losses = []
    for i, alpha in enumerate(alphas):
        print(f"Alpha: {i + 1}")
        interpolated_state_dict = {}

        for key in state_dict_1.keys():
            interpolated_state_dict[key] = ((1 - alpha) * state_dict_1[key].float() + alpha * state_dict_2[key].float())

        model = copy.deepcopy(model_a)
        model.load_state_dict(interpolated_state_dict)
        model.to(device)
        model.eval()

        train_ds, test_ds = load_data()
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
        nOutputNeurons = 1 if len(train_ds.classes) == 2 else len(train_ds.classes)
        criterion = nn.BCEWithLogitsLoss() if nOutputNeurons == 1 else nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

        losses.append(loss)

    plt.plot(alphas, losses)
    plt.xlabel('alpha')
    plt.ylabel('loss')
    plt.savefig('linear_interpolation_loss.png')