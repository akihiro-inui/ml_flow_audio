import os
import time
from typing import Tuple
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.common_logger import logger


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # TODO: Something is wrong. Fix it. Torch version of Input size for Convolution
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(57, 6), stride=(1, 1), padding_mode="replicate", padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 3))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 1), padding_mode="replicate", padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.flatten1 = nn.Flatten()
        self.dense1 = nn.Linear(153, 16)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(16, 16)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


def evaluate(
        model: nn.Module,
        test_dataloader: DataLoader,
        criterion,
        writer: SummaryWriter,
        epoch: int,
        device: str = "cpu",
) -> Tuple[float, float]:
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels)

    accuracy = 100 * float(correct / total)
    loss = total_loss / 10000

    writer.add_scalar("Loss/test", loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)
    logger.info(f"Accuracy: {accuracy}, Loss: {loss}")
    return accuracy, float(loss)


def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion,
        optimizer,
        writer: SummaryWriter,
        epochs: int = 10,
        model_directory: str = "/opt/gtzan/model/",
        device: str = "cpu",
):
    logger.info("Start training...")
    for epoch in range(epochs):
        epoch_to_vis = epoch+1
        running_loss = 0.0
        logger.info(f"Starting epoch: {epoch_to_vis}")
        epoch_start = time.time()
        start = time.time()
        for i, data in enumerate(train_dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        for i, data in enumerate(train_dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("data/total_loss", float(loss.item()), (epoch + 1) * i)
            writer.add_scalar("Loss/train", float(running_loss / (i + 1)), (epoch + 1) * i)

            if i % 200 == 199:
                end = time.time()
                logger.info(f"[{epoch_to_vis}, {i}] loss: {running_loss / 200} duration: {end - start}")
                running_loss = 0.0
                start = time.time()
        epoch_end = time.time()
        logger.info(f"Finished epoch {epoch_to_vis}: duration in seconds: {epoch_end - epoch_start}")

        accuracy, loss = evaluate(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            device=device,
        )
        checkpoints = os.path.join(model_directory, f"epoch_{epoch}_loss_{loss}.pth")
        logger.info(f"Save checkpoints: {checkpoints}")
        torch.save(model.state_dict(), checkpoints)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
