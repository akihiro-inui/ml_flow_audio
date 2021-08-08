import os
import time
from typing import Tuple
import mlflow
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.common_logger import logger


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, self.resnet.conv1.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
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
        scheduler,
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
            spectrogram, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(spectrogram)
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
            device=device
        )
        scheduler.step()
        checkpoints = os.path.join(model_directory, f"epoch_{epoch}_loss_{loss}.pth")
        logger.info(f"Save checkpoints: {checkpoints}")
        torch.save(model.state_dict(), checkpoints)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Loss", loss)
