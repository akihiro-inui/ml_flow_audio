import logging
import os
import time
from typing import Tuple
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from scipy.io import wavfile
import torchaudio
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from configuration import PreprocessConfigurations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GTZANDataset(Dataset):
    def __init__(self, data_directory, transform):
        super().__init__()
        self.label2int = PreprocessConfigurations.label2int
        self.data_directory = data_directory
        self.transform = transform
        self.audio_array_list = []
        self.label_list = []
        self.__load_audio_files_and_labels()

    def __len__(self):
        return len(self.audio_array_list)

    def __getitem__(self, index):
        audio_array = self.audio_array_list[index]

        audio_tensor = self.transform(audio_array)
        label = self.label_list[index]

        return audio_tensor, label

    def __load_audio_files_and_labels(self):
        """
        File paths list as ["~/something.wav", "~/something_else.wav"]
        label_list as [0,0,0,0,1,1,1,1,....]
        """
        file_path_list = []
        sub_folder_names = [folder_name for folder_name in os.listdir(self.data_directory) if not folder_name.startswith('.')]
        for sub_folder_name in sub_folder_names:
            class_int = self.label2int[sub_folder_name]
            file_path_list.extend([os.path.join(self.data_directory, sub_folder_name, file_name) for file_name in os.listdir(os.path.join(self.data_directory, sub_folder_name))])
            self.label_list.extend([int(class_int) for _ in os.listdir(os.path.join(self.data_directory, sub_folder_name))])

        for file_path in file_path_list:
            try:
                data, _ = librosa.load(file_path, duration=10)
                self.audio_array_list.append(data)
            except Exception as err:
                logger.error(f"Failed to load file: {file_path}: {err}")
        logger.info(f"loaded: {len(self.label_list)} data")


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 4)
        self.maxpool1 = nn.MaxPool2d(4)
        self.dropout1 = nn.Dropout2d(0.25)
        self.flatten1 = nn.Flatten()
        self.dense1 = nn.Linear(75392, 32)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
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
        checkpoints_directory: str = "/opt/gtzan/model/",
        device: str = "cpu",
):
    logger.info("start training...")
    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"starting epoch: {epoch}")
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
                logger.info(f"[{epoch}, {i}] loss: {running_loss / 200} duration: {end - start}")
                running_loss = 0.0
                start = time.time()
        epoch_end = time.time()
        logger.info(f"[{epoch}] duration in seconds: {epoch_end - epoch_start}")

        _, loss = evaluate(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            device=device,
        )
        checkpoints = os.path.join(checkpoints_directory, f"epoch_{epoch}_loss_{loss}.pth")
        logger.info(f"save checkpoints: {checkpoints}")
        torch.save(model.state_dict(), checkpoints)
