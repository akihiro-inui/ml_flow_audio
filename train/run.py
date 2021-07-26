import os
import mlflow
import argparse
import numpy as np
import mlflow.pytorch
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from constants import MODEL_ENUM

from train.dataset import GTZANDataSet
from train.models import SimpleModel, evaluate, train
from utils.feature_extraction import extract_melspectrogram
from utils.common_logger import logger


def get_model(model_type: str, device: str):
    """
    Get ML or DSP model from MODEL_ENUM
    :param model_type: Name of model
    :param device: Device "cpu" or "cuda:0"
    :return: Model
    """
    try:
        if model_type == MODEL_ENUM.SIMPLE_MODEL.value:
            model = SimpleModel().to(device)
        else:
            raise ValueError("Unknown model")
        model.eval()
        mlflow.pytorch.log_model(model, "model")
        return model
    except ValueError:
        logger.error(f"Could not find the model: {MODEL_ENUM.SIMPLE_MODEL.value}")
    except Exception as err:
        logger.error(f"Could not load model: {err}")


def get_transformer(sample_rate: int, duration: int):
    """
    Get Transformer that is applied to input data
    :param sample_rate: Sample rate
    :param duration: Duration in seconds
    :return: Torch transform
    """
    return transforms.Compose([
        lambda x: x[0:sample_rate*duration],  # Clip to {duration} seconds
        lambda x: x.astype(np.float32) / np.max(x),  # Normalize time domain signal to to -1 to 1
        lambda x: extract_melspectrogram(audio=x, sampling_rate=sample_rate),
        lambda x: Tensor(x)
    ])


def get_data_loader(data_directory_path: str, transformer, batch_size: int, shuffle: bool = True, num_workers: int = 0)\
        -> DataLoader:
    """
    Get dataset
    :param data_directory_path: Directory path where files exist e.g. data/gtzan/preprocess/train
    :param transformer: Torch transform function
    :param batch_size: Batch size
    :param shuffle: Boolean to shuffle dataset order
    :param num_workers: Number of workers for data loader
    :return: Torch DataLoader class
    """
    try:
        train_dataset = GTZANDataSet(data_directory=data_directory_path, transform=transformer)
        return DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers)

    except Exception as err:
        logger.error(f"Failed to get DataLoader: {err}")


def start_run(
        mlflow_experiment_id: int,
        data_directory: str,
        model_directory: str,
        tensorboard_directory: str,
        batch_size: int,
        num_workers: int,
        epochs: int,
        learning_rate: float,
        model_type: str,
):
    """
    Run Model training
    :param mlflow_experiment_id: Int experiment ID
    :param data_directory: Directory path to dataset
    :param model_directory: Directory path to save model checkpoints
    :param tensorboard_directory: Directory path to store tensorboard
    :param batch_size: Batch size for training
    :param num_workers: Number of workers for Data Loader
    :param epochs: Number of epochs for training
    :param learning_rate: Learning rate
    :param model_type: Name of model e.g. "simple"

    This function executes the followings:
    1. Get available device, cpu or gpu
    2. Get transformer
    3. Get train & test dataset
    4. Get model
    5. Train model
    6. Evaluate model performance
    7. Save model file and checkpoints
    """
    # 1. Get available device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir=tensorboard_directory)

    # 2. Get transformer
    transform = get_transformer(sample_rate=8000, duration=30)

    # 3. Get training & test dataset
    logger.info("Loading training dataset")
    train_dataloader = get_data_loader(data_directory_path=os.path.join(data_directory, "train"),
                                       transformer=transform,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)

    logger.info("Loading test dataset")
    test_dataloader = get_data_loader(data_directory_path=os.path.join(data_directory, "test"),
                                      transformer=transform,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

    # 4. Get model
    model = get_model(model_type, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        writer=writer,
        model_directory=model_directory,
        device=device,
    )

    # 6. Evaluate model performance
    accuracy, loss = evaluate(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        writer=writer,
        epoch=epochs,
        device=device,
    )
    logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")

    writer.close()

    # 7. Save model file and checkpoints
    model_file_path = os.path.join(model_directory, f"gtzan_{mlflow_experiment_id}.pth")
    torch.save(model.state_dict(), model_file_path)

    # MLFlow log
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_workers", num_workers)
    mlflow.log_param("device", device)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    mlflow.log_artifact(model_file_path)
    mlflow.log_artifacts(tensorboard_directory, artifact_path="tensorboard")


def main():
    # Arg parser
    parser = argparse.ArgumentParser(
        description="Train GTZAN Genre Classification Model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default="./data/gtzan/preprocess",
        help="data directory",
    )
    parser.add_argument(
        "--model_directory",
        type=str,
        default="./data/gtzan/model/",
        help="model directory",
    )
    parser.add_argument(
        "--tensorboard_directory",
        type=str,
        default="./data/gtzan/tensorboard/",
        help="tensorboard directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_ENUM.SIMPLE_MODEL.value,
        choices=[
            MODEL_ENUM.SIMPLE_MODEL.value
        ],
        help="simple, vgg11 or vgg16",
    )
    args = parser.parse_args()

    # Get args
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
    data_directory = args.data_directory

    # Create empty directories to save model files
    os.makedirs(args.model_directory, exist_ok=True)
    os.makedirs(args.tensorboard_directory, exist_ok=True)

    # Start training
    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        data_directory=data_directory,
        model_directory=args.model_directory,
        tensorboard_directory=args.tensorboard_directory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
