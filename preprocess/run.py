import os
import shutil
import mlflow
import argparse
import torchaudio
from distutils.dir_util import copy_tree
from configurations import PreprocessConfigurations
from utils.file_read_write import write_class_to_file, create_sub_directories


def process_cache_data(args: argparse):
    """
    Process cache data with given cached data ID
    Copy cache data into data directory
    :param args: Argparser see variables in main()
    """
    cached_artifact_directory = os.path.join(
        "/tmp/mlruns",
        args.cached_data_id,
        "artifacts/data_directory",
    )

    # Copy cache data
    copy_tree(
        cached_artifact_directory,
        args.data_directory,
    )


def process_data(args: argparse):
    """
    Pre-process dataset
    1. Create empty directories for train, test data
    2. Load/Download original dataset
    3. Get label names and create sub-directories under train and test with label names
    4. Copy train and test files into their directories from original dataset
    5. Write out metadata as json files (label, file_names etc)

    :param args: Argparser see variables in main()
    """
    train_output_destination = os.path.join(args.data_directory, "train")
    test_output_destination = os.path.join(args.data_directory, "test")

    # 1. Make directories to store dataset
    os.makedirs(args.data_directory, exist_ok=True)
    os.makedirs(train_output_destination, exist_ok=True)
    os.makedirs(test_output_destination, exist_ok=True)

    # 2. Load or Download dataset
    training_dataset = torchaudio.datasets.GTZAN(root=args.data_directory,
                                                 subset="training",
                                                 download=False)

    test_dataset = torchaudio.datasets.GTZAN(root=args.data_directory,
                                             subset="testing",
                                             download=False)

    # 3. Prepare dict where key is class as integer and values are file names
    label_place_holder = {i: [] for i in range(10)}

    # 3. Get label names from directory names
    label_names_list = [sub_folder.name for sub_folder in os.scandir(os.path.join(args.data_directory, "genres")) if sub_folder.is_dir()]

    # 3. Create sub-folders under train and test folder e.g. ~/train/label1, ~/train/label2, ~/test/label1, ~/test/label2
    create_sub_directories(train_output_destination, label_names_list)
    create_sub_directories(test_output_destination, label_names_list)

    # 4. Copy files into train folder
    # TODO: Make this clean
    for file_name in training_dataset._walker:
        genre = file_name.split(".")[0]
        shutil.copy(f"{train_output_destination}/{genre}/{file_name}.wav", f"{train_output_destination}/{genre}/{file_name}.wav")
        label_place_holder[int(PreprocessConfigurations.label2int[genre])].append(f"{training_dataset._path}/{genre}/{file_name}.wav")

    # 4. Copy files into test folder
    # TODO: Make this clean
    for file_name in test_dataset._walker:
        genre = file_name.split(".")[0]
        shutil.copy(f"{train_output_destination}/{genre}/{file_name}.wav", f"{test_output_destination}/{genre}/{file_name}.wav")
        label_place_holder[int(PreprocessConfigurations.label2int[genre])].append(f"{test_dataset._path}/{genre}/{file_name}.wav")

    # 5. Write out config to files
    write_class_to_file(PreprocessConfigurations.classes, os.path.join(args.data_directory, "classes.json"))
    write_class_to_file(label_place_holder, os.path.join(args.data_directory, "meta_train.json"))
    write_class_to_file(label_place_holder, os.path.join(args.data_directory, "meta_test.json"))


def main():
    # Arg parser
    parser = argparse.ArgumentParser(
        description="Make dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="gtzan",
        help="gtzan or something else; default gtzan",
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default="./data/gtzan/preprocess/",
        help="data directory",
    )
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache. Default ''",
    )
    args = parser.parse_args()

    # Main process
    process_cache_data(args) if args.cached_data_id else process_data(args)

    # Log for MLFlow
    mlflow.log_artifacts(
        args.data_directory,
        artifact_path="data_directory",
    )


if __name__ == "__main__":
    main()
