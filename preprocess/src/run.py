import argparse
import json
import os
import shutil
from glob import glob
from distutils.dir_util import copy_tree

import mlflow
import torchaudio
from configuration import PreprocessConfigurations


def main():
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
        "--downstream",
        type=str,
        default="../data/gtzan/preprocess/",
        help="downstream directory",
    )
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )
    args = parser.parse_args()

    downstream_directory = args.downstream

    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )
    else:
        train_output_destination = os.path.join(
            downstream_directory,
            "train",
        )
        test_output_destination = os.path.join(
            downstream_directory,
            "test",
        )
        gtzan_directory = os.path.join(
            downstream_directory,
            "gtzan-batches-py",
        )

        os.makedirs(downstream_directory, exist_ok=True)
        os.makedirs(train_output_destination, exist_ok=True)
        os.makedirs(test_output_destination, exist_ok=True)
        os.makedirs(gtzan_directory, exist_ok=True)

        training_dataset = torchaudio.datasets.GTZAN(root=downstream_directory,
                                                     subset="training",
                                                     download=False)

        test_dataset = torchaudio.datasets.GTZAN(root=downstream_directory,
                                                 subset="testing",
                                                 download=False)

        # Prepare dict where key is class as integer and values are file names
        meta_train = {i: [] for i in range(10)}
        meta_test = {i: [] for i in range(10)}

        # Create sub-folders in train and test
        genre_names_list = [sub_folder.name for sub_folder in os.scandir(os.path.join(downstream_directory, "genres")) if sub_folder.is_dir()]
        for genre in genre_names_list:
            os.mkdir(f"{train_output_destination}/{genre}")
            os.mkdir(f"{test_output_destination}/{genre}")

        # Make dictionary where key is class string and value is integer
        for file_name in training_dataset._walker:
            genre = file_name.split(".")[0]
            shutil.copy(f"{training_dataset._path}/{genre}/{file_name}.wav", f"{train_output_destination}/{genre}/{file_name}.wav")
            meta_train[int(PreprocessConfigurations.label2int[genre])].append(f"{training_dataset._path}/{genre}/{file_name}.wav")

        # Make dictionary where key is class string and value is integer
        for file_name in test_dataset._walker:
            genre = file_name.split(".")[0]
            shutil.copy(f"{test_dataset._path}/{genre}/{file_name}.wav", f"{test_output_destination}/{genre}/{file_name}.wav")
            meta_test[int(PreprocessConfigurations.label2int[genre])].append(f"{test_dataset._path}/{genre}/{file_name}.wav")

        classes_filepath = os.path.join(
            downstream_directory,
            "classes.json",
        )
        meta_train_filepath = os.path.join(
            downstream_directory,
            "meta_train.json",
        )
        meta_test_filepath = os.path.join(
            downstream_directory,
            "meta_test.json",
        )
        with open(classes_filepath, "w") as f:
            json.dump(PreprocessConfigurations.classes, f)
        with open(meta_train_filepath, "w") as f:
            json.dump(meta_train, f)
        with open(meta_test_filepath, "w") as f:
            json.dump(meta_test, f)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )


if __name__ == "__main__":
    main()
