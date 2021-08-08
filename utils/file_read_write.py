import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import ClassVar
from utils.common_logger import logger


def write_class_to_file(input_class: ClassVar, output_json_file_path: str):
    """
    Write Python class to output json file
    :param input_class: Input class to write out
    :param output_json_file_path: Output json file path
    """
    try:
        with open(output_json_file_path, "w") as f:
            json.dump(input_class.classes, f)
    except Exception as err:
        logger.error(f"Failed to write out the file: {err}")


def create_sub_directories(main_directory_path: str, sub_directory_names_list: list):
    """
    Create empty sub-directories under main_directory_path
    :param main_directory_path: Main directory path to create sub-directories underneath
    :param sub_directory_names_list: List of sub-directory names
    """
    try:
        for sub_name in sub_directory_names_list:
            os.makedirs(f"{main_directory_path}/{sub_name}", exist_ok=True)
            os.makedirs(f"{main_directory_path}/{sub_name}", exist_ok=True)
    except Exception as err:
        logger.error(f"Failed to write out the file: {err}")


def find_files(input_directory: str, extension: str):
    """
    Find all file paths with the given file extension.
    :param input_directory: Find all files under this directory
    :param extension: File extension to match
    :return: file_paths_list: list of file path
    """
    # Get all filenames under the directory
    file_paths_list = []
    for file_path in tqdm(Path(input_directory).glob(f'**/*{extension}')):
        file_paths_list.append(file_path)
    return file_paths_list


def write_dict_to_json(input_dictionary: dict, output_json_file_path: str):
    """
    Write out dictionary to json file
    :param input_dictionary: Input dictionary
    :param output_json_file_path: Output json file path
    """
    try:
        with open(output_json_file_path, "w") as outfile:
            json.dump(input_dictionary, outfile, indent=4)
    except Exception as err:
        logger.error(f"Failed to write out the file: {err}")
