import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import librosa
from torch.utils.data import Dataset
from configurations import PreprocessConfigurations
from utils.common_logger import logger


class GTZANDataSet(Dataset):
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
                data, sample_rate = librosa.load(file_path)
                self.audio_array_list.append(data)
            except Exception as err:
                logger.error(f"Failed to load file: {file_path}: {err}")
        logger.info(f"Loaded: {len(self.label_list)} samples")
