import os
import argparse
from tqdm import tqdm
from utils.file_read_write import find_files, write_dict_to_json
from mir_library.dsp_model.key_detection import KeyDetection
from mir_library.data.file_loader import FileLoader


def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data_directory",
        type=str,
        default="./data/gtzan/preprocess/genres",
        help="data directory",
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Sampling Rate",
    )

    parser.add_argument(
        "--out_file_path",
        type=str,
        default="./key.json",
        help="Output file path",
    )

    args = parser.parse_args()

    # Define key detection model
    KeyDetectionModel = KeyDetection(bins_per_octave=24, corr_threshold=0.9)

    # Run on target data
    result = {}
    for file_path in tqdm(find_files(args.data_directory, ".wav")):
        key_profile = run_one(KeyDetectionModel, file_path, args.sample_rate)
        result.update({os.path.split(file_path)[1]: key_profile['key']})

    # Write out result
    write_dict_to_json(result, args.out_file_path)


def run_one(model: KeyDetection, file_path: str, sampling_rate: int) -> dict:
    """
    Run key detection model on target file
    :param model: Key detection model
    :param file_path: Input file path
    :param sampling_rate: Sample rate
    :return: Key profile
    """
    audio_array = FileLoader.load_audio(file_path)
    return model.apply(audio_array, sampling_rate)


if __name__ == "__main__":
    main()
