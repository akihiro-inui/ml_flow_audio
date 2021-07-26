import librosa
import numpy as np
# from matplotlib.plot import plt
from utils.common_logger import logger


def extract_melspectrogram(audio: np.ndarray, sampling_rate: int, num_fft: int = 2048, num_of_mels: int = 60,
                           log_scale: bool = True, normalize: bool = True, visualize: bool = False):
    """
    Compute MelSpectrogram with fixed length
    :param audio: Input audio numpy array
    :param sampling_rate: Sampling rate
    :param num_fft: Number of fft size
    :param num_of_mels: Number of mel-filter coefficients
    :param log_scale: Log scale mel-spectrogram
    :param normalize: Normalize mel-spectrogram
    :param visualize: Visualize mel-spectrogram
    :return: Mel-spectrogram
    """
    try:
        # Compute a mel-scaled spectrogram
        # Extract mel-spectrogram from entire audio file
        mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                         sr=sampling_rate,
                                                         n_fft=num_fft,
                                                         n_mels=num_of_mels)

        # Convert to log scale
        if log_scale:
            mel_spectrogram = np.log(mel_spectrogram + 1e-9)

        # Normalize
        if normalize is True:
            mel_spectrogram = librosa.util.normalize(mel_spectrogram)

        # Visualize
        if visualize:
            # TODO: Do this
            print("Visualize Mel-spectrogram here")
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),
            #                          y_axis='mel',
            #                          fmax=self.cfg.sample_rate,
            #                          x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()
        return mel_spectrogram
    except Exception as err:
        logger.error(f"Failed to extract mel-spectrogram: {err}")
