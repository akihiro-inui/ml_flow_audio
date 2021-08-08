import numpy as np


def spec_to_image(spectrogram: np.ndarray, eps=1e-6):
    """
    Normalize spectrogram
    :param spec:
    :param eps:
    :return:
    """
    mean = spectrogram.mean()
    std = spectrogram.std()
    spec_norm = (spectrogram - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled
