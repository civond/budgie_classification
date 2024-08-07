import toml
import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import scipy.signal as signal
import cv2
import os

# Make Directories
def make_directories(dir_name):
    if os.path.exists(dir_name):
        print(f"{dir_name} already exists.")
    else:
        print(f"Writing: {dir_name}.")
        voc_dir = os.path.join(dir_name, "voc")
        noise_dir = os.path.join(dir_name, "noise")
        os.mkdir(dir_name)
        os.mkdir(voc_dir)
        os.mkdir(noise_dir)

# Filter audio
def load_filter_audio(path):
    [y, fs] = lr.load(path, sr=None)
    [b,a] = signal.cheby2(N=8,
                        rs=25,
                        Wn=300,
                        btype='high',
                        fs=fs)
    y = signal.filtfilt(b,a,y)
    return [y, fs]


def generate_spectrogram(audio, fs):
    stftMat_mel = lr.feature.melspectrogram(
                            y = audio, 
                            sr=fs,
                            n_fft=2048, 
                            win_length=int(fs*0.008),
                            hop_length=int(fs*0.001), 
                            center=True, 
                            window='hann',
                            n_mels=225,
                            fmax=10000)
    return stftMat_mel


# Convert spectrogramn to dB and rotate 180 degrees
def spec2dB(spec, show_img = False):

    # Piezo Spectrogram
    db = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    #db = db**2 # originally no squaring
    normalized_image = cv2.normalize(db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    normalized_image = clahe.apply(normalized_image)
    
    # Input CLAHE image into BGR channel
    rgb_image = np.zeros((normalized_image.shape[0], 
                          normalized_image.shape[1], 
                          3), dtype=np.uint8)
    
    rgb_image[:, :, 0] = normalized_image
    rgb_image[:, :, 1] = normalized_image
    rgb_image[:, :, 2] = normalized_image

    rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)

    if show_img == True:
        cv2.imshow("piezo", normalized_image)#, colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rgb_image