import os
import numpy as np
import torch
from torch.utils.data import Dataset
from morlet_feature import wavelet_spectrogram

def load_data_by_recording_id(data_dir, eeg_recording_id, n_channels=22, scale_factor=1e6, verbose=False):
    data_path = os.path.join(data_dir, f"{eeg_recording_id}.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")
    loaded_npz = np.load(data_path, allow_pickle=True)
    eeg_channel_load = loaded_npz['channel'][:n_channels]
    eeg_data = np.vstack(loaded_npz['data'])[:n_channels,:] * scale_factor
    if verbose:
        print(f"{eeg_recording_id} loaded from {data_path}")
    return eeg_data, eeg_channel_load

class ScalpEEG_Wavelet3D_Dataset(Dataset):
    """
    Loads raw EEG data and returns windowed segments.
    Wavelet computation is moved outside the dataset.
    """
    def __init__(
        self,
        data_dir,
        patient_id,
        eeg_recording_id,
        label,
        window_in_sec=10,
        mode="continuous",
        step_size_in_sec=10,
        random_sample_num=8,
        sample_rate=200,
    ):
        """
        Args:
            data_dir (str): directory for the data
            patient_id: patient identifier
            eeg_recording_id: ID for the specific EEG recording
            label (int or float): Label for the recording
            window_size (int): number of samples in each window
            mode (str): "continuous" or "random"
            step_size (int): step size for continuous mode
            random_sample_num (int): number of random windows if mode="random"
            sample_rate (int): sampling rate
        """
        super().__init__()
        self.mode = mode
        self.patient_id = patient_id
        self.eeg_recording_id = eeg_recording_id
        self.sample_rate = sample_rate
        self.label = label
        self.window_size_in_frames = int(window_in_sec * sample_rate)

        # 1) Load data
        original_waveform, eeg_channel_load = load_data_by_recording_id(data_dir, eeg_recording_id)

        self.original_waveform = original_waveform
        self.eeg_channels = eeg_channel_load
        self.n_channels, self.total_frames = original_waveform.shape

        # 2) Prepare indexing for windows
        if self.mode == "random":
            self.length = int(random_sample_num)
        else:  # continuous
            self.step_size_in_frames = int(step_size_in_sec * sample_rate)
            max_start = self.total_frames - self.window_size_in_frames
            self.length = int(max_start // self.step_size_in_frames + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 1) Determine start index
        if self.mode == "random":
            start_ind = np.random.randint(0, self.total_frames - self.window_size_in_frames)
        else:  # continuous
            start_ind = index * self.step_size_in_frames

        end_ind = start_ind + self.window_size_in_frames

        # print(f"Start index: {start_ind}, End index: {end_ind}, wav shape: {self.original_waveform.shape}")

        # 2) Get windowed segment => shape (n_channels, window_size)
        window_data = self.original_waveform[:, start_ind : end_ind].copy()

        # 3) Build sample dict
        sample_dict = {
            "patient_id": self.patient_id,
            "recording_id": self.eeg_recording_id,
            "start_ind": start_ind,
            "end_ind": end_ind,
            "label": self.label,
            "waveform": window_data          # Raw EEG window
        }

        return sample_dict
