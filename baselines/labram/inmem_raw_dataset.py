import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_data_by_recording_id(
    data_dir,
    recording_id,
    patient_id=None,
    n_channels=19,
    scale_factor=1.0,
    verbose=False
):
    """
    Loads single .npz file:
      - 'channel' array
      - 'data' array => shape (n_channels, T)
    Scales by 'scale_factor'.
    Returns raw_data (C,T) and channel_array (optional).
    """
    path = os.path.join(data_dir, f"{recording_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    channel_arr = loaded["channel"][:n_channels]    # e.g. shape=(19,)
    raw_data    = loaded["data"][:n_channels, :]    # shape=(19, T)

    raw_data = raw_data * scale_factor

    if verbose:
        print(f"Loaded: {recording_id}, shape={raw_data.shape}, channels={channel_arr.shape}")

    return raw_data, channel_arr


class InMemoryRandomDataset(Dataset):
    """
    In-memory dataset that:
      1) Preloads all recordings at init into self.memory
      2) For 'train', each __getitem__ => random recording & random window
      3) For 'val'/'test', step-based windows (index_map)
      4) Returns a dict with keys:
         ["patient_id", "recording_id", "start_ind", "end_ind", "label", "waveform"]
         where 'waveform' has shape (C, seq_len, 200).
    """

    def __init__(
        self,
        data_dir,
        info_list,   # list of (patient_id, recording_id, label)
        mode="train",
        sample_rate=200,
        window_sec=30,
        step_sec=30,
        n_channels=19,
        scale_factor=1.0,
        train_iterations=10000,
        verbose=False,
        return_dict=False
    ):
        """
        Args:
            data_dir (str): folder with {recording_id}.npz
            recording_list (list[tuple]): e.g. [(patient_id, rec_id, label), ...]
            mode (str): 'train','val','test'
            sample_rate (int): e.g. 200
            window_sec (int): length of window in seconds
            step_sec (int): step size in seconds for val/test
            n_channels (int): how many channels to load from .npz
            scale_factor (float): multiply raw data
            train_iterations (int): notional length for random train
            verbose (bool): prints info if True
        """
        super().__init__()
        self.data_dir = data_dir
        self.recording_list = info_list
        self.mode = mode
        self.sample_rate = sample_rate
        self.window_frames = int(window_sec * sample_rate)
        self.step_frames   = int(step_sec   * sample_rate)
        self.n_channels = n_channels
        self.scale_factor= scale_factor
        self.verbose = verbose
        self.train_iterations = train_iterations
        self.return_dict = return_dict

        # Load everything into memory
        # We'll store a list of dict => each:
        #   {
        #     'patient_id': str,
        #     'recording_id': str,
        #     'label': int,
        #     'raw_data': shape (C, T),
        #     'T': int (total frames)
        #   }
        self.memory = []
        for (p_id, r_id, lbl) in tqdm(info_list, 
                                      total = len(info_list),
                                      desc=f"[{mode}] Loading data"):
            try:
                raw_data, ch_arr = load_data_by_recording_id(
                    data_dir, r_id,
                    n_channels=n_channels,
                    scale_factor=scale_factor,
                    verbose=False
                )
            except FileNotFoundError:
                if verbose:
                    print(f"[WARN] Missing {r_id}, skipping.")
                continue

            c, total_frames = raw_data.shape
            self.memory.append({
                "patient_id": p_id,
                "recording_id": r_id,
                "label": lbl,
                "raw_data": raw_data,
                "T": total_frames
            })

        # If train => random windows on-the-fly (no index map).
        # If val/test => step-based index_map
        if self.mode == "train":
            self._length = train_iterations
        else:
            self.index_map = []
            for mem_idx, info in enumerate(self.memory):
                T = info["T"]
                if T < self.window_frames:
                    continue
                max_start = T - self.window_frames
                n_steps = max_start // self.step_frames + 1
                for s_i in range(n_steps):
                    start_ind = s_i * self.step_frames
                    self.index_map.append((mem_idx, start_ind))
            self._length = len(self.index_map)
            if verbose:
                print(f"[{mode}] total windows => {self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """
        Returns a dictionary:
          {
            "patient_id": str,
            "recording_id": str,
            "start_ind": int,
            "end_ind": int,
            "label": int,
            "waveform": shape (C, seq_len, 200) float
          }
        """
        if self.mode == "train":
            # random approach
            mem_idx = np.random.randint(0, len(self.memory))
            rec_dict = self.memory[mem_idx]
            T = rec_dict["T"]
            if T < self.window_frames:
                # skip => re-call
                return self.__getitem__(idx)
            max_start = T - self.window_frames
            start_ind = np.random.randint(0, max_start+1)
        else:
            # val/test
            mem_idx, start_ind = self.index_map[idx]
            rec_dict = self.memory[mem_idx]

        raw_data = rec_dict["raw_data"]   # shape (C, T)
        label    = rec_dict["label"]
        patient_id   = rec_dict["patient_id"]
        recording_id = rec_dict["recording_id"]

        end_ind = start_ind + self.window_frames
        window_data = raw_data[:, start_ind:end_ind]  # shape (C, window_frames)

        if self.return_dict:
            sample_dict = {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "start_ind": start_ind,
                "end_ind": end_ind,
                "label": label,
                "waveform": window_data
            }
            return sample_dict
        else:
            return window_data, label

    @staticmethod
    def collate_fn(batch_list):
        """
        batch_list => list of sample_dict
        We'll stack the waveforms => (B,C,seq_len,200)
        We'll stack labels => (B,)
        The rest => keep as lists
        """
        # collect all waveforms, labels, etc.
        waveforms = []
        labels    = []
        patient_ids = []
        recording_ids = []
        start_inds = []
        end_inds   = []

        for d in batch_list:
            waveforms.append(d["waveform"])
            labels.append(d["label"])
            patient_ids.append(d["patient_id"])
            recording_ids.append(d["recording_id"])
            start_inds.append(d["start_ind"])
            end_inds.append(d["end_ind"])

        waveforms_np = np.stack(waveforms, axis=0)  # (B,C,seq_len,200)
        waveforms_t  = torch.from_numpy(waveforms_np).float()
        labels_t     = torch.tensor(labels, dtype=torch.long)

        # We return a dictionary for the whole batch, so the trainer can parse it
        batch_out = {
            "waveform": waveforms_t,        # shape (B,C,seq_len,200)
            "label": labels_t,              # shape (B,)
            "patient_id": patient_ids,      # list[str]
            "recording_id": recording_ids,  # list[str]
            "start_ind": start_inds,        # list[int]
            "end_ind": end_inds,            # list[int]
        }
        return batch_out


class LoadDataset:
    """
    Minimal wrapper that constructs train/val/test DataLoaders from lists
    of (patient_id, recording_id, label).
    """
    def __init__(self, params):
        self.params = params
        self.data_dir = params.datasets_dir
        self.batch_size = params.batch_size

        # Suppose we have e.g.:
        #  params.train_list => [(ptid, recid, lbl), ...]
        #  params.val_list   => ...
        #  params.test_list  => ...
        # or we might load these from CSV or somewhere else
        self.train_list = params.train_list
        self.val_list   = params.val_list
        self.test_list  = params.test_list

        # config
        self.sample_rate    = 200
        self.window_sec     = 30
        self.step_sec       = 30
        self.n_channels     = 19
        self.scale_factor   = 1.0/100
        self.train_iterations = 10000
        self.verbose = False

    def get_data_loader(self):
        train_ds = InMemoryRandomDataset(
            data_dir=self.data_dir,
            info_list=self.train_list,   # => list of (ptid, recid, label)
            mode="train",
            sample_rate=self.sample_rate,
            window_sec=self.window_sec,
            step_sec=self.step_sec,
            n_channels=self.n_channels,
            scale_factor=self.scale_factor,
            train_iterations=self.train_iterations,
            verbose=self.verbose
        )
        val_ds   = InMemoryRandomDataset(
            data_dir=self.data_dir,
            info_list=self.val_list,
            mode="val",
            sample_rate=self.sample_rate,
            window_sec=self.window_sec,
            step_sec=self.step_sec,
            n_channels=self.n_channels,
            scale_factor=self.scale_factor,
            verbose=self.verbose
        )
        test_ds  = InMemoryRandomDataset(
            data_dir=self.data_dir,
            info_list=self.test_list,
            mode="test",
            sample_rate=self.sample_rate,
            window_sec=self.window_sec,
            step_sec=self.step_sec,
            n_channels=self.n_channels,
            scale_factor=self.scale_factor,
            verbose=self.verbose
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            collate_fn=train_ds.collate_fn
        )
        val_loader   = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=val_ds.collate_fn
        )
        test_loader  = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=test_ds.collate_fn
        )

        print(f"[DATA] train_ds len={len(train_ds)}  val_ds len={len(val_ds)}  test_ds len={len(test_ds)}")
        loaders = {
            'train': train_loader,
            'val':   val_loader,
            'test':  test_loader
        }
        return loaders
