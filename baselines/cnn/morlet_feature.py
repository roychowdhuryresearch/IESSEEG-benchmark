import numpy as np
import  math
from scipy.interpolate import interp1d
import scipy.linalg as LA
import numpy as np 
from skimage.transform import resize
from multiprocessing import Process
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mne
import torch
import time

def create_extended_sig(sig):
    s_len = len(sig)
    s_halflen = int(np.ceil(s_len/2)) + 1
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    start_win = -start_win[::-1] + sig[0]
    end_win = -end_win[::-1] + sig[-1]
    final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
    if len(final_sig)%2 == 0:
        final_sig = final_sig[:-1]
    return final_sig

def compute_spectrum(org_sig, ps_SampleRate = 2000, ps_FreqSeg = 512, ps_MinFreqHz = 10, ps_MaxFreqHz = 500, ps_StDevCycles = 3, device='cuda'):
    extend_sig = create_extended_sig(org_sig)
    extend_sig = torch.from_numpy(extend_sig).to(device)
    
    s_Len = len(extend_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    v_WAxis = torch.linspace(0, 2*np.pi, s_Len, device=device)[:-1]* ps_SampleRate
    v_WAxisHalf = v_WAxis[:s_HalfLen].repeat(ps_FreqSeg, 1)
    v_FreqAxis = torch.linspace(ps_MaxFreqHz, ps_MinFreqHz, steps=ps_FreqSeg, device=device)
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=device)
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5*torch.pow(v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)), 2) * (s_StDevSec**2).view(-1, 1))
    v_WinFFT = v_WinFFT * np.sqrt(s_Len)/ torch.norm(v_WinFFT, dim = -1).view(-1, 1)
    v_InputSignalFFT = torch.fft.fft(extend_sig)
    res = torch.fft.ifft(v_InputSignalFFT.view(1,-1)* v_WinFFT)/torch.sqrt(s_StDevSec).view(-1,1)
    ii, jj = int(len(org_sig)//2), int(len(org_sig)//2 + len(org_sig))
    res = torch.abs(res[:, ii:jj])
    return res.cpu().numpy()  # Convert to numpy before returning

def construct_coding(raw_signal, length=2000):
    index = np.arange(len(raw_signal))
    intensity_image = np.zeros((int(length), int(length)))
    intensity_image[index, :] = raw_signal
    return intensity_image

def wavelet_spectrogram(
    eeg_data,
    sampling_rate=200,
    freq_seg=128,
    min_freq=10,
    max_freq=100,
    stdev_cycles=3,
):
    """
    Converts a multi-channel EEG array of shape (n_channels, n_times)
    into a wavelet-based spectrogram array of shape:
      (n_channels, freq_seg, n_times),
    where freq_seg is the number of frequency steps.

    Args:
      eeg_data: np.ndarray, shape => (n_channels, n_times)
      sampling_rate: int, e.g. 200
      freq_seg: int, number of frequency steps (like # of scales)
      min_freq, max_freq: min/max frequencies for wavelet transform
      stdev_cycles: the Gaussian's std-dev factor in cycles

    Returns:
      A NumPy array of shape (n_channels, freq_seg, n_times).
    """
    n_channels, n_times = eeg_data.shape
    spectrograms = []

    for ch in range(n_channels):
        channel_data = eeg_data[ch, :]
        spec_2d = compute_spectrum(
            channel_data,
            ps_SampleRate=sampling_rate,
            ps_FreqSeg=freq_seg,
            ps_MinFreqHz=min_freq,
            ps_MaxFreqHz=max_freq,
            ps_StDevCycles=stdev_cycles,
        )
        # spec_2d => (freq_seg, ~n_times)
        spectrograms.append(spec_2d[None, ...])  # add channel dim => (1, freq_seg, ~n_times)

    # Stack along channel dimension => (n_channels, freq_seg, ~n_times)
    return np.concatenate(spectrograms, axis=0)

def make_wavelet_3d_batch(
    waveform_batch,
    sampling_rate=200,
    freq_seg=64,
    min_freq=1,
    max_freq=100,
    stdev_cycles=3,
    use_channel_as_in_dim=False,
    device='cuda'
):
    """
    Convert a batch of EEG waveforms to wavelet spectrograms.
    
    Args:
        waveform_batch: torch.Tensor of shape (batch_size, n_channels, n_times)
        sampling_rate: int, sampling rate of the EEG
        freq_seg: int, number of frequency steps
        min_freq, max_freq: float, frequency range
        stdev_cycles: int, wavelet cycles parameter
        use_channel_as_in_dim: bool, whether to use channels as input dimension
        device: str, device to compute on ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor of shape:
            if use_channel_as_in_dim:
                (batch_size, n_channels, freq_seg, n_times, 1)
            else:
                (batch_size, 1, n_channels, freq_seg, n_times)
    """
    batch_size, n_channels, n_times = waveform_batch.shape
    
    # Move input to device if not already there
    if waveform_batch.device != device:
        waveform_batch = waveform_batch.to(device)
    
    # Initialize output tensor
    if use_channel_as_in_dim:
        wavelet_3d = torch.zeros(batch_size, n_channels, freq_seg, n_times, 1, device=device)
    else:
        wavelet_3d = torch.zeros(batch_size, 1, n_channels, freq_seg, n_times, device=device)
    
    # Process each sample in the batch
    for b in range(batch_size):
        # Get current sample
        current_waveform = waveform_batch[b]  # (n_channels, n_times)
        
        # Compute wavelet spectrogram for each channel
        for ch in range(n_channels):
            channel_data = current_waveform[ch]
            
            # Compute spectrum
            spec_2d = compute_spectrum(
                channel_data.cpu().numpy(),  # Convert to numpy for current implementation
                ps_SampleRate=sampling_rate,
                ps_FreqSeg=freq_seg,
                ps_MinFreqHz=min_freq,
                ps_MaxFreqHz=max_freq,
                ps_StDevCycles=stdev_cycles,
                device=device
            )
            
            # Convert spec_2d to torch tensor and move to device
            spec_2d_tensor = torch.from_numpy(spec_2d).to(device)
            
            # Store in appropriate shape
            if use_channel_as_in_dim:
                wavelet_3d[b, ch, :, :, 0] = spec_2d_tensor
            else:
                wavelet_3d[b, 0, ch, :, :] = spec_2d_tensor
    
    return wavelet_3d

def create_extended_sig_gpu(sig):
    """
    GPU version of create_extended_sig.
    Args:
        sig: torch.Tensor of shape (n_times,)
    Returns:
        torch.Tensor of extended signal
    """
    s_len = len(sig)
    s_halflen = int(math.ceil(s_len/2)) + 1
    
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    
    start_win = -start_win.flip(0) + sig[0]
    end_win = -end_win.flip(0) + sig[-1]
    
    final_sig = torch.cat((start_win[:-1], sig, end_win[1:]))
    if len(final_sig) % 2 == 0:
        final_sig = final_sig[:-1]
    return final_sig

def compute_spectrum_gpu(sig, sampling_rate=2000, freq_seg=512, min_freq=10, max_freq=500, stdev_cycles=3):
    """
    GPU version of compute_spectrum.
    Args:
        sig: torch.Tensor of shape (n_times,)
        sampling_rate: int, sampling rate
        freq_seg: int, number of frequency steps
        min_freq, max_freq: float, frequency range
        stdev_cycles: int, wavelet cycles parameter
    Returns:
        torch.Tensor of shape (freq_seg, n_times)
    """
    # Create extended signal
    extend_sig = create_extended_sig_gpu(sig)
    
    # Get signal length
    s_len = len(extend_sig)
    s_half_len = math.floor(s_len/2) + 1
    
    # Create frequency axis
    freq_axis = torch.linspace(max_freq, min_freq, freq_seg, device=sig.device)
    
    # Create time axis
    time_axis = torch.linspace(0, 2*math.pi, s_len, device=sig.device)[:-1] * sampling_rate
    time_axis_half = time_axis[:s_half_len].repeat(freq_seg, 1)
    
    # Compute standard deviation in seconds
    stdev_sec = (1 / freq_axis) * stdev_cycles
    
    # Create window FFT
    win_fft = torch.zeros(freq_seg, s_len, device=sig.device)
    freq_axis_2d = freq_axis.view(-1, 1)
    stdev_sec_2d = stdev_sec.view(-1, 1)
    
    # Compute Gaussian window
    win_fft[:, :s_half_len] = torch.exp(
        -0.5 * torch.pow(
            time_axis_half - (2 * math.pi * freq_axis_2d),
            2
        ) * (stdev_sec_2d**2)
    )
    
    # Normalize window
    win_fft = win_fft * math.sqrt(s_len) / torch.norm(win_fft, dim=-1).view(-1, 1)
    
    # Compute FFT of input signal
    input_fft = torch.fft.fft(extend_sig)
    
    # Compute wavelet transform
    res = torch.fft.ifft(
        input_fft.view(1, -1) * win_fft
    ) / torch.sqrt(stdev_sec_2d)
    
    # Extract relevant portion
    ii, jj = int(len(sig)//2), int(len(sig)//2 + len(sig))
    res = torch.abs(res[:, ii:jj])
    
    return res

def make_wavelet_3d_batch_gpu(
    waveform_batch,
    sampling_rate=200,
    freq_seg=64,
    min_freq=1,
    max_freq=100,
    stdev_cycles=3,
    use_channel_as_in_dim=False,
    device='cuda',
    chunk_size=128,
):
    """
    GPU version of make_wavelet_3d_batch.
    Convert a batch of EEG waveforms to wavelet spectrograms on GPU.

    Vectorized across (batch, channel) in chunks, instead of either (a) a
    nested Python loop over every individual signal, or (b) one giant
    all-at-once batched op. (a) was the original implementation:
    batch_size*n_channels (~1400+ at batch_size=64, n_channels=22)
    separate small CUDA allocations per training batch, which caused
    severe allocator fragmentation and could OOM even though true peak
    memory need was modest -- on top of being far slower than necessary,
    since the wavelet filter bank (win_fft) only depends on signal length
    and the frequency params (identical for every signal) yet was being
    recomputed on every single one of those 1400+ iterations. (b), tried
    first, batches all signals into one FFT/iFFT call -- exact and fast,
    but needs a genuinely huge intermediate tensor (S, freq_seg, s_len)
    that OOMs at realistic batch sizes (~32GB at batch=64, n_channels=22,
    n_times=6000). Chunking gives the best of both: the filter bank is
    still built once per call, and each chunk's FFT/iFFT is a single
    vectorized op, bounding peak memory to chunk_size signals at a time.

    Args:
        waveform_batch: torch.Tensor of shape (batch_size, n_channels, n_times)
        sampling_rate: int, sampling rate of the EEG
        freq_seg: int, number of frequency steps
        min_freq, max_freq: float, frequency range
        stdev_cycles: int, wavelet cycles parameter
        use_channel_as_in_dim: bool, whether to use channels as input dimension
        device: str, device to compute on ('cuda' or 'cpu')
        chunk_size: int, number of (sample, channel) signals to transform
            per vectorized FFT/iFFT call. Lower uses less peak memory but
            is slower; higher is faster but uses more memory.

    Returns:
        torch.Tensor of shape:
            if use_channel_as_in_dim:
                (batch_size, n_channels, freq_seg, n_times, 1)
            else:
                (batch_size, 1, n_channels, freq_seg, n_times)
    """
    batch_size, n_channels, n_times = waveform_batch.shape

    # Raw .npz waveform data loads as float64 (numpy's default) with no
    # cast anywhere upstream; force float32 here so the whole pipeline
    # (win_fft, model weights) stays consistently float32 regardless of
    # the caller's input dtype -- otherwise fft() on a float64 signal
    # promotes to complex128, and abs() of that yields a float64 result
    # that mismatches the model's float32 conv weights.
    waveform_batch = waveform_batch.float()

    # Move input to device if not already there
    if waveform_batch.device != torch.device(device):
        waveform_batch = waveform_batch.to(device)

    # Pre-compute frequency axis and standard deviation (independent of the
    # actual signal data)
    freq_axis = torch.linspace(max_freq, min_freq, freq_seg, device=device)
    stdev_sec = (1 / freq_axis) * stdev_cycles  # (freq_seg,)

    # Flatten (batch, channel) into one "signals" axis -- the transform is
    # applied identically and independently to each 1D signal.
    sig_all = waveform_batch.reshape(batch_size * n_channels, n_times)  # (S, T)
    n_signals = sig_all.shape[0]

    # --- vectorized create_extended_sig (mirrors edges, matches original) ---
    s_halflen = int(math.ceil(n_times / 2)) + 1

    def extend(sig):
        start_win = sig[:, :s_halflen] - sig[:, :1]
        end_win = sig[:, n_times - s_halflen - 1:] - sig[:, -1:]
        start_win = -start_win.flip(-1) + sig[:, :1]
        end_win = -end_win.flip(-1) + sig[:, -1:]
        extend_sig = torch.cat((start_win[:, :-1], sig, end_win[:, 1:]), dim=-1)
        if extend_sig.shape[-1] % 2 == 0:
            extend_sig = extend_sig[:, :-1]
        return extend_sig

    s_len = extend(sig_all[:1]).shape[-1]  # same for every signal (fixed n_times)
    s_half_len = math.floor(s_len / 2) + 1

    # --- filter bank: identical for every signal in this call, built once ---
    time_axis = torch.linspace(0, 2 * math.pi, s_len, device=device)[:-1] * sampling_rate
    time_axis_half = time_axis[:s_half_len].repeat(freq_seg, 1)  # (freq_seg, s_half_len)
    freq_axis_2d = freq_axis.view(-1, 1)
    stdev_sec_2d = stdev_sec.view(-1, 1)

    win_fft = torch.zeros(freq_seg, s_len, device=device)
    win_fft[:, :s_half_len] = torch.exp(
        -0.5 * torch.pow(time_axis_half - (2 * math.pi * freq_axis_2d), 2) * (stdev_sec_2d ** 2)
    )
    win_fft = win_fft * math.sqrt(s_len) / torch.norm(win_fft, dim=-1).view(-1, 1)  # (freq_seg, s_len)
    win_fft_b = win_fft.unsqueeze(0)  # (1, freq_seg, s_len)
    inv_sqrt_stdev = torch.sqrt(stdev_sec_2d).unsqueeze(0)  # (1, freq_seg, 1)

    ii, jj = int(n_times // 2), int(n_times // 2 + n_times)

    # --- batched FFT / wavelet convolution, chunked over the signals axis ---
    out_chunks = []
    for start in range(0, n_signals, chunk_size):
        chunk = sig_all[start:start + chunk_size]
        extend_sig = extend(chunk)
        input_fft = torch.fft.fft(extend_sig, dim=-1)  # (c, s_len)
        res = torch.fft.ifft(
            input_fft.unsqueeze(1) * win_fft_b  # (c, freq_seg, s_len)
        ) / inv_sqrt_stdev
        res = torch.abs(res[:, :, ii:jj])  # (c, freq_seg, n_times)
        out_chunks.append(res)

    res = torch.cat(out_chunks, dim=0).reshape(batch_size, n_channels, freq_seg, n_times)

    if use_channel_as_in_dim:
        wavelet_3d = res.unsqueeze(-1)  # (batch, n_channels, freq_seg, n_times, 1)
    else:
        wavelet_3d = res.unsqueeze(1)  # (batch, 1, n_channels, freq_seg, n_times)

    return wavelet_3d

def compare_wavelet_implementations(waveform_batch, sampling_rate=200, freq_seg=64, min_freq=1, max_freq=100, stdev_cycles=3, device='cuda'):
    """
    Compare the original and GPU implementations of wavelet transform.
    Args:
        waveform_batch: torch.Tensor of shape (batch_size, n_channels, n_times)
        ... (other parameters same as make_wavelet_3d_batch)
    Returns:
        tuple of (original_result, gpu_result, max_diff)
    """
    # Original implementation
    orig_result = make_wavelet_3d_batch(
        waveform_batch,
        sampling_rate=sampling_rate,
        freq_seg=freq_seg,
        min_freq=min_freq,
        max_freq=max_freq,
        stdev_cycles=stdev_cycles,
        use_channel_as_in_dim=False,
        device=device
    )
    
    # GPU implementation
    gpu_result = make_wavelet_3d_batch_gpu(
        waveform_batch,
        sampling_rate=sampling_rate,
        freq_seg=freq_seg,
        min_freq=min_freq,
        max_freq=max_freq,
        stdev_cycles=stdev_cycles,
        use_channel_as_in_dim=False,
        device=device
    )
    
    # Compare results
    max_diff = torch.max(torch.abs(orig_result - gpu_result))
    
    return orig_result, gpu_result, max_diff