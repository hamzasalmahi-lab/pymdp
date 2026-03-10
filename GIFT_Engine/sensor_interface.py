import numpy as np
from scipy import signal
import mne

# --- MNE PREPROCESSING CONFIGURATION ---
SAMPLING_RATE = 100  # Hz (adjusted for quick processing)
BANDPASS_LOWCUT = 1.0   # Hz
BANDPASS_HIGHCUT = 40.0  # Hz
NOTCH_FREQ = 60.0  # Hz (electrical noise)
ALPHA_LOWCUT = 8.0  # Hz
ALPHA_HIGHCUT = 12.0  # Hz

def apply_bandpass_filter(eeg_chunk, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT, 
                          sampling_rate=SAMPLING_RATE, order=4):
    """
    Apply Butterworth bandpass filter (1-40 Hz) to EEG data.
    
    Parameters:
    -----------
    eeg_chunk : array-like
        Raw EEG signal
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    sampling_rate : int
        Sampling rate in Hz
    order : int
        Filter order
    
    Returns:
    --------
    filtered_eeg : array-like
        Bandpass filtered EEG
    """
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_eeg = signal.filtfilt(b, a, eeg_chunk)
    return filtered_eeg

def apply_notch_filter(eeg_chunk, notch_freq=NOTCH_FREQ, sampling_rate=SAMPLING_RATE, quality=30):
    """
    Apply notch filter to remove 60Hz electrical noise.
    
    Parameters:
    -----------
    eeg_chunk : array-like
        EEG signal
    notch_freq : float
        Frequency to remove (Hz)
    sampling_rate : int
        Sampling rate in Hz
    quality : int
        Quality factor (Q)
    
    Returns:
    --------
    filtered_eeg : array-like
        Notch filtered EEG
    """
    nyquist = sampling_rate / 2.0
    w0 = notch_freq / nyquist
    
    # Ensure valid range
    w0 = max(w0, 0.001)
    w0 = min(w0, 0.999)
    
    b, a = signal.iirnotch(w0, quality)
    filtered_eeg = signal.filtfilt(b, a, eeg_chunk)
    return filtered_eeg

def calculate_alpha_band_power(eeg_chunk, sampling_rate=SAMPLING_RATE):
    """
    Calculate power in the alpha frequency band (8-12 Hz).
    
    Uses Welch's method for robust power spectral density estimation.
    
    Parameters:
    -----------
    eeg_chunk : array-like
        Preprocessed EEG signal
    sampling_rate : int
        Sampling rate in Hz
    
    Returns:
    --------
    alpha_power : float
        Power in the alpha band
    alpha_power_normalized : float
        Alpha power normalized by total power (0-1 range)
    """
    # Compute power spectral density using Welch's method
    freqs, psd = signal.welch(eeg_chunk, fs=sampling_rate, nperseg=len(eeg_chunk))
    
    # Find indices for alpha band (8-12 Hz)
    alpha_mask = (freqs >= ALPHA_LOWCUT) & (freqs <= ALPHA_HIGHCUT)
    alpha_power = np.sum(psd[alpha_mask])
    
    # Normalized alpha power (total power = 1.0)
    total_power = np.sum(psd)
    alpha_power_normalized = alpha_power / (total_power + 1e-10)  # Add small epsilon to avoid division by zero
    
    return alpha_power, alpha_power_normalized

def get_observation_from_eeg(raw_data_chunk, preprocessing=True):
    """
    Extract observation from raw EEG chunk.
    
    Pipeline:
    1. Bandpass filter (1-40 Hz) - removes DC drift and high-frequency noise
    2. Notch filter (60 Hz) - removes electrical noise
    3. Calculate alpha band power (8-12 Hz)
    4. Classify as Stable (high alpha) or Unstable (low alpha)
    
    Parameters:
    -----------
    raw_data_chunk : array-like
        Raw EEG signal (typically 100 samples = 1 second)
    preprocessing : bool
        Whether to apply preprocessing (True for inference, False for testing)
    
    Returns:
    --------
    observation : int
        0 = Stable state (high alpha power)
        1 = Unstable state (low alpha power)
    
    Notes:
    ------
    Alpha band (8-12 Hz) is associated with:
    - Relaxed, eyes-closed states (high alpha)
    - Attention, cognitive load, anxiety (low/suppressed alpha)
    """
    
    if preprocessing and len(raw_data_chunk) > 10:
        # Apply preprocessing pipeline
        eeg_filtered = apply_bandpass_filter(raw_data_chunk)
        eeg_filtered = apply_notch_filter(eeg_filtered)
    else:
        eeg_filtered = raw_data_chunk
    
    # Calculate alpha band power
    alpha_power, alpha_normalized = calculate_alpha_band_power(eeg_filtered)
    
    # Classify: High alpha (>0.10 normalized) = Stable, Low alpha = Unstable
    # Lowered threshold for increased sensitivity to high-frequency noise
    threshold = 0.10
    observation = 0 if alpha_normalized >= threshold else 1
    
    return observation, alpha_normalized, alpha_power
