from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.stats import entropy
import numpy as np
import scipy.io

def bandpass_filter(signal, fs=250, lowcut=0.5, highcut=40):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_ecg_features(mat_file):
    data = scipy.io.loadmat(mat_file)
    if 'val' not in data:
        return None
    ecg_signal = bandpass_filter(data['val'][0])
    
    # Statistical features
    mean_val = np.mean(ecg_signal)
    std_dev = np.std(ecg_signal)
    skewness = np.mean((ecg_signal - mean_val) ** 3) / std_dev ** 3
    kurtosis = np.mean((ecg_signal - mean_val) ** 4) / std_dev ** 4
    peak_to_peak = np.ptp(ecg_signal)

    # Time-domain features
    peaks, _ = find_peaks(ecg_signal, distance=200)
    heart_rate = len(peaks)
    rr_intervals = np.diff(peaks) / 250
    mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    std_rr = np.std(rr_intervals) if len(rr_intervals) > 0 else 0

    # Frequency-domain features
    freqs, psd = welch(ecg_signal, fs=250)
    dominant_freq = freqs[np.argmax(psd)]
    spectral_entropy = entropy(psd)

    return [mean_val, std_dev, skewness, kurtosis, peak_to_peak, heart_rate, mean_rr, std_rr, dominant_freq, spectral_entropy]