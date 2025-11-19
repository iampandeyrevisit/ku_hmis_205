# Lab 2 Quick Reference Card
## ECG Signal Loading and Visualization - Student Cheat Sheet

---

## Essential WFDB Commands

### Load Record
```python
import wfdb

# From PhysioNet
record = wfdb.rdrecord('3000003', 
                       pn_dir='mimic3wdb-matched/1.0/p00/p000020/3000003_0003')

# From local files
record = wfdb.rdrecord('3000003', pn_dir='path/to/local/data')
```

### Access Record Information
```python
record.sig_name        # Signal names ['II', 'V', 'ABP', ...]
record.fs              # Sampling frequency (Hz)
record.sig_len         # Number of samples
record.n_sig           # Number of signals
record.units           # Units for each signal ['mV', 'mmHg', ...]
record.p_signal        # The actual signal data (numpy array)
```

### Access Signal Data
```python
# All signals (2D array: samples × channels)
all_signals = record.p_signal

# Single signal (e.g., first channel)
ecg_signal = record.p_signal[:, 0]

# Time array (seconds)
time = np.arange(len(ecg_signal)) / record.fs

# Specific segment
start_sec = 10
duration_sec = 5
start_sample = int(start_sec * record.fs)
end_sample = int((start_sec + duration_sec) * record.fs)
segment = record.p_signal[start_sample:end_sample, 0]
```

---

## Basic Plotting

### Single Lead ECG
```python
import matplotlib.pyplot as plt
import numpy as np

# Extract 10 seconds
duration = 10
samples = int(duration * record.fs)
ecg = record.p_signal[:samples, 0]
time = np.arange(samples) / record.fs

# Plot
plt.figure(figsize=(15, 5))
plt.plot(time, ecg, 'b-', linewidth=0.8)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG Signal')
plt.grid(True, alpha=0.3)
plt.show()
```

### Multiple Leads (Subplots)
```python
fig, axes = plt.subplots(3, 1, figsize=(15, 9))

for i in range(3):
    axes[i].plot(time, record.p_signal[:samples, i])
    axes[i].set_ylabel(record.sig_name[i])
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.show()
```

---

## Peak Detection (R-peaks)

```python
from scipy.signal import find_peaks

# Simple peak detection
threshold = np.mean(ecg) + 0.5 * np.std(ecg)
peaks, _ = find_peaks(ecg, 
                      height=threshold,
                      distance=int(0.4 * record.fs))

# Plot with peaks
plt.figure(figsize=(15, 5))
plt.plot(time, ecg, 'b-', linewidth=0.8)
plt.plot(time[peaks], ecg[peaks], 'ro', markersize=8)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG with Detected R-peaks')
plt.grid(True)
plt.show()

# Calculate heart rate
num_beats = len(peaks)
heart_rate = (num_beats / duration) * 60
print(f"Heart Rate: {heart_rate:.1f} bpm")
```

---

## Basic Statistics

```python
# Signal statistics
mean_val = np.mean(ecg)
std_val = np.std(ecg)
min_val = np.min(ecg)
max_val = np.max(ecg)
range_val = max_val - min_val

print(f"Mean: {mean_val:.3f}")
print(f"Std: {std_val:.3f}")
print(f"Range: [{min_val:.3f}, {max_val:.3f}]")

# RR intervals (if peaks detected)
if len(peaks) > 1:
    rr_intervals = np.diff(peaks) / record.fs * 1000  # milliseconds
    print(f"Mean RR: {np.mean(rr_intervals):.1f} ms")
    print(f"RR Variability: {np.std(rr_intervals):.1f} ms")
```

---

## Common Tasks

### Find ECG Lead by Name
```python
ecg_idx = None
for i, name in enumerate(record.sig_name):
    if 'II' in name or 'ECG' in name:
        ecg_idx = i
        break

if ecg_idx is not None:
    ecg = record.p_signal[:, ecg_idx]
    print(f"Found ECG at index {ecg_idx}: {record.sig_name[ecg_idx]}")
```

### Calculate Duration
```python
duration_seconds = record.sig_len / record.fs
duration_minutes = duration_seconds / 60
print(f"Recording duration: {duration_minutes:.2f} minutes")
```

### Extract Time Window
```python
def extract_window(signal, fs, start_time, duration):
    """Extract time window from signal"""
    start_sample = int(start_time * fs)
    end_sample = int((start_time + duration) * fs)
    return signal[start_sample:end_sample]

# Example: Get 5 seconds starting at 30 seconds
window = extract_window(ecg, record.fs, start_time=30, duration=5)
```

---

## Common Errors and Fixes

### Error: "Record not found"
```python
# Check path is correct (case-sensitive!)
# Verify internet connection
# Ensure PhysioNet credentials are set up
```

### Error: "Index out of bounds"
```python
# Check signal has enough samples
if end_sample > len(record.p_signal):
    end_sample = len(record.p_signal)
```

### Error: "No peaks detected"
```python
# Adjust threshold
threshold = np.mean(ecg) + 0.3 * np.std(ecg)  # Lower threshold

# Adjust minimum distance
min_distance = int(0.3 * record.fs)  # Allow closer peaks
```

### Error: Plot doesn't show
```python
# Add plt.show() at the end
plt.show()

# Or in Jupyter, use magic command:
%matplotlib inline
```

---

## 💡 Quick Tips

### Sampling Rate Shortcuts
- **125 Hz:** Common ECG sampling rate in MIMIC
- **1 second = 125 samples** at 125 Hz
- **10 seconds = 1250 samples**

### Normal ECG Values
- **Heart Rate:** 60-100 bpm (normal)
- **QRS Duration:** 80-120 ms (normal)
- **PR Interval:** 120-200 ms (normal)
- **RR Interval:** 600-1000 ms (normal, 60-100 bpm)

### Signal Quality Indicators
- **Good:** No flat lines, clear QRS, minimal noise
- **Fair:** Some artifacts, but QRS visible
- **Poor:** Flat lines, saturation, excessive noise

---

## 📝 Report Template Structure

```markdown
# Lab 2 Report: ECG Analysis

## 1. Introduction
- Brief background on ECG
- Objectives of the lab

## 2. Methods
- Data source (MIMIC-III record number)
- Analysis approach
- Tools used (Python, wfdb, etc.)

## 3. Results
### 3.1 Signal Characteristics
- Duration, sampling rate
- Signal quality assessment

### 3.2 Heart Rate Analysis
- Calculated heart rate
- Number of beats detected
- RR intervals

### 3.3 Waveform Morphology
- QRS duration
- Observations about P, QRS, T waves

## 4. Discussion
- Clinical interpretation
- Challenges encountered
- Interesting findings

## 5. Figures
- Include all required plots with captions

## 6. Conclusion
- Summary of findings
- What you learned
```

---

## 🆘 Getting Help

### Where to Get Help:
- **Documentation:**
  - WFDB: https://wfdb.readthedocs.io/
  - Matplotlib: https://matplotlib.org/
  - NumPy: https://numpy.org/doc/


---

## ✅ Pre-Lab Checklist

Before starting Lab 2:
- [ ] MIMIC-III access
- [ ] Python 3.7+ installed
- [ ] Required packages installed (run verify_setup.py)
      OR 
- [ ] Have a Colab account

---

## 📚 Key Imports for Lab 2

```python
# Standard imports for all exercises
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import find_peaks
from IPython.display import display

# Optional: prevent warnings
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline  # For Jupyter
```

---

## 🎓 Learning Objectives Reminder

By the end of Lab 2, you should be able to:
1. ✓ Load ECG records from MIMIC database
2. ✓ Visualize ECG signals professionally
3. ✓ Identify P, QRS, and T waves
4. ✓ Calculate heart rate from ECG
5. ✓ Assess signal quality
6. ✓ Compare different ECG leads
7. ✓ Generate analysis reports

---


**Version:** 1.0 | **For:** Lab 2 | **Course:** Medical Signal Processing
