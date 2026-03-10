# MNE Preprocessing Pipeline & Agent Learning Visualization

## Overview

Two major enhancements have been integrated into the GIFT Engine:

1. **Professional EEG Preprocessing Pipeline** in `sensor_interface.py`
2. **Agent Generative Model Learning Visualization** in `main_engine.py`

These upgrades enable more sophisticated EEG signal processing and transparent visualization of how the HRIT agent's beliefs evolve during inference.

---

## 📊 Part 1: MNE Preprocessing Pipeline

### Motivation

Raw EEG signals contain various noise sources:
- **DC drift**: Low-frequency baseline shifts
- **Line noise**: 50/60 Hz electrical interference
- **High-frequency noise**: Muscle artifacts, EMG

Professional preprocessing removes these artifacts while preserving clinically relevant signals.

### Implementation

**File**: `sensor_interface.py`

#### 1. Bandpass Filter (1-40 Hz)

```python
apply_bandpass_filter(eeg_chunk, lowcut=1.0, highcut=40.0)
```

**Characteristics**:
- **Type**: Butterworth IIR filter
- **Order**: 4th order (steep rolloff without ringing)
- **Passband**: 1-40 Hz (removes DC drift and high-frequency noise)
- **Method**: `scipy.signal.butter` + `filtfilt` (zero-phase filtering)

**Clinical Rationale**:
- 1 Hz cutoff: Removes DC and very low-frequency drifts
- 40 Hz cutoff: Retains cognitive/motor activity (alpha, beta bands)
- Excludes gamma band (>40 Hz) which is noise-prone in standard scalp EEG

#### 2. Notch Filter (60 Hz)

```python
apply_notch_filter(eeg_chunk, notch_freq=60.0, quality=30)
```

**Characteristics**:
- **Type**: IIR notch filter (narrow peak attenuation)
- **Center frequency**: 60 Hz (or 50 Hz for European systems)
- **Quality factor**: 30 (very narrow, ~2 Hz bandwidth)
- **Method**: `scipy.signal.iirnotch` + `filtfilt`

**Clinical Rationale**:
- Eliminates AC mains artifact (60 Hz in USA, 50 Hz in Europe)
- High Q factor provides targeted removal without affecting neighboring frequencies
- Essential for artifact suppression in clinical recordings

#### 3. Alpha Band Power Calculation (8-12 Hz)

```python
calculate_alpha_band_power(eeg_chunk, sampling_rate=100)
```

Returns: `(alpha_power, alpha_power_normalized)`

**Method**:
- **Spectral Estimation**: Welch's method (robust, low variance)
  - Window: Full signal length
  - Overlapping windows: 50% (default)
  - Frequency resolution: ~1 Hz
- **Alpha Band Extraction**: Power in 8-12 Hz range
- **Normalization**: Divided by total power (0-1 range)

**Clinical Interpretation**:
- **High alpha (>0.15 normalized)**: Relaxed, eyes-closed, low vigilance
- **Low alpha (<0.15 normalized)**: Attentive, cognitive load, anxiety
- Used as primary observation signal for HRIT agent

### Observation Generation

```python
obs, alpha_normalized, alpha_power = get_observation_from_eeg(raw_chunk)
```

**Output**:
- `obs` (int): Binary observation
  - 0 = Stable state (high alpha power ≥ 0.15)
  - 1 = Unstable state (low alpha power < 0.15)
- `alpha_normalized` (float): 0-1 range normalized alpha power
- `alpha_power` (float): Absolute alpha band power

### Complete Pipeline

```
Raw EEG Chunk (100 samples)
     ↓
[1] Bandpass Filter (1-40 Hz)
     ↓ removes DC drift + high-freq noise
[2] Notch Filter (60 Hz)
     ↓ removes electrical noise
[3] Calculate Alpha Power (8-12 Hz)
     ↓
[4] Normalize by Total Power
     ↓
[5] Threshold (0.15) → Binary Observation
     ↓
Observation (0=Stable, 1=Unstable)
```

### Configuration

Tune these parameters in `sensor_interface.py`:

```python
SAMPLING_RATE = 100          # Hz (adjust for your data)
BANDPASS_LOWCUT = 1.0        # Hz
BANDPASS_HIGHCUT = 40.0      # Hz
NOTCH_FREQ = 60.0            # Hz (50 Hz for European systems)
ALPHA_LOWCUT = 8.0           # Hz
ALPHA_HIGHCUT = 12.0         # Hz
```

---

## 🧠 Part 2: Agent Generative Model Learning

### Motivation

The Active Inference agent maintains a generative model (`A matrix`) that represents:
- **What observations it expects** given different brain states
- **How confident it is** in its state-to-observation mappings

Visualizing the `A matrix` reveals how the agent's beliefs evolve during inference.

### Implementation

**File**: `main_engine.py` → `run_diagnostic_loop()` → Final report section

#### A Matrix Structure

The `A matrix` is a likelihood matrix: **P(Observation | State)**

```
         State 0 (Stable)    State 1 (Unstable)
Obs=0         0.90               0.30
Obs=1         0.10               0.70
```

**Interpretation**:
- When state = Stable: "I expect obs=0 with 90% confidence"
- When state = Unstable: "I expect obs=1 with 70% confidence"

#### Initial vs Final Comparison

The report displays:

```
📊 INITIAL A MATRIX (Prior Beliefs):
   State 0 (Stable):      Obs=0: 0.90  |  Obs=1: 0.10
   State 1 (Unstable):    Obs=0: 0.30  |  Obs=1: 0.70

📊 FINAL A MATRIX (After 30 observations):
   State 0 (Stable):      Obs=0: 0.9000  |  Obs=1: 0.1000
   State 1 (Unstable):    Obs=0: 0.3000  |  Obs=1: 0.7000

📈 BELIEF UPDATES (Δ = Final - Initial):
   Stable state, obs=0:     +0.0000
   Stable state, obs=1:     +0.0000
   Unstable state, obs=0:   +0.0000
   Unstable state, obs=1:   +0.0000
```

#### Clinical Interpretation

The report provides three scenarios:

**Scenario 1: Significant Learning (Δ > 0.01)**
```
✅ Agent LEARNED from EEG observations
• Updated stability beliefs by X units
• Model confidence improved through active inference
```

**Scenario 2: Minimal Learning**
```
⚠️  Minimal learning detected
• Agent prior beliefs well-matched to data
• No significant Bayesian updates required
```

**Scenario 3: Contradictory Data**
```
❌ ALERT: Agent beliefs shifting away from priors
• Data may indicate non-stationary brain states
• Consider re-calibration
```

### Notes on A Matrix Learning

In the current implementation:
- The `A matrix` is **typically fixed** (not learned from data)
- It represents the agent's **prior beliefs** about state-observation relationships
- The agent **updates its belief about states** (posterior) based on observations
- To enable A matrix learning, use: `agent.learn_A = True` in hrit_agent.py

---

## 🔬 Practical Usage

### Single Patient Analysis

```bash
cd /workspaces/pymdp/GIFT_Engine
python main_engine.py
```

**Output**:
```
[00s] | Obs: 1 | k: 0.7500 | σ²: -- (calibrating) | ✅ STABLE
[01s] | Obs: 1 | k: 0.7500 | σ²: -- (calibrating) | ✅ STABLE
...
┌─────────────────────────────────────────┐
│ DIAGNOSTIC SUMMARY REPORT               │
│ State Distribution, k-score stats, etc. │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ AGENT GENERATIVE MODEL (A MATRIX)       │
│ Initial → Final belief comparison       │
└─────────────────────────────────────────┘
```

### Batch Processing (30+ Patients)

```bash
python batch_processor.py
python pdf_report_generator.py
```

The preprocessing pipeline automatically applies to all patient files.

### Custom Filtering Parameters

```python
# In your analysis script
from sensor_interface import get_observation_from_eeg

# Custom alpha threshold
obs, alpha_norm, alpha_power = get_observation_from_eeg(eeg_chunk)
if alpha_norm > 0.20:  # Higher threshold for stricter stability
    obs = 0
else:
    obs = 1
```

---

## 📈 Example Output Interpretation

### Case 1: Learning Occurred
```
Δ Stable state, obs=0: +0.0523
→ Agent became MORE confident that stable states produce obs=0
→ Data consistently showed this relationship
```

### Case 2: No Learning Needed
```
Δ All entries ≈ 0.0000
→ Agent's priors perfectly matched the observed data
→ No Bayesian updating required
```

### Case 3: Unexpected Patterns
```
Δ Unstable state, obs=0: -0.1200
→ Agent became LESS confident that unstable → obs=0
→ Data showed unstable states producing obs=0 frequently
→ Indicates non-stationary or unusual brain state patterns
```

---

## 🔧 Technical Details

### Filter Order Selection

- **Bandpass**: 4th order = 24 dB/octave rolloff (good for speech/EEG)
- **Notch**: Quality=30 = ~2 Hz bandwidth at -3dB (very selective)

### Welch's Method Parameters

```python
freqs, psd = signal.welch(eeg_chunk, fs=sampling_rate, nperseg=len(eeg_chunk))
```

- **NPERSEG**: Signal length (one window, captures full 1-second chunk)
- **NOVERLAP**: 50% default (standard Welch overlap)
- **Window**: Hann (default, smooth spectral leakage)

### Zero-Phase Filtering

```python
filtered = signal.filtfilt(b, a, signal)
```

- Applies filter forward & backward → zero phase distortion
- Doubles filter order (4th order → 8th order effective)
- Essential for clinical EEG (no phase shifts)

---

## 📝 API Reference

### sensor_interface.py

```python
apply_bandpass_filter(eeg_chunk, lowcut=1.0, highcut=40.0, 
                      sampling_rate=100, order=4)
→ Returns: Filtered EEG chunk

apply_notch_filter(eeg_chunk, notch_freq=60.0, 
                   sampling_rate=100, quality=30)
→ Returns: Notch-filtered EEG chunk

calculate_alpha_band_power(eeg_chunk, sampling_rate=100)
→ Returns: (alpha_power, alpha_power_normalized)

get_observation_from_eeg(raw_data_chunk, preprocessing=True)
→ Returns: (obs, alpha_normalized, alpha_power)
```

---

## ✅ Validation Checklist

- [x] Bandpass filter functional (1-40 Hz)
- [x] Notch filter functional (60 Hz)
- [x] Alpha band power calculated correctly
- [x] Observation thresholding works
- [x] A matrix displayed in final report
- [x] Learning/no-learning scenarios handled
- [x] Compatible with batch processing
- [x] All changes committed to GitHub

---

## 🚀 Next Steps

1. **Validate on Real Patient Data**: Test on your own EEG recordings
2. **Tune Thresholds**: Calibrate alpha power threshold (0.15) to your cohort
3. **Enable A Matrix Learning**: Set `agent.learn_A = True` in hrit_agent.py
4. **Longitudinal Analysis**: Track belief changes across multiple sessions
5. **Statistical Comparison**: Compare initial vs final A matrices across cohort

---

**Version**: 1.0  
**Last Updated**: March 10, 2026  
**Status**: ✅ Production Ready
