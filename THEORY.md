# DSP Theory & Parameter Guide

Complete explanation of the signal processing algorithms and parameters used for music segmentation.

## 📐 How It Works

### Overview

The tool uses **short-time energy analysis** to detect changes in music loudness over time, then applies **adaptive thresholding** to identify intro/main/outro boundaries.

### Processing Pipeline

```
Audio Input
    ↓
[1] Framing (windowing)
    ↓
[2] RMS Energy Computation
    ↓
[3] Moving Average Smoothing
    ↓
[4] Adaptive Thresholding
    ↓
[5] Gap Merging
    ↓
[6] Boundary Detection
    ↓
Intro/Main/Outro Timestamps
```

---

## 1️⃣ Framing (Windowing)

**What:** Divide audio into overlapping frames

**Parameters:**
- **Frame Length:** 2048 samples (~46ms at 44.1kHz)
- **Hop Length:** 512 samples (~12ms at 44.1kHz)
- **Overlap:** 75% (typical for audio analysis)

**Why:** Can't analyze entire song at once - need time-varying energy profile

**Math:**
```
Number of frames = 1 + (N - frame_len) / hop_len
```

---

## 2️⃣ RMS Energy Computation

**What:** Calculate Root Mean Square energy for each frame

**Formula:**
```
RMS[i] = sqrt(mean(x[n]²))
      = sqrt((1/N) × Σ(x[n]²))
```

**Why:** RMS represents the "loudness" or "power" of the signal in that frame

**Result:** Array of energy values over time

---

## 3️⃣ Moving Average Smoothing

**Parameter:** `--smooth_sec` (default: 1.0 seconds)

**What:** Apply moving average filter to smooth RMS curve

**Formula:**
```
y[n] = (1/M) × Σ(x[n-k]) for k=0 to M-1
```

This is **FIR filtering via convolution**:
```
y = x * h
where h = [1/M, 1/M, ..., 1/M]  (M times)
```

**Why:**
- Removes rapid fluctuations/noise
- Reveals overall energy trends
- Makes boundary detection more stable

**Window Length:**
```
M = smooth_sec × (fs / hop_len)
```

**Effect of Parameter:**
- **Larger** (2.0s): More stable, less noise, but less precise
- **Smaller** (0.5s): More precise, but noisier, less stable
- **Default** (1.0s): Good balance for most music

**Example:**
```
Raw RMS:     ▲ ▲ ▲▲  ▲ ▲▲▲  ▲  (noisy)
Smoothed:    ─────────────────  (stable trend)
```

---

## 4️⃣ Adaptive Thresholding

**Parameters:**
- `--thr_mode` (default: percentile)
- `--thr_value` (default: 60)

**What:** Compute a threshold to separate "active" from "quiet" frames

### Mode 1: Percentile (Recommended)

**Formula:**
```
threshold = percentile(rms_smooth, thr_value)
```

**Example:** `thr_value=60` means:
- 60% of frames are below threshold ("quiet")
- 40% of frames are above threshold ("active")

**Advantages:**
- Robust to outliers
- Works well for most music
- Interpretable (60 = 60th percentile)

### Mode 2: Statistics

**Formula:**
```
threshold = median(rms_smooth) + (thr_value × std(rms_smooth))
```

**Example:** `thr_value=0.5` means:
- Threshold = median + 0.5 standard deviations

**Advantages:**
- Based on distribution statistics
- Can be more sensitive to dynamics

**Which to use?**
- **Percentile:** For most songs (default)
- **Statistics:** If percentile doesn't work well

---

## 5️⃣ Gap Merging

**Parameter:** `--gap_merge_sec` (default: 1.0 seconds)

**What:** Fill small inactive gaps surrounded by active regions (morphological closing)

**Example:**
```
Before: |ACTIVE|gap|ACTIVE|gap|ACTIVE|  (choppy)
After:  |======== ACTIVE ============|  (smooth)
```

**Why:**
- Songs may have brief quiet moments (not real boundaries)
- Creates more stable, continuous regions
- Prevents false boundary detection

**Effect of Parameter:**
- **Larger** (2.0s): Merge longer gaps, very stable
- **Smaller** (0.5s): Strict separation, may be choppy
- **Default** (1.0s): Good balance

---

## 6️⃣ Boundary Detection

**Parameter:** `--min_sec` (default: 5.0 seconds)

**What:** Find first and last sustained active regions

### Detection Rules:

**Intro End:**
- First time signal becomes "active" (above threshold)
- Must stay active for at least `min_sec` seconds continuously

**Outro Start:**
- Last time signal stops being "active" (goes below threshold)
- Must have been active for at least `min_sec` seconds before

**Main Section:**
- Everything between intro_end and outro_start

### Edge Cases:

If intro_end not detected:
```
intro_end = 10% of total duration
```

If outro_start not detected:
```
outro_start = 90% of total duration
```

Ensure valid ordering:
```
0 ≤ intro_end < outro_start ≤ duration
```

---

## ⚙️ Parameter Reference

### Smoothing Window (`--smooth_sec`)

**Range:** 0.1 - 10.0 seconds
**Default:** 1.0s

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.5s | Precise, but noisy | Very short sections |
| 1.0s | Balanced | Most pop/rock |
| 2.0s | Stable, smooth | Gradual intro/outro |
| 3.0s | Very stable | Classical music |

### Min Section Duration (`--min_sec`)

**Range:** 1 - 30 seconds
**Default:** 5.0s

| Value | Effect | Use Case |
|-------|--------|----------|
| 3s | Allows short sections | Electronic/dance |
| 5s | Typical sections | Most music |
| 8s | Long sections only | Classical, long intros |
| 10s | Very conservative | Avoid false positives |

### Threshold Mode (`--thr_mode`)

| Mode | Formula | When to Use |
|------|---------|-------------|
| **percentile** | `percentile(rms, value)` | Default, works for most |
| **stats** | `median + value×std` | If percentile fails |

### Threshold Value (`--thr_value`)

**For Percentile Mode:**

| Value | Sensitivity | Effect |
|-------|-------------|--------|
| 50 | High | More sections detected |
| 60 | Normal | Default |
| 70 | Low | Only loud sections |

**For Stats Mode:**

| Value | Effect |
|-------|--------|
| 0.3 | More sensitive |
| 0.5 | Balanced |
| 0.8 | Less sensitive |

### Gap Merge (`--gap_merge_sec`)

**Range:** 0 - 5.0 seconds
**Default:** 1.0s

| Value | Effect |
|-------|--------|
| 0.5s | Strict separation |
| 1.0s | Balanced |
| 2.0s | Very smooth |

---

## 🎯 Tuning Guide

### Problem: Intro Detected Too Late

**Symptoms:**
- Intro end timestamp is after the song gets loud
- Missing the quiet opening

**Solutions:**
```bash
# Lower threshold (more sensitive)
--thr_value 50

# Shorter minimum section
--min_sec 3
```

### Problem: Outro Detected Too Early

**Symptoms:**
- Outro start is in the middle of the song
- Quiet chorus mistaken for outro

**Solutions:**
```bash
# Require longer sections
--min_sec 8

# Higher threshold (less sensitive)
--thr_value 70
```

### Problem: Boundaries Are Jumpy/Unstable

**Symptoms:**
- Boundaries change when re-running
- Multiple rapid transitions

**Solutions:**
```bash
# More smoothing
--smooth_sec 2.0

# Merge longer gaps
--gap_merge_sec 2.0

# Require longer sections
--min_sec 7
```

### Problem: No Clear Boundaries Detected

**Symptoms:**
- Intro_end and outro_start at default positions (10%/90%)
- Energy curve is flat

**Diagnosis:**
- Song may have constant energy throughout
- No clear intro/outro structure

**Solutions:**
```bash
# Try statistics mode
--thr_mode stats --thr_value 0.3

# More aggressive detection
--min_sec 2 --thr_value 45
```

---

## 🎵 Genre-Specific Settings

### Pop/Rock (Default)
```bash
--smooth_sec 1.0 --min_sec 5.0 --thr_mode percentile --thr_value 60
```

### Classical Music
```bash
--smooth_sec 2.5 --min_sec 10.0 --thr_mode stats --thr_value 0.4
```

### Electronic/Dance
```bash
--smooth_sec 1.5 --min_sec 4.0 --thr_mode percentile --thr_value 65
```

### Acoustic/Folk
```bash
--smooth_sec 1.2 --min_sec 5.0 --thr_mode percentile --thr_value 55
```

### Metal/Hard Rock
```bash
--smooth_sec 0.8 --min_sec 4.0 --thr_mode percentile --thr_value 70
```

---

## 📊 Understanding the Plots

### RMS Plot

**Elements:**
- **Gray line:** Raw RMS (noisy)
- **Blue line:** Smoothed RMS (what algorithm uses)
- **Red dashed:** Threshold
- **Shaded regions:** Detected segments
  - Green = Intro
  - Yellow = Main
  - Purple = Outro

**What to Look For:**
- Does smoothed line follow the energy trend?
- Is threshold at a good level?
- Do boundaries align with energy changes?

### Waveform Plot

**Elements:**
- **Black waveform:** Original audio
- **Vertical dashed lines:** Boundaries
- **Shaded regions:** Segments

**What to Look For:**
- Do boundaries correspond to visual changes?
- Is intro quieter than main?
- Does outro fade out?

### Spectrogram

**Elements:**
- **Color intensity:** Frequency content
- **White dashed lines:** Boundaries

**What to Look For:**
- Frequency changes at boundaries
- Intro may have less high-frequency content
- Outro may show fade-out across frequencies

---

## 🧪 Example Analysis

### Typical Pop Song

**Audio Characteristics:**
- Quiet intro: RMS ~ 0.2
- Loud main: RMS ~ 0.7
- Fade-out outro: RMS 0.7 → 0.1

**Parameters:**
```bash
--smooth_sec 1.0 --min_sec 5.0 --thr_value 60
```

**Threshold Calculation:**
```
60th percentile of [0.2, 0.2, ..., 0.7, 0.7, ..., 0.5, 0.3, 0.1]
= approximately 0.5
```

**Detection:**
- Intro end: When RMS first exceeds 0.5 for 5s
- Outro start: When RMS last drops below 0.5 for 5s

**Result:**
```
Intro:  00:00 - 00:08 (quiet opening)
Main:   00:08 - 03:12 (verses, choruses)
Outro:  03:12 - 03:30 (fade out)
```

---

## 🎓 Signal Processing Concepts

### Windowing
- **Purpose:** Analyze signal in short segments
- **Trade-off:** Time resolution vs frequency resolution
- **Application:** Short-time analysis of non-stationary signals

### FIR Filtering
- **Type:** Moving average is a simple FIR filter
- **Properties:** Linear phase, stable, no feedback
- **Application:** Smoothing time-series data

### Convolution
- **Operation:** y = x * h
- **Interpretation:** Weighted average with kernel h
- **Application:** Filtering, smoothing

### Adaptive Thresholding
- **Purpose:** Robust detection across different songs
- **Advantage:** No fixed threshold needed
- **Application:** Activity detection, onset detection

### Morphological Processing
- **Operation:** Closing (dilation followed by erosion)
- **Purpose:** Fill small gaps, smooth boundaries
- **Application:** Binary signal processing

---

## 📚 References

- **Short-Time Energy:** Rabiner & Schafer, "Theory and Applications of Digital Speech Processing"
- **Moving Average Filters:** Smith, "The Scientist and Engineer's Guide to Digital Signal Processing"
- **Music Structure Analysis:** Paulus et al., "Music Structure Analysis"
- **Adaptive Thresholding:** Otsu's method, Sauvola thresholding

---

## 🔬 Limitations & Future Work

### Current Limitations:

1. **Energy-based only** - Doesn't consider harmony, melody, rhythm
2. **Assumes standard structure** - Intro-main-outro
3. **Single threshold** - Can't detect multiple sections (verse, chorus)
4. **Manual tuning** - Parameters need adjustment per song

### Possible Enhancements:

1. **Spectral features** - Use MFCCs or chroma features
2. **Beat tracking** - Align boundaries to beats
3. **Novelty detection** - Detect changes in timbre/harmony
4. **Multi-level segmentation** - Detect verse, chorus, bridge
5. **Auto-parameter tuning** - ML to predict best parameters

---

**Made with classical DSP** 🎵 **Fully explainable signal processing!**
