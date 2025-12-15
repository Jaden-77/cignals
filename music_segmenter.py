#!/usr/bin/env python3
"""
================================================================================
MUSIC SEGMENTATION - INTRO/MAIN/OUTRO DETECTION
================================================================================

PROJECT: Signals & Systems - Automatic Music Structure Analysis
APPROACH: Short-Time Energy (RMS) + Smoothing + Adaptive Thresholding (No AI)

INSTALLATION:
    pip install -r requirements.txt

USAGE:
    python music_segmenter.py --input path/to/song.wav
    python music_segmenter.py --input song.wav --smooth_sec 1.5 --min_sec 6
    python music_segmenter.py --input song.wav --thr_mode stats --thr_value 0.5

OUTPUTS:
    - output/segments.json         (timestamps in seconds)
    - output/segments.txt          (human-readable mm:ss format)
    - output/rms_plot.png          (RMS energy with segmentation)
    - output/waveform_plot.png     (waveform with shaded regions)
    - output/spectrogram.png       (optional spectrogram)

THEORY:
    Short-Time Energy Analysis:
        - Divide audio into overlapping frames
        - Compute RMS (Root Mean Square) energy per frame
        - Smooth with moving average filter (FIR convolution)
        - Apply adaptive threshold to detect "active" regions

    Segmentation Rules:
        - Intro: From start until sustained energy increase
        - Main: Between intro end and outro start
        - Outro: From last sustained energy decrease to end

PARAMETERS:
    --frame_len: Frame size for RMS computation (default: 2048)
    --hop_len: Hop size between frames (default: 512)
    --smooth_sec: Smoothing window duration (default: 1.0 seconds)
    --min_sec: Minimum section duration (default: 5.0 seconds)
    --thr_mode: Threshold mode (percentile|stats, default: percentile)
    --thr_value: Threshold value (default: 60 for percentile)
    --gap_merge_sec: Merge gaps shorter than this (default: 1.0 seconds)

LIMITATIONS:
    - Songs with very dynamic structure may need parameter tuning
    - Quiet choruses or long breaks can affect detection
    - Classical/ambient music may not have clear sections
    - Parameters optimized for pop/rock structure

================================================================================
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal as scipy_signal

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_audio(file_path):
    """
    Load audio file and convert to mono if stereo.

    Args:
        file_path: Path to audio file (.wav or .mp3)

    Returns:
        audio: numpy array (1D mono), float32 in range [-1, 1]
        fs: sampling rate (int)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Try loading with soundfile (handles WAV natively)
    try:
        audio, fs = sf.read(file_path, dtype='float32')
    except Exception as e:
        # If soundfile fails, try MP3 with pydub
        if file_path.lower().endswith('.mp3'):
            try:
                from pydub import AudioSegment
                print("Converting MP3 to WAV format...")
                audio_segment = AudioSegment.from_mp3(file_path)

                # Export to temporary WAV
                temp_wav = "temp_converted.wav"
                audio_segment.export(temp_wav, format="wav")

                # Load the converted WAV
                audio, fs = sf.read(temp_wav, dtype='float32')

                # Clean up temp file
                os.remove(temp_wav)
                print("MP3 conversion successful!")
            except Exception as mp3_error:
                print(f"Error loading MP3: {mp3_error}")
                print("Make sure ffmpeg is installed for MP3 support.")
                raise
        else:
            raise e

    # Convert stereo to mono by averaging channels
    if audio.ndim == 2:
        print(f"  Converting stereo to mono (averaging channels)")
        audio = np.mean(audio, axis=1)

    # Normalize to [-1, 1]
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, fs


def frame_rms(audio, fs, frame_len=2048, hop_len=512):
    """
    Compute short-time RMS (Root Mean Square) energy.

    RMS Energy Theory:
        For each frame i containing samples x[n]:
        RMS[i] = sqrt(mean(x[n]^2))

    This captures the energy/loudness of the signal over time.

    Args:
        audio: 1D audio array
        fs: sampling rate
        frame_len: frame size in samples
        hop_len: hop size in samples

    Returns:
        rms: RMS energy array (one value per frame)
        times: Time array (in seconds) for each frame
    """
    # Number of frames
    n_frames = 1 + (len(audio) - frame_len) // hop_len

    # Initialize RMS array
    rms = np.zeros(n_frames)

    # Compute RMS for each frame
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]

        # RMS = sqrt(mean(x^2))
        rms[i] = np.sqrt(np.mean(frame ** 2))

    # Compute time for each frame (center of frame)
    times = np.arange(n_frames) * hop_len / fs + (frame_len / 2) / fs

    return rms, times


def smooth_signal(rms, win_frames):
    """
    Smooth RMS signal using moving average (FIR filter via convolution).

    Moving Average Theory:
        y[n] = (1/M) * sum(x[n-k]) for k=0 to M-1

    This is equivalent to convolution with a rectangular window:
        y = x * h, where h = [1/M, 1/M, ..., 1/M]

    The moving average acts as a low-pass FIR filter, removing rapid
    fluctuations and revealing overall energy trends.

    Args:
        rms: RMS energy array
        win_frames: window length in frames

    Returns:
        rms_smooth: smoothed RMS array
    """
    if win_frames < 1:
        win_frames = 1

    # Create rectangular window (normalized to sum=1)
    window = np.ones(win_frames) / win_frames

    # Convolve with 'same' mode to preserve length
    rms_smooth = np.convolve(rms, window, mode='same')

    return rms_smooth


def compute_threshold(rms_smooth, mode='percentile', value=60):
    """
    Compute adaptive threshold for activity detection.

    Two modes:
        1. percentile: threshold = percentile(rms, value)
           Example: value=60 means 60th percentile

        2. stats: threshold = median(rms) + value * std(rms)
           Example: value=0.5 means median + 0.5 standard deviations

    Args:
        rms_smooth: smoothed RMS array
        mode: 'percentile' or 'stats'
        value: threshold parameter

    Returns:
        threshold: scalar threshold value
    """
    if mode == 'percentile':
        threshold = np.percentile(rms_smooth, value)
    elif mode == 'stats':
        median = np.median(rms_smooth)
        std = np.std(rms_smooth)
        threshold = median + value * std
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

    return threshold


def merge_small_gaps(active_mask, gap_merge_frames):
    """
    Merge small gaps in active regions (morphological closing).

    If there's a short inactive period surrounded by active periods,
    fill it in to create a continuous active region.

    Args:
        active_mask: boolean array (True = active, False = inactive)
        gap_merge_frames: maximum gap size to merge (in frames)

    Returns:
        merged_mask: boolean array with small gaps filled
    """
    merged = active_mask.copy()

    # Find transitions
    diff = np.diff(np.concatenate([[False], merged, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Merge gaps between consecutive active regions
    for i in range(len(starts) - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start

        if gap_size <= gap_merge_frames:
            merged[gap_start:gap_end] = True

    return merged


def find_segments(active_mask, times, min_frames, gap_merge_frames):
    """
    Find intro/main/outro segments based on activity mask.

    Segmentation Logic:
        - Intro: From start until first sustained activity
        - Main: Between intro_end and outro_start
        - Outro: From last sustained activity to end

    Args:
        active_mask: boolean array (True = active)
        times: time array in seconds
        min_frames: minimum sustained frames for section boundary
        gap_merge_frames: merge gaps shorter than this

    Returns:
        segments: dict with intro_end, outro_start (in seconds)
    """
    # Merge small gaps to stabilize detection
    active_mask = merge_small_gaps(active_mask, gap_merge_frames)

    duration = times[-1]

    # Find intro_end: first sustained active region
    intro_end = None
    active_count = 0

    for i, is_active in enumerate(active_mask):
        if is_active:
            active_count += 1
            if active_count >= min_frames:
                intro_end = times[i - min_frames + 1]
                break
        else:
            active_count = 0

    # Find outro_start: last sustained active region
    outro_start = None
    active_count = 0

    for i in range(len(active_mask) - 1, -1, -1):
        if active_mask[i]:
            active_count += 1
            if active_count >= min_frames:
                outro_start = times[i + min_frames - 1]
                break
        else:
            active_count = 0

    # Handle edge cases
    if intro_end is None or intro_end < 0:
        intro_end = duration * 0.10  # Default to 10% of duration

    if outro_start is None or outro_start > duration:
        outro_start = duration * 0.90  # Default to 90% of duration

    # Ensure valid ordering
    intro_end = max(0, min(intro_end, duration))
    outro_start = max(intro_end + 1, min(outro_start, duration))

    return {
        'intro_end': intro_end,
        'outro_start': outro_start,
        'duration': duration
    }


def format_timestamp(seconds):
    """
    Convert seconds to mm:ss format.

    Args:
        seconds: time in seconds

    Returns:
        formatted string "mm:ss"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def save_outputs(segments, output_dir='output'):
    """
    Save segmentation results to JSON and TXT files.

    Args:
        segments: dict with intro_end, outro_start, duration
        output_dir: output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON (machine-readable)
    json_path = os.path.join(output_dir, 'segments.json')
    with open(json_path, 'w') as f:
        json.dump(segments, f, indent=2)

    # Save TXT (human-readable)
    txt_path = os.path.join(output_dir, 'segments.txt')
    with open(txt_path, 'w') as f:
        f.write("MUSIC SEGMENTATION RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Duration:  {format_timestamp(segments['duration'])}\n\n")

        f.write(f"Intro:   00:00 - {format_timestamp(segments['intro_end'])}\n")
        f.write(f"Main:    {format_timestamp(segments['intro_end'])} - {format_timestamp(segments['outro_start'])}\n")
        f.write(f"Outro:   {format_timestamp(segments['outro_start'])} - {format_timestamp(segments['duration'])}\n")

    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


def plot_results(audio, fs, rms, rms_smooth, times, threshold, segments, active_mask):
    """
    Generate visualization plots.

    Creates:
        1. RMS plot with threshold and shaded segments
        2. Waveform plot with shaded segments
        3. Spectrogram (optional)

    Args:
        audio: original audio signal
        fs: sampling rate
        rms: raw RMS energy
        rms_smooth: smoothed RMS energy
        times: time array for RMS
        threshold: threshold value used
        segments: segmentation results
        active_mask: boolean activity mask
    """
    intro_end = segments['intro_end']
    outro_start = segments['outro_start']
    duration = segments['duration']

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # ===== Plot 1: RMS Energy with Segmentation =====
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot RMS curves
    ax.plot(times, rms, alpha=0.3, label='Raw RMS', linewidth=0.5, color='gray')
    ax.plot(times, rms_smooth, label='Smoothed RMS', linewidth=2, color='blue')

    # Plot threshold line
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')

    # Shade segments
    ax.axvspan(0, intro_end, alpha=0.2, color='green', label='Intro')
    ax.axvspan(intro_end, outro_start, alpha=0.2, color='yellow', label='Main')
    ax.axvspan(outro_start, duration, alpha=0.2, color='purple', label='Outro')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('RMS Energy', fontsize=12)
    ax.set_title('Short-Time RMS Energy with Music Segmentation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/rms_plot.png', dpi=150)
    plt.close()

    # ===== Plot 2: Waveform with Segmentation =====
    fig, ax = plt.subplots(figsize=(14, 6))

    # Time axis for waveform
    waveform_time = np.arange(len(audio)) / fs

    # Plot waveform
    ax.plot(waveform_time, audio, linewidth=0.5, color='black', alpha=0.7)

    # Shade segments
    ax.axvspan(0, intro_end, alpha=0.2, color='green', label='Intro')
    ax.axvspan(intro_end, outro_start, alpha=0.2, color='yellow', label='Main')
    ax.axvspan(outro_start, duration, alpha=0.2, color='purple', label='Outro')

    # Add vertical lines at boundaries
    ax.axvline(intro_end, color='green', linestyle='--', linewidth=2, label=f'Intro End ({format_timestamp(intro_end)})')
    ax.axvline(outro_start, color='purple', linestyle='--', linewidth=2, label=f'Outro Start ({format_timestamp(outro_start)})')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Audio Waveform with Segmentation Boundaries', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/waveform_plot.png', dpi=150)
    plt.close()

    # ===== Plot 3: Spectrogram =====
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute and plot spectrogram
    f, t, Sxx = scipy_signal.spectrogram(audio, fs, nperseg=2048)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')

    # Shade segments (with transparency)
    ax.axvspan(0, intro_end, alpha=0.15, color='green')
    ax.axvspan(intro_end, outro_start, alpha=0.15, color='yellow')
    ax.axvspan(outro_start, duration, alpha=0.15, color='purple')

    # Add boundary lines
    ax.axvline(intro_end, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(outro_start, color='white', linestyle='--', linewidth=2, alpha=0.8)

    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Spectrogram with Segmentation Boundaries', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 8000])  # Focus on lower frequencies

    plt.tight_layout()
    plt.savefig('output/spectrogram.png', dpi=150)
    plt.close()

    print(f"  Saved: output/rms_plot.png")
    print(f"  Saved: output/waveform_plot.png")
    print(f"  Saved: output/spectrogram.png")


def print_report(segments, params):
    """
    Print segmentation report to terminal.

    Args:
        segments: segmentation results
        params: processing parameters
    """
    print("\n" + "=" * 70)
    print("SEGMENTATION REPORT")
    print("=" * 70)

    print(f"\nAudio Properties:")
    print(f"  Sample Rate: {params['fs']} Hz")
    print(f"  Duration: {format_timestamp(segments['duration'])} ({segments['duration']:.2f} seconds)")

    print(f"\nProcessing Parameters:")
    print(f"  Frame Length: {params['frame_len']} samples")
    print(f"  Hop Length: {params['hop_len']} samples")
    print(f"  Smoothing Window: {params['smooth_sec']:.2f} seconds")
    print(f"  Min Section Duration: {params['min_sec']:.2f} seconds")
    print(f"  Threshold Mode: {params['thr_mode']}")
    print(f"  Threshold Value: {params['thr_value']}")
    print(f"  Gap Merge: {params['gap_merge_sec']:.2f} seconds")

    print(f"\nDetected Segments:")
    print(f"  Intro:  00:00 -> {format_timestamp(segments['intro_end'])} ({segments['intro_end']:.2f}s)")
    print(f"  Main:   {format_timestamp(segments['intro_end'])} -> {format_timestamp(segments['outro_start'])} ({segments['outro_start'] - segments['intro_end']:.2f}s)")
    print(f"  Outro:  {format_timestamp(segments['outro_start'])} -> {format_timestamp(segments['duration'])} ({segments['duration'] - segments['outro_start']:.2f}s)")

    print("\n" + "=" * 70)


def main():
    """
    Main processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Automatic Music Segmentation (Intro/Main/Outro)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file (.wav or .mp3)')
    parser.add_argument('--frame_len', type=int, default=2048,
                        help='Frame length for RMS (default: 2048)')
    parser.add_argument('--hop_len', type=int, default=512,
                        help='Hop length for RMS (default: 512)')
    parser.add_argument('--smooth_sec', type=float, default=1.0,
                        help='Smoothing window duration in seconds (default: 1.0)')
    parser.add_argument('--min_sec', type=float, default=5.0,
                        help='Minimum section duration in seconds (default: 5.0)')
    parser.add_argument('--thr_mode', type=str, default='percentile',
                        choices=['percentile', 'stats'],
                        help='Threshold mode (default: percentile)')
    parser.add_argument('--thr_value', type=float, default=60,
                        help='Threshold value (default: 60 for percentile)')
    parser.add_argument('--gap_merge_sec', type=float, default=1.0,
                        help='Merge gaps shorter than this (default: 1.0)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MUSIC SEGMENTATION - INTRO/MAIN/OUTRO DETECTION")
    print("=" * 70)

    # Step 1: Load audio
    print(f"\n[1/6] Loading audio: {args.input}")
    try:
        audio, fs = load_audio(args.input)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    duration = len(audio) / fs
    print(f"  Loaded: {duration:.2f}s, {fs} Hz, Mono")

    # Step 2: Compute short-time RMS
    print(f"\n[2/6] Computing short-time RMS energy")
    print(f"  Frame length: {args.frame_len} samples")
    print(f"  Hop length: {args.hop_len} samples")

    rms, times = frame_rms(audio, fs, args.frame_len, args.hop_len)
    print(f"  Computed {len(rms)} frames")

    # Step 3: Smooth RMS
    print(f"\n[3/6] Smoothing RMS with moving average")
    win_frames = int(args.smooth_sec * fs / args.hop_len)
    print(f"  Smoothing window: {args.smooth_sec}s ({win_frames} frames)")

    rms_smooth = smooth_signal(rms, win_frames)

    # Step 4: Compute threshold
    print(f"\n[4/6] Computing adaptive threshold")
    threshold = compute_threshold(rms_smooth, args.thr_mode, args.thr_value)
    print(f"  Mode: {args.thr_mode}")
    print(f"  Value: {args.thr_value}")
    print(f"  Threshold: {threshold:.6f}")

    # Create activity mask
    active_mask = rms_smooth >= threshold

    # Step 5: Find segments
    print(f"\n[5/6] Detecting segments")
    min_frames = int(args.min_sec * fs / args.hop_len)
    gap_merge_frames = int(args.gap_merge_sec * fs / args.hop_len)

    print(f"  Min section: {args.min_sec}s ({min_frames} frames)")
    print(f"  Gap merge: {args.gap_merge_sec}s ({gap_merge_frames} frames)")

    segments = find_segments(active_mask, times, min_frames, gap_merge_frames)

    # Step 6: Save and visualize
    print(f"\n[6/6] Saving outputs and generating visualizations")

    save_outputs(segments)
    plot_results(audio, fs, rms, rms_smooth, times, threshold, segments, active_mask)

    # Print report
    params = {
        'fs': fs,
        'frame_len': args.frame_len,
        'hop_len': args.hop_len,
        'smooth_sec': args.smooth_sec,
        'min_sec': args.min_sec,
        'thr_mode': args.thr_mode,
        'thr_value': args.thr_value,
        'gap_merge_sec': args.gap_merge_sec
    }
    print_report(segments, params)

    print("\nProcessing complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


"""
================================================================================
PROJECT WRITE-UP
================================================================================

REAL-WORLD PROBLEM:
    Automatic music segmentation is useful for:
    - Music streaming services (preview generation)
    - DJ software (finding intro/outro for mixing)
    - Audio editing (quick navigation to sections)
    - Music analysis and research

SIGNAL PROCESSING THEORY USED:

    1. Short-Time Energy (RMS):
        - Divide signal into overlapping frames (windowing)
        - Compute RMS = sqrt(mean(x^2)) for each frame
        - Provides time-varying energy profile

    2. Moving Average Filter (FIR Convolution):
        - Smooth RMS with rectangular window
        - y[n] = (1/M) sum(x[n-k]) for k=0..M-1
        - Acts as low-pass filter, removes noise/fluctuations
        - Implemented via convolution: y = x * h

    3. Adaptive Thresholding:
        - Percentile-based: robust to outliers
        - Statistics-based: median + k*std
        - Separates "active" from "quiet" regions

    4. Time-Domain Analysis:
        - No frequency decomposition needed
        - Fast and efficient
        - Works on energy envelope

METHODOLOGY:
    1. Load and normalize audio
    2. Compute short-time RMS energy (frame-based)
    3. Smooth with moving average (convolution)
    4. Compute adaptive threshold
    5. Detect sustained active regions
    6. Apply segmentation rules:
       - Intro ends at first sustained activity
       - Outro starts at last sustained activity
       - Main is between intro and outro

LIMITATIONS:
    - Songs with very dynamic structure (quiet chorus, loud verse) may confuse
      the detector
    - Classical/ambient music without clear energy changes won't segment well
    - Parameters (threshold, min_sec) may need tuning per genre
    - Assumes typical pop/rock structure (intro -> main -> outro)
    - No semantic understanding (can't detect verse vs chorus)

ADVANTAGES OVER AI:
    - No training data needed
    - Works immediately on any audio
    - Fast processing (seconds)
    - Fully explainable (every step is clear)
    - Parameters are interpretable and tunable

FUTURE IMPROVEMENTS:
    - Add beat tracking for more precise boundaries
    - Use spectral features (not just energy)
    - Detect verse/chorus using repetition analysis
    - Machine learning for parameter auto-tuning

================================================================================
"""
