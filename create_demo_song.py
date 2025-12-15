#!/usr/bin/env python3
"""
Generate a synthetic demo song for testing music segmentation.

Creates a simple song structure with:
- Quiet intro (5 seconds)
- Loud main section (30 seconds)
- Fade-out outro (5 seconds)
"""

import numpy as np
import soundfile as sf
import os

def create_demo_song():
    """
    Create a synthetic song with clear intro/main/outro structure.
    """
    print("\n" + "=" * 70)
    print("DEMO SONG GENERATOR")
    print("=" * 70)

    fs = 44100  # Sample rate

    # === INTRO (5 seconds, quiet) ===
    print("\nGenerating Intro (5 seconds, quiet)...")
    intro_duration = 5
    intro_samples = int(intro_duration * fs)
    t_intro = np.linspace(0, intro_duration, intro_samples)

    # Quiet sine waves (simulating soft intro)
    intro = 0.1 * np.sin(2 * np.pi * 220 * t_intro)  # A3
    intro += 0.05 * np.sin(2 * np.pi * 330 * t_intro)  # E4

    # === MAIN (30 seconds, loud) ===
    print("Generating Main section (30 seconds, loud)...")
    main_duration = 30
    main_samples = int(main_duration * fs)
    t_main = np.linspace(0, main_duration, main_samples)

    # Loud complex signal (simulating energetic main section)
    main = 0.5 * np.sin(2 * np.pi * 220 * t_main)  # A3
    main += 0.3 * np.sin(2 * np.pi * 440 * t_main)  # A4
    main += 0.2 * np.sin(2 * np.pi * 660 * t_main)  # E5

    # Add some rhythm/beats (simulating drums)
    beat_freq = 2  # 2 Hz = 120 BPM
    beats = 0.4 * np.sin(2 * np.pi * beat_freq * t_main)
    beats = np.clip(beats, 0, 1)  # Rectify
    main += 0.3 * beats * np.sin(2 * np.pi * 80 * t_main)  # Bass drum effect

    # === OUTRO (5 seconds, fade out) ===
    print("Generating Outro (5 seconds, fade out)...")
    outro_duration = 5
    outro_samples = int(outro_duration * fs)
    t_outro = np.linspace(0, outro_duration, outro_samples)

    # Same as main but fading out
    outro = 0.5 * np.sin(2 * np.pi * 220 * t_outro)
    outro += 0.3 * np.sin(2 * np.pi * 440 * t_outro)

    # Apply fade-out envelope
    fade_out = np.linspace(1, 0, outro_samples)
    outro *= fade_out

    # === CONCATENATE ===
    print("Concatenating sections...")
    song = np.concatenate([intro, main, outro])

    # Normalize
    song = song / (np.max(np.abs(song)) + 1e-8) * 0.95

    # === SAVE ===
    os.makedirs('samples', exist_ok=True)
    output_path = 'samples/demo_song.wav'

    sf.write(output_path, song, fs, subtype='FLOAT')

    print(f"\n[OK] Saved: {output_path}")
    print(f"  Duration: {len(song) / fs:.1f} seconds")
    print(f"  Sample rate: {fs} Hz")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # === EXPECTED RESULTS ===
    print("\n" + "=" * 70)
    print("EXPECTED SEGMENTATION")
    print("=" * 70)
    print(f"\nIntro:  00:00 - 00:05 (quiet opening)")
    print(f"Main:   00:05 - 00:35 (loud energetic section)")
    print(f"Outro:  00:35 - 00:40 (fade out)")
    print("\n" + "=" * 70)

    print("\nNext step: Run the segmenter!")
    print("  python music_segmenter.py --input samples/demo_song.wav")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    create_demo_song()
