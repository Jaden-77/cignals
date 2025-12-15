#!/usr/bin/env python3
"""
Flask Web Application for Music Segmentation
Allows users to upload music files and get intro/main/outro timestamps
"""

import os
import time
import json
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import segmentation functions
from music_segmenter import (
    load_audio,
    frame_rms,
    smooth_signal,
    compute_threshold,
    find_segments,
    format_timestamp
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_music_file(input_path, frame_len=2048, hop_len=512, smooth_sec=1.0,
                       min_sec=5.0, thr_mode='percentile', thr_value=60, gap_merge_sec=1.0):
    """
    Process audio file and detect segments.

    Returns:
        dict with segments and plot paths
    """
    try:
        # Load audio
        audio, fs = load_audio(input_path)
        duration = float(len(audio) / fs)

        # Compute RMS
        rms, times = frame_rms(audio, fs, frame_len, hop_len)

        # Smooth
        win_frames = int(smooth_sec * fs / hop_len)
        rms_smooth = smooth_signal(rms, win_frames)

        # Threshold
        threshold = float(compute_threshold(rms_smooth, thr_mode, thr_value))

        # Active mask
        active_mask = rms_smooth >= threshold

        # Find segments
        min_frames = int(min_sec * fs / hop_len)
        gap_merge_frames = int(gap_merge_sec * fs / hop_len)

        segments = find_segments(active_mask, times, min_frames, gap_merge_frames)

        # Generate unique filenames
        timestamp = str(int(time.time()))
        plot_filename = f'segmentation_{timestamp}.png'
        plot_path = os.path.join(app.config['OUTPUT_FOLDER'], plot_filename)

        # Create visualization
        create_plot(times, rms_smooth, threshold, segments, plot_path)

        # Format results
        result = {
            'success': True,
            'intro_end': float(segments['intro_end']),
            'outro_start': float(segments['outro_start']),
            'duration': float(segments['duration']),
            'intro_end_formatted': format_timestamp(segments['intro_end']),
            'outro_start_formatted': format_timestamp(segments['outro_start']),
            'duration_formatted': format_timestamp(segments['duration']),
            'main_duration': float(segments['outro_start'] - segments['intro_end']),
            'plot_file': plot_filename,
            'sample_rate': int(fs),
            'threshold': threshold
        }

        return result

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_plot(times, rms_smooth, threshold, segments, output_path):
    """Create RMS plot with segmentation."""
    intro_end = segments['intro_end']
    outro_start = segments['outro_start']
    duration = segments['duration']

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot RMS
    ax.plot(times, rms_smooth, linewidth=2, color='blue', label='Smoothed RMS')

    # Threshold line
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.4f}')

    # Shade segments
    ax.axvspan(0, intro_end, alpha=0.25, color='green', label='Intro')
    ax.axvspan(intro_end, outro_start, alpha=0.25, color='yellow', label='Main')
    ax.axvspan(outro_start, duration, alpha=0.25, color='purple', label='Outro')

    # Boundary lines
    ax.axvline(intro_end, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(outro_start, color='purple', linestyle='--', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('RMS Energy', fontsize=11)
    ax.set_title('Music Segmentation Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    # Check if file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only WAV and MP3 files are allowed.'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    unique_filename = f"{timestamp}_{filename}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(input_path)

    # Get parameters
    frame_len = int(request.form.get('frame_len', 2048))
    hop_len = int(request.form.get('hop_len', 512))
    smooth_sec = float(request.form.get('smooth_sec', 1.0))
    min_sec = float(request.form.get('min_sec', 5.0))
    thr_mode = request.form.get('thr_mode', 'percentile')
    thr_value = float(request.form.get('thr_value', 60))
    gap_merge_sec = float(request.form.get('gap_merge_sec', 1.0))

    # Validate parameters
    if frame_len < 256 or frame_len > 8192:
        return jsonify({'error': 'Frame length must be between 256-8192'}), 400
    if hop_len < 64 or hop_len > 4096:
        return jsonify({'error': 'Hop length must be between 64-4096'}), 400
    if smooth_sec < 0.1 or smooth_sec > 10:
        return jsonify({'error': 'Smoothing window must be between 0.1-10 seconds'}), 400
    if min_sec < 1 or min_sec > 30:
        return jsonify({'error': 'Min section must be between 1-30 seconds'}), 400

    # Process the audio
    result = process_music_file(
        input_path, frame_len, hop_len, smooth_sec,
        min_sec, thr_mode, thr_value, gap_merge_sec
    )

    # Clean up uploaded file
    try:
        os.remove(input_path)
    except:
        pass

    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download plot file."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/view/<filename>')
def view_plot(filename):
    """View plot file in browser."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old output files (older than 1 hour)."""
    try:
        current_time = time.time()
        output_dir = app.config['OUTPUT_FOLDER']

        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:
                    os.remove(file_path)

        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MUSIC SEGMENTATION - WEB INTERFACE")
    print("="*70)
    print("\nStarting Flask web server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
