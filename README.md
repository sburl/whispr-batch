# Audio Transcription Tool

A Python-based GUI application for transcribing audio files using OpenAI's Whisper model. This tool provides a user-friendly interface for batch processing audio files with flexible settings and real-time progress tracking.

## Features

- **Multiple Model Support**: Choose from different Whisper models (tiny, base, small, medium, large-v3) based on your needs
- **Batch Processing**: Process multiple audio files in a queue
- **Flexible File Management**:
  - Add files at any time (even during processing when paused)
  - Drag and drop to reorder files in the queue
  - Remove files from the queue
  - Individual model selection for each file
  - Individual timestamp settings for each file
- **Processing Controls**:
  - Start/Pause/Resume/Stop functionality
  - Real-time progress tracking
  - Estimated time remaining
  - Pause to add more files
- **Output Options**:
  - Optional timestamps for each segment
  - Saves transcriptions in the same directory as source files
  - Clear progress and status updates

## Requirements

- Python 3.8 or higher
- FFmpeg installed on your system
- Required Python packages (install via `pip install -r requirements.txt`):
  - openai-whisper
  - tkinter
  - librosa
  - numpy
  - torch

## Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure FFmpeg is installed on your system:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Running the Application

1. Make sure you're in the project directory and your virtual environment is activated:
   ```bash
   cd [repository-name]
   source venv/bin/activate  # On macOS/Linux
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. Run the application:
   ```bash
   python transcribe_gui.py
   ```

3. The GUI will open and you can start using the application.

## Usage

1. **Add Audio Files**:
   - Click "Add Audio Files" to select one or more audio files
   - Files will be added to the queue with default settings
   - You can add more files at any time when processing is paused

2. **Configure Settings**:
   - Select default model (tiny, base, small, medium, large-v3)
   - Toggle default timestamps setting
   - Individual files can have their own model and timestamp settings

3. **File Queue Management**:
   - Drag and drop files to reorder them
   - Select files and use "Remove" to delete them from the queue
   - Use "Toggle Timestamps" to change timestamp settings for selected files
   - Use "Change Model" to set different models for selected files
   - All changes can be made while processing is paused

4. **Processing Controls**:
   - Click "Start" to begin processing the queue
   - Use "Pause" to temporarily stop processing
   - While paused, you can:
     - Add more files
     - Reorder the queue
     - Change file settings
   - Click "Resume" to continue processing
   - Use "Stop" to end processing completely

5. **Monitor Progress**:
   - View real-time progress in the progress bar
   - See estimated time remaining
   - Check status updates for each file
   - View completion messages and output file locations

## Model Information

- **tiny**: ~75MB download, ~1GB in memory
  - Best for: Quick transcriptions, short audio, clear speech, English only
- **base**: ~142MB download, ~1GB in memory
  - Best for: General purpose, good balance of speed and accuracy
- **small**: ~466MB download, ~2GB in memory
  - Best for: Multiple languages, moderate accuracy needed
- **medium**: ~1.5GB download, ~5GB in memory
  - Best for: Complex audio, multiple speakers, high accuracy needed
- **large-v3**: ~3GB download, ~10GB in memory
  - Best for: Professional use, maximum accuracy, complex audio

## Notes

- Models are downloaded and stored locally in `~/.cache/whisper/`
- Processing speed varies by model and hardware
- Each file can have its own model and timestamp settings
- Files can be added to the queue at any time when processing is paused
- The application will automatically switch models when processing files with different model settings
- Estimated processing times are approximate and may vary based on your system

## Troubleshooting

- If you encounter any issues with FFmpeg, ensure it's properly installed and accessible in your system's PATH
- For memory issues, try using a smaller model or processing fewer files at once
- If the application becomes unresponsive, use the Stop button and restart processing
- Check the status messages for any error information
