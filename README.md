# Audio Transcription App

A simple GUI application that uses OpenAI's Whisper model to transcribe audio files. Select your files, choose your options, and get transcriptions with timestamps.

## Features

- Simple GUI interface for file selection
- Supports various audio formats (WAV, MP3, MPEG, MP4, M4A)
- Includes timestamps for each segment of speech
- Multiple Whisper model options (tiny to large-v3)
- Real-time status updates and progress information
- Detailed transcription statistics
- Saves transcriptions to text files
- Pause/Resume and Stop functionality
- Flexible file queue management
- Per-file timestamp control
- File reordering and removal

## Installation

1. Install Python 3.10 and tkinter:
```bash
brew install python@3.10 python-tk@3.10
```

2. Set up the virtual environment:
```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip (using python -m pip to ensure we use the right pip)
python -m pip install --upgrade pip

# Install requirements
python -m pip install -r requirements.txt
```

3. Download the Whisper model (large-v3, ~10GB):
```bash
# Create models directory
mkdir -p models

# Download the model with progress bar (this may take a while depending on your internet speed)
python -c "from tqdm import tqdm; import whisper; print('Downloading large-v3 model (this may take a while)...'); model = whisper.load_model('large-v3', download_root='models', in_memory=False)"

# Verify the model was downloaded (should show the model path)
python -c "import whisper; print('Model location:', whisper._download(whisper._MODELS['large-v3'], 'models', in_memory=False))"
```

## Usage

1. Activate the virtual environment (if not already activated):
```bash
source .venv/bin/activate
```

2. Run the application:
```bash
python transcribe_gui.py
```

3. In the application:
   - Select your preferred Whisper model from the dropdown
   - Set default timestamp preference (can be changed per file)
   - Click "Add Audio Files" to choose your audio files
   - Manage your file queue:
     - Use ↑/↓ buttons to reorder files
     - Use "Remove" to delete files from the queue
     - Use "Toggle Timestamps" to change timestamp settings for selected files
   - Click "Start" to begin processing
   - Use "Pause/Resume" to temporarily stop/continue processing
   - Use "Stop" to cancel the current batch
   - Watch the status updates and transcriptions appear in real-time

4. When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## File Queue Management

The application provides several ways to manage your transcription queue:

- **Adding Files**: Click "Add Audio Files" to select multiple files. They'll be added to the queue without starting processing.
- **Reordering**: Select a file and use the ↑/↓ buttons to move it up or down in the queue.
- **Removing Files**: Select one or more files and click "Remove" to delete them from the queue.
- **Timestamp Control**: 
  - Set default timestamp preference in the options
  - Select files and use "Toggle Timestamps" to change their individual settings
  - Each file shows its timestamp status in the queue

## Processing Controls

The application provides full control over the transcription process:

- **Start**: Begin processing the current queue of files
- **Pause/Resume**: 
  - Pause the current transcription
  - Resume from where it left off
  - Processing can be paused at any time
- **Stop**: 
  - Cancel the current batch
  - Completed files remain processed
  - Unprocessed files remain in the queue

## Models

Available Whisper models with their characteristics and recommended use cases:

### tiny (~75MB download, ~1GB in memory)
- Fastest model, least accurate
- Best for: Quick transcriptions, short audio, clear speech
- Recommended for: English-only content, simple audio
- Processing speed: ~2.5x real-time

### base (~142MB download, ~1GB in memory)
- Good balance of speed and accuracy
- Best for: General purpose transcription
- Recommended for: Most everyday transcription needs
- Processing speed: ~2x real-time

### small (~466MB download, ~2GB in memory)
- Better accuracy, supports multiple languages
- Best for: Multiple languages, moderate accuracy needed
- Recommended for: International content, mixed language audio
- Processing speed: ~1.5x real-time

### medium (~1.5GB download, ~5GB in memory)
- High accuracy, good for complex audio
- Best for: Complex audio, multiple speakers
- Recommended for: Professional content, interviews, meetings
- Processing speed: ~1x real-time

### large-v3 (~3GB download, ~10GB in memory)
- Best accuracy, professional quality
- Best for: Professional use, maximum accuracy
- Recommended for: Critical content, complex audio, multiple speakers
- Processing speed: ~0.6x real-time

Models are stored in your home directory under `.cache/whisper/`. The download only happens once per model.

## Performance and Timing

Transcription time varies based on several factors:
- Model size: Larger models are more accurate but slower
- Audio length: Longer files take proportionally longer to process
- CPU/GPU: Processing speed depends on your hardware

For example:
- A 5-minute audio file using the large-v3 model might take 8-10 minutes to transcribe
- The same file using the tiny model might take 2-3 minutes but with lower accuracy

## Status Updates

The application provides detailed status information during transcription:
- Audio file length and duration
- Selected model and estimated processing time
- Number of speech segments detected
- Progress through multiple files (if selected)
- Clear indication when transcription is complete
- Location of saved transcription files

## Output Format

Transcriptions are saved as text files with the following information:
- File name and total duration
- Number of speech segments
- Timestamps for each segment (if enabled)
- Full transcription text
- Saved in the same directory as the source audio file

## Notes

- Models are stored in your home directory under `.cache/whisper/`
- Transcriptions are saved in the same directory as the source audio files
- The virtual environment (`.venv`) contains all required Python packages
- The application shows detailed status updates but cannot show real-time progress during transcription
- You can pause/resume processing at any time
- Files can be reordered or removed from the queue before processing
- Each file can have its own timestamp setting
