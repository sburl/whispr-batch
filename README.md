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
   - Choose whether to include timestamps
   - Click "Select Audio Files" to choose your audio files
   - Watch the status updates and transcriptions appear in real-time

4. When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Models

Available Whisper models with their characteristics and recommended use cases:

### tiny (~75MB download, ~1GB in memory)
- Fastest model, least accurate
- Best for: Quick transcriptions, short audio, clear speech
- Recommended for: English-only content, simple audio
- Processing speed: ~2-3x real-time

### base (~142MB download, ~1GB in memory)
- Good balance of speed and accuracy
- Best for: General purpose transcription
- Recommended for: Most everyday transcription needs
- Processing speed: ~1.5-2x real-time

### small (~466MB download, ~2GB in memory)
- Better accuracy, supports multiple languages
- Best for: Multiple languages, moderate accuracy needed
- Recommended for: International content, mixed language audio
- Processing speed: ~1-1.5x real-time

### medium (~1.5GB download, ~5GB in memory)
- High accuracy, good for complex audio
- Best for: Complex audio, multiple speakers
- Recommended for: Professional content, interviews, meetings
- Processing speed: ~0.7-1x real-time

### large-v3 (~3GB download, ~10GB in memory)
- Best accuracy, professional quality
- Best for: Professional use, maximum accuracy
- Recommended for: Critical content, complex audio, multiple speakers
- Processing speed: ~0.5-0.7x real-time

Models are stored locally in the `models` directory next to the script. The download only happens once per model.

## Performance and Timing

Transcription time varies based on several factors:
- Model size: Larger models are more accurate but slower
- Audio length: Longer files take proportionally longer to process
- CPU/GPU: Processing speed depends on your hardware

For example:
- A 5-minute audio file using the large-v3 model might take 7-10 minutes to transcribe
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

- The large-v3 model (~3GB download, ~10GB in memory) is downloaded during installation
- Models are stored in the `models` directory in the same folder as the script
- Transcriptions are saved in the same directory as the source audio files
- The virtual environment (`.venv`) contains all required Python packages
- The application shows detailed status updates but cannot show real-time progress during transcription
