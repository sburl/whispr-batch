# Audio Transcription App

A simple GUI application that uses OpenAI's Whisper model to transcribe audio files. Select your files, choose your options, and get transcriptions with timestamps.

## Features

- Simple GUI interface for file selection
- Supports various audio formats (WAV, MP3, MPEG, MP4, M4A)
- Includes timestamps for each segment of speech
- Multiple Whisper model options (tiny to large-v3)
- Real-time transcription display
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
   - Watch the transcriptions appear in real-time

4. When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Models

Available Whisper models (from fastest to most accurate):
- tiny: ~75MB (compressed), ~1GB (in memory), fastest, least accurate
- base: ~142MB (compressed), ~1GB (in memory), good balance of speed and accuracy
- small: ~466MB (compressed), ~2GB (in memory), better accuracy, slower
- medium: ~1.5GB (compressed), ~5GB (in memory), high accuracy, slower
- large-v3: ~3GB (compressed), ~10GB (in memory), best accuracy, slowest (default)

Models are stored locally in the `models` directory next to the script. The download only happens once per model.

## Performance and Timing

Transcription time varies based on several factors:
- Model size: Larger models are more accurate but slower
- Audio length: Longer files take proportionally longer to process
- CPU/GPU: Processing speed depends on your hardware

Approximate transcription speeds (on CPU):
- tiny: ~2-3x real-time (e.g., 1 minute of audio takes 20-30 seconds)
- base: ~1.5-2x real-time
- small: ~1-1.5x real-time
- medium: ~0.7-1x real-time
- large-v3: ~0.5-0.7x real-time (e.g., 1 minute of audio takes 1.5-2 minutes)

For example:
- A 5-minute audio file using the large-v3 model might take 7-10 minutes to transcribe
- The same file using the tiny model might take 2-3 minutes but with lower accuracy

Note: These are approximate times and may vary based on your system's specifications and the complexity of the audio content.

## Notes

- The large-v3 model (~3GB download, ~10GB in memory) is downloaded during installation
- Models are stored in the `models` directory in the same folder as the script
- Transcriptions are saved in the same directory as the source audio files
- The virtual environment (`.venv`) contains all required Python packages # whispr-batch
# whispr-batch
