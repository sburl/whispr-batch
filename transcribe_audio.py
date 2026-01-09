#!/usr/bin/env python3

import argparse
from pathlib import Path

from audio_transcribe_core import (
    load_model,
    transcribe_file,
)

def transcribe_audio(file_path, model_name="large-v3", include_timestamps=True, model=None):
    """Transcribe audio file using faster-whisper"""
    # Allow caller to supply a pre-loaded model so we don't reload per file
    model = model or load_model(model_name, device="auto")
    
    print(f"Transcribing: {file_path}")
    result = transcribe_file(
        str(file_path),
        model_name=model_name,
        include_timestamps=include_timestamps,
        device="auto",
        model=model,
        task="transcribe"
    )
    return result.text

def process_directory(directory_path, model_name="large-v3", include_timestamps=True):
    """Process all audio files in the given directory"""
    audio_extensions = {'.wav', '.mp3', '.mpeg', '.mp4', '.m4a'}
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Load the model once for the entire run to avoid repeated downloads and RAM spikes
    print(f"Loading faster-whisper model: {model_name}")
    model = load_model(model_name, device="auto")
    
    # Create output directory
    output_dir = directory / "transcriptions"
    output_dir.mkdir(exist_ok=True)
    
    # Process each audio file
    for file_path in directory.glob("*"):
        if file_path.suffix.lower() in audio_extensions:
            print(f"\nProcessing: {file_path.name}")
            try:
                transcription = transcribe_audio(file_path, model_name, include_timestamps, model=model)
                
                # Save transcription to file
                output_file = output_dir / f"{file_path.stem}_transcription.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcription)
                print(f"Transcription saved to: {output_file}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using faster-whisper")
    parser.add_argument("directory", help="Directory containing audio files to transcribe")
    parser.add_argument("--model", default="large-v3", 
                      choices=["tiny", "base", "small", "medium", "large-v3"],
                      help="faster-whisper model to use (default: large-v3)")
    parser.add_argument("--no-timestamps", action="store_true",
                      help="Disable timestamps in output")
    
    args = parser.parse_args()
    
    try:
        process_directory(args.directory, args.model, not args.no_timestamps)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
