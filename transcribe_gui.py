#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
from pathlib import Path
import whisper
from datetime import timedelta
import queue
import os
import librosa
from typing import List, Dict

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription")
        self.root.geometry("1000x800")
        
        # Create message queue for thread-safe updates
        self.queue = queue.Queue()
        
        # Control flags
        self.is_processing = False
        self.is_paused = False
        self.should_stop = False
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left and right frames
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Options frame
        self.options_frame = ttk.LabelFrame(self.left_frame, text="Options", padding="5")
        self.options_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Model selection
        ttk.Label(self.options_frame, text="Model:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="base")
        self.model_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            state="readonly",
            width=10
        )
        self.model_combo.grid(row=0, column=1, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Default timestamps checkbox
        self.timestamps_var = tk.BooleanVar(value=True)
        self.timestamps_check = ttk.Checkbutton(
            self.options_frame,
            text="Default Timestamps",
            variable=self.timestamps_var
        )
        self.timestamps_check.grid(row=0, column=2, padx=5)
        
        # File selection button
        self.select_button = ttk.Button(
            self.left_frame, 
            text="Add Audio Files",
            command=self.select_files
        )
        self.select_button.grid(row=1, column=0, pady=5)
        
        # File list frame
        self.file_list_frame = ttk.LabelFrame(self.left_frame, text="Files to Process", padding="5")
        self.file_list_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File list
        self.file_list = ttk.Treeview(
            self.file_list_frame,
            columns=("status", "timestamps"),
            show="headings",
            height=10
        )
        self.file_list.heading("status", text="Status")
        self.file_list.heading("timestamps", text="Timestamps")
        self.file_list.column("status", width=200)
        self.file_list.column("timestamps", width=80)
        self.file_list.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File list scrollbar
        self.file_list_scrollbar = ttk.Scrollbar(
            self.file_list_frame,
            orient=tk.VERTICAL,
            command=self.file_list.yview
        )
        self.file_list_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_list.configure(yscrollcommand=self.file_list_scrollbar.set)
        
        # File list buttons frame
        self.file_buttons_frame = ttk.Frame(self.file_list_frame)
        self.file_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        # File list control buttons
        self.move_up_btn = ttk.Button(
            self.file_buttons_frame,
            text="↑",
            width=3,
            command=self.move_file_up
        )
        self.move_up_btn.grid(row=0, column=0, padx=2)
        
        self.move_down_btn = ttk.Button(
            self.file_buttons_frame,
            text="↓",
            width=3,
            command=self.move_file_down
        )
        self.move_down_btn.grid(row=0, column=1, padx=2)
        
        self.remove_btn = ttk.Button(
            self.file_buttons_frame,
            text="Remove",
            command=self.remove_selected_file
        )
        self.remove_btn.grid(row=0, column=2, padx=2)
        
        self.toggle_timestamps_btn = ttk.Button(
            self.file_buttons_frame,
            text="Toggle Timestamps",
            command=self.toggle_selected_timestamps
        )
        self.toggle_timestamps_btn.grid(row=0, column=3, padx=2)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.left_frame)
        self.control_frame.grid(row=3, column=0, pady=5)
        
        # Control buttons
        self.start_btn = ttk.Button(
            self.control_frame,
            text="Start",
            command=self.start_processing
        )
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.pause_btn = ttk.Button(
            self.control_frame,
            text="Pause",
            command=self.toggle_pause,
            state=tk.DISABLED
        )
        self.pause_btn.grid(row=0, column=1, padx=2)
        
        self.stop_btn = ttk.Button(
            self.control_frame,
            text="Stop",
            command=self.stop_processing,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=2, padx=2)
        
        # Transcription text area
        self.text_area = scrolledtext.ScrolledText(
            self.right_frame,
            wrap=tk.WORD,
            width=80,
            height=30
        )
        self.text_area.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model download progress frame
        self.model_progress_frame = ttk.LabelFrame(self.right_frame, text="Model Download Progress", padding="5")
        self.model_progress_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        self.model_progress_frame.grid_remove()
        
        # Model download progress bar
        self.model_progress = ttk.Progressbar(
            self.model_progress_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate'
        )
        self.model_progress.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        # Model progress label
        self.model_progress_label = ttk.Label(self.model_progress_frame, text="")
        self.model_progress_label.grid(row=0, column=1)
        
        # File progress frame
        self.progress_frame = ttk.LabelFrame(self.right_frame, text="File Processing Progress", padding="5")
        self.progress_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # File progress bar
        self.progress = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate'
        )
        self.progress.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        # File progress label
        self.progress_label = ttk.Label(self.progress_frame, text="0%")
        self.progress_label.grid(row=0, column=1)
        
        # Status label
        self.status_label = ttk.Label(self.right_frame, text="Ready")
        self.status_label.grid(row=3, column=0, pady=5)
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(2, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)
        self.options_frame.columnconfigure(3, weight=1)
        self.progress_frame.columnconfigure(0, weight=1)
        self.model_progress_frame.columnconfigure(0, weight=1)
        self.file_list_frame.columnconfigure(0, weight=1)
        self.file_list_frame.rowconfigure(0, weight=1)
        
        # Start checking the queue
        self.check_queue()
        
        # Show model info
        self.show_model_info()

    def move_file_up(self):
        """Move selected file up in the queue"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        item = selected[0]
        index = self.file_list.index(item)
        if index > 0:
            self.file_list.move(item, "", index - 1)

    def move_file_down(self):
        """Move selected file down in the queue"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        item = selected[0]
        index = self.file_list.index(item)
        if index < len(self.file_list.get_children()) - 1:
            self.file_list.move(item, "", index + 1)

    def remove_selected_file(self):
        """Remove selected file from the queue"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        for item in selected:
            self.file_list.delete(item)

    def toggle_selected_timestamps(self):
        """Toggle timestamps for selected file"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        for item in selected:
            current = self.file_list.item(item)["values"][1]
            new_value = "No" if current == "Yes" else "Yes"
            values = list(self.file_list.item(item)["values"])
            values[1] = new_value
            self.file_list.item(item, values=values)

    def start_processing(self):
        """Start processing the file queue"""
        if not self.file_list.get_children():
            return
        
        self.is_processing = True
        self.should_stop = False
        self.is_paused = False
        
        # Update button states
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.select_button.configure(state=tk.DISABLED)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_queue)
        thread.daemon = True
        thread.start()

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        self.pause_btn.configure(text="Resume" if self.is_paused else "Pause")
        self.queue.put(("status", "Paused" if self.is_paused else "Resuming..."))

    def stop_processing(self):
        """Stop processing the queue"""
        self.should_stop = True
        self.is_processing = False
        self.is_paused = False
        
        # Update button states
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.select_button.configure(state=tk.NORMAL)
        
        self.queue.put(("status", "Processing stopped"))

    def process_queue(self):
        """Process the file queue"""
        model_name = self.model_var.get()
        
        try:
            # Load model once for all files
            self.queue.put(("text", "Loading model...\n"))
            model = self.load_model(model_name)
            self.queue.put(("text", "Model loaded, starting transcription...\n\n"))
            
            # Get all files from the queue
            files = []
            for item in self.file_list.get_children():
                values = self.file_list.item(item)["values"]
                files.append((item, values[0].split(" - ")[0], values[1] == "Yes", values[2]))  # Include full path
            
            total_files = len(files)
            
            for i, (item, filename, include_timestamps, file_path) in enumerate(files, 1):
                if self.should_stop:
                    break
                
                # Update file status
                self.queue.put(("file_status", (i-1, f"{filename} - Processing")))
                
                # Update progress
                progress = (i/total_files) * 100
                self.queue.put(("progress", progress))
                self.queue.put(("status", f"Processing file {i} of {total_files}: {filename}"))
                self.queue.put(("text", f"\nStarting transcription of {filename}...\n"))
                
                try:
                    # Get audio duration
                    duration = librosa.get_duration(path=file_path)
                    total_minutes = int(duration / 60)
                    self.queue.put(("text", f"Audio length: {total_minutes} minutes\n"))
                    
                    # Calculate estimated processing time based on model
                    model_speeds = {
                        "tiny": 2.5,      # ~2.5x real-time
                        "base": 2.0,      # ~2x real-time
                        "small": 1.5,     # ~1.5x real-time
                        "medium": 1.0,    # ~1x real-time
                        "large-v3": 0.6   # ~0.6x real-time
                    }
                    speed_factor = model_speeds[model_name]
                    est_minutes = int((duration / 60) / speed_factor)
                    
                    # Show model and processing info
                    self.queue.put(("text", f"Using {model_name} model\n"))
                    self.queue.put(("text", f"Estimated processing time: {est_minutes} minutes\n"))
                    self.queue.put(("text", "Transcription in progress...\n"))
                    self.queue.put(("text", "This may take a while. The application will update when complete.\n\n"))
                    
                    # Transcribe file
                    result = model.transcribe(file_path)
                    
                    # Show processing stats
                    self.queue.put(("text", "Transcription complete!\n"))
                    self.queue.put(("text", f"Found {len(result['segments'])} segments of speech\n"))
                    
                    # After transcription, show the segments with their timestamps
                    if include_timestamps:
                        self.queue.put(("text", "Formatting output with timestamps...\n"))
                        # Format with timestamps
                        formatted_text = []
                        for segment in result["segments"]:
                            start_time = self.format_timestamp(segment["start"])
                            end_time = self.format_timestamp(segment["end"])
                            text = segment["text"].strip()
                            formatted_text.append(f"[{start_time} --> {end_time}] {text}")
                        transcription = "\n".join(formatted_text)
                    else:
                        transcription = result["text"]
                    
                    # Save to file in the same directory as the source file
                    output_file = Path(file_path).parent / f"{Path(file_path).stem}_transcription.txt"
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    
                    # Update text area with completion info
                    self.queue.put(("text", f"\n=== {filename} ===\n"))
                    self.queue.put(("text", f"Total duration: {self.format_timestamp(duration)}\n"))
                    self.queue.put(("text", f"Number of segments: {len(result['segments'])}\n"))
                    self.queue.put(("text", f"Saved to: {output_file}\n\n"))
                    self.queue.put(("text", f"{transcription}\n"))
                    self.queue.put(("status", f"Saved transcription to: {output_file}"))
                    
                    # Update file status
                    self.queue.put(("file_status", (i-1, f"{filename} - Complete")))
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.queue.put(("text", f"\n=== {filename} ===\n{error_msg}\n"))
                    self.queue.put(("status", error_msg))
                    self.queue.put(("file_status", (i-1, f"{filename} - Error")))
                
                # Check for pause
                while self.is_paused and not self.should_stop:
                    self.root.after(100, lambda: None)
            
            if not self.should_stop:
                self.queue.put(("status", "All transcriptions complete!"))
            
        except Exception as e:
            self.queue.put(("status", f"Error: {str(e)}"))
            self.queue.put(("text", f"\nError during processing: {str(e)}\n"))
        
        finally:
            self.is_processing = False
            self.queue.put(("progress", 0))
            # Reset button states
            self.queue.put(("button_state", ("start", tk.NORMAL)))
            self.queue.put(("button_state", ("pause", tk.DISABLED)))
            self.queue.put(("button_state", ("stop", tk.DISABLED)))
            self.queue.put(("button_state", ("select", tk.NORMAL)))

    def select_files(self):
        """Open file dialog to select audio files"""
        filetypes = (
            ("Audio files", "*.wav *.mp3 *.mpeg *.mp4 *.m4a"),
            ("All files", "*.*")
        )
        
        files = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=filetypes
        )
        
        if files:
            # Clear text area
            self.text_area.delete(1.0, tk.END)
            
            # Add files to list
            for file_path in files:
                filename = os.path.basename(file_path)
                self.file_list.insert("", tk.END, text=filename, values=(
                    f"{filename} - Pending",
                    "Yes" if self.timestamps_var.get() else "No",
                    file_path  # Store full path as hidden value
                ))
            
            # Reset progress
            self.progress["value"] = 0
            self.progress_label["text"] = "0%"
            
            # Show model info
            self.show_model_info()

    def check_queue(self):
        """Check the queue for updates"""
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                
                if msg_type == "text":
                    self.text_area.insert(tk.END, msg_data)
                    self.text_area.see(tk.END)
                elif msg_type == "progress":
                    self.progress["value"] = msg_data
                    self.progress_label["text"] = f"{int(msg_data)}%"
                elif msg_type == "model_progress":
                    self.model_progress["value"] = msg_data
                elif msg_type == "model_progress_label":
                    self.model_progress_label["text"] = msg_data
                elif msg_type == "status":
                    self.status_label["text"] = msg_data
                elif msg_type == "file_status":
                    index, status = msg_data
                    item = self.file_list.get_children()[index]
                    values = list(self.file_list.item(item)["values"])
                    values[0] = status
                    self.file_list.item(item, values=values)
                elif msg_type == "show_model_progress":
                    if msg_data:
                        self.model_progress_frame.grid()
                    else:
                        self.model_progress_frame.grid_remove()
                elif msg_type == "button_state":
                    button, state = msg_data
                    if button == "start":
                        self.start_btn.configure(state=state)
                    elif button == "pause":
                        self.pause_btn.configure(state=state)
                    elif button == "stop":
                        self.stop_btn.configure(state=state)
                    elif button == "select":
                        self.select_button.configure(state=state)
                
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)

    def on_model_change(self, event=None):
        """Update model info when model selection changes"""
        # Clear text area
        self.text_area.delete(1.0, tk.END)
        # Show new model info
        self.show_model_info()

    def show_model_info(self):
        """Show information about the selected model"""
        model_name = self.model_var.get()
        model_sizes = {
            "tiny": "~75MB download, ~1GB in memory",
            "base": "~142MB download, ~1GB in memory",
            "small": "~466MB download, ~2GB in memory",
            "medium": "~1.5GB download, ~5GB in memory",
            "large-v3": "~3GB download, ~10GB in memory"
        }
        
        model_use_cases = {
            "tiny": "Best for: Quick transcriptions, short audio, clear speech, English only",
            "base": "Best for: General purpose, good balance of speed and accuracy",
            "small": "Best for: Multiple languages, moderate accuracy needed",
            "medium": "Best for: Complex audio, multiple speakers, high accuracy needed",
            "large-v3": "Best for: Professional use, maximum accuracy, complex audio"
        }
        
        # Check which models are downloaded using Whisper's cache location
        downloaded_models = []
        cache_dir = os.path.expanduser("~/.cache/whisper")
        for model in ["tiny", "base", "small", "medium", "large-v3"]:
            model_path = os.path.join(cache_dir, f"{model}.pt")
            if os.path.exists(model_path):
                downloaded_models.append(model)
        
        # Update status and text area
        self.queue.put(("status", f"Selected model: {model_name} ({model_sizes[model_name]})"))
        
        # Build model info text
        info_text = f"Selected model: {model_name}\n"
        info_text += f"Size: {model_sizes[model_name]}\n"
        info_text += f"Use case: {model_use_cases[model_name]}\n"
        info_text += "The model will be downloaded and run locally.\n\n"
        
        # Add downloaded models info
        if downloaded_models:
            info_text += "Downloaded models:\n"
            for model in downloaded_models:
                info_text += f"- {model}: {model_sizes[model]}\n"
            info_text += "\n"
        else:
            info_text += "No models downloaded yet. The selected model will be downloaded when you start transcription.\n\n"
        
        # Add model selection guidance
        info_text += "Model Selection Guide:\n"
        info_text += "- tiny: Fastest, least accurate, English only\n"
        info_text += "- base: Good balance of speed and accuracy\n"
        info_text += "- small: Better accuracy, supports multiple languages\n"
        info_text += "- medium: High accuracy, good for complex audio\n"
        info_text += "- large-v3: Best accuracy, professional quality\n\n"
        
        self.queue.put(("text", info_text))

    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=round(seconds)))

    def load_model(self, model_name):
        """Load the Whisper model with download status"""
        try:
            # Get the model path in Whisper's cache
            model_path = os.path.expanduser(f"~/.cache/whisper/{model_name}.pt")
            
            # Check if model exists
            if not os.path.exists(model_path):
                self.queue.put(("show_model_progress", True))
                self.queue.put(("status", f"Downloading {model_name} model..."))
                self.queue.put(("text", f"\nDownloading {model_name} model...\nThis is a one-time download. The model will be stored locally at:\n{model_path}\n\n"))
                self.queue.put(("model_progress", 0))
                self.queue.put(("model_progress_label", "Starting download..."))
            
            # Load the model
            model = whisper.load_model(model_name)
            
            self.queue.put(("model_progress", 100))
            self.queue.put(("model_progress_label", "Complete"))
            self.queue.put(("status", f"Model {model_name} loaded successfully"))
            self.queue.put(("text", f"Model {model_name} loaded successfully!\n\n"))
            self.queue.put(("show_model_progress", False))
            return model
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.queue.put(("status", error_msg))
            self.queue.put(("text", f"\nError: {error_msg}\n"))
            self.queue.put(("show_model_progress", False))
            raise

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 