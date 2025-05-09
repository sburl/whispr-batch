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
import time

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
        
        # Time tracking
        self.remaining_time = 0
        self.total_time = 0
        self.start_time = None
        
        # Model speeds for time estimation
        self.model_speeds = {
            "tiny": 2.5,      # ~2.5x real-time
            "base": 2.0,      # ~2x real-time
            "small": 1.5,     # ~1.5x real-time
            "medium": 1.0,    # ~1x real-time
            "large-v3": 0.6   # ~0.6x real-time
        }
        
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
        self.timestamps_var = tk.BooleanVar(value=False)
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
            columns=("filename", "status", "timestamps", "model", "est_time"),
            show="headings",
            height=10
        )
        self.file_list.heading("filename", text="Filename")
        self.file_list.heading("status", text="Status")
        self.file_list.heading("timestamps", text="Timestamps")
        self.file_list.heading("model", text="Model")
        self.file_list.heading("est_time", text="Est. Time")
        self.file_list.column("filename", width=150)
        self.file_list.column("status", width=100)
        self.file_list.column("timestamps", width=80)
        self.file_list.column("model", width=80)
        self.file_list.column("est_time", width=80)
        self.file_list.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Enable drag and drop
        self.file_list.bind('<ButtonPress-1>', self.on_drag_start)
        self.file_list.bind('<B1-Motion>', self.on_drag_motion)
        self.file_list.bind('<ButtonRelease-1>', self.on_drag_release)
        
        # File list scrollbar
        self.file_list_scrollbar = ttk.Scrollbar(
            self.file_list_frame,
            orient=tk.VERTICAL,
            command=self.file_list.yview
        )
        self.file_list_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_list.configure(yscrollcommand=self.file_list_scrollbar.set)
        
        # Total time estimate label
        self.total_time_label = ttk.Label(self.file_list_frame, text="Total estimated time: --:--")
        self.total_time_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # File list buttons frame
        self.file_buttons_frame = ttk.Frame(self.file_list_frame)
        self.file_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        # File list control buttons
        self.remove_btn = ttk.Button(
            self.file_buttons_frame,
            text="Remove",
            command=self.remove_selected_file
        )
        self.remove_btn.grid(row=0, column=0, padx=2)
        
        self.toggle_timestamps_btn = ttk.Button(
            self.file_buttons_frame,
            text="Toggle Timestamps",
            command=self.toggle_selected_timestamps
        )
        self.toggle_timestamps_btn.grid(row=0, column=1, padx=2)
        
        self.change_model_btn = ttk.Button(
            self.file_buttons_frame,
            text="Change Model",
            command=self.change_selected_model
        )
        self.change_model_btn.grid(row=0, column=2, padx=2)
        
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

    def remove_selected_file(self):
        """Remove selected file from the queue"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        for item in selected:
            self.file_list.delete(item)
        
        # Update total time estimate
        self.update_total_time_estimate()

    def toggle_selected_timestamps(self):
        """Toggle timestamps for selected file"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        for item in selected:
            values = list(self.file_list.item(item)["values"])
            if len(values) >= 3:  # Ensure we have the timestamps value
                current = values[2]
                new_value = "No" if current == "Yes" else "Yes"
                values[2] = new_value
                self.file_list.item(item, values=values)

    def start_processing(self):
        """Start processing the file queue"""
        if not self.file_list.get_children():
            return
        
        self.is_processing = True
        self.should_stop = False
        self.is_paused = False
        self.start_time = time.time()
        
        # Update button states
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.select_button.configure(state=tk.DISABLED)
        
        # Start the remaining time updates
        self.update_remaining_time()
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_queue)
        thread.daemon = True
        thread.start()

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        self.pause_btn.configure(text="Resume" if self.is_paused else "Pause")
        
        if self.is_paused:
            self.start_time = None
            # Enable file selection when paused
            self.select_button.configure(state=tk.NORMAL)
            self.queue.put(("status", "Paused - You can add more files"))
            self.queue.put(("text", "\nProcessing paused. You can add more files or click 'Resume' to continue.\n"))
        else:
            # Reset start time when resuming
            self.start_time = time.time()
            # Disable file selection when resuming
            self.select_button.configure(state=tk.DISABLED)
            self.queue.put(("status", "Resuming..."))
            self.queue.put(("text", "\nResuming processing...\n"))

    def stop_processing(self):
        """Stop processing the queue"""
        self.should_stop = True
        self.is_processing = False
        self.is_paused = False
        self.start_time = None
        
        # Update button states
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.select_button.configure(state=tk.NORMAL)
        
        # Reset time display
        self.update_total_time_estimate()
        
        self.queue.put(("status", "Processing stopped"))

    def process_queue(self):
        """Process the file queue"""
        try:
            # Load model once for all files
            self.queue.put(("text", "Loading model...\n"))
            current_model_name = self.model_var.get()
            model = self.load_model(current_model_name)
            self.queue.put(("text", "Model loaded, starting transcription...\n\n"))
            
            # Get all files from the queue
            files = []
            for item in self.file_list.get_children():
                try:
                    values = self.file_list.item(item)["values"]
                    if len(values) < 6:  # Ensure we have all required values
                        self.queue.put(("text", f"\nError: Invalid file entry in queue\n"))
                        continue
                        
                    filename = values[0]
                    status = values[1]
                    include_timestamps = values[2] == "Yes"
                    file_path = values[5]
                    file_model = values[3]
                    
                    # Skip non-local files
                    if status == "Not Local":
                        self.queue.put(("text", f"\nSkipping non-local file: {filename}\n"))
                        self.queue.put(("file_status", (len(files), "Not Local")))
                        continue
                    
                    # Convert to absolute path if needed
                    file_path = os.path.abspath(file_path)
                    
                    # Verify file still exists and is not empty
                    if not self.is_local_file(file_path):
                        self.queue.put(("text", f"\nError: File is not local: {filename}\n"))
                        self.queue.put(("file_status", (len(files), "Not Local")))
                        continue
                        
                    if os.path.getsize(file_path) == 0:
                        self.queue.put(("text", f"\nError: File is empty: {filename}\n"))
                        self.queue.put(("file_status", (len(files), "Error")))
                        continue
                        
                    files.append((item, filename, include_timestamps, file_path, file_model))
                except Exception as e:
                    self.queue.put(("text", f"\nError processing queue entry: {str(e)}\n"))
                    continue
            
            if not files:
                self.queue.put(("text", "\nNo valid files to process\n"))
                return
                
            total_files = len(files)
            
            for i, (item, filename, include_timestamps, file_path, file_model) in enumerate(files, 1):
                if self.should_stop:
                    break
                
                # Update file status
                self.queue.put(("file_status", (i-1, "Processing")))
                
                # Update progress
                progress = (i/total_files) * 100
                self.queue.put(("progress", progress))
                self.queue.put(("status", f"Processing file {i} of {total_files}: {filename}"))
                self.queue.put(("text", f"\nStarting transcription of {filename}...\n"))
                
                try:
                    # Verify file still exists and is not empty
                    if not self.is_local_file(file_path):
                        raise FileNotFoundError(f"File is not local: {filename}")
                    if os.path.getsize(file_path) == 0:
                        raise ValueError(f"File is empty: {filename}")
                    
                    # Get audio duration using ffprobe
                    import subprocess
                    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    duration = float(result.stdout.strip())
                    
                    if duration <= 0:
                        raise ValueError(f"Invalid duration for {filename}")
                        
                    total_minutes = int(duration / 60)
                    self.queue.put(("text", f"Audio length: {total_minutes} minutes\n"))
                    
                    # Calculate estimated processing time based on model
                    speed_factor = self.model_speeds[file_model]
                    est_minutes = int((duration / 60) / speed_factor)
                    
                    # Show model and processing info
                    self.queue.put(("text", f"Using {file_model} model\n"))
                    self.queue.put(("text", f"Estimated processing time: {est_minutes} minutes\n"))
                    self.queue.put(("text", "Transcription in progress...\n"))
                    self.queue.put(("text", "This may take a while. The application will update when complete.\n\n"))
                    
                    # Load the correct model for this file if different from current
                    if file_model != current_model_name:
                        self.queue.put(("text", f"Loading {file_model} model for this file...\n"))
                        model = self.load_model(file_model)
                        current_model_name = file_model
                    
                    # Transcribe file
                    result = model.transcribe(file_path)
                    
                    # After transcription, format with timestamps if needed
                    if include_timestamps:
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
                    self.queue.put(("text", f"Transcription complete!\n"))
                    self.queue.put(("text", f"Saved to: {output_file}\n\n"))
                    self.queue.put(("status", f"Saved transcription to: {output_file}"))
                    
                    # Update file status
                    self.queue.put(("file_status", (i-1, "Complete")))
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.queue.put(("text", f"\n=== {filename} ===\n{error_msg}\n"))
                    self.queue.put(("status", error_msg))
                    self.queue.put(("file_status", (i-1, "Error")))
                
                # Check for pause
                while self.is_paused and not self.should_stop:
                    self.queue.put(("status", "Paused"))
                    time.sleep(0.1)  # Reduce CPU usage while paused
                    self.root.update()  # Keep UI responsive
                    if self.should_stop:  # Check if stop was requested while paused
                        break
            
            if not self.should_stop:
                self.queue.put(("status", "All transcriptions complete!"))
            
        except Exception as e:
            self.queue.put(("status", f"Error: {str(e)}"))
            self.queue.put(("text", f"\nError during processing: {str(e)}\n"))
        
        finally:
            self.is_processing = False
            self.is_paused = False  # Reset pause state
            self.start_time = None
            self.queue.put(("progress", 0))
            # Reset button states
            self.queue.put(("button_state", ("start", tk.NORMAL)))
            self.queue.put(("button_state", ("pause", tk.DISABLED)))
            self.queue.put(("button_state", ("stop", tk.DISABLED)))
            self.queue.put(("button_state", ("select", tk.NORMAL)))
            # Reset time display
            self.update_total_time_estimate()

    def select_files(self):
        """Open file dialog to select audio files"""
        # Check if we're processing and not paused
        if self.is_processing and not self.is_paused:
            self.queue.put(("text", "\nCannot add files while processing is active.\n"))
            self.queue.put(("text", "Please click 'Pause' first to add more files.\n\n"))
            self.queue.put(("status", "Click 'Pause' to add more files"))
            return

        filetypes = (
            ("Audio files", "*.wav *.mp3 *.mpeg *.mp4 *.m4a"),
            ("All files", "*.*")
        )
        
        files = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=filetypes
        )
        
        if files:
            # Only clear text area if not processing
            if not self.is_processing:
                self.text_area.delete(1.0, tk.END)
            
            # Add files to list
            for file_path in files:
                try:
                    # Convert to absolute path
                    file_path = os.path.abspath(file_path)
                    filename = os.path.basename(file_path)
                    
                    # Check if file is local
                    if not self.is_local_file(file_path):
                        self.queue.put(("text", f"\nWarning: File is not local (may be in iCloud Drive): {filename}\n"))
                        # Add file to list with "Not Local" status
                        self.file_list.insert("", tk.END, values=(
                            filename,  # Display name
                            "Not Local",  # Status
                            "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                            self.model_var.get(),  # Model
                            "--:--",  # Estimated time
                            file_path  # Full path (hidden)
                        ))
                        continue
                    
                    # Try to get duration to verify file is valid
                    try:
                        # Use a more efficient method to get duration
                        import subprocess
                        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        duration = float(result.stdout.strip())
                        
                        if duration <= 0:
                            raise ValueError("Invalid duration")
                            
                        total_minutes = int(duration / 60)
                        speed_factor = self.model_speeds[self.model_var.get()]
                        est_minutes = int((duration / 60) / speed_factor)
                        est_time = self.format_time_estimate(est_minutes)
                    except Exception as e:
                        self.queue.put(("text", f"\nWarning: Could not get duration for {filename}: {str(e)}\n"))
                        est_time = "--:--"
                    
                    # Add file to list with full path
                    self.file_list.insert("", tk.END, values=(
                        filename,  # Display name
                        "Pending",  # Status
                        "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                        self.model_var.get(),  # Model
                        est_time,  # Estimated time
                        file_path  # Full path (hidden)
                    ))
                    
                except Exception as e:
                    self.queue.put(("text", f"\nError adding file {file_path}: {str(e)}\n"))
                    continue
            
            # Reset progress if not processing
            if not self.is_processing:
                self.progress["value"] = 0
                self.progress_label["text"] = "0%"
            
            # Update total time estimate
            self.update_total_time_estimate()
            
            # Show model info only if not processing
            if not self.is_processing:
                self.show_model_info()
            
            # If we're paused, remind user to resume
            if self.is_paused:
                self.queue.put(("text", "\nFiles added successfully. Click 'Resume' to continue processing.\n"))
                self.queue.put(("status", "Files added. Click 'Resume' to continue"))

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
                    values[1] = status
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
        # Only update the status bar
        model_name = self.model_var.get()
        model_sizes = {
            "tiny": "~75MB download, ~1GB in memory",
            "base": "~142MB download, ~1GB in memory",
            "small": "~466MB download, ~2GB in memory",
            "medium": "~1.5GB download, ~5GB in memory",
            "large-v3": "~3GB download, ~10GB in memory"
        }
        self.status_label["text"] = f"Selected model: {model_name} ({model_sizes[model_name]})"

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
        
        # Update status
        self.status_label["text"] = f"Selected model: {model_name} ({model_sizes[model_name]})"
        
        # Only show full info in text area if it's empty
        if not self.text_area.get(1.0, tk.END).strip():
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
            
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, info_text)

    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=round(seconds)))

    def format_time_estimate(self, minutes):
        """Format time estimate in HH:MM format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def update_total_time_estimate(self):
        """Update the total time estimate based on all files"""
        total_minutes = 0
        for item in self.file_list.get_children():
            values = self.file_list.item(item)["values"]
            if len(values) >= 5 and values[4] != "--:--":
                try:
                    hours, minutes = map(int, values[4].split(":"))
                    total_minutes += hours * 60 + minutes
                except ValueError:
                    continue
        
        self.total_time = total_minutes
        if not self.is_processing:
            self.remaining_time = total_minutes
            self.total_time_label["text"] = f"Total estimated time: {self.format_time_estimate(total_minutes)}"

    def update_remaining_time(self):
        """Update the remaining time display"""
        if self.is_processing and not self.is_paused and self.start_time:
            elapsed_minutes = (time.time() - self.start_time) / 60
            self.remaining_time = max(0, self.total_time - elapsed_minutes)
            self.total_time_label["text"] = f"Time remaining: {self.format_time_estimate(int(self.remaining_time))}"
        
        # Schedule next update
        if self.is_processing:
            self.root.after(1000, self.update_remaining_time)

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

    def change_selected_model(self):
        """Change model for selected files"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        # Create a dialog for model selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Model")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Add model selection
        ttk.Label(dialog, text="Select new model:").pack(pady=5)
        model_var = tk.StringVar(value=self.model_var.get())
        model_combo = ttk.Combobox(
            dialog,
            textvariable=model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            state="readonly",
            width=10
        )
        model_combo.pack(pady=5)
        
        def apply_model_change():
            new_model = model_var.get()
            for item in selected:
                try:
                    values = list(self.file_list.item(item)["values"])
                    if len(values) < 6:  # Ensure we have all required values
                        continue
                        
                    status = values[1]
                    # Only change model for pending files
                    if status == "Pending":
                        try:
                            duration = librosa.get_duration(path=values[5])
                            total_minutes = int(duration / 60)
                            speed_factor = self.model_speeds[new_model]
                            est_minutes = int((duration / 60) / speed_factor)
                            values[3] = new_model  # Update model
                            values[4] = self.format_time_estimate(est_minutes)  # Update time estimate
                            self.file_list.item(item, values=values)
                        except Exception as e:
                            print(f"Error updating time estimate: {e}")
                            values[3] = new_model
                            values[4] = "--:--"
                            self.file_list.item(item, values=values)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
            
            # Update total time estimate
            self.update_total_time_estimate()
            dialog.destroy()
        
        # Add buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Apply",
            command=apply_model_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)

    def on_drag_start(self, event):
        """Start dragging an item"""
        # Get the item under the cursor
        item = self.file_list.identify_row(event.y)
        if not item:
            return
        
        # Store the item being dragged
        self.drag_item = item
        
        # Get the item's values
        values = self.file_list.item(item)["values"]
        if values[1] != "Pending":  # Only allow dragging pending items
            self.drag_item = None
            return
        
        # Store the initial position
        self.drag_start_index = self.file_list.index(item)
        
        # Change the item's appearance
        self.file_list.tag_configure('dragging', background='#e0e0e0')
        self.file_list.item(item, tags=('dragging',))

    def on_drag_motion(self, event):
        """Handle dragging motion"""
        if not hasattr(self, 'drag_item') or not self.drag_item:
            return
        
        # Get the item under the cursor
        target = self.file_list.identify_row(event.y)
        if not target:
            return
        
        # Get the target's values
        target_values = self.file_list.item(target)["values"]
        if target_values[1] != "Pending":  # Only allow dropping on pending items
            return
        
        # Get the target's position
        target_index = self.file_list.index(target)
        
        # Move the item
        if target_index != self.drag_start_index:
            self.file_list.move(self.drag_item, "", target_index)
            self.drag_start_index = target_index

    def on_drag_release(self, event):
        """Handle end of drag"""
        if hasattr(self, 'drag_item') and self.drag_item:
            # Remove the dragging tag
            self.file_list.item(self.drag_item, tags=())
            self.drag_item = None

    def is_local_file(self, file_path):
        """Check if a file is local or in iCloud Drive"""
        try:
            # Check if file is in iCloud Drive
            if "iCloud Drive" in file_path:
                return False
            # Check if file exists and is accessible
            return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
        except Exception:
            return False

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 