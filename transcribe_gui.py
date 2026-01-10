#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
from pathlib import Path
import queue
import os
import librosa
import time

from whisper_batch_core import (
    format_timestamp as core_format_timestamp,
    load_model as core_load_model,
    render_plain_text,
    render_timestamped_text,
    transcribe_segments,
)
import platform
import sys

# --- Environment sanity checks for macOS/Torch ---------------------------------
# If running on Apple Silicon ensure a native arm64 build of PyTorch is present.
# x86-64 wheels executed via Rosetta crash with a misleading loader error.
# We intercept that scenario to display actionable instructions instead.

def _check_pytorch_arch():
    try:
        import torch  # noqa: F401 – we only need the import side-effects
    except OSError as exc:
        if "have instead 16" in str(exc) and platform.machine() == "arm64":
            sys.stderr.write(
                "\n🚫 Detected x86-64 PyTorch wheel running under Rosetta.\n"
                "Please reinstall the native arm64 wheel:\n\n"
                "    pip uninstall -y torch\n"
                "    pip install --no-cache-dir --force-reinstall torch==2.1.0 "
                "--index-url https://download.pytorch.org/whl/cpu\n\n"
                "Then run the program again.\n"
            )
            sys.exit(1)
    except ModuleNotFoundError:
        # torch not installed – setup is still in progress; skip check
        pass

_check_pytorch_arch()

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WhisperBatch")
        self.root.geometry("1000x800")
        
        # Create message queue for thread-safe updates
        self.queue = queue.Queue()
        
        # Control flags
        self.is_processing = False
        self.is_paused = False
        self.should_stop = False
        self.worker_thread = None
        
        # Task queue and progress tracking
        self.task_queue = queue.Queue()
        self.progress_lock = threading.Lock()
        self.total_tasks = 0
        self.completed_tasks = 0
        
        # Time tracking
        self.start_time = None
        self.total_elapsed_seconds = 0
        self.pause_start_time = None
        self.processing_completed = False
        self.transcribe_start_time = None
        self.transcribe_filename = None
        self.transcribe_timer_id = None
        
        # Base model speeds for time estimation (CPU float32 baseline)
        self.base_model_speeds = {
            "tiny": 2.5,      # ~2.5x real-time
            "base": 2.0,      # ~2x real-time
            "small": 1.5,     # ~1.5x real-time
            "medium": 1.0,    # ~1x real-time
            "large-v3": 0.6   # ~0.6x real-time
        }
        self.model_speeds = dict(self.base_model_speeds)
        # On Apple-silicon with int8 inference the effective speed is much faster.
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self.model_speeds = {
                "tiny": 15.0,
                "base": 10.0,
                "small": 6.0,
                "medium": 4.0,
                "large-v3": 2.0,
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

        # Device selection
        ttk.Label(self.options_frame, text="Device:").grid(row=1, column=0, padx=5, pady=(5, 0))
        self.device_options = {
            "Auto (recommended)": "auto",
            "CPU": "cpu",
            "CUDA (NVIDIA GPU)": "cuda"
        }
        self.device_var = tk.StringVar(value="Auto (recommended)")
        self.device_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.device_var,
            values=list(self.device_options.keys()),
            state="readonly",
            width=18
        )
        self.device_combo.grid(row=1, column=1, padx=5, pady=(5, 0), sticky=tk.W)
        self.device_combo.bind('<<ComboboxSelected>>', self.on_device_change)

        # Compute type selection
        ttk.Label(self.options_frame, text="Compute:").grid(row=1, column=2, padx=5, pady=(5, 0))
        self.compute_label_to_type = {
            "Auto (recommended)": None,
            "Fast (float16)": "float16",
            "Balanced (int8_float16)": "int8_float16",
            "Memory Saver (int8)": "int8",
            "Precise (float32)": "float32"
        }
        self.compute_var = tk.StringVar(value="Auto (recommended)")
        self.compute_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.compute_var,
            values=[],
            state="readonly",
            width=22
        )
        self.compute_combo.grid(row=1, column=3, padx=5, pady=(5, 0), sticky=tk.W)
        self.compute_combo.bind('<<ComboboxSelected>>', self.on_compute_change)
        self.refresh_compute_options()
        self.update_speed_factors()
        
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
            columns=("filename", "status", "timestamps", "model"),
            show="headings",
            height=10
        )
        self.file_list.heading("filename", text="Filename")
        self.file_list.heading("status", text="Status")
        self.file_list.heading("timestamps", text="Timestamps")
        self.file_list.heading("model", text="Model")
        self.file_list.column("filename", width=200)
        self.file_list.column("status", width=120)
        self.file_list.column("timestamps", width=100)
        self.file_list.column("model", width=100)
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
        
        # Time label removed per user request
        self.total_time_label = ttk.Label(self.file_list_frame, text="")
        self.total_time_label.grid(row=2, column=0, columnspan=2, pady=5)
        self.total_time_label.grid_remove()
        
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
        
        # Elapsed time label near control buttons
        self.elapsed_time_label = ttk.Label(self.control_frame, text="Elapsed: 0s")
        self.elapsed_time_label.grid(row=0, column=3, padx=10)
        
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
        
        # Hidden progress bar (kept for internal state updates)
        self.progress_frame = ttk.Frame(self.right_frame)
        self.progress_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        self.progress_frame.grid_remove()
        self.progress = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate'
        )
        self.progress.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
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

    def reset_progress_tracking(self):
        """Reset counters and progress UI"""
        with self.progress_lock:
            self.total_tasks = 0
            self.completed_tasks = 0
        if hasattr(self, 'progress'):
            self.progress["value"] = 0
        if hasattr(self, 'progress_label'):
            self.progress_label["text"] = "0%"

    def enqueue_task_from_values(self, item_id, values):
        """Push a pending file into the worker queue"""
        if len(values) < 5:
            return
        filename, status, timestamps_flag, file_model, file_path = values
        if status != "Pending":
            return
        include_timestamps = timestamps_flag == "Yes"
        file_path = os.path.abspath(file_path)
        with self.progress_lock:
            self.total_tasks += 1
            total = self.total_tasks
            completed = self.completed_tasks
        percent = int((completed / total) * 100) if total else 0
        self.task_queue.put({
            "item_id": item_id,
            "filename": filename,
            "include_timestamps": include_timestamps,
            "file_path": file_path,
            "file_model": file_model
        })
        self.queue.put(("progress", percent))

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
            values = list(self.file_list.item(item)["values"])
            if len(values) >= 3:  # Ensure we have the timestamps value
                current = values[2]
                new_value = "No" if current == "Yes" else "Yes"
                values[2] = new_value
                self.file_list.item(item, values=values)

    def start_processing(self):
        """Start processing the file queue"""
        # If already processing but paused, just resume
        if self.is_processing and self.is_paused:
            self.is_paused = False
            self.start_time = time.time()
            self.select_button.configure(state=tk.DISABLED)
            self.queue.put(("status", "Resuming..."))
            self.queue.put(("text", "\nResuming processing...\n"))
            return

        if not self.file_list.get_children() or self.is_processing:
            return
        
        # Fresh run
        self.is_processing = True
        self.should_stop = False
        self.is_paused = False
        self.total_elapsed_seconds = 0
        self.pause_start_time = None
        self.processing_completed = False
        self.start_time = time.time()
        self.reset_progress_tracking()
        # Recreate the task queue for this run
        self.task_queue = queue.Queue()
        self.worker_initial_model = self.model_var.get()
        self.worker_device = self.get_selected_device()
        self.worker_compute_type = self.get_selected_compute_type()
        self.worker_model_speeds = dict(self.model_speeds)
        
        # Snapshot pending files on the main thread and enqueue them
        for item_id in self.file_list.get_children():
            values = self.file_list.item(item_id)["values"]
            self.enqueue_task_from_values(item_id, values)
        
        if self.total_tasks == 0:
            self.queue.put(("status", "No pending files to process"))
            self.is_processing = False
            self.start_btn.configure(state=tk.NORMAL)
            return
        
        # Update button states
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.select_button.configure(state=tk.DISABLED)
        self.device_combo.configure(state=tk.DISABLED)
        self.compute_combo.configure(state=tk.DISABLED)
        
        # Start the elapsed time updates
        self.update_remaining_time()
        
        # Start processing in a separate thread
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        self.pause_btn.configure(text="Resume" if self.is_paused else "Pause")
        
        if self.is_paused:
            # Pause - accumulate elapsed time
            if self.start_time:
                self.total_elapsed_seconds += int(time.time() - self.start_time)
                self.pause_start_time = time.time()
            # Enable file selection when paused
            self.select_button.configure(state=tk.NORMAL)
            self.queue.put(("status", "Paused - You can add more files"))
            self.queue.put(("text", "\nProcessing paused. You can add more files or click 'Resume' to continue.\n"))
        else:
            # Resume - restart timer from where we left off
            if self.pause_start_time:
                # Don't add pause time to elapsed
                self.pause_start_time = None
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
        self.total_elapsed_seconds = 0
        self.pause_start_time = None
        self.processing_completed = False
        self.elapsed_time_label["text"] = "Elapsed: 0s"
        
        # Update button states
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.select_button.configure(state=tk.NORMAL)
        self.device_combo.configure(state="readonly")
        self.compute_combo.configure(state="readonly")
        
        # Reset time display
        self.update_total_time_estimate()
        
        self.queue.put(("status", "Processing stopped"))

    def process_queue(self):
        """Process the file queue"""
        try:
            # Load model once initially; will swap if per-file model differs
            self.queue.put(("text", "Initializing faster-whisper (first time startup, ~10-30 seconds)...\n"))
            self.queue.put(("status", "Loading faster-whisper library..."))
            current_model_name = getattr(self, "worker_initial_model", self.model_var.get())
            model = self.load_model(
                current_model_name,
                device=getattr(self, "worker_device", self.get_selected_device()),
                compute_type=getattr(self, "worker_compute_type", self.get_selected_compute_type())
            )
            self.queue.put(("text", "Model loaded, starting transcription...\n\n"))
            
            pause_notified = False
            while not self.should_stop:
                # Respect pause requests
                while self.is_paused and not self.should_stop:
                    if not pause_notified:
                        self.queue.put(("status", "Paused"))
                        pause_notified = True
                    time.sleep(0.1)
                
                if self.should_stop:
                    break
                
                pause_notified = False
                
                try:
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    # Nothing left to do
                    with self.progress_lock:
                        if self.total_tasks == 0 or self.completed_tasks >= self.total_tasks:
                            break
                    continue
                
                item_id = task["item_id"]
                filename = task["filename"]
                include_timestamps = task["include_timestamps"]
                file_path = task["file_path"]
                file_model = task["file_model"]
                
                # Update file status
                self.queue.put(("file_status", (item_id, "Processing")))
                
                # Update status/progress text
                self.queue.put(("status", f"Processing: {filename}"))
                self.queue.put(("text", f"\nStarting transcription of {filename}...\n"))
                
                try:
                    # Verify file still exists and is accessible
                    if not self.is_local_file(file_path):
                        raise FileNotFoundError(f"File is not accessible: {filename}")
                    
                    # Get audio duration using ffprobe
                    import subprocess
                    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode != 0:
                        raise ValueError(f"Invalid audio file: {filename}")
                    duration = float(result.stdout.strip())
                    
                    if duration <= 0:
                        raise ValueError(f"Invalid duration for {filename}")
                        
                    total_minutes = int(duration / 60)
                    self.queue.put(("text", f"Audio length: {total_minutes} minutes\n"))
                    
                    # Show model and processing info
                    self.queue.put(("text", f"Using {file_model} model\n"))
                    self.queue.put(("text", "Transcription in progress...\n"))
                    self.queue.put(("text", "This may take a while. The application will update when complete.\n\n"))

                    # Load the correct model for this file if different from current
                    if file_model != current_model_name:
                        self.queue.put(("text", f"Loading {file_model} model for this file...\n"))
                        model = self.load_model(
                            file_model,
                            device=self.worker_device,
                            compute_type=self.worker_compute_type
                        )
                        current_model_name = file_model

                    # Start tracking elapsed time
                    transcribe_start = time.time()
                    self.queue.put(("transcribe_start", (filename, transcribe_start)))

                    # Transcribe file
                    segments, _info = transcribe_segments(model, file_path, task="transcribe")

                    # Stop tracking elapsed time
                    self.queue.put(("transcribe_end", None))
                    
                    # After transcription, format with timestamps if needed
                    if include_timestamps:
                        transcription = render_timestamped_text(segments)
                    else:
                        transcription = render_plain_text(segments)
                    
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
                    self.queue.put(("file_status", (item_id, "Complete")))
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.queue.put(("text", f"\n=== {filename} ===\n{error_msg}\n"))
                    self.queue.put(("status", error_msg))
                    self.queue.put(("file_status", (item_id, "Error")))
                
                finally:
                    with self.progress_lock:
                        self.completed_tasks += 1
                        total = self.total_tasks
                        completed = self.completed_tasks
                    percent = int((completed / total) * 100) if total else 0
                    self.queue.put(("progress", percent))
                
                # Check for stop or pause after each file
                if self.should_stop:
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
            # Reset button states
            self.queue.put(("button_state", ("start", tk.NORMAL)))
            self.queue.put(("button_state", ("pause", tk.DISABLED)))
            self.queue.put(("button_state", ("stop", tk.DISABLED)))
            self.queue.put(("button_state", ("select", tk.NORMAL)))
            self.queue.put(("device_state", ("readonly", "readonly")))
            self.queue.put(("processing_complete", None))

    def select_files(self):
        """Open file dialog to select audio files"""
        try:
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
            
            try:
                files = filedialog.askopenfilenames(
                    title="Select audio files",
                    filetypes=filetypes
                )
            except Exception as e:
                self.queue.put(("text", f"\nError opening file dialog: {str(e)}\n"))
                self.queue.put(("status", "Error selecting files"))
                return
            
            if not files:  # User cancelled or no files selected
                return
            
            # Only clear text area if not processing
            if not self.is_processing:
                self.text_area.delete(1.0, tk.END)
            
            # Add files to list
            for file_path in files:
                try:
                    # Convert to absolute path
                    file_path = os.path.abspath(file_path)
                    filename = os.path.basename(file_path)
                    
                    # Check if file is accessible
                    if not self.is_local_file(file_path):
                        self.queue.put(("text", f"\nWarning: Cannot access file: {filename}\n"))
                        if "iCloud Drive" in file_path:
                            self.queue.put(("text", "This file is in iCloud Drive and needs to be downloaded first.\n"))
                            self.queue.put(("text", "Please download it from iCloud Drive before trying again.\n\n"))
                        else:
                            self.queue.put(("text", "Please make sure the file exists and you have permission to access it.\n\n"))
                        
                        # Add file to list with "Not Accessible" status
                        self.file_list.insert("", tk.END, values=(
                            filename,  # Display name
                            "Not Accessible",  # Status
                            "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                            self.model_var.get(),  # Model
                            file_path  # Full path (hidden)
                        ))
                        continue
                    
                    # Try to get duration to verify file is valid
                    try:
                        # Use a more efficient method to get duration with timeout
                        import subprocess
                        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)  # 5 second timeout
                        
                        if result.returncode != 0:
                            raise ValueError("Invalid audio file")
                            
                        duration = float(result.stdout.strip())
                        
                        if duration <= 0:
                            raise ValueError("Invalid duration")
                            
                    except (subprocess.TimeoutExpired, ValueError, subprocess.CalledProcessError) as e:
                        self.queue.put(("text", f"\nWarning: Invalid audio file: {filename}\n"))
                        self.queue.put(("text", "This file appears to be corrupted or is not a valid audio file.\n\n"))
                        
                        # Add file to list with "Invalid" status
                        self.file_list.insert("", tk.END, values=(
                            filename,  # Display name
                            "Invalid",  # Status
                            "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                            self.model_var.get(),  # Model
                            file_path  # Full path (hidden)
                        ))
                        continue
                    except Exception as e:
                        self.queue.put(("text", f"\nWarning: Could not process {filename}: {str(e)}\n"))
                        self.queue.put(("text", "This file may be corrupted or in an unsupported format.\n\n"))
                        
                        # Add file to list with "Error" status
                        self.file_list.insert("", tk.END, values=(
                            filename,  # Display name
                            "Error",  # Status
                            "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                            self.model_var.get(),  # Model
                            file_path  # Full path (hidden)
                        ))
                        continue
                    
                    # Add file to list with full path
                    item_id = self.file_list.insert("", tk.END, values=(
                        filename,  # Display name
                        "Pending",  # Status
                        "Yes" if self.timestamps_var.get() else "No",  # Timestamps
                        self.model_var.get(),  # Model
                        file_path  # Full path (hidden)
                    ))
                    
                    # If we're paused mid-run, enqueue the new task so it processes after resume
                    if self.is_processing and self.is_paused:
                        self.enqueue_task_from_values(item_id, self.file_list.item(item_id)["values"])
                    
                except Exception as e:
                    self.queue.put(("text", f"\nError adding file {file_path}: {str(e)}\n"))
                    continue
            
            # Reset progress if not processing
            if not self.is_processing and hasattr(self, 'progress'):
                self.progress["value"] = 0
                if hasattr(self, 'progress_label'):
                    self.progress_label["text"] = "0%"
            
            # Show model info only if not processing
            if not self.is_processing:
                self.show_model_info()
            
            # If we're paused, remind user to resume
            if self.is_paused:
                self.queue.put(("text", "\nFiles added successfully. Click 'Resume' to continue processing.\n"))
                self.queue.put(("status", "Files added. Click 'Resume' to continue"))
                
        except Exception as e:
            self.queue.put(("text", f"\nUnexpected error during file selection: {str(e)}\n"))
            self.queue.put(("status", "Error selecting files"))
            # Reset any state that might have been left in an invalid state
            self.is_processing = False
            self.is_paused = False
            self.start_time = None
            if hasattr(self, 'progress'):
                self.progress["value"] = 0
            if hasattr(self, 'progress_label'):
                self.progress_label["text"] = "0%"
            self.start_btn.configure(state=tk.NORMAL)
            self.pause_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.DISABLED)
            self.select_button.configure(state=tk.NORMAL)

    def update_transcribe_elapsed_time(self):
        """Update the status display during transcription (elapsed time removed per user request)"""
        if self.transcribe_start_time and self.transcribe_filename:
            self.status_label["text"] = f"Transcribing {self.transcribe_filename}..."
            # Schedule next update
            self.transcribe_timer_id = self.root.after(1000, self.update_transcribe_elapsed_time)

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
                elif msg_type == "transcribe_start":
                    filename, start_time = msg_data
                    self.transcribe_filename = filename
                    self.transcribe_start_time = start_time
                    # Cancel any existing timer
                    if self.transcribe_timer_id:
                        self.root.after_cancel(self.transcribe_timer_id)
                    # Start the elapsed time updates
                    self.update_transcribe_elapsed_time()
                elif msg_type == "transcribe_end":
                    # Cancel the elapsed time timer
                    if self.transcribe_timer_id:
                        self.root.after_cancel(self.transcribe_timer_id)
                        self.transcribe_timer_id = None
                    self.transcribe_start_time = None
                    self.transcribe_filename = None
                elif msg_type == "file_status":
                    item_id, status = msg_data
                    if not self.file_list.exists(item_id):
                        continue
                    item = item_id
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
                elif msg_type == "device_state":
                    device_state, compute_state = msg_data
                    self.device_combo.configure(state=device_state)
                    self.compute_combo.configure(state=compute_state)
                elif msg_type == "processing_complete":
                    # Processing completed naturally - show "Done!"
                    self.processing_completed = True
                    self.elapsed_time_label["text"] = "Done!"

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
        device_label = self.device_var.get()
        compute_label = self.compute_var.get()
        self.status_label["text"] = (
            f"Selected model: {model_name} ({model_sizes[model_name]}), "
            f"{device_label}, {compute_label}"
        )

    def on_device_change(self, event=None):
        """Update compute options and time estimates when device changes"""
        self.refresh_compute_options()
        self.update_speed_factors()
        self.show_model_info()

    def on_compute_change(self, event=None):
        """Update time estimates when compute type changes"""
        self.update_speed_factors()
        self.show_model_info()

    def refresh_compute_options(self):
        """Update compute choices based on the selected device"""
        device = self.get_selected_device()
        if device == "cuda":
            compute_choices = [
                "Auto (recommended)",
                "Fast (float16)",
                "Balanced (int8_float16)",
                "Memory Saver (int8)",
                "Precise (float32)"
            ]
        else:
            compute_choices = [
                "Auto (recommended)",
                "Memory Saver (int8)",
                "Precise (float32)"
            ]
        current = self.compute_var.get()
        self.compute_combo["values"] = compute_choices
        if current not in compute_choices:
            self.compute_var.set("Auto (recommended)")

    def update_speed_factors(self):
        """Update model speed estimates based on device/compute selection"""
        device = self.get_selected_device()
        compute_type = self.get_selected_compute_type()

        device_multiplier = {
            "cpu": 1.0,
            "cuda": 3.5,
            "auto": 1.0
        }.get(device, 1.0)

        if device == "cuda":
            compute_multiplier = {
                None: 1.0,
                "float16": 1.2,
                "int8_float16": 1.1,
                "int8": 0.9,
                "float32": 0.85
            }.get(compute_type, 1.0)
        else:
            compute_multiplier = {
                None: 1.0,
                "int8": 1.2,
                "float32": 1.0
            }.get(compute_type, 1.0)

        scale = device_multiplier * compute_multiplier
        self.model_speeds = {
            name: speed * scale for name, speed in self.base_model_speeds.items()
        }

    def refresh_estimates_for_queue(self):
        """Recalculate estimates (removed - estimates were inaccurate)"""
        # Estimates removed per user request
        pass

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
        
        # Check which models are downloaded using faster-whisper's cache location
        downloaded_models = []
        for model in ["tiny", "base", "small", "medium", "large-v3"]:
            model_path = self.get_model_cache_dir(model)
            if os.path.isdir(model_path):
                downloaded_models.append(model)
        
        # Update status (model info removed per user request)
        if not self.is_processing:
            self.status_label["text"] = "Ready"
        
        # Only show full info in text area if it's empty
        if not self.text_area.get(1.0, tk.END).strip():
            # Build model info text
            info_text = ""
            info_text += f"Size: {model_sizes[model_name]}\n"
            info_text += f"Use case: {model_use_cases[model_name]}\n"
            info_text += "The model will be downloaded and run locally with faster-whisper.\n\n"
            
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
        return core_format_timestamp(seconds)

    def format_time_estimate(self, minutes):
        """Format time estimate in HH:MM format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def update_total_time_estimate(self):
        """Update the total time estimate (removed - estimates were inaccurate)"""
        # Estimates removed per user request
        pass

    def update_remaining_time(self):
        """Update the elapsed time display near control buttons"""
        if self.start_time:
            if self.is_paused and self.pause_start_time:
                # Paused - use accumulated time
                elapsed = self.total_elapsed_seconds
            else:
                # Running - calculate from start
                elapsed = int(time.time() - self.start_time) + self.total_elapsed_seconds
            self.elapsed_time_label["text"] = f"Elapsed: {elapsed}s"
            # Keep updating as long as start_time is set
            self.root.after(1000, self.update_remaining_time)
        elif hasattr(self, 'processing_completed') and self.processing_completed:
            # Processing finished naturally - show "Done!"
            self.elapsed_time_label["text"] = "Done!"
            self.processing_completed = False  # Reset flag

    def load_model(self, model_name, device=None, compute_type=None):
        """Load the faster-whisper model with download status"""
        try:
            # Get the model path in faster-whisper's cache
            model_path = self.get_model_cache_dir(model_name)

            # Check if model exists
            if not os.path.isdir(model_path):
                self.queue.put(("show_model_progress", True))
                self.queue.put(("status", f"Downloading {model_name} model..."))
                self.queue.put(("text", f"\nDownloading {model_name} model...\nThis is a one-time download. The model will be stored locally at:\n{model_path}\n\n"))
                self.queue.put(("model_progress", 0))
                self.queue.put(("model_progress_label", "Starting download..."))

            # Load the model
            model = core_load_model(
                model_name,
                device=device or "auto",
                compute_type=compute_type
            )
            
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
                    if len(values) < 5:  # Ensure we have all required values
                        continue
                        
                    status = values[1]
                    # Only change model for pending files
                    if status == "Pending":
                        values[3] = new_model  # Update model
                        self.file_list.item(item, values=values)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
            
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
        """Check if a file is accessible and readable"""
        try:
            # Try to open the file for reading
            with open(file_path, 'rb') as f:
                # Try to read a small chunk to verify access
                f.read(1024)
            return True
        except (IOError, OSError):
            return False

    def get_selected_device(self):
        """Map friendly device label to faster-whisper device string"""
        return self.device_options.get(self.device_var.get(), "auto")

    def get_selected_compute_type(self):
        """Map friendly compute label to faster-whisper compute_type"""
        return self.compute_label_to_type.get(self.compute_var.get())

    def get_model_cache_dir(self, model_name):
        """Get faster-whisper cache directory for a model"""
        cache_root = os.path.expanduser("~/.cache/huggingface/hub")
        return os.path.join(cache_root, f"models--Systran--faster-whisper-{model_name}")

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
