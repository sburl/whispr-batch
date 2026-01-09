# WhisperBatch

A Python package and GUI application for transcribing audio files using [faster-whisper](https://github.com/Systran/faster-whisper). Use the GUI for batch processing with a user-friendly interface, or install the package to use the transcription API in your own Python projects.

---

## Highlights
- **Multiple Model Support** – tiny, base, small, medium, large-v3
- **Batch Queue** – add/reorder/remove files while paused
- **Progress & ETA** – per-file status plus global remaining-time estimate
- **Timestamps** – optional per-segment timecodes
- **Cross-platform** – macOS, Linux, Windows (Python ≥ 3.8)

---

## Quick-start (all platforms)
```bash
# clone project
cd AudioTranscribe

# one-step setup (creates .venv, installs deps, handles Apple-silicon quirks)
chmod +x setup.sh
./setup.sh

# activate env & run GUI
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python transcribe_gui.py
```

---

## Apple-Silicon notes (M-series Macs)
1. The setup script auto-detects arm64 and ensures the **native** PyTorch CPU wheel is installed (`torch==2.1.0`).
2. When you choose **Device = Auto** (default) the program now *automatically falls back* to **CPU + int8** compute-type. This avoids current CTranslate2/Metal seg-faults while still running ~2× real-time on an M1/M2/M3.
3. Once a stable CTranslate2 Metal backend is released the GUI will switch back to GPU automatically.

If you ever see the linker error below you are running an x86-64 wheel under Rosetta:
```
macOS 26 (2601) or later required, have instead 16 (1601) !
```
Fix with:
```bash
pip uninstall -y torch
pip install --no-cache-dir --force-reinstall torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Requirements
- Python 3.8 – 3.12 (3.13 currently lacks binary wheels for NumPy/PyTorch)
- FFmpeg in PATH
- Required pip packages (installed by `setup.sh`):
  - faster-whisper
  - torch (CPU wheel by default)
  - librosa, numpy, tqdm, requests

---

## Running headless CLI
Batch-transcribe an entire directory without the GUI:
```bash
python transcribe_audio.py /path/to/folder --model base
```
(Add `--no-timestamps` to disable timestamps.)

---

## Core Package

The non-GUI transcription utilities live in `audio_transcribe_core/`. Install locally:

```bash
pip install -e .
```

Example usage:

```python
from audio_transcribe_core import transcribe_file

result = transcribe_file("path/to/audio.wav", model_name="base")
text = result.text
```

API options:
- `model_name`: model size such as `tiny`, `base`, `small`, `medium`, `large-v3`
- `device`: `auto`, `cpu`, or `cuda`
- `compute_type`: `float16`, `int8_float16`, `int8`, or `float32`
- `include_timestamps`: include segment timestamps in the output text
- `task`: `transcribe` or `translate`

---

## Troubleshooting
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: _tkinter` | Reinstall Homebrew `python@3.x` **after** `brew install tcl-tk`, or use `/usr/bin/python3`. |
| `macOS 26 / 16` loader abort | You installed an x86-64 PyTorch wheel – reinstall arm64 CPU wheel (see above). |
| Segmentation fault on model load | Automatically mitigated by CPU fallback; update faster-whisper & CTranslate2 when new GPU wheels land. |

---

MIT License © 2026
