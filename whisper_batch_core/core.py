from typing import Iterable, List, Optional, Tuple

import platform

from faster_whisper import WhisperModel

from .types import TranscriptSegment, TranscriptionResult


def load_model(model_name: str, device: str = "auto", compute_type: Optional[str] = None) -> WhisperModel:
    """Load a faster-whisper model with the requested device/compute settings."""
    if device == "auto" and platform.system() == "Darwin" and platform.machine() == "arm64":
        kwargs = {"device": "cpu", "compute_type": "int8"}
    else:
        kwargs = {"device": device}
    if compute_type:
        kwargs["compute_type"] = compute_type
    return WhisperModel(model_name, **kwargs)


def transcribe_segments(
    model: WhisperModel,
    audio_path: str,
    task: str = "transcribe"
) -> Tuple[List[TranscriptSegment], object]:
    """Transcribe audio and return a list of segments plus metadata info."""
    segments, info = model.transcribe(audio_path, task=task)
    return [TranscriptSegment.from_whisper(segment) for segment in segments], info


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_timestamped_text(segments: Iterable[TranscriptSegment]) -> str:
    """Render transcript with per-segment timestamps."""
    formatted_text = []
    for segment in segments:
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        formatted_text.append(f"[{start_time} --> {end_time}] {text}")
    return "\n".join(formatted_text)


def render_plain_text(segments: Iterable[TranscriptSegment]) -> str:
    """Render transcript as a single text block without timestamps."""
    text_parts = [segment.text.strip() for segment in segments]
    return " ".join(text_parts).strip()


def transcribe_file(
    audio_path: str,
    model_name: str = "large-v3",
    include_timestamps: bool = True,
    device: str = "auto",
    compute_type: Optional[str] = None,
    model: Optional[WhisperModel] = None,
    task: str = "transcribe"
) -> TranscriptionResult:
    """Transcribe a single audio file and return text plus segments metadata."""
    model = model or load_model(model_name, device=device, compute_type=compute_type)
    segments, info = transcribe_segments(model, audio_path, task=task)
    if include_timestamps:
        text = render_timestamped_text(segments)
    else:
        text = render_plain_text(segments)
    return TranscriptionResult(text=text, segments=segments, info=info)
