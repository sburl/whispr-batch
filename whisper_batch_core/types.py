from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    @classmethod
    def from_whisper(cls, segment: object) -> "TranscriptSegment":
        return cls(
            start=float(segment.start),
            end=float(segment.end),
            text=str(segment.text),
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: List[TranscriptSegment]
    info: object
