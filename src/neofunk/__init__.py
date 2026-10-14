"""neofunk — ubah suara mulut (humming, nyanyi, "solo gitar" pakai mulut) jadi not musik."""

from neofunk.notes import Note, Transcription
from neofunk.pipeline import TranscribeConfig, transcribe

__all__ = ["Note", "Transcription", "TranscribeConfig", "transcribe"]
__version__ = "0.1.0"
