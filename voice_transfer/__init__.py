"""
AI Voice Transfer Framework

Make any LLM write like a specific human, not like an AI.

Architecture:
    Voice Corpus -> Register Library -> Lexicon Brand -> Draft ->
    Fingerprint Scan -> Self-Critique -> Ship

Modules:
    extractor: Voice extraction from human-written corpus
    register_library: Multiple voice modes for different contexts
    lexicon_brand: Prescriptive voice rules with executable tests
    fingerprint_detector: LLM writing tic detection and scoring
    scorer: Voice fidelity scoring against a profile
    pipeline: End-to-end voice transfer pipeline
"""

__version__ = "0.1.0"
__author__ = "David Dabert"

from voice_transfer.extractor import VoiceCorpus, RegisterProfile
from voice_transfer.register_library import Register, RegisterLibrary
from voice_transfer.lexicon_brand import LexiconModule, LexiconBrand
from voice_transfer.fingerprint_detector import FingerprintDetector, FingerprintMatch
from voice_transfer.scorer import VoiceFidelityScorer, FidelityReport
from voice_transfer.pipeline import VoiceTransferPipeline, DraftRequest, DraftResult

__all__ = [
    "VoiceCorpus",
    "RegisterProfile",
    "Register",
    "RegisterLibrary",
    "LexiconModule",
    "LexiconBrand",
    "FingerprintDetector",
    "FingerprintMatch",
    "VoiceFidelityScorer",
    "FidelityReport",
    "VoiceTransferPipeline",
    "DraftRequest",
    "DraftResult",
]
