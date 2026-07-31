"""
Voice Fidelity Scorer

Scores a draft against a voice profile on multiple dimensions:
- Vocabulary match (0-100)
- Structure match (0-100)
- Register match (0-100)
- Fingerprint count (lower is better)
- Overall voice fidelity score

The scorer bridges the gap between extraction (what does the voice
look like?) and validation (does this draft match?). It produces a
numeric score that can be used programmatically and a human-readable
report that can be reviewed by the writer.

The scoring is deliberately conservative. A score of 70 means "this
is recognizably in the right voice with room for improvement." A
score of 90 means "a reader familiar with the source voice would
not flag this as written by someone else." A score of 100 is not
the target; perfect scores usually indicate overfitting to the
profile at the expense of natural expression.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from voice_transfer.extractor import (
    VoiceCorpus,
    RegisterProfile,
    VocabularyProfile,
    SentenceProfile,
    PunctuationProfile,
    LATINATE_SUFFIXES,
    SAXON_MARKERS,
)
from voice_transfer.fingerprint_detector import FingerprintDetector, DetectionReport
from voice_transfer.lexicon_brand import LexiconBrand, SelfCritiqueReport


@dataclass
class DimensionScore:
    """Score for a single dimension of voice fidelity."""

    dimension: str
    score: float  # 0-100
    details: str
    weight: float = 1.0

    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class FidelityReport:
    """Complete voice fidelity report."""

    dimensions: list[DimensionScore] = field(default_factory=list)
    fingerprint_report: Optional[DetectionReport] = None
    critique_report: Optional[SelfCritiqueReport] = None
    overall_score: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Voice Fidelity Report ===",
            f"Overall Score: {self.overall_score:.1f}/100",
            "",
        ]

        for dim in self.dimensions:
            bar = self._bar(dim.score)
            lines.append(f"{dim.dimension:25s} {bar} {dim.score:.1f}/100")
            if dim.details:
                lines.append(f"{'':25s} {dim.details}")

        if self.fingerprint_report and not self.fingerprint_report.clean:
            lines.append("")
            lines.append(
                f"LLM Fingerprints: {self.fingerprint_report.total_fingerprints} detected"
            )
            lines.append(
                f"  High severity: {self.fingerprint_report.high_severity_count}"
            )

        if self.critique_report:
            lines.append("")
            lines.append(
                f"Lexicon Brand: {self.critique_report.hard_fails} hard fails, "
                f"{self.critique_report.soft_fails} soft fails"
            )

        lines.append("")
        lines.append(self._verdict())

        return "\n".join(lines)

    def _bar(self, score: float, width: int = 20) -> str:
        """Visual score bar."""
        filled = int(score / 100 * width)
        empty = width - filled
        return f"[{'#' * filled}{'.' * empty}]"

    def _verdict(self) -> str:
        """Human-readable verdict based on overall score."""
        if self.overall_score >= 90:
            return "VERDICT: Strong voice match. Ready to ship with minor review."
        elif self.overall_score >= 75:
            return "VERDICT: Good voice match. Review flagged items before shipping."
        elif self.overall_score >= 60:
            return "VERDICT: Partial voice match. Significant revision needed."
        elif self.overall_score >= 40:
            return "VERDICT: Weak voice match. Consider redrafting from scratch."
        else:
            return "VERDICT: Voice mismatch. This does not sound like the source."


class VoiceFidelityScorer:
    """
    Scores a draft against a voice profile.

    Dimensions scored:
    1. Vocabulary match (latinate ratio, distinctive words)
    2. Sentence structure match (length distribution)
    3. Punctuation match (semicolons, dashes, splices)
    4. Fingerprint score (absence of LLM tics)
    5. Lexicon brand compliance (if brand provided)

    Usage:
        scorer = VoiceFidelityScorer(profile)
        report = scorer.score("Draft text here...")
        print(report.summary())
    """

    def __init__(
        self,
        profile: RegisterProfile,
        fingerprint_detector: Optional[FingerprintDetector] = None,
        lexicon_brand: Optional[LexiconBrand] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.profile = profile
        self.detector = fingerprint_detector or FingerprintDetector()
        self.brand = lexicon_brand
        self.weights = weights or {
            "vocabulary": 1.0,
            "structure": 1.0,
            "punctuation": 0.8,
            "fingerprints": 1.5,
            "lexicon_brand": 1.2,
        }

    def score(self, text: str) -> FidelityReport:
        """
        Score a draft against the voice profile.

        Returns a FidelityReport with dimension scores and overall score.
        """
        report = FidelityReport()

        # 1. Vocabulary match
        vocab_score = self._score_vocabulary(text)
        report.dimensions.append(vocab_score)

        # 2. Sentence structure match
        structure_score = self._score_structure(text)
        report.dimensions.append(structure_score)

        # 3. Punctuation match
        punct_score = self._score_punctuation(text)
        report.dimensions.append(punct_score)

        # 4. Fingerprint detection
        fp_report = self.detector.scan(text)
        report.fingerprint_report = fp_report
        fp_score = self._score_fingerprints(fp_report, text)
        report.dimensions.append(fp_score)

        # 5. Lexicon brand compliance
        if self.brand:
            critique = self.brand.critique(text)
            report.critique_report = critique
            brand_score = self._score_brand(critique)
            report.dimensions.append(brand_score)

        # Compute overall score (weighted average)
        total_weight = sum(d.weight for d in report.dimensions)
        if total_weight > 0:
            report.overall_score = (
                sum(d.weighted_score() for d in report.dimensions) / total_weight
            )
        else:
            report.overall_score = 0.0

        return report

    def _score_vocabulary(self, text: str) -> DimensionScore:
        """Score vocabulary match against profile."""
        words = text.lower().split()
        if not words:
            return DimensionScore("Vocabulary", 0.0, "No text to analyze")

        # Compute latinate ratio of draft
        content_words = [w for w in words if w not in SAXON_MARKERS and len(w) > 3]
        if not content_words:
            return DimensionScore("Vocabulary", 50.0, "No content words found")

        draft_latinate = sum(
            1 for w in content_words
            if any(w.endswith(suf) for suf in LATINATE_SUFFIXES)
        ) / len(content_words)

        target_latinate = self.profile.vocabulary.latinate_ratio

        # Score based on distance from target
        distance = abs(draft_latinate - target_latinate)
        # Perfect match = 100, each 0.05 difference = -10 points
        vocab_score = max(0, 100 - distance * 200)

        # Bonus for using distinctive words from the profile
        draft_word_set = set(words)
        distinctive_used = sum(
            1 for w in self.profile.vocabulary.distinctive_words
            if w in draft_word_set
        )
        if self.profile.vocabulary.distinctive_words:
            distinctive_ratio = distinctive_used / len(
                self.profile.vocabulary.distinctive_words
            )
            vocab_score = min(100, vocab_score + distinctive_ratio * 20)

        details = (
            f"Draft latinate ratio: {draft_latinate:.2%} "
            f"(target: {target_latinate:.2%})"
        )

        return DimensionScore(
            "Vocabulary",
            vocab_score,
            details,
            weight=self.weights.get("vocabulary", 1.0),
        )

    def _score_structure(self, text: str) -> DimensionScore:
        """Score sentence structure match against profile."""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if len(sentences) < 2:
            return DimensionScore("Structure", 50.0, "Not enough sentences to analyze")

        lengths = [len(s.split()) for s in sentences if len(s.split()) >= 2]
        if not lengths:
            return DimensionScore("Structure", 50.0, "No valid sentences found")

        draft_mean = statistics.mean(lengths)
        target_mean = self.profile.sentences.mean_length

        # Score based on mean length distance
        mean_distance = abs(draft_mean - target_mean)
        mean_score = max(0, 100 - mean_distance * 5)

        # Score based on variation similarity
        if len(lengths) > 1 and self.profile.sentences.std_dev > 0:
            draft_std = statistics.stdev(lengths)
            target_std = self.profile.sentences.std_dev
            std_distance = abs(draft_std - target_std)
            std_score = max(0, 100 - std_distance * 5)
        else:
            std_score = 50.0

        structure_score = mean_score * 0.6 + std_score * 0.4

        details = (
            f"Draft mean length: {draft_mean:.1f} "
            f"(target: {target_mean:.1f})"
        )

        return DimensionScore(
            "Structure",
            structure_score,
            details,
            weight=self.weights.get("structure", 1.0),
        )

    def _score_punctuation(self, text: str) -> DimensionScore:
        """Score punctuation usage match against profile."""
        word_count = len(text.split())
        if word_count == 0:
            return DimensionScore("Punctuation", 50.0, "No text to analyze")

        factor = 1000.0 / word_count

        draft_semicolons = text.count(";") * factor
        draft_emdashes = (
            text.count("\u2014") + len(re.findall(r"(?<!\-)--(?!\-)", text))
        ) * factor

        target_semicolons = self.profile.punctuation.semicolons_per_1000_words
        target_emdashes = self.profile.punctuation.em_dashes_per_1000_words

        # Score semicolons
        semi_distance = abs(draft_semicolons - target_semicolons)
        semi_score = max(0, 100 - semi_distance * 15)

        # Score em-dashes
        dash_distance = abs(draft_emdashes - target_emdashes)
        dash_score = max(0, 100 - dash_distance * 15)

        punct_score = semi_score * 0.5 + dash_score * 0.5

        details = (
            f"Semicolons: {draft_semicolons:.1f}/1000w "
            f"(target: {target_semicolons:.1f}); "
            f"Em-dashes: {draft_emdashes:.1f}/1000w "
            f"(target: {target_emdashes:.1f})"
        )

        return DimensionScore(
            "Punctuation",
            punct_score,
            details,
            weight=self.weights.get("punctuation", 0.8),
        )

    def _score_fingerprints(
        self, fp_report: DetectionReport, text: str
    ) -> DimensionScore:
        """Score based on absence of LLM fingerprints."""
        word_count = len(text.split())
        if word_count == 0:
            return DimensionScore("Fingerprints", 100.0, "No text to analyze")

        # Start at 100, subtract for each fingerprint
        score = 100.0
        score -= fp_report.high_severity_count * 15
        score -= fp_report.medium_severity_count * 8
        score -= fp_report.low_severity_count * 3
        score = max(0, score)

        details = (
            f"{fp_report.total_fingerprints} fingerprints "
            f"(H:{fp_report.high_severity_count} "
            f"M:{fp_report.medium_severity_count} "
            f"L:{fp_report.low_severity_count})"
        )

        return DimensionScore(
            "Fingerprints",
            score,
            details,
            weight=self.weights.get("fingerprints", 1.5),
        )

    def _score_brand(self, critique: SelfCritiqueReport) -> DimensionScore:
        """Score based on lexicon brand compliance."""
        score = critique.overall_score * 100

        details = (
            f"Brand score: {critique.overall_score:.2f}, "
            f"{critique.hard_fails} hard fails, "
            f"{critique.soft_fails} soft fails"
        )

        return DimensionScore(
            "Lexicon Brand",
            score,
            details,
            weight=self.weights.get("lexicon_brand", 1.2),
        )
