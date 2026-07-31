"""
LLM Fingerprint Detector

Detects and flags common AI writing patterns that make text
sound machine-generated rather than human-written.

Based on empirical observation of ten recurring LLM tics,
cross-referenced with:
- Register Analysis for Style Transfer (arXiv 2505.00679, 2025)
- Voice Cloning is Style Transfer (arXiv 2605.16578, 2025)
- Cultural Marker Erasure (arXiv 2602.22145, February 2026)

The core insight: LLMs do not produce "bad" writing. They produce
writing that converges toward the statistical center of their training
distribution. That center sounds like a competent, anonymous, mildly
enthusiastic professional writer. The problem is not quality but
identity: every LLM output sounds the same, and none of them sound
like the person who asked.

The fingerprints catalogued here are not errors. They are the specific
stylistic moves that mark text as LLM-generated to a reader who has
seen enough AI output to recognize the patterns. Each fingerprint has
a detection method (regex or heuristic), a severity score, and a
suggested fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FingerprintMatch:
    """A single detected LLM fingerprint in a text."""

    fingerprint_id: int
    fingerprint_name: str
    description: str
    matched_text: str
    position: int  # character position in the source text
    severity: float  # 0.0 to 1.0
    suggested_fix: str

    def __str__(self) -> str:
        sev_label = "HIGH" if self.severity >= 0.7 else "MED" if self.severity >= 0.4 else "LOW"
        return (
            f"[{sev_label}] #{self.fingerprint_id} {self.fingerprint_name}\n"
            f"  Found: \"{self.matched_text[:100]}\"\n"
            f"  Fix: {self.suggested_fix}"
        )


@dataclass
class DetectionReport:
    """Full fingerprint detection report for a text."""

    matches: list[FingerprintMatch] = field(default_factory=list)
    total_fingerprints: int = 0
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    clean: bool = True

    @property
    def fingerprint_density(self) -> float:
        """Fingerprints per 100 words (lower is better)."""
        return self.total_fingerprints  # caller should normalize by word count

    def summary(self) -> str:
        """Human-readable summary."""
        if self.clean:
            return "No LLM fingerprints detected."

        lines = [
            f"=== Fingerprint Detection Report ===",
            f"Total fingerprints: {self.total_fingerprints}",
            f"  High severity: {self.high_severity_count}",
            f"  Medium severity: {self.medium_severity_count}",
            f"  Low severity: {self.low_severity_count}",
            "",
        ]
        for match in self.matches:
            lines.append(str(match))
            lines.append("")

        return "\n".join(lines)


class FingerprintDetector:
    """
    Detects LLM writing fingerprints in text.

    The ten canonical fingerprints:
    1. Staccato three-word closes
    2. Hook-then-reveal openers
    3. "Here is what makes this different" formulas
    4. "The [class] you would [verb] is that..." constructions
    5. Semicolon-list credential dumps
    6. Aphoristic closes
    7. Em-dash overuse
    8. Theatrical tricolons
    9. Generic-plausible specificity
    10. Mechanical parallelism

    Usage:
        detector = FingerprintDetector()
        report = detector.scan("text to analyze...")
        print(report.summary())
    """

    def __init__(self, custom_patterns: Optional[list[dict]] = None) -> None:
        self._detectors = self._build_default_detectors()
        if custom_patterns:
            for p in custom_patterns:
                self.add_pattern(
                    fingerprint_id=p.get("id", 99),
                    name=p["name"],
                    pattern=p["pattern"],
                    severity=p.get("severity", 0.5),
                    fix=p.get("fix", ""),
                    description=p.get("description", ""),
                    is_regex=p.get("is_regex", True),
                )

    def add_pattern(
        self,
        fingerprint_id: int,
        name: str,
        pattern: str,
        severity: float = 0.5,
        fix: str = "",
        description: str = "",
        is_regex: bool = True,
    ) -> None:
        """Add a custom fingerprint detection pattern."""
        self._detectors.append({
            "id": fingerprint_id,
            "name": name,
            "pattern": pattern,
            "severity": severity,
            "fix": fix,
            "description": description or name,
            "is_regex": is_regex,
        })

    def scan(self, text: str) -> DetectionReport:
        """
        Scan text for all known LLM fingerprints.

        Returns a DetectionReport with all matches.
        """
        report = DetectionReport()

        # Run each detector
        for detector in self._detectors:
            matches = self._run_detector(text, detector)
            report.matches.extend(matches)

        # Run heuristic detectors that cannot be expressed as simple regex
        report.matches.extend(self._detect_staccato_close(text))
        report.matches.extend(self._detect_aphoristic_close(text))
        report.matches.extend(self._detect_emdash_overuse(text))
        report.matches.extend(self._detect_theatrical_tricolon(text))
        report.matches.extend(self._detect_generic_specificity(text))
        report.matches.extend(self._detect_credential_dump(text))

        # Compute summary statistics
        report.total_fingerprints = len(report.matches)
        report.high_severity_count = sum(
            1 for m in report.matches if m.severity >= 0.7
        )
        report.medium_severity_count = sum(
            1 for m in report.matches if 0.4 <= m.severity < 0.7
        )
        report.low_severity_count = sum(
            1 for m in report.matches if m.severity < 0.4
        )
        report.clean = report.total_fingerprints == 0

        return report

    def _run_detector(
        self, text: str, detector: dict
    ) -> list[FingerprintMatch]:
        """Run a single regex-based detector."""
        matches = []

        if not detector.get("is_regex", True):
            # Simple string search
            pattern_lower = detector["pattern"].lower()
            text_lower = text.lower()
            start = 0
            while True:
                idx = text_lower.find(pattern_lower, start)
                if idx == -1:
                    break
                end = min(len(text), idx + len(detector["pattern"]) + 30)
                context_start = max(0, idx - 15)
                matched = text[context_start:end]
                matches.append(FingerprintMatch(
                    fingerprint_id=detector["id"],
                    fingerprint_name=detector["name"],
                    description=detector["description"],
                    matched_text=matched,
                    position=idx,
                    severity=detector["severity"],
                    suggested_fix=detector["fix"],
                ))
                start = idx + len(detector["pattern"])
        else:
            try:
                for match in re.finditer(
                    detector["pattern"], text, re.IGNORECASE | re.MULTILINE
                ):
                    matches.append(FingerprintMatch(
                        fingerprint_id=detector["id"],
                        fingerprint_name=detector["name"],
                        description=detector["description"],
                        matched_text=match.group()[:120],
                        position=match.start(),
                        severity=detector["severity"],
                        suggested_fix=detector["fix"],
                    ))
            except re.error:
                pass  # skip malformed regex

        return matches

    def _build_default_detectors(self) -> list[dict]:
        """Build the default set of regex-based fingerprint detectors."""
        return [
            # 2. Hook-then-reveal openers
            {
                "id": 2,
                "name": "hook_then_reveal",
                "description": (
                    "Hook-then-reveal opener: 'X is the theatre. Y is "
                    "the physics.' A two-sentence construction where the "
                    "first sentence names a surface phenomenon and the "
                    "second names the 'real' mechanism underneath."
                ),
                "pattern": (
                    r"^[A-Z][^.!?]{5,40}\s+is\s+the\s+\w+\.\s+"
                    r"[A-Z][^.!?]{5,40}\s+is\s+the\s+\w+\."
                ),
                "severity": 0.8,
                "fix": (
                    "Open with the thesis directly. Do not stage the "
                    "insight with a two-part metaphor."
                ),
                "is_regex": True,
            },

            # 3. "Here is what makes this different"
            {
                "id": 3,
                "name": "here_is_what_makes_different",
                "description": (
                    "'Here is what makes this different' formula. "
                    "A self-referential claim of uniqueness that "
                    "substitutes assertion for evidence."
                ),
                "pattern": (
                    r"here\s+is\s+what\s+makes?\s+(?:this|it|the|that)"
                    r"\s+(?:different|unique|special|stand\s+out|matter)"
                ),
                "severity": 0.9,
                "fix": (
                    "Delete the formula. Show the difference through "
                    "evidence and let the reader conclude."
                ),
                "is_regex": True,
            },

            # 4. "The [class] you would [verb] is that..."
            {
                "id": 4,
                "name": "the_class_you_would",
                "description": (
                    "'The [class] you would [verb] is that...' "
                    "construction. Positions the reader as needing "
                    "a tour guide."
                ),
                "pattern": (
                    r"the\s+\w+\s+you\s+would\s+\w+\s+is\s+that"
                ),
                "severity": 0.7,
                "fix": (
                    "State the point directly. 'The pattern is X' "
                    "rather than 'The pattern you would notice is that X.'"
                ),
                "is_regex": True,
            },

            # 10. Mechanical parallelism
            {
                "id": 10,
                "name": "mechanical_parallelism",
                "description": (
                    "Mechanical parallelism: 'On one hand X. On the "
                    "other hand Y.' Signals balanced-sounding analysis "
                    "that is actually formulaic."
                ),
                "pattern": (
                    r"on\s+(?:the\s+)?one\s+hand\b.{5,100}?"
                    r"on\s+the\s+other\s+hand\b"
                ),
                "severity": 0.6,
                "fix": (
                    "State both positions without the mechanical frame. "
                    "'X is true. But Y complicates it.' is more natural."
                ),
                "is_regex": True,
            },

            # LLM vocabulary fingerprints (supplementary)
            {
                "id": 11,
                "name": "llm_vocab_delve",
                "description": "'Delve' is the most overrepresented word in LLM output.",
                "pattern": r"\bdelv(?:e|es|ing|ed)\b",
                "severity": 0.9,
                "fix": "Replace with 'examine,' 'look at,' 'explore,' or delete.",
                "is_regex": True,
            },
            {
                "id": 12,
                "name": "llm_vocab_landscape",
                "description": "'Landscape' as metaphor (the X landscape) is an LLM fingerprint.",
                "pattern": r"\b(?:the|this|today'?s?|current|evolving)\s+\w*\s*landscape\b",
                "severity": 0.7,
                "fix": "Replace with a concrete description of the domain.",
                "is_regex": True,
            },
            {
                "id": 13,
                "name": "llm_vocab_tapestry",
                "description": "'Tapestry' as metaphor is heavily overrepresented in LLM output.",
                "pattern": r"\btapestry\b",
                "severity": 0.8,
                "fix": "Replace with a concrete noun.",
                "is_regex": True,
            },
            {
                "id": 14,
                "name": "llm_vocab_beacon",
                "description": "'Beacon' as metaphor is heavily overrepresented in LLM output.",
                "pattern": r"\bbeacon\b",
                "severity": 0.7,
                "fix": "Replace with 'example,' 'model,' or a concrete description.",
                "is_regex": True,
            },
            {
                "id": 15,
                "name": "llm_vocab_navigate",
                "description": "'Navigate' (metaphorical) is overrepresented in LLM output.",
                "pattern": r"\bnavigate?(?:s|d|ing)?\s+(?:the|this|these|those|complex|challenging)\b",
                "severity": 0.6,
                "fix": "Replace with 'handle,' 'manage,' 'work through,' or a domain-specific verb.",
                "is_regex": True,
            },
            {
                "id": 16,
                "name": "llm_vocab_foster",
                "description": "'Foster' is overrepresented in LLM output (especially 'fostering').",
                "pattern": r"\bfoster(?:s|ed|ing)?\b",
                "severity": 0.5,
                "fix": "Replace with 'build,' 'create,' 'support,' 'encourage.'",
                "is_regex": True,
            },
            {
                "id": 17,
                "name": "llm_vocab_leverage",
                "description": "'Leverage' (verb) is overrepresented in LLM output.",
                "pattern": r"\bleverag(?:e|es|ed|ing)\b",
                "severity": 0.6,
                "fix": "Replace with 'use,' 'apply,' 'draw on.'",
                "is_regex": True,
            },
            {
                "id": 18,
                "name": "llm_vocab_robust",
                "description": "'Robust' is overrepresented in LLM output.",
                "pattern": r"\brobust\b",
                "severity": 0.5,
                "fix": "Replace with 'strong,' 'solid,' 'reliable,' or a specific quality.",
                "is_regex": True,
            },
            {
                "id": 19,
                "name": "llm_vocab_holistic",
                "description": "'Holistic' is overrepresented in LLM output.",
                "pattern": r"\bholistic(?:ally)?\b",
                "severity": 0.6,
                "fix": "Replace with 'complete,' 'whole-system,' or describe what you mean concretely.",
                "is_regex": True,
            },
            {
                "id": 20,
                "name": "llm_vocab_nuanced",
                "description": "'Nuanced' is overrepresented in LLM output.",
                "pattern": r"\bnuanced?\b",
                "severity": 0.5,
                "fix": "Delete, or describe the actual nuance.",
                "is_regex": True,
            },
            {
                "id": 21,
                "name": "llm_opener_in_todays",
                "description": "'In today's [X]' opener is an LLM fingerprint.",
                "pattern": r"^In\s+today'?s?\s+\w+",
                "severity": 0.7,
                "fix": "Open with the thesis, not with a temporal frame.",
                "is_regex": True,
            },
            {
                "id": 22,
                "name": "llm_opener_in_era",
                "description": "'In an era of/where' opener is an LLM fingerprint.",
                "pattern": r"^In\s+an?\s+(?:era|age|world|time)\s+(?:of|where|when)",
                "severity": 0.7,
                "fix": "Open with the thesis directly.",
                "is_regex": True,
            },
            {
                "id": 23,
                "name": "llm_excited_opener",
                "description": "'Excited to/Thrilled to/Honored to' opener is an LLM fingerprint.",
                "pattern": r"^(?:I'?m?\s+)?(?:excited|thrilled|honored|humbled|delighted)\s+to\b",
                "severity": 0.8,
                "fix": "State what you did or what happened. No performative emotion.",
                "is_regex": True,
            },
            {
                "id": 24,
                "name": "llm_intersection",
                "description": "'At the intersection of X and Y' is an LLM fingerprint.",
                "pattern": r"at\s+the\s+intersection\s+of\b",
                "severity": 0.8,
                "fix": "Name the specific overlap. 'Combining X and Y' or just state the work.",
                "is_regex": True,
            },
            {
                "id": 25,
                "name": "llm_deep_dive",
                "description": "'Deep dive' is overrepresented in LLM output.",
                "pattern": r"\bdeep\s+div(?:e|es|ed|ing)\b",
                "severity": 0.6,
                "fix": "Replace with 'detailed analysis,' 'close look,' or 'examination.'",
                "is_regex": True,
            },
            {
                "id": 26,
                "name": "llm_unpack",
                "description": "'Unpack' (metaphorical) is overrepresented in LLM output.",
                "pattern": r"\bunpack(?:s|ed|ing)?\s+(?:this|that|the|what|how|why)\b",
                "severity": 0.5,
                "fix": "Replace with 'explain,' 'examine,' or just state the content.",
                "is_regex": True,
            },
            {
                "id": 27,
                "name": "llm_resonate",
                "description": "'Resonate(s)' is overrepresented in LLM output.",
                "pattern": r"\bresonat(?:e|es|ed|ing)\b",
                "severity": 0.5,
                "fix": "Delete or replace with a concrete description of the effect.",
                "is_regex": True,
            },
            {
                "id": 28,
                "name": "llm_underscore",
                "description": "'Underscore(s)' (verb) is overrepresented in LLM output.",
                "pattern": r"\bunderscore?(?:s|d|ing)?\b",
                "severity": 0.4,
                "fix": "Replace with 'shows,' 'confirms,' 'proves.'",
                "is_regex": True,
            },
        ]

    # --- Heuristic detectors (not expressible as simple regex) ---

    def _detect_staccato_close(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 1: Staccato three-word closes.

        Pattern: "Bold. Brave. Necessary." or "Simple. Direct. Powerful."
        Three short sentences (1-3 words each) in sequence at or near
        the end of the text, used for rhetorical punch.
        """
        matches = []
        lines = text.strip().split("\n")
        if not lines:
            return matches

        # Check last 3 lines
        tail = lines[-3:] if len(lines) >= 3 else lines
        tail_text = " ".join(l.strip() for l in tail)

        # Pattern: Word. Word. Word. (each 1-3 words)
        staccato_pattern = re.compile(
            r"(?:^|\.\s+)"
            r"([A-Z][a-z]{1,12}(?:\s+\w{1,10}){0,2})\.\s+"
            r"([A-Z][a-z]{1,12}(?:\s+\w{1,10}){0,2})\.\s+"
            r"([A-Z][a-z]{1,12}(?:\s+\w{1,10}){0,2})\.\s*$"
        )

        m = staccato_pattern.search(tail_text)
        if m:
            matched = f"{m.group(1)}. {m.group(2)}. {m.group(3)}."
            matches.append(FingerprintMatch(
                fingerprint_id=1,
                fingerprint_name="staccato_close",
                description=(
                    "Staccato three-word close. Three short punchy "
                    "sentences used for rhetorical emphasis at the end. "
                    "This is the single most recognizable LLM writing tic."
                ),
                matched_text=matched,
                position=len(text) - len(tail_text) + (m.start() if m else 0),
                severity=0.9,
                suggested_fix=(
                    "Delete the staccato close entirely. End with the "
                    "final substantive point or a concrete next step."
                ),
            ))

        return matches

    def _detect_aphoristic_close(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 6: Aphoristic closes.

        Pattern: ending with a fortune-cookie sentence that could
        apply to any topic. "And that's what real leadership looks like."
        "The best time to plant a tree was twenty years ago."
        """
        matches = []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return matches

        last = sentences[-1].strip()
        last_lower = last.lower()

        aphoristic_patterns = [
            r"and\s+that'?s?\s+(?:what|why|how|where)\s+.{5,}",
            r"the\s+(?:real|true|best|greatest|only)\s+.{5,30}\s+is\s+",
            r"at\s+the\s+end\s+of\s+the\s+day",
            r"when\s+all\s+is\s+said\s+and\s+done",
            r"this\s+is\s+(?:what|why|how)\s+(?:real|true)\s+",
            r"(?:that|this)\s+makes?\s+all\s+the\s+difference",
            r"the\s+rest\s+(?:is|will\s+be)\s+(?:history|details)",
        ]

        for pattern in aphoristic_patterns:
            if re.search(pattern, last_lower):
                matches.append(FingerprintMatch(
                    fingerprint_id=6,
                    fingerprint_name="aphoristic_close",
                    description=(
                        "Aphoristic close: a fortune-cookie wisdom "
                        "sentence at the end that could apply to any topic."
                    ),
                    matched_text=last[:100],
                    position=len(text) - len(last),
                    severity=0.7,
                    suggested_fix=(
                        "Delete the aphorism. The last sentence should "
                        "be the final piece of evidence, a concrete "
                        "proposal, or the restated thesis."
                    ),
                ))
                break

        return matches

    def _detect_emdash_overuse(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 7: Em-dash overuse.

        LLMs insert em-dash parentheticals at a much higher rate than
        most human writers. More than 3 per 1000 words is suspicious;
        more than 5 is almost certainly LLM.
        """
        matches = []
        em_dashes = text.count("\u2014")
        double_hyphens = len(re.findall(r"(?<!\-)--(?!\-)", text))
        total_dashes = em_dashes + double_hyphens
        word_count = len(text.split())

        if word_count > 50 and total_dashes > 0:
            per_1000 = total_dashes / word_count * 1000
            if per_1000 > 5.0:
                matches.append(FingerprintMatch(
                    fingerprint_id=7,
                    fingerprint_name="emdash_overuse",
                    description=(
                        f"Em-dash overuse: {total_dashes} em-dashes in "
                        f"{word_count} words ({per_1000:.1f} per 1000). "
                        f"Threshold: 5.0 per 1000 is almost certainly LLM."
                    ),
                    matched_text=f"({total_dashes} em-dashes detected)",
                    position=0,
                    severity=0.7,
                    suggested_fix=(
                        "Replace most em-dashes with semicolons, commas, "
                        "or restructure to eliminate the parenthetical."
                    ),
                ))
            elif per_1000 > 3.0:
                matches.append(FingerprintMatch(
                    fingerprint_id=7,
                    fingerprint_name="emdash_overuse",
                    description=(
                        f"Elevated em-dash usage: {total_dashes} in "
                        f"{word_count} words ({per_1000:.1f} per 1000). "
                        f"Above the 3.0 per 1000 caution threshold."
                    ),
                    matched_text=f"({total_dashes} em-dashes detected)",
                    position=0,
                    severity=0.4,
                    suggested_fix=(
                        "Consider reducing em-dash usage. Check if the "
                        "source voice actually uses this many."
                    ),
                ))

        return matches

    def _detect_theatrical_tricolon(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 8: Theatrical tricolons.

        Three items listed for rhythm rather than meaning.
        "She was smart, determined, and relentless."
        The third item adds no information; it exists for cadence.
        """
        matches = []

        # Pattern: adjective, adjective, and adjective
        tricolon_pattern = re.compile(
            r"\b(\w+),\s+(\w+),\s+and\s+(\w+)\b"
        )

        for m in tricolon_pattern.finditer(text):
            word1, word2, word3 = m.group(1), m.group(2), m.group(3)
            # Heuristic: if all three words are roughly the same length
            # and all could be adjectives, this is suspicious
            lengths = [len(word1), len(word2), len(word3)]
            if all(3 <= l <= 12 for l in lengths):
                # Check if they share suffixes (adjective-like)
                adj_suffixes = ("ed", "ful", "ive", "ous", "ant", "ent", "al", "ic", "able", "ible", "less")
                adj_count = sum(
                    1 for w in (word1, word2, word3)
                    if any(w.lower().endswith(s) for s in adj_suffixes)
                )
                if adj_count >= 2:
                    matches.append(FingerprintMatch(
                        fingerprint_id=8,
                        fingerprint_name="theatrical_tricolon",
                        description=(
                            "Theatrical tricolon: three items listed for "
                            "rhythm rather than meaning. The third item "
                            "often adds no information."
                        ),
                        matched_text=m.group(),
                        position=m.start(),
                        severity=0.5,
                        suggested_fix=(
                            "Keep only the two items that carry distinct "
                            "meaning. If the third adds nothing, cut it."
                        ),
                    ))

        return matches

    def _detect_generic_specificity(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 9: Generic-plausible specificity.

        Text that sounds specific but could apply to anyone or any
        organization. "With over a decade of experience in the field,
        their approach combines rigorous analysis with practical
        implementation."
        """
        matches = []

        generic_patterns = [
            (
                r"with\s+(?:over|more\s+than)\s+(?:a\s+)?decade\s+of\s+experience",
                "Generic decade-of-experience claim",
            ),
            (
                r"(?:combines?|blends?|marries?|bridges?)\s+\w+\s+(?:with|and)\s+\w+",
                "Generic 'combines X with Y' claim",
            ),
            (
                r"(?:passionate|passionate\s+about|deeply\s+committed\s+to)\s+\w+",
                "Generic passion/commitment claim",
            ),
            (
                r"(?:unique|unparalleled|exceptional)\s+(?:blend|combination|mix)\s+of",
                "Generic 'unique blend of' claim",
            ),
            (
                r"track\s+record\s+of\s+(?:success|delivering|achieving|driving)",
                "Generic track-record claim",
            ),
            (
                r"proven\s+(?:ability|track\s+record|expertise|results)\s+in",
                "Generic 'proven ability' claim",
            ),
        ]

        for pattern, desc in generic_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(FingerprintMatch(
                    fingerprint_id=9,
                    fingerprint_name="generic_specificity",
                    description=(
                        f"Generic-plausible specificity: {desc}. "
                        f"Sounds specific but could describe anyone."
                    ),
                    matched_text=m.group()[:80],
                    position=m.start(),
                    severity=0.6,
                    suggested_fix=(
                        "Replace with a concrete, falsifiable claim. "
                        "Name the specific programme, the specific number, "
                        "the specific outcome."
                    ),
                ))

        return matches

    def _detect_credential_dump(self, text: str) -> list[FingerprintMatch]:
        """
        Fingerprint 5: Semicolon-list credential dumps.

        "multilingual; cross-cultural; senior-tested; governance-ready"
        A list of adjectives or credentials separated by semicolons,
        deployed for density rather than argument.
        """
        matches = []

        # Pattern: three or more semicolon-separated items of similar length
        # that look like credential lists
        semicolon_list_pattern = re.compile(
            r"((?:\w[\w\s-]{2,25};\s*){2,}\w[\w\s-]{2,25})"
        )

        for m in semicolon_list_pattern.finditer(text):
            items = [i.strip() for i in m.group().split(";")]
            # Check if items look like credentials (short, no verbs)
            if all(len(item.split()) <= 4 for item in items if item):
                if len(items) >= 3:
                    matches.append(FingerprintMatch(
                        fingerprint_id=5,
                        fingerprint_name="credential_dump",
                        description=(
                            "Semicolon-list credential dump: a list of "
                            "adjectives or credentials separated by "
                            "semicolons, deployed for density."
                        ),
                        matched_text=m.group()[:100],
                        position=m.start(),
                        severity=0.6,
                        suggested_fix=(
                            "Integrate the credentials into a sentence "
                            "where each one earns its place with evidence. "
                            "Or keep only the two most relevant."
                        ),
                    ))

        return matches
