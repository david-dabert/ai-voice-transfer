"""
Lexicon Brand Builder

Converts descriptive voice analysis into prescriptive rules.
Each rule is executable at draft time and returns a pass/fail
with specific violations.

The difference between a VoiceCorpus (descriptive) and a LexiconBrand
(prescriptive) is the difference between "this person uses semicolons
3.2 times per 1000 words" and "use semicolons instead of em-dashes
for parenthetical insertions." The corpus tells you what the voice
looks like; the brand tells the LLM what to do.

A LexiconBrand is a set of LexiconModules. Each module owns one
dimension of the voice (vocabulary, connectors, imagery, punctuation,
etc.) and provides a test function that can be run against any draft.
The self-critique runner iterates all modules, collects violations,
and returns a scored report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ModuleViolation:
    """A single violation found by a lexicon module."""

    module_name: str
    description: str
    location: str  # the offending text or context
    severity: float  # 0.0 to 1.0 (1.0 = hard fail)
    suggestion: str = ""

    def __str__(self) -> str:
        sev = "HARD" if self.severity >= 0.8 else "SOFT" if self.severity >= 0.4 else "NOTE"
        parts = [f"[{sev}] {self.module_name}: {self.description}"]
        if self.location:
            parts.append(f"  Found: \"{self.location[:80]}\"")
        if self.suggestion:
            parts.append(f"  Fix: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class ModuleResult:
    """Result of running a single lexicon module against a draft."""

    module_name: str
    passed: bool
    violations: list[ModuleViolation] = field(default_factory=list)
    score: float = 1.0  # 1.0 = perfect, 0.0 = total failure

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        header = f"{self.module_name}: {status} (score: {self.score:.2f})"
        if self.violations:
            violation_text = "\n".join(f"  {v}" for v in self.violations)
            return f"{header}\n{violation_text}"
        return header


@dataclass
class SelfCritiqueReport:
    """Complete self-critique report across all lexicon modules."""

    module_results: list[ModuleResult] = field(default_factory=list)
    overall_score: float = 1.0
    hard_fails: int = 0
    soft_fails: int = 0
    total_violations: int = 0

    def passed(self) -> bool:
        """True if no hard fails."""
        return self.hard_fails == 0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Self-Critique Report ===",
            f"Overall score: {self.overall_score:.2f}",
            f"Hard fails: {self.hard_fails}",
            f"Soft fails: {self.soft_fails}",
            f"Total violations: {self.total_violations}",
            "",
        ]
        for result in self.module_results:
            lines.append(str(result))
            lines.append("")
        return "\n".join(lines)


# Type alias for test functions
TestFunction = Callable[[str], list[ModuleViolation]]


@dataclass
class LexiconModule:
    """
    A single prescriptive voice rule with an executable test.

    Each module owns one dimension of the voice and provides:
    - A human-readable rule description
    - A test function that checks a draft for violations
    - Examples of correct and incorrect usage
    """

    name: str
    rule: str
    test_function: TestFunction
    examples_correct: list[str] = field(default_factory=list)
    examples_incorrect: list[str] = field(default_factory=list)
    severity: float = 0.5  # default module severity

    def test(self, text: str) -> ModuleResult:
        """Run this module's test against a draft."""
        violations = self.test_function(text)
        passed = not any(v.severity >= 0.8 for v in violations)
        score = 1.0
        if violations:
            avg_severity = sum(v.severity for v in violations) / len(violations)
            score = max(0.0, 1.0 - avg_severity * (len(violations) / 10.0))
        return ModuleResult(
            module_name=self.name,
            passed=passed,
            violations=violations,
            score=score,
        )


class LexiconBrand:
    """
    A collection of LexiconModules that together define the
    prescriptive voice rules for a specific human.

    Usage:
        brand = LexiconBrand.with_defaults()
        # or
        brand = LexiconBrand()
        brand.add_module(my_custom_module)

        report = brand.critique("Draft text here...")
        if report.passed():
            print("Ready to ship")
        else:
            print(report.summary())
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._modules: list[LexiconModule] = []

    def add_module(self, module: LexiconModule) -> None:
        """Add a lexicon module."""
        self._modules.append(module)

    def remove_module(self, name: str) -> None:
        """Remove a module by name."""
        self._modules = [m for m in self._modules if m.name != name]

    @property
    def module_count(self) -> int:
        return len(self._modules)

    @property
    def module_names(self) -> list[str]:
        return [m.name for m in self._modules]

    def get_module(self, name: str) -> Optional[LexiconModule]:
        """Get a module by name."""
        for m in self._modules:
            if m.name == name:
                return m
        return None

    def critique(self, text: str) -> SelfCritiqueReport:
        """
        Run all modules against a draft and return a SelfCritiqueReport.

        This is the primary method. It iterates every module,
        collects violations, computes the overall score, and
        returns a structured report.
        """
        report = SelfCritiqueReport()

        for module in self._modules:
            result = module.test(text)
            report.module_results.append(result)
            report.total_violations += len(result.violations)

            for v in result.violations:
                if v.severity >= 0.8:
                    report.hard_fails += 1
                elif v.severity >= 0.4:
                    report.soft_fails += 1

        # Overall score: average of module scores
        if report.module_results:
            report.overall_score = (
                sum(r.score for r in report.module_results)
                / len(report.module_results)
            )
        else:
            report.overall_score = 1.0

        return report

    def to_prompt_instructions(self) -> str:
        """Export all modules as LLM prompt constraints."""
        lines = [f"Voice Brand Rules ({self.name}):"]
        for i, module in enumerate(self._modules, 1):
            lines.append(f"\n{i}. {module.name}")
            lines.append(f"   Rule: {module.rule}")
            if module.examples_correct:
                lines.append(f"   Correct: \"{module.examples_correct[0]}\"")
            if module.examples_incorrect:
                lines.append(f"   Incorrect: \"{module.examples_incorrect[0]}\"")
        return "\n".join(lines)

    @classmethod
    def with_defaults(cls, name: str = "default") -> "LexiconBrand":
        """
        Create a LexiconBrand with the fifteen default modules.

        These modules cover the most common dimensions of voice
        and can be customized after creation.
        """
        brand = cls(name=name)
        for module in _build_default_modules():
            brand.add_module(module)
        return brand


# --- Default module builders ---


def _build_default_modules() -> list[LexiconModule]:
    """Build the fifteen default lexicon modules."""
    return [
        _module_01_base_register(),
        _module_02_connector_discipline(),
        _module_03_imagery_preferences(),
        _module_04_self_reference(),
        _module_05_opening_formulas(),
        _module_06_closing_formulas(),
        _module_07_punctuation_rules(),
        _module_08_sentence_structure(),
        _module_09_forbidden_vocabulary(),
        _module_10_forbidden_structures(),
        _module_11_signature_phrases(),
        _module_12_humor_tone(),
        _module_13_cultural_references(),
        _module_14_formality_calibration(),
        _module_15_imperfection_preservation(),
    ]


def _module_01_base_register() -> LexiconModule:
    """Module 1: Base Register (vocabulary origin preferences)."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        # Check for vocabulary that is too uniformly latinate or saxon.
        # This is a heuristic: flag if the text has zero semicolons
        # in 500+ words (suggests the register is not being maintained)
        words = text.split()
        if len(words) > 200 and text.count(";") == 0:
            violations.append(ModuleViolation(
                module_name="base_register",
                description=(
                    "No semicolons in a text over 200 words. "
                    "If the source voice uses semicolons, this is a drift."
                ),
                location="(entire text)",
                severity=0.3,
                suggestion="Replace some comma-joined clauses with semicolons.",
            ))
        return violations

    return LexiconModule(
        name="base_register",
        rule=(
            "Maintain the vocabulary origin balance of the source voice. "
            "If the source is latinate-dominant, do not simplify to saxon. "
            "If the source uses semicolons, use semicolons."
        ),
        test_function=test,
        examples_correct=[
            "The institutional architecture requires recalibration; "
            "the current configuration does not serve the stated objective."
        ],
        examples_incorrect=[
            "The setup needs fixing because it does not work right now."
        ],
        severity=0.5,
    )


def _module_02_connector_discipline() -> LexiconModule:
    """Module 2: Connector Discipline (allowed/banned transition words)."""

    BANNED_CONNECTORS = [
        "moreover", "furthermore", "additionally", "consequently",
        "nevertheless", "nonetheless", "indeed", "certainly",
        "undoubtedly", "essentially", "fundamentally", "notably",
        "significantly", "importantly", "crucially", "interestingly",
        "surprisingly", "remarkably", "strikingly",
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        text_lower = text.lower()
        for connector in BANNED_CONNECTORS:
            if connector in text_lower:
                # Find the context
                idx = text_lower.index(connector)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(connector) + 20)
                context = text[start:end]
                violations.append(ModuleViolation(
                    module_name="connector_discipline",
                    description=f"Banned connector: '{connector}'",
                    location=context,
                    severity=0.6,
                    suggestion=(
                        f"Remove '{connector}' or replace with a "
                        f"simpler conjunction (and, but, so, yet)."
                    ),
                ))
        return violations

    return LexiconModule(
        name="connector_discipline",
        rule=(
            "Never use LLM-overrepresented transition words. "
            "Replace with simple conjunctions or restructure the sentence. "
            "Banned: moreover, furthermore, additionally, consequently, "
            "nevertheless, nonetheless, indeed, certainly, undoubtedly, "
            "essentially, fundamentally, notably, significantly, "
            "importantly, crucially, interestingly, surprisingly, "
            "remarkably, strikingly."
        ),
        test_function=test,
        examples_correct=["The data shows a gap; the team has not addressed it."],
        examples_incorrect=[
            "Moreover, the data significantly shows a crucial gap."
        ],
        severity=0.6,
    )


def _module_03_imagery_preferences() -> LexiconModule:
    """Module 3: Imagery Preferences (concrete vs abstract)."""

    ABSTRACT_FILLERS = [
        "landscape", "paradigm", "ecosystem", "synergy", "tapestry",
        "beacon", "cornerstone", "journey", "space", "arena",
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        text_lower = text.lower()
        for filler in ABSTRACT_FILLERS:
            if filler in text_lower:
                idx = text_lower.index(filler)
                start = max(0, idx - 15)
                end = min(len(text), idx + len(filler) + 15)
                violations.append(ModuleViolation(
                    module_name="imagery_preferences",
                    description=f"Abstract filler metaphor: '{filler}'",
                    location=text[start:end],
                    severity=0.7,
                    suggestion=f"Replace '{filler}' with a concrete noun.",
                ))
        return violations

    return LexiconModule(
        name="imagery_preferences",
        rule=(
            "Use concrete imagery over abstract metaphors. "
            "Never use: landscape, paradigm, ecosystem, synergy, "
            "tapestry, beacon, cornerstone, journey (metaphorical), "
            "space (metaphorical), arena (metaphorical)."
        ),
        test_function=test,
        examples_correct=["The programme covers six countries and twelve partners."],
        examples_incorrect=["This initiative sits at the heart of a rich ecosystem."],
        severity=0.7,
    )


def _module_04_self_reference() -> LexiconModule:
    """Module 4: Self-Reference Style."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        # Flag excessive self-reference (more than 5 "I" per 100 words)
        words = text.split()
        if len(words) > 50:
            i_count = sum(1 for w in words if w.lower() in ("i", "i'm", "i've", "i'd", "i'll"))
            ratio = i_count / len(words) * 100
            if ratio > 5.0:
                violations.append(ModuleViolation(
                    module_name="self_reference",
                    description=(
                        f"Excessive first-person singular: {ratio:.1f}% "
                        f"of words are self-referential ({i_count} instances)"
                    ),
                    location="(throughout text)",
                    severity=0.4,
                    suggestion="Restructure some sentences to reduce I-density.",
                ))
        return violations

    return LexiconModule(
        name="self_reference",
        rule=(
            "Self-reference should be measured and purposeful. "
            "Avoid I-stacking (multiple sentences starting with 'I' in sequence). "
            "First-person singular should not exceed 5% of total words."
        ),
        test_function=test,
        severity=0.4,
    )


def _module_05_opening_formulas() -> LexiconModule:
    """Module 5: Opening Formulas by Context."""

    BANNED_OPENERS = [
        r"^excited to",
        r"^thrilled to",
        r"^honored to",
        r"^humbled to",
        r"^i'?m? delighted",
        r"^in today'?s",
        r"^in an era",
        r"^in a world",
        r"^have you ever",
        r"^what if i told you",
        r"^picture this",
        r"^imagine",
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        first_line = text.strip().split("\n")[0].strip() if text.strip() else ""
        first_lower = first_line.lower()

        for pattern in BANNED_OPENERS:
            if re.match(pattern, first_lower):
                violations.append(ModuleViolation(
                    module_name="opening_formulas",
                    description=f"Banned opening formula detected",
                    location=first_line[:80],
                    severity=0.8,
                    suggestion=(
                        "Open with a concrete observation, a fact, "
                        "or the thesis directly. No performative openers."
                    ),
                ))
                break

        return violations

    return LexiconModule(
        name="opening_formulas",
        rule=(
            "Never open with performative excitement, rhetorical "
            "questions, or 'In today's/In an era/In a world' frames. "
            "Open with the substance: a fact, an observation, a thesis."
        ),
        test_function=test,
        examples_correct=[
            "The demand for women-only shared accommodation at our "
            "Bordeaux property is real, repeated, and currently unmet."
        ],
        examples_incorrect=[
            "In today's hospitality landscape, an exciting new "
            "paradigm is emerging around gender-inclusive dormitories."
        ],
        severity=0.8,
    )


def _module_06_closing_formulas() -> LexiconModule:
    """Module 6: Closing Formulas by Context."""

    BANNED_CLOSERS = [
        r"thoughts\??$",
        r"what do you think\??$",
        r"agree\??$",
        r"let that sink in\.?$",
        r"mic drop\.?$",
        r"that'?s? it\.? that'?s? the (?:tweet|post)\.?$",
        r"the future is [\w\s]+\.?$",
        r"and that makes all the difference\.?$",
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return violations
        last_line = lines[-1].lower()

        for pattern in BANNED_CLOSERS:
            if re.search(pattern, last_line):
                violations.append(ModuleViolation(
                    module_name="closing_formulas",
                    description="Banned closing formula detected",
                    location=lines[-1][:80],
                    severity=0.8,
                    suggestion=(
                        "Close with a concrete next step, a restated "
                        "thesis, or the final piece of evidence. "
                        "No engagement bait."
                    ),
                ))
                break

        return violations

    return LexiconModule(
        name="closing_formulas",
        rule=(
            "Never close with engagement bait (Thoughts?), "
            "aphoristic wisdom (Let that sink in), or mic-drop "
            "formulas. Close with substance: the final argument, "
            "a concrete proposal, or the restated thesis."
        ),
        test_function=test,
        severity=0.8,
    )


def _module_07_punctuation_rules() -> LexiconModule:
    """Module 7: Punctuation Rules."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []

        # Count em-dashes
        em_dash_count = text.count("\u2014") + len(re.findall(r"(?<!\-)--(?!\-)", text))
        word_count = len(text.split())

        if word_count > 0 and em_dash_count > 0:
            per_1000 = em_dash_count / word_count * 1000
            if per_1000 > 3.0:
                violations.append(ModuleViolation(
                    module_name="punctuation_rules",
                    description=(
                        f"Em-dash overuse: {em_dash_count} em-dashes "
                        f"({per_1000:.1f} per 1000 words)"
                    ),
                    location="(throughout text)",
                    severity=0.6,
                    suggestion="Replace some em-dashes with semicolons or commas.",
                ))

        # Exclamation marks in professional text
        exclamation_count = text.count("!")
        if exclamation_count > 1:
            violations.append(ModuleViolation(
                module_name="punctuation_rules",
                description=f"Multiple exclamation marks: {exclamation_count}",
                location="(throughout text)",
                severity=0.3,
                suggestion="Remove exclamation marks. Let the content carry the emphasis.",
            ))

        return violations

    return LexiconModule(
        name="punctuation_rules",
        rule=(
            "Semicolons for clause joining. Commas for lists and "
            "light pauses. Em-dashes sparingly or not at all "
            "(check source profile). No multiple exclamation marks."
        ),
        test_function=test,
        severity=0.6,
    )


def _module_08_sentence_structure() -> LexiconModule:
    """Module 8: Sentence Structure Preferences."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        sentences = re.split(r"[.!?]+\s+", text)

        # Check for monotonous sentence length (all roughly the same)
        if len(sentences) >= 5:
            lengths = [len(s.split()) for s in sentences if len(s.split()) >= 2]
            if lengths:
                import statistics
                try:
                    cv = statistics.stdev(lengths) / statistics.mean(lengths)
                    if cv < 0.2:
                        violations.append(ModuleViolation(
                            module_name="sentence_structure",
                            description=(
                                f"Monotonous sentence length (CV={cv:.2f}). "
                                f"All sentences are roughly the same length."
                            ),
                            location="(structural pattern)",
                            severity=0.5,
                            suggestion=(
                                "Vary sentence length. Mix short declarative "
                                "sentences with longer subordinated ones."
                            ),
                        ))
                except statistics.StatisticsError:
                    pass

        return violations

    return LexiconModule(
        name="sentence_structure",
        rule=(
            "Vary sentence length deliberately. Mix short declarative "
            "sentences with longer subordinated ones. Avoid monotonous "
            "rhythm (coefficient of variation below 0.2 is a red flag)."
        ),
        test_function=test,
        severity=0.5,
    )


def _module_09_forbidden_vocabulary() -> LexiconModule:
    """Module 9: Forbidden Vocabulary."""

    FORBIDDEN = [
        "delve", "delves", "delving",
        "leverage", "leveraging", "leveraged",
        "robust", "holistic", "comprehensive",
        "nuanced", "multifaceted", "streamline",
        "pivotal", "foster", "fosters", "fostering",
        "navigate", "navigating", "navigates",
        "resonate", "resonates", "resonating",
        "underscore", "underscores", "underscoring",
        "unpack", "unpacking", "unpacks",
        "deep dive",
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        text_lower = text.lower()

        for word in FORBIDDEN:
            if word in text_lower:
                idx = text_lower.index(word)
                start = max(0, idx - 15)
                end = min(len(text), idx + len(word) + 15)
                violations.append(ModuleViolation(
                    module_name="forbidden_vocabulary",
                    description=f"LLM-fingerprint word: '{word}'",
                    location=text[start:end],
                    severity=0.8,
                    suggestion=f"Replace '{word}' with a simpler, more specific word.",
                ))

        return violations

    return LexiconModule(
        name="forbidden_vocabulary",
        rule=(
            "Never use words that are overrepresented in LLM output: "
            "delve, leverage, robust, holistic, comprehensive, nuanced, "
            "multifaceted, streamline, pivotal, foster, navigate, "
            "resonate, underscore, unpack, deep dive."
        ),
        test_function=test,
        severity=0.8,
    )


def _module_10_forbidden_structures() -> LexiconModule:
    """Module 10: Forbidden Structures."""

    FORBIDDEN_PATTERNS = [
        (
            r"(?:sits?|standing|positioned?)\s+at\s+(?:the|an?)\s+intersection",
            "sits at an intersection",
        ),
        (
            r"the\s+\w+\s+you\s+would\s+\w+\s+is\s+that",
            "The [X] you would [verb] is that...",
        ),
        (
            r"here\s+is\s+what\s+makes?\s+this\s+different",
            "Here is what makes this different",
        ),
        (
            r"\w+\.\s+\w+\.\s+\w+\.$",
            "Staccato three-word close (Bold. Brave. Necessary.)",
        ),
        (
            r"on\s+one\s+hand.{5,60}on\s+the\s+other\s+hand",
            "On one hand... on the other hand (mechanical parallelism)",
        ),
    ]

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        for pattern, label in FORBIDDEN_PATTERNS:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                violations.append(ModuleViolation(
                    module_name="forbidden_structures",
                    description=f"Forbidden structure: {label}",
                    location=match.group()[:80],
                    severity=0.9,
                    suggestion="Restructure the sentence to avoid this LLM pattern.",
                ))
        return violations

    return LexiconModule(
        name="forbidden_structures",
        rule=(
            "Never use these LLM-signature structures: "
            "'sits at an intersection,' "
            "'The [X] you would [verb] is that,' "
            "'Here is what makes this different,' "
            "three-word staccato closes, "
            "'On one hand... on the other hand.'"
        ),
        test_function=test,
        severity=0.9,
    )


def _module_11_signature_phrases() -> LexiconModule:
    """Module 11: Signature Phrases (placeholder for user customization)."""

    def test(text: str) -> list[ModuleViolation]:
        # This module is customized per voice. Default is no-op.
        return []

    return LexiconModule(
        name="signature_phrases",
        rule=(
            "Use the person's signature phrases sparingly and naturally. "
            "Never force them. Never use them more than once per document. "
            "Configure this module with the actual phrases."
        ),
        test_function=test,
        severity=0.3,
    )


def _module_12_humor_tone() -> LexiconModule:
    """Module 12: Humor and Tone Markers."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        # Detect generic LLM humor attempts
        generic_humor = [
            "spoiler alert",
            "plot twist",
            "hot take",
            "pro tip",
            "fun fact",
            "newsflash",
            "breaking news",
        ]
        text_lower = text.lower()
        for phrase in generic_humor:
            if phrase in text_lower:
                violations.append(ModuleViolation(
                    module_name="humor_tone",
                    description=f"Generic internet humor: '{phrase}'",
                    location=phrase,
                    severity=0.5,
                    suggestion="Remove. If humor is needed, use the source person's actual humor style.",
                ))
        return violations

    return LexiconModule(
        name="humor_tone",
        rule=(
            "Humor must match the source person's actual style. "
            "Never inject generic internet humor (spoiler alert, "
            "plot twist, hot take, pro tip, fun fact)."
        ),
        test_function=test,
        severity=0.5,
    )


def _module_13_cultural_references() -> LexiconModule:
    """Module 13: Cultural Reference Style."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        # Flag generic-plausible cultural references
        generic_refs = [
            "as einstein once said",
            "as gandhi once said",
            "as martin luther king",
            "as steve jobs famously",
            "as the saying goes",
            "as they say",
            "an old proverb says",
        ]
        text_lower = text.lower()
        for ref in generic_refs:
            if ref in text_lower:
                violations.append(ModuleViolation(
                    module_name="cultural_references",
                    description=f"Generic cultural reference: '{ref}'",
                    location=ref,
                    severity=0.6,
                    suggestion=(
                        "Use cultural references specific to the source person's "
                        "actual repertoire. Generic quotation attribution is an "
                        "LLM fingerprint."
                    ),
                ))
        return violations

    return LexiconModule(
        name="cultural_references",
        rule=(
            "Cultural references must come from the source person's "
            "actual repertoire. No generic Einstein/Gandhi/Jobs quotes. "
            "No 'as the saying goes.' If the person references Nietzsche "
            "and the 1755 Lisbon earthquake, use those; do not substitute "
            "with more recognizable references."
        ),
        test_function=test,
        severity=0.6,
    )


def _module_14_formality_calibration() -> LexiconModule:
    """Module 14: Formality Calibration by Recipient."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []
        # Check for formality inconsistency: mixing "Hey" with "Respectfully"
        has_casual = bool(re.search(r"\b(hey|gonna|wanna|gotta|kinda)\b", text, re.IGNORECASE))
        has_formal = bool(re.search(
            r"\b(respectfully|pursuant|hereby|heretofore|notwithstanding)\b",
            text, re.IGNORECASE,
        ))
        if has_casual and has_formal:
            violations.append(ModuleViolation(
                module_name="formality_calibration",
                description="Formality inconsistency: casual and formal markers mixed",
                location="(throughout text)",
                severity=0.7,
                suggestion="Choose one formality level and hold it consistently.",
            ))
        return violations

    return LexiconModule(
        name="formality_calibration",
        rule=(
            "Formality must be consistent within a single piece. "
            "Do not mix casual markers (hey, gonna, kinda) with "
            "formal markers (pursuant, hereby, notwithstanding)."
        ),
        test_function=test,
        severity=0.7,
    )


def _module_15_imperfection_preservation() -> LexiconModule:
    """Module 15: Imperfection Preservation (anti-AI signatures)."""

    def test(text: str) -> list[ModuleViolation]:
        violations = []

        # Check if text is too "clean" (suspiciously perfect formatting)
        sentences = re.split(r"[.!?]+\s+", text)
        if len(sentences) >= 10:
            # All sentences start with capital, all end with period
            # No comma splices, no sentence fragments: suspicious
            all_start_capital = all(
                s[0].isupper() for s in sentences if s
            )
            all_proper_length = all(
                5 <= len(s.split()) <= 30 for s in sentences if s.strip()
            )
            if all_start_capital and all_proper_length:
                violations.append(ModuleViolation(
                    module_name="imperfection_preservation",
                    description=(
                        "Text is suspiciously uniform. Real human writing "
                        "has occasional short sentences, varying paragraph "
                        "lengths, and structural variety."
                    ),
                    location="(structural uniformity)",
                    severity=0.3,
                    suggestion=(
                        "If the source voice has characteristic imperfections "
                        "(comma splices, sentence fragments, occasional missing "
                        "spaces), preserve them. They are anti-AI signatures."
                    ),
                ))

        return violations

    return LexiconModule(
        name="imperfection_preservation",
        rule=(
            "Preserve the source voice's characteristic imperfections. "
            "If they use comma splices intentionally, keep them. If they "
            "sometimes omit spaces after commas, preserve that. If they "
            "use 'i.e;' with a semicolon, do not correct it. These "
            "imperfections are anti-AI signatures that mark the text "
            "as human-written."
        ),
        test_function=test,
        severity=0.3,
    )
