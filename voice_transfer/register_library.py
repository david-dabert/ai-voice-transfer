"""
Register Library

Most people write in multiple distinct registers:
- Professional email vs personal message
- Report vs presentation
- Social media vs formal letter

A "register" is not a tone adjective. It is a measurable configuration
of vocabulary, sentence structure, punctuation, formality level, and
recipient-class awareness. The same person writing a condolence note
and a project status update uses the same underlying voice but a
different register of that voice.

LLMs collapse all registers into one output unless explicitly
constrained. This module classifies and manages distinct voice modes,
allowing the pipeline to select the correct register before drafting
and to validate that the output matches the selected register.

The Register Library is the bridge between "this person has one voice"
and "this person deploys that voice differently depending on context."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class FormalityLevel(Enum):
    """Formality spectrum for register classification."""
    INTIMATE = 1
    CASUAL = 2
    INFORMAL_PROFESSIONAL = 3
    FORMAL_PROFESSIONAL = 4
    CEREMONIAL = 5


@dataclass
class RegisterMarker:
    """
    A specific, testable marker that indicates a register is active.

    Markers are the atoms of register detection. Each marker is a
    pattern (word, phrase, structural element) that appears in one
    register and not in others.
    """
    pattern: str
    description: str
    is_regex: bool = False
    weight: float = 1.0

    def matches(self, text: str) -> bool:
        """Check if this marker is present in the given text."""
        if self.is_regex:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        return self.pattern.lower() in text.lower()


@dataclass
class Register:
    """
    A distinct voice mode with its own markers, anti-markers,
    sample phrases, and recipient class.

    A Register is not a persona. It is a configuration of the same
    voice for a specific context. The voice remains constant; the
    register adjusts formality, vocabulary density, sentence length,
    and structural choices.
    """

    name: str
    description: str
    formality: FormalityLevel = FormalityLevel.FORMAL_PROFESSIONAL

    # What marks this register as active
    markers: list[RegisterMarker] = field(default_factory=list)

    # What must NOT appear in this register
    anti_markers: list[RegisterMarker] = field(default_factory=list)

    # Concrete examples of this register in action
    sample_phrases: list[str] = field(default_factory=list)

    # Who receives text in this register
    recipient_classes: list[str] = field(default_factory=list)

    # Structural constraints
    max_sentence_length: Optional[int] = None
    min_sentence_length: Optional[int] = None
    preferred_punctuation: list[str] = field(default_factory=list)
    forbidden_punctuation: list[str] = field(default_factory=list)

    # Vocabulary constraints
    vocabulary_notes: str = ""
    forbidden_words: list[str] = field(default_factory=list)
    preferred_words: list[str] = field(default_factory=list)

    # Opening and closing constraints
    opening_formulas: list[str] = field(default_factory=list)
    closing_formulas: list[str] = field(default_factory=list)
    forbidden_openings: list[str] = field(default_factory=list)
    forbidden_closings: list[str] = field(default_factory=list)

    def score_match(self, text: str) -> float:
        """
        Score how well a text matches this register.

        Returns a float between 0.0 and 1.0.
        Positive markers increase the score; anti-markers decrease it.
        """
        if not text.strip():
            return 0.0

        marker_score = 0.0
        marker_total = 0.0

        for marker in self.markers:
            marker_total += marker.weight
            if marker.matches(text):
                marker_score += marker.weight

        # Anti-markers are penalties
        anti_penalty = 0.0
        for anti in self.anti_markers:
            if anti.matches(text):
                anti_penalty += anti.weight

        if marker_total == 0:
            base_score = 0.5
        else:
            base_score = marker_score / marker_total

        # Apply penalties (each anti-marker reduces score by 0.1, capped)
        penalty_factor = max(0.0, 1.0 - (anti_penalty * 0.1))
        return min(1.0, max(0.0, base_score * penalty_factor))

    def validate(self, text: str) -> list[str]:
        """
        Validate a draft against this register's constraints.

        Returns a list of violation descriptions. Empty list means
        the draft passes validation.
        """
        violations = []
        text_lower = text.lower()

        # Check forbidden words
        for word in self.forbidden_words:
            if word.lower() in text_lower:
                violations.append(
                    f"Forbidden word in {self.name} register: '{word}'"
                )

        # Check forbidden punctuation
        for punct in self.forbidden_punctuation:
            if punct in text:
                violations.append(
                    f"Forbidden punctuation in {self.name} register: '{punct}'"
                )

        # Check anti-markers
        for anti in self.anti_markers:
            if anti.matches(text):
                violations.append(
                    f"Anti-marker detected in {self.name} register: "
                    f"{anti.description}"
                )

        # Check sentence length constraints
        sentences = re.split(r"[.!?]+\s+", text)
        for sentence in sentences:
            word_count = len(sentence.split())
            if self.max_sentence_length and word_count > self.max_sentence_length:
                violations.append(
                    f"Sentence too long for {self.name} register: "
                    f"{word_count} words (max {self.max_sentence_length}). "
                    f"Sentence: '{sentence[:60]}...'"
                )
            if self.min_sentence_length and word_count < self.min_sentence_length:
                if word_count >= 2:  # ignore fragments
                    violations.append(
                        f"Sentence too short for {self.name} register: "
                        f"{word_count} words (min {self.min_sentence_length}). "
                        f"Sentence: '{sentence[:60]}'"
                    )

        # Check forbidden openings
        first_sentence = sentences[0].strip() if sentences else ""
        for forbidden in self.forbidden_openings:
            if first_sentence.lower().startswith(forbidden.lower()):
                violations.append(
                    f"Forbidden opening in {self.name} register: "
                    f"starts with '{forbidden}'"
                )

        # Check forbidden closings
        last_sentence = sentences[-1].strip() if sentences else ""
        for forbidden in self.forbidden_closings:
            if last_sentence.lower().endswith(forbidden.lower()):
                violations.append(
                    f"Forbidden closing in {self.name} register: "
                    f"ends with '{forbidden}'"
                )

        return violations

    def to_prompt_instructions(self) -> str:
        """
        Export this register as LLM prompt instructions.
        """
        lines = [f"Register: {self.name}", f"Formality: {self.formality.name}"]

        if self.description:
            lines.append(f"Description: {self.description}")

        if self.vocabulary_notes:
            lines.append(f"Vocabulary: {self.vocabulary_notes}")

        if self.preferred_punctuation:
            lines.append(
                f"Preferred punctuation: {', '.join(self.preferred_punctuation)}"
            )

        if self.forbidden_punctuation:
            lines.append(
                f"Forbidden punctuation: {', '.join(self.forbidden_punctuation)}"
            )

        if self.forbidden_words:
            lines.append(f"Forbidden words: {', '.join(self.forbidden_words[:20])}")

        if self.max_sentence_length:
            lines.append(f"Max sentence length: {self.max_sentence_length} words")

        if self.opening_formulas:
            lines.append(
                f"Opening style examples: "
                + "; ".join(f'"{o}"' for o in self.opening_formulas[:3])
            )

        if self.forbidden_openings:
            lines.append(
                f"Never open with: "
                + "; ".join(f'"{o}"' for o in self.forbidden_openings[:5])
            )

        if self.sample_phrases:
            lines.append("Voice samples from this register:")
            for phrase in self.sample_phrases[:5]:
                lines.append(f'  - "{phrase}"')

        return "\n".join(lines)


class RegisterLibrary:
    """
    Manages multiple registers for a single voice.

    The library provides:
    1. Storage of named registers
    2. Register selection based on context
    3. Register validation of drafts
    4. Best-match detection for unclassified text

    Usage:
        library = RegisterLibrary()
        library.add_register(professional_register)
        library.add_register(personal_register)

        # Select by name
        reg = library.select("professional")

        # Select by context
        reg = library.select_by_context(
            recipient="senior manager",
            formality=FormalityLevel.FORMAL_PROFESSIONAL
        )

        # Validate a draft
        violations = library.validate("draft text...", register_name="professional")

        # Detect which register a text belongs to
        match = library.detect_register("some text...")
    """

    def __init__(self) -> None:
        self._registers: dict[str, Register] = {}
        self._default: Optional[str] = None

    def add_register(self, register: Register, is_default: bool = False) -> None:
        """Add a register to the library."""
        self._registers[register.name] = register
        if is_default or self._default is None:
            self._default = register.name

    def remove_register(self, name: str) -> None:
        """Remove a register by name."""
        if name in self._registers:
            del self._registers[name]
            if self._default == name:
                self._default = next(iter(self._registers), None)

    def list_registers(self) -> list[str]:
        """Return all register names."""
        return list(self._registers.keys())

    def get(self, name: str) -> Optional[Register]:
        """Get a register by name."""
        return self._registers.get(name)

    def select(self, name: str) -> Register:
        """
        Select a register by name. Raises KeyError if not found.
        """
        if name not in self._registers:
            available = ", ".join(self._registers.keys())
            raise KeyError(
                f"Register '{name}' not found. Available: {available}"
            )
        return self._registers[name]

    @property
    def default_register(self) -> Optional[Register]:
        """Return the default register."""
        if self._default:
            return self._registers.get(self._default)
        return None

    def select_by_context(
        self,
        recipient: str = "",
        formality: Optional[FormalityLevel] = None,
        context: str = "",
    ) -> Register:
        """
        Select the best register based on context clues.

        Priority:
        1. Exact recipient class match
        2. Formality level match
        3. Context keyword match in description
        4. Default register
        """
        candidates: list[tuple[float, Register]] = []

        for register in self._registers.values():
            score = 0.0

            # Recipient match
            if recipient:
                recipient_lower = recipient.lower()
                for rc in register.recipient_classes:
                    if rc.lower() in recipient_lower or recipient_lower in rc.lower():
                        score += 3.0
                        break

            # Formality match
            if formality and register.formality == formality:
                score += 2.0
            elif formality:
                diff = abs(register.formality.value - formality.value)
                score += max(0, 1.5 - diff * 0.5)

            # Context keyword match
            if context:
                context_lower = context.lower()
                desc_lower = register.description.lower()
                matching_words = sum(
                    1 for word in context_lower.split()
                    if word in desc_lower and len(word) > 3
                )
                score += matching_words * 0.5

            candidates.append((score, register))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        if self.default_register:
            return self.default_register

        raise ValueError("No registers available and no default set")

    def validate(self, text: str, register_name: str) -> list[str]:
        """
        Validate text against a named register.

        Returns list of violation descriptions.
        """
        register = self.select(register_name)
        return register.validate(text)

    def detect_register(self, text: str) -> tuple[str, float]:
        """
        Detect which register a text most likely belongs to.

        Returns (register_name, confidence_score).
        """
        best_name = ""
        best_score = -1.0

        for name, register in self._registers.items():
            score = register.score_match(text)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    def to_prompt_instructions(self, register_name: Optional[str] = None) -> str:
        """
        Export register instructions for LLM prompt injection.

        If register_name is given, exports only that register.
        Otherwise, exports all registers with selection guidance.
        """
        if register_name:
            register = self.select(register_name)
            return register.to_prompt_instructions()

        lines = ["Available voice registers:"]
        for name, register in self._registers.items():
            default_marker = " (DEFAULT)" if name == self._default else ""
            lines.append(f"\n--- {name}{default_marker} ---")
            lines.append(register.to_prompt_instructions())

        return "\n".join(lines)


# --- Factory functions for common register configurations ---


def create_professional_email_register(
    forbidden_words: Optional[list[str]] = None,
    sample_phrases: Optional[list[str]] = None,
) -> Register:
    """Create a standard professional email register."""
    return Register(
        name="professional_email",
        description=(
            "Formal professional correspondence. Clear, direct, "
            "respectful. No casual language. Proper salutations."
        ),
        formality=FormalityLevel.FORMAL_PROFESSIONAL,
        markers=[
            RegisterMarker("respectfully", "Formal courtesy marker"),
            RegisterMarker("please find", "Formal attachment reference"),
            RegisterMarker(
                r"^(Dear|Monsieur|Madame|Mr\.|Mrs\.|Ms\.)",
                "Formal salutation",
                is_regex=True,
            ),
        ],
        anti_markers=[
            RegisterMarker("hey", "Casual greeting"),
            RegisterMarker("gonna", "Colloquial contraction"),
            RegisterMarker("wanna", "Colloquial contraction"),
            RegisterMarker("lol", "Internet slang"),
            RegisterMarker("!", "Exclamation mark (avoid in formal register)"),
        ],
        recipient_classes=["manager", "client", "institution", "gatekeeper"],
        forbidden_words=forbidden_words or [],
        sample_phrases=sample_phrases or [],
        max_sentence_length=45,
        forbidden_openings=["Hey", "Hi there", "So,"],
        forbidden_closings=["Cheers", "Later", "xo"],
    )


def create_personal_message_register(
    sample_phrases: Optional[list[str]] = None,
) -> Register:
    """Create a warm personal message register."""
    return Register(
        name="personal_message",
        description=(
            "Warm, direct, personal correspondence. Plain language. "
            "No professional jargon. Commas and periods only. "
            "No semicolons."
        ),
        formality=FormalityLevel.CASUAL,
        markers=[
            RegisterMarker(
                r"^(Hi|Hello|Dear)\s+[A-Z]",
                "Personal greeting with first name",
                is_regex=True,
            ),
        ],
        anti_markers=[
            RegisterMarker("pursuant to", "Legal jargon"),
            RegisterMarker("heretofore", "Archaic formal"),
            RegisterMarker("please be advised", "Bureaucratic formula"),
        ],
        recipient_classes=["friend", "family", "personal"],
        forbidden_punctuation=[";"],
        sample_phrases=sample_phrases or [],
        vocabulary_notes="Plain, direct, warm. Short words over long ones.",
    )


def create_analytical_essay_register(
    sample_phrases: Optional[list[str]] = None,
) -> Register:
    """Create an analytical/essay register for long-form writing."""
    return Register(
        name="analytical_essay",
        description=(
            "Analytical long-form writing. Latinate vocabulary permitted. "
            "Complex sentence structures. Evidence-based reasoning. "
            "Semicolons for clause joining."
        ),
        formality=FormalityLevel.FORMAL_PROFESSIONAL,
        markers=[
            RegisterMarker("evidence", "Analytical marker"),
            RegisterMarker("analysis", "Analytical marker"),
            RegisterMarker(";", "Semicolon usage"),
        ],
        anti_markers=[
            RegisterMarker("!", "Exclamation (too emphatic for analysis)"),
            RegisterMarker(
                r"\b(amazing|incredible|awesome)\b",
                "Hyperbolic adjective",
                is_regex=True,
            ),
        ],
        recipient_classes=["reader", "publication", "academic"],
        sample_phrases=sample_phrases or [],
        preferred_punctuation=[";", ":"],
        vocabulary_notes=(
            "Latinate register. Precision over simplicity. "
            "Subordination depth of 2-3 clauses acceptable."
        ),
        min_sentence_length=6,
    )


def create_social_media_register(
    sample_phrases: Optional[list[str]] = None,
) -> Register:
    """Create a social media / LinkedIn register."""
    return Register(
        name="social_media",
        description=(
            "Public social media writing. Compression essay style. "
            "Each paragraph earns the next. No filler. "
            "Opening line must hook without gimmick."
        ),
        formality=FormalityLevel.INFORMAL_PROFESSIONAL,
        markers=[
            RegisterMarker(
                r"^\S.{10,80}$",
                "Short punchy opening line",
                is_regex=True,
            ),
        ],
        anti_markers=[
            RegisterMarker(
                "Here is what makes this different",
                "LLM formula opener",
            ),
            RegisterMarker(
                r"^(Excited to|Thrilled to|Honored to|Humbled to)",
                "Generic LinkedIn opener",
                is_regex=True,
            ),
            RegisterMarker(
                "thoughts?",
                "Generic engagement bait closer (Thoughts?)",
            ),
        ],
        recipient_classes=["public", "linkedin", "social"],
        sample_phrases=sample_phrases or [],
        forbidden_openings=[
            "Excited to announce",
            "Thrilled to share",
            "I'm honored to",
            "I'm humbled to",
        ],
        forbidden_closings=[
            "Thoughts?",
            "What do you think?",
            "Agree?",
            "Drop a comment below",
        ],
    )
