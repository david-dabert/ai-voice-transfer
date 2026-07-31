"""
Voice Transfer Pipeline

End-to-end pipeline:
1. Load voice corpus
2. Extract register profiles
3. Build lexicon brand
4. Accept draft request (text, target register, recipient context)
5. Generate draft (with LLM)
6. Run fingerprint detection
7. Run self-critique against lexicon brand
8. Return scored result with specific fix suggestions

This module ties together all components of the voice transfer
framework into a single, configurable pipeline. Each step can be
run independently or as part of the full chain.

The pipeline does not call an LLM directly. It produces the
constraints that should be injected into the LLM prompt, and it
validates the output after generation. The actual LLM call is the
responsibility of the caller, because different users will use
different providers (OpenAI, Anthropic, local models, etc.).

The pipeline's value is in what surrounds the generation: the
extraction that produces specific constraints, and the validation
that catches the drift back toward the training distribution median.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

from voice_transfer.extractor import VoiceCorpus, RegisterProfile
from voice_transfer.register_library import Register, RegisterLibrary, FormalityLevel
from voice_transfer.lexicon_brand import LexiconBrand, SelfCritiqueReport
from voice_transfer.fingerprint_detector import FingerprintDetector, DetectionReport
from voice_transfer.scorer import VoiceFidelityScorer, FidelityReport


@dataclass
class DraftRequest:
    """A request to generate text in a specific voice."""

    content_prompt: str  # what to write about
    target_register: str = ""  # which register to use
    recipient: str = ""  # who is the recipient
    formality: Optional[FormalityLevel] = None
    context: str = ""  # additional context
    max_length: Optional[int] = None  # word count target
    language: str = "en"  # target language

    def describe(self) -> str:
        """Human-readable description of the request."""
        parts = [f"Content: {self.content_prompt}"]
        if self.target_register:
            parts.append(f"Register: {self.target_register}")
        if self.recipient:
            parts.append(f"Recipient: {self.recipient}")
        if self.formality:
            parts.append(f"Formality: {self.formality.name}")
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.max_length:
            parts.append(f"Target length: ~{self.max_length} words")
        return "; ".join(parts)


@dataclass
class DraftResult:
    """Result of the voice transfer pipeline."""

    # The draft text (either generated or provided for validation)
    draft: str

    # The constraints that were (or should be) injected into the LLM prompt
    prompt_constraints: str

    # Validation results
    fidelity_report: Optional[FidelityReport] = None
    fingerprint_report: Optional[DetectionReport] = None
    critique_report: Optional[SelfCritiqueReport] = None

    # Metadata
    register_used: str = ""
    request: Optional[DraftRequest] = None

    @property
    def score(self) -> float:
        """Overall voice fidelity score (0-100)."""
        if self.fidelity_report:
            return self.fidelity_report.overall_score
        return 0.0

    @property
    def passed(self) -> bool:
        """True if the draft passes all hard constraints."""
        if self.critique_report and not self.critique_report.passed():
            return False
        if self.fingerprint_report and self.fingerprint_report.high_severity_count > 0:
            return False
        return True

    def summary(self) -> str:
        """Human-readable summary of the result."""
        lines = [
            "=== Voice Transfer Result ===",
            f"Register: {self.register_used}",
            f"Score: {self.score:.1f}/100",
            f"Status: {'PASS' if self.passed else 'NEEDS REVISION'}",
        ]

        if self.fidelity_report:
            lines.append("")
            lines.append(self.fidelity_report.summary())

        if not self.passed:
            lines.append("")
            lines.append("--- Required Fixes ---")
            if self.critique_report:
                for result in self.critique_report.module_results:
                    for v in result.violations:
                        if v.severity >= 0.8:
                            lines.append(f"  HARD: {v.description}")
                            if v.suggestion:
                                lines.append(f"    Fix: {v.suggestion}")

            if self.fingerprint_report:
                for match in self.fingerprint_report.matches:
                    if match.severity >= 0.7:
                        lines.append(f"  FINGERPRINT: {match.fingerprint_name}")
                        lines.append(f"    Found: \"{match.matched_text[:60]}\"")
                        lines.append(f"    Fix: {match.suggested_fix}")

        return "\n".join(lines)


class VoiceTransferPipeline:
    """
    End-to-end voice transfer pipeline.

    The pipeline has two modes:

    1. CONSTRAINT MODE: Given a DraftRequest, produce the prompt
       constraints that should be injected into the LLM system prompt.
       The caller handles the actual LLM call.

    2. VALIDATION MODE: Given a draft text, validate it against the
       voice profile and return a scored report with fix suggestions.

    Usage (constraint mode):
        pipeline = VoiceTransferPipeline()
        pipeline.load_corpus(["sample1...", "sample2..."])
        constraints = pipeline.build_constraints(request)
        # Caller injects constraints into LLM prompt and generates

    Usage (validation mode):
        pipeline = VoiceTransferPipeline()
        pipeline.load_corpus(["sample1...", "sample2..."])
        result = pipeline.validate("draft text...", register_name="professional")
        print(result.summary())

    Usage (full pipeline with custom generator):
        pipeline = VoiceTransferPipeline()
        pipeline.load_corpus(["sample1...", "sample2..."])
        result = pipeline.run(request, generator=my_llm_function)
        print(result.summary())
    """

    def __init__(
        self,
        name: str = "default",
        custom_fingerprints: Optional[list[dict]] = None,
    ) -> None:
        self.name = name
        self._corpus: Optional[VoiceCorpus] = None
        self._profile: Optional[RegisterProfile] = None
        self._register_library: Optional[RegisterLibrary] = None
        self._brand: Optional[LexiconBrand] = None
        self._detector = FingerprintDetector(custom_patterns=custom_fingerprints)
        self._scorer: Optional[VoiceFidelityScorer] = None

    # --- Setup methods ---

    def load_corpus(
        self,
        samples: list[str],
        name: str = "voice",
        description: str = "",
    ) -> RegisterProfile:
        """
        Load text samples and extract a voice profile.

        Returns the extracted RegisterProfile.
        """
        self._corpus = VoiceCorpus(name=name, description=description)
        self._corpus.add_samples(samples)
        self._profile = self._corpus.extract()

        # Initialize scorer with the profile
        self._scorer = VoiceFidelityScorer(
            profile=self._profile,
            fingerprint_detector=self._detector,
            lexicon_brand=self._brand,
        )

        return self._profile

    def load_profile(self, profile: RegisterProfile) -> None:
        """Load a pre-extracted profile directly."""
        self._profile = profile
        self._scorer = VoiceFidelityScorer(
            profile=self._profile,
            fingerprint_detector=self._detector,
            lexicon_brand=self._brand,
        )

    def set_register_library(self, library: RegisterLibrary) -> None:
        """Set the register library for multi-register voice handling."""
        self._register_library = library

    def set_lexicon_brand(self, brand: LexiconBrand) -> None:
        """Set the lexicon brand for prescriptive rule checking."""
        self._brand = brand
        # Rebuild scorer with brand
        if self._profile:
            self._scorer = VoiceFidelityScorer(
                profile=self._profile,
                fingerprint_detector=self._detector,
                lexicon_brand=self._brand,
            )

    def use_default_brand(self) -> LexiconBrand:
        """Initialize and set the default 15-module lexicon brand."""
        self._brand = LexiconBrand.with_defaults(name=self.name)
        if self._profile:
            self._scorer = VoiceFidelityScorer(
                profile=self._profile,
                fingerprint_detector=self._detector,
                lexicon_brand=self._brand,
            )
        return self._brand

    # --- Core pipeline methods ---

    def build_constraints(
        self,
        request: Optional[DraftRequest] = None,
        register_name: Optional[str] = None,
    ) -> str:
        """
        Build the LLM prompt constraints for a given request.

        This is CONSTRAINT MODE. The output should be injected
        into the LLM system prompt before generation.

        Returns a string of constraints.
        """
        constraints_parts = []

        # 1. Profile-level constraints
        if self._profile:
            constraints_parts.append("=== Voice Profile Constraints ===")
            constraints_parts.append(self._profile.to_prompt_constraints())

        # 2. Register-level constraints
        if self._register_library and register_name:
            try:
                register = self._register_library.select(register_name)
                constraints_parts.append("")
                constraints_parts.append("=== Register Constraints ===")
                constraints_parts.append(register.to_prompt_instructions())
            except KeyError:
                pass
        elif self._register_library and request:
            register = self._register_library.select_by_context(
                recipient=request.recipient,
                formality=request.formality,
                context=request.context,
            )
            constraints_parts.append("")
            constraints_parts.append(f"=== Register: {register.name} ===")
            constraints_parts.append(register.to_prompt_instructions())

        # 3. Lexicon brand constraints
        if self._brand:
            constraints_parts.append("")
            constraints_parts.append("=== Lexicon Brand Rules ===")
            constraints_parts.append(self._brand.to_prompt_instructions())

        # 4. Anti-fingerprint instructions
        constraints_parts.append("")
        constraints_parts.append("=== Anti-Fingerprint Rules ===")
        constraints_parts.append(self._anti_fingerprint_instructions())

        return "\n".join(constraints_parts)

    def validate(
        self,
        draft: str,
        register_name: Optional[str] = None,
        request: Optional[DraftRequest] = None,
    ) -> DraftResult:
        """
        Validate a draft against the voice profile.

        This is VALIDATION MODE. The draft has already been generated
        (by an LLM or by hand) and needs to be checked.

        Returns a DraftResult with scores and fix suggestions.
        """
        result = DraftResult(
            draft=draft,
            prompt_constraints=self.build_constraints(request, register_name),
            register_used=register_name or "default",
            request=request,
        )

        # Run fingerprint detection
        result.fingerprint_report = self._detector.scan(draft)

        # Run lexicon brand critique
        if self._brand:
            result.critique_report = self._brand.critique(draft)

        # Run fidelity scoring
        if self._scorer:
            result.fidelity_report = self._scorer.score(draft)

        # Register validation
        if self._register_library and register_name:
            violations = self._register_library.validate(draft, register_name)
            if violations:
                # Add register violations to the result as notes
                if result.fidelity_report:
                    for v in violations:
                        result.fidelity_report.dimensions.append(
                            __import__(
                                "voice_transfer.scorer", fromlist=["DimensionScore"]
                            ).DimensionScore(
                                dimension="Register Compliance",
                                score=max(0, 100 - len(violations) * 20),
                                details="; ".join(violations[:3]),
                                weight=1.0,
                            )
                        )

        return result

    def run(
        self,
        request: DraftRequest,
        generator: Optional[Callable[[str, str], str]] = None,
        register_name: Optional[str] = None,
        max_iterations: int = 1,
    ) -> DraftResult:
        """
        Run the full pipeline: build constraints, generate, validate.

        The generator function takes (system_prompt, user_prompt) and
        returns the generated text. If no generator is provided, only
        the constraints are built (useful for manual generation).

        If max_iterations > 1, the pipeline will re-generate with
        fix suggestions appended to the prompt until the draft passes
        or iterations are exhausted.
        """
        # Build constraints
        constraints = self.build_constraints(request, register_name)

        if generator is None:
            # No generator: return constraints only
            return DraftResult(
                draft="",
                prompt_constraints=constraints,
                register_used=register_name or request.target_register or "default",
                request=request,
            )

        # Generate and validate loop
        current_constraints = constraints
        best_result: Optional[DraftResult] = None

        for iteration in range(max_iterations):
            # Build prompts
            system_prompt = (
                f"You are writing as a specific person. Follow these "
                f"voice constraints exactly.\n\n{current_constraints}"
            )
            user_prompt = request.content_prompt
            if request.max_length:
                user_prompt += f"\n\nTarget length: approximately {request.max_length} words."

            # Generate
            draft = generator(system_prompt, user_prompt)

            # Validate
            result = self.validate(
                draft,
                register_name=register_name or request.target_register,
                request=request,
            )

            if best_result is None or result.score > best_result.score:
                best_result = result

            if result.passed:
                break

            # If not passed and more iterations available, append fixes
            if iteration < max_iterations - 1:
                fix_instructions = self._extract_fix_instructions(result)
                current_constraints = (
                    constraints + "\n\n=== Revision Instructions ===\n" + fix_instructions
                )

        return best_result or DraftResult(
            draft="",
            prompt_constraints=constraints,
            register_used=register_name or request.target_register or "default",
            request=request,
        )

    # --- Utility methods ---

    def _anti_fingerprint_instructions(self) -> str:
        """Generate anti-fingerprint instructions for the LLM prompt."""
        return (
            "Do NOT use any of these LLM writing patterns:\n"
            "- Staccato three-word closes (Bold. Brave. Necessary.)\n"
            "- Hook-then-reveal openers (X is the theatre. Y is the physics.)\n"
            "- 'Here is what makes this different' formulas\n"
            "- 'The [class] you would [verb] is that...' constructions\n"
            "- Semicolon-list credential dumps (multilingual; cross-cultural; senior-tested)\n"
            "- Aphoristic closes (fortune-cookie wisdom at the end)\n"
            "- Em-dash overuse (no more than 2 per 500 words)\n"
            "- Theatrical tricolons (three items for rhythm, not for meaning)\n"
            "- Generic-plausible specificity (sounds specific but could apply to anyone)\n"
            "- Mechanical parallelism (On one hand X. On the other hand Y.)\n"
            "- Words: delve, leverage, robust, holistic, nuanced, foster, navigate, landscape, tapestry, beacon\n"
            "- Openers: 'In today's...', 'In an era of...', 'Excited to...', 'Thrilled to...'\n"
            "- Generic references: 'as Einstein once said', 'as the saying goes'\n"
            "- Engagement bait closers: 'Thoughts?', 'What do you think?', 'Agree?'"
        )

    def _extract_fix_instructions(self, result: DraftResult) -> str:
        """Extract specific fix instructions from a validation result."""
        fixes = []

        if result.critique_report:
            for module_result in result.critique_report.module_results:
                for v in module_result.violations:
                    if v.severity >= 0.6:
                        fixes.append(f"- {v.description}: {v.suggestion}")

        if result.fingerprint_report:
            for match in result.fingerprint_report.matches:
                if match.severity >= 0.5:
                    fixes.append(
                        f"- Remove LLM fingerprint '{match.fingerprint_name}': "
                        f"{match.suggested_fix}"
                    )

        return "\n".join(fixes[:15])  # cap at 15 fixes per iteration

    @property
    def profile(self) -> Optional[RegisterProfile]:
        return self._profile

    @property
    def register_library(self) -> Optional[RegisterLibrary]:
        return self._register_library

    @property
    def brand(self) -> Optional[LexiconBrand]:
        return self._brand

    def profile_summary(self) -> str:
        """Return a summary of the loaded profile."""
        if self._profile:
            return self._profile.summary()
        return "No profile loaded."
