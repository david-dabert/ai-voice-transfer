"""
Voice Extractor

Analyzes a corpus of human-written text to extract:
- Vocabulary preferences (latinate vs saxon, formal vs casual)
- Sentence structure patterns (length distribution, subordination depth)
- Punctuation habits (semicolons vs dashes, comma splices, etc.)
- Signature phrases and formulations
- Forbidden patterns (what the person NEVER writes)

The extractor operates on the premise that voice is not tone. Tone is
a single adjective ("casual," "formal," "warm"). Voice is a statistical
fingerprint: the specific distribution of sentence lengths, the ratio
of semicolons to em-dashes, the words that appear in the top 1% but
would not appear in a generic LLM output, the structures that are
systematically absent.

An LLM prompted with "write in a casual tone" produces the training
distribution median of casual. An LLM prompted with a RegisterProfile
extracted from a real corpus produces something measurably closer to
the source voice, because the constraints are specific, quantified,
and falsifiable.
"""

from __future__ import annotations

import re
import string
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# --- Vocabulary classification heuristics ---

# Common Latinate prefixes and suffixes in English
LATINATE_SUFFIXES = (
    "tion", "sion", "ment", "ance", "ence", "ity", "ous", "ive",
    "able", "ible", "al", "ial", "ual", "ular", "ular", "ate",
    "ify", "ise", "ize", "ure",
)

SAXON_MARKERS = {
    "the", "a", "an", "and", "but", "or", "so", "yet", "for", "nor",
    "in", "on", "at", "to", "by", "up", "off", "out", "with", "from",
    "into", "onto", "upon", "over", "under", "about", "after", "before",
    "between", "through", "during", "without", "within", "along",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "need", "dare", "ought",
    "I", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "who", "whom", "whose", "which",
    "what", "where", "when", "why", "how",
    "not", "no", "nor", "neither", "never", "none", "nothing",
    "here", "there", "now", "then", "still", "just", "only",
    "all", "each", "every", "both", "few", "many", "much", "some",
    "any", "most", "other", "another",
    "good", "bad", "big", "small", "old", "new", "long", "short",
    "high", "low", "right", "wrong", "true", "false",
    "get", "go", "come", "make", "take", "give", "put", "set",
    "say", "tell", "ask", "know", "think", "see", "look", "find",
    "want", "let", "keep", "help", "start", "stop", "try", "work",
    "run", "walk", "stand", "sit", "lie", "fall", "rise", "hold",
    "turn", "move", "open", "close", "cut", "break", "build",
    "thing", "man", "woman", "child", "world", "life", "time",
    "day", "night", "year", "way", "part", "place", "hand", "eye",
    "head", "face", "body", "word", "name", "home", "house", "room",
    "door", "side", "end", "point", "line", "water", "fire", "earth",
    "light", "dark",
}

# Common LLM transition words (overrepresented in model outputs)
LLM_TRANSITION_WORDS = {
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "notwithstanding",
    "indeed", "certainly", "undoubtedly", "unquestionably",
    "essentially", "fundamentally", "ultimately", "inherently",
    "notably", "significantly", "importantly", "crucially",
    "interestingly", "surprisingly", "remarkably", "strikingly",
    "delve", "delves", "delving",
    "landscape", "paradigm", "ecosystem", "synergy",
    "leverage", "leveraging", "leveraged",
    "robust", "holistic", "comprehensive", "nuanced",
    "foster", "fostering", "fosters",
    "navigate", "navigating", "navigates",
    "multifaceted", "streamline", "pivotal",
    "tapestry", "beacon", "cornerstone",
}


@dataclass
class PunctuationProfile:
    """Statistical profile of punctuation usage."""

    semicolons_per_1000_words: float = 0.0
    em_dashes_per_1000_words: float = 0.0
    en_dashes_per_1000_words: float = 0.0
    colons_per_1000_words: float = 0.0
    exclamation_per_1000_words: float = 0.0
    question_per_1000_words: float = 0.0
    ellipsis_per_1000_words: float = 0.0
    parenthetical_per_1000_words: float = 0.0
    comma_splice_candidates: int = 0
    dash_to_semicolon_ratio: float = 0.0

    def describe(self) -> str:
        """Return a human-readable summary of the punctuation profile."""
        lines = []
        if self.semicolons_per_1000_words > 3.0:
            lines.append("Heavy semicolon user (above 3 per 1000 words)")
        elif self.semicolons_per_1000_words > 1.0:
            lines.append("Moderate semicolon user")
        else:
            lines.append("Light or no semicolon use")

        if self.em_dashes_per_1000_words > 3.0:
            lines.append("Heavy em-dash user")
        elif self.em_dashes_per_1000_words > 1.0:
            lines.append("Moderate em-dash user")
        else:
            lines.append("Light or no em-dash use")

        if self.dash_to_semicolon_ratio > 2.0:
            lines.append("Prefers dashes over semicolons for parentheticals")
        elif self.dash_to_semicolon_ratio < 0.5 and self.semicolons_per_1000_words > 1.0:
            lines.append("Prefers semicolons over dashes for parentheticals")

        if self.comma_splice_candidates > 0:
            lines.append(
                f"Comma splices detected: {self.comma_splice_candidates} "
                f"(may be intentional voice signature)"
            )

        return "; ".join(lines) if lines else "No distinctive punctuation patterns"


@dataclass
class SentenceProfile:
    """Statistical profile of sentence structure."""

    mean_length: float = 0.0
    median_length: float = 0.0
    std_dev: float = 0.0
    min_length: int = 0
    max_length: int = 0
    short_ratio: float = 0.0  # sentences under 8 words
    medium_ratio: float = 0.0  # sentences 8-20 words
    long_ratio: float = 0.0  # sentences 21-40 words
    very_long_ratio: float = 0.0  # sentences over 40 words
    subordination_markers_per_sentence: float = 0.0

    def describe(self) -> str:
        """Return a human-readable summary."""
        style_label = "mixed"
        if self.mean_length < 12:
            style_label = "terse"
        elif self.mean_length > 25:
            style_label = "expansive"
        elif self.long_ratio > 0.3:
            style_label = "complex-leaning"

        return (
            f"Sentence style: {style_label}. "
            f"Mean {self.mean_length:.1f} words, "
            f"median {self.median_length:.1f}, "
            f"range [{self.min_length}, {self.max_length}]. "
            f"Distribution: {self.short_ratio:.0%} short, "
            f"{self.medium_ratio:.0%} medium, "
            f"{self.long_ratio:.0%} long, "
            f"{self.very_long_ratio:.0%} very long."
        )


@dataclass
class VocabularyProfile:
    """Statistical profile of vocabulary usage."""

    total_words: int = 0
    unique_words: int = 0
    type_token_ratio: float = 0.0
    latinate_ratio: float = 0.0
    top_50_words: list[tuple[str, int]] = field(default_factory=list)
    distinctive_words: list[str] = field(default_factory=list)
    llm_transition_count: int = 0
    llm_transitions_found: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Return a human-readable summary."""
        register = "mixed"
        if self.latinate_ratio > 0.25:
            register = "latinate-dominant"
        elif self.latinate_ratio < 0.10:
            register = "saxon-dominant"

        lexical_density = "sparse"
        if self.type_token_ratio > 0.6:
            lexical_density = "dense"
        elif self.type_token_ratio > 0.4:
            lexical_density = "moderate"

        parts = [
            f"Vocabulary register: {register} (latinate ratio {self.latinate_ratio:.2%})",
            f"Lexical density: {lexical_density} (TTR {self.type_token_ratio:.3f})",
            f"Corpus size: {self.total_words} words, {self.unique_words} unique",
        ]

        if self.llm_transitions_found:
            parts.append(
                f"LLM-overrepresented transitions present: "
                f"{', '.join(self.llm_transitions_found[:10])}"
            )

        if self.distinctive_words:
            parts.append(
                f"Distinctive vocabulary (top 20): "
                f"{', '.join(self.distinctive_words[:20])}"
            )

        return "; ".join(parts)


@dataclass
class RegisterProfile:
    """
    Complete voice profile extracted from a corpus.

    This is the primary output of the VoiceCorpus extractor.
    It captures the statistical fingerprint of a human voice
    in enough detail to constrain an LLM at generation time.
    """

    name: str = ""
    source_description: str = ""
    sample_count: int = 0

    vocabulary: VocabularyProfile = field(default_factory=VocabularyProfile)
    sentences: SentenceProfile = field(default_factory=SentenceProfile)
    punctuation: PunctuationProfile = field(default_factory=PunctuationProfile)

    signature_phrases: list[str] = field(default_factory=list)
    signature_openings: list[str] = field(default_factory=list)
    signature_closings: list[str] = field(default_factory=list)

    anti_patterns: list[str] = field(default_factory=list)
    absent_transitions: list[str] = field(default_factory=list)

    raw_notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a complete human-readable summary of the voice profile."""
        sections = [
            f"=== Voice Profile: {self.name} ===",
            f"Source: {self.source_description}",
            f"Samples analyzed: {self.sample_count}",
            "",
            "--- Vocabulary ---",
            self.vocabulary.describe(),
            "",
            "--- Sentence Structure ---",
            self.sentences.describe(),
            "",
            "--- Punctuation ---",
            self.punctuation.describe(),
        ]

        if self.signature_phrases:
            sections.append("")
            sections.append("--- Signature Phrases ---")
            for phrase in self.signature_phrases[:15]:
                sections.append(f"  - \"{phrase}\"")

        if self.signature_openings:
            sections.append("")
            sections.append("--- Signature Openings ---")
            for opening in self.signature_openings[:10]:
                sections.append(f"  - \"{opening}\"")

        if self.signature_closings:
            sections.append("")
            sections.append("--- Signature Closings ---")
            for closing in self.signature_closings[:10]:
                sections.append(f"  - \"{closing}\"")

        if self.anti_patterns:
            sections.append("")
            sections.append("--- Anti-Patterns (never does this) ---")
            for pattern in self.anti_patterns:
                sections.append(f"  - {pattern}")

        if self.absent_transitions:
            sections.append("")
            sections.append("--- Absent Transitions (never uses these) ---")
            for t in self.absent_transitions[:20]:
                sections.append(f"  - {t}")

        return "\n".join(sections)

    def to_prompt_constraints(self) -> str:
        """
        Export the profile as a set of constraints suitable for
        insertion into an LLM system prompt.

        This is the bridge between extraction and generation.
        """
        constraints = []

        # Vocabulary constraints
        if self.vocabulary.latinate_ratio > 0.20:
            constraints.append(
                "Use a latinate register. Prefer words of Latin or French "
                "origin over Germanic equivalents where natural."
            )
        elif self.vocabulary.latinate_ratio < 0.10:
            constraints.append(
                "Use plain, direct language. Prefer short Germanic words "
                "over Latin-derived alternatives."
            )

        # Sentence length constraints
        if self.sentences.mean_length > 20:
            constraints.append(
                f"Write longer sentences. Target mean sentence length "
                f"around {self.sentences.mean_length:.0f} words. "
                f"Allow sentences up to {self.sentences.max_length} words."
            )
        elif self.sentences.mean_length < 12:
            constraints.append(
                f"Write short, direct sentences. Target mean sentence "
                f"length around {self.sentences.mean_length:.0f} words."
            )

        # Punctuation constraints
        if self.punctuation.semicolons_per_1000_words > 2.0:
            constraints.append(
                "Use semicolons freely to join related independent clauses."
            )
        if self.punctuation.em_dashes_per_1000_words < 1.0:
            constraints.append(
                "Avoid em-dashes. Use semicolons or commas instead "
                "for parenthetical insertions."
            )
        elif self.punctuation.em_dashes_per_1000_words > 3.0:
            constraints.append(
                "Use em-dashes for parenthetical insertions and "
                "mid-sentence pivots."
            )

        # Signature phrases
        if self.signature_phrases:
            phrases = "; ".join(f'"{p}"' for p in self.signature_phrases[:5])
            constraints.append(
                f"Characteristic phrases from this voice (use sparingly, "
                f"not in every paragraph): {phrases}"
            )

        # Anti-patterns
        if self.absent_transitions:
            absent = ", ".join(self.absent_transitions[:15])
            constraints.append(
                f"NEVER use these words or transitions (they are absent "
                f"from the source voice and mark the output as AI-generated): "
                f"{absent}"
            )

        if self.anti_patterns:
            for ap in self.anti_patterns[:10]:
                constraints.append(f"AVOID: {ap}")

        return "\n".join(f"- {c}" for c in constraints)


class VoiceCorpus:
    """
    Ingests human-written text samples and extracts a statistical
    voice profile (RegisterProfile).

    Usage:
        corpus = VoiceCorpus(name="My Voice")
        corpus.add_sample("First piece of writing...")
        corpus.add_sample("Second piece of writing...")
        profile = corpus.extract()
        print(profile.summary())
        print(profile.to_prompt_constraints())
    """

    def __init__(self, name: str = "unnamed", description: str = ""):
        self.name = name
        self.description = description
        self._samples: list[str] = []

    def add_sample(self, text: str, label: str = "") -> None:
        """Add a text sample to the corpus."""
        cleaned = text.strip()
        if cleaned:
            self._samples.append(cleaned)

    def add_samples(self, texts: list[str]) -> None:
        """Add multiple text samples at once."""
        for text in texts:
            self.add_sample(text)

    @property
    def full_text(self) -> str:
        """Concatenated text of all samples."""
        return "\n\n".join(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def extract(self) -> RegisterProfile:
        """
        Run the full extraction pipeline and return a RegisterProfile.

        This is the primary method. It runs:
        1. Tokenization
        2. Vocabulary analysis
        3. Sentence structure analysis
        4. Punctuation analysis
        5. Pattern extraction (openings, closings, signature phrases)
        6. Anti-pattern detection
        """
        if not self._samples:
            raise ValueError("No samples in corpus. Add samples before extracting.")

        full = self.full_text
        words = self._tokenize_words(full)
        sentences = self._split_sentences(full)

        profile = RegisterProfile(
            name=self.name,
            source_description=self.description,
            sample_count=self.sample_count,
        )

        profile.vocabulary = self._analyze_vocabulary(words)
        profile.sentences = self._analyze_sentences(sentences)
        profile.punctuation = self._analyze_punctuation(full, len(words))

        profile.signature_phrases = self._extract_recurring_phrases(words)
        profile.signature_openings = self._extract_openings()
        profile.signature_closings = self._extract_closings()

        profile.anti_patterns = self._detect_anti_patterns(full, words)
        profile.absent_transitions = self._detect_absent_transitions(words)

        return profile

    # --- Tokenization ---

    def _tokenize_words(self, text: str) -> list[str]:
        """Split text into lowercase word tokens, stripping punctuation."""
        raw_tokens = text.split()
        words = []
        for token in raw_tokens:
            cleaned = token.strip(string.punctuation + "\u2014\u2013\u2018\u2019\u201c\u201d")
            if cleaned:
                words.append(cleaned.lower())
        return words

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences. Uses a simple heuristic:
        split on period, exclamation, question mark followed by
        whitespace and a capital letter, or by end of string.
        """
        # Normalize line breaks
        text = re.sub(r"\n{2,}", ". ", text)
        text = re.sub(r"\n", " ", text)

        # Split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00FF\u00AB])", text)
        sentences = [s.strip() for s in raw if s.strip() and len(s.strip().split()) >= 2]
        return sentences

    # --- Vocabulary analysis ---

    def _analyze_vocabulary(self, words: list[str]) -> VocabularyProfile:
        """Analyze vocabulary distribution."""
        total = len(words)
        if total == 0:
            return VocabularyProfile()

        freq = Counter(words)
        unique = len(freq)
        ttr = unique / total if total > 0 else 0.0

        # Latinate ratio: proportion of non-function words with latinate suffixes
        content_words = [w for w in words if w not in SAXON_MARKERS and len(w) > 3]
        latinate_count = sum(
            1 for w in content_words
            if any(w.endswith(suf) for suf in LATINATE_SUFFIXES)
        )
        latinate_ratio = latinate_count / len(content_words) if content_words else 0.0

        # Top 50 by frequency
        top_50 = freq.most_common(50)

        # Distinctive words: words that appear 3+ times but are not in the
        # top 200 most common English words (approximated by SAXON_MARKERS)
        distinctive = [
            w for w, c in freq.items()
            if c >= 3 and w not in SAXON_MARKERS and len(w) > 4
        ]
        distinctive.sort(key=lambda w: freq[w], reverse=True)

        # LLM transition detection
        llm_found = [w for w in LLM_TRANSITION_WORDS if w in freq]

        return VocabularyProfile(
            total_words=total,
            unique_words=unique,
            type_token_ratio=ttr,
            latinate_ratio=latinate_ratio,
            top_50_words=top_50,
            distinctive_words=distinctive[:30],
            llm_transition_count=sum(freq[w] for w in llm_found),
            llm_transitions_found=llm_found,
        )

    # --- Sentence analysis ---

    def _analyze_sentences(self, sentences: list[str]) -> SentenceProfile:
        """Analyze sentence length distribution and structure."""
        if not sentences:
            return SentenceProfile()

        lengths = [len(s.split()) for s in sentences]
        total = len(lengths)

        subordination_markers = [
            "because", "although", "though", "while", "whereas",
            "since", "unless", "until", "if", "when", "where",
            "which", "who", "whom", "whose", "that",
        ]

        sub_counts = []
        for s in sentences:
            words_lower = s.lower().split()
            count = sum(1 for w in words_lower if w in subordination_markers)
            sub_counts.append(count)

        short = sum(1 for l in lengths if l < 8)
        medium = sum(1 for l in lengths if 8 <= l <= 20)
        long = sum(1 for l in lengths if 21 <= l <= 40)
        very_long = sum(1 for l in lengths if l > 40)

        return SentenceProfile(
            mean_length=statistics.mean(lengths),
            median_length=statistics.median(lengths),
            std_dev=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            min_length=min(lengths),
            max_length=max(lengths),
            short_ratio=short / total,
            medium_ratio=medium / total,
            long_ratio=long / total,
            very_long_ratio=very_long / total,
            subordination_markers_per_sentence=(
                statistics.mean(sub_counts) if sub_counts else 0.0
            ),
        )

    # --- Punctuation analysis ---

    def _analyze_punctuation(self, text: str, word_count: int) -> PunctuationProfile:
        """Analyze punctuation usage patterns."""
        if word_count == 0:
            return PunctuationProfile()

        factor = 1000.0 / word_count

        semicolons = text.count(";")
        em_dashes = text.count("\u2014") + len(re.findall(r"(?<!\-)---?(?!\-)", text))
        en_dashes = text.count("\u2013")
        colons = text.count(":")
        exclamations = text.count("!")
        questions = text.count("?")
        ellipses = text.count("...") + text.count("\u2026")
        parentheticals = text.count("(")

        # Comma splice detection: comma followed by a space and a pronoun
        # or common subject that starts a new independent clause
        comma_splice_pattern = re.compile(
            r",\s+(?:I|he|she|it|we|they|you|this|that|there|here|"
            r"the|a|an|my|his|her|its|our|their)\s+(?:is|am|are|was|were|"
            r"have|has|had|do|does|did|will|would|can|could|shall|should|"
            r"may|might|must)\b",
            re.IGNORECASE,
        )
        comma_splices = len(comma_splice_pattern.findall(text))

        dash_count = em_dashes + en_dashes
        dash_semicolon_ratio = (
            dash_count / semicolons if semicolons > 0 else float(dash_count)
        )

        return PunctuationProfile(
            semicolons_per_1000_words=semicolons * factor,
            em_dashes_per_1000_words=em_dashes * factor,
            en_dashes_per_1000_words=en_dashes * factor,
            colons_per_1000_words=colons * factor,
            exclamation_per_1000_words=exclamations * factor,
            question_per_1000_words=questions * factor,
            ellipsis_per_1000_words=ellipses * factor,
            parenthetical_per_1000_words=parentheticals * factor,
            comma_splice_candidates=comma_splices,
            dash_to_semicolon_ratio=dash_semicolon_ratio,
        )

    # --- Pattern extraction ---

    def _extract_recurring_phrases(self, words: list[str]) -> list[str]:
        """
        Extract recurring multi-word phrases (bigrams and trigrams)
        that appear across multiple samples.
        """
        phrases: list[str] = []

        # Bigram extraction
        bigram_counter: Counter[str] = Counter()
        for i in range(len(words) - 1):
            if words[i] not in SAXON_MARKERS or words[i + 1] not in SAXON_MARKERS:
                bigram = f"{words[i]} {words[i + 1]}"
                bigram_counter[bigram] += 1

        # Trigram extraction
        trigram_counter: Counter[str] = Counter()
        for i in range(len(words) - 2):
            has_content = any(w not in SAXON_MARKERS for w in words[i : i + 3])
            if has_content:
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                trigram_counter[trigram] += 1

        # Keep phrases that appear 3+ times
        for phrase, count in bigram_counter.most_common(50):
            if count >= 3:
                phrases.append(phrase)

        for phrase, count in trigram_counter.most_common(50):
            if count >= 3:
                phrases.append(phrase)

        return phrases[:30]

    def _extract_openings(self) -> list[str]:
        """Extract the first sentence of each sample as opening patterns."""
        openings = []
        for sample in self._samples:
            sentences = self._split_sentences(sample)
            if sentences:
                first = sentences[0].strip()
                if len(first.split()) >= 3:
                    openings.append(first)
        return openings

    def _extract_closings(self) -> list[str]:
        """Extract the last sentence of each sample as closing patterns."""
        closings = []
        for sample in self._samples:
            sentences = self._split_sentences(sample)
            if sentences:
                last = sentences[-1].strip()
                if len(last.split()) >= 3:
                    closings.append(last)
        return closings

    # --- Anti-pattern detection ---

    def _detect_anti_patterns(self, text: str, words: list[str]) -> list[str]:
        """
        Detect structural patterns that are systematically absent
        from the corpus. Their absence is a voice signal.
        """
        anti_patterns = []
        text_lower = text.lower()

        # Check for absence of common LLM constructions
        checks = [
            ("Exclamation marks", text.count("!") == 0, "Never uses exclamation marks"),
            (
                "Question marks",
                text.count("?") == 0,
                "Never uses rhetorical questions in writing",
            ),
            (
                "First person plural",
                "we " not in text_lower and "our " not in text_lower,
                "Never uses first-person plural (we/our)",
            ),
            (
                "Bullet points",
                "- " not in text and "* " not in text,
                "Never uses bullet-point lists",
            ),
            (
                "Em-dashes",
                "\u2014" not in text and " -- " not in text and " - " not in text.replace("- ", ""),
                "Never uses em-dashes or en-dashes as parentheticals",
            ),
        ]

        for _name, condition, description in checks:
            if condition:
                anti_patterns.append(description)

        return anti_patterns

    def _detect_absent_transitions(self, words: list[str]) -> list[str]:
        """
        Find LLM-overrepresented transition words that are absent
        from the corpus. These are strong anti-signals.
        """
        word_set = set(words)
        absent = [t for t in sorted(LLM_TRANSITION_WORDS) if t not in word_set]
        return absent
