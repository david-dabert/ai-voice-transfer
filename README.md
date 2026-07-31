# AI Voice Transfer

**Make any LLM write like you, not like an AI.**

## The Problem

Every LLM output sounds the same. Not bad, not wrong, but the same. The same em-dash parentheticals. The same "In today's rapidly evolving landscape." The same three-word staccato close. The same twelve words that no human uses at the rate LLMs use them: *delve, leverage, robust, holistic, nuanced, foster, navigate, tapestry, beacon, landscape, resonate, underscore.*

The output converges toward the statistical center of the training distribution. That center sounds like a competent, anonymous, mildly enthusiastic professional writer. The problem is not quality. It is identity. None of it sounds like you.

## Why Current Approaches Fail

**"Write in a casual tone"** produces the training distribution median of casual. It does not produce *your* casual.

**"Mimic this style"** with a few example sentences gives the model a direction, not a boundary. It drifts back toward the median within two paragraphs.

**"You are a writer who..."** followed by adjectives (witty, direct, analytical) produces the model's interpolation of those adjectives, not a real person's voice.

The failure is architectural. A system prompt describes *what* to aim for. It does not describe *what to avoid*, does not quantify the target, and does not validate the output. Voice transfer requires all three.

## The Solution

AI Voice Transfer is a framework that treats voice as a measurable, transferable profile rather than a tone adjective. It operates in five stages:

1. **Voice Extraction** - Analyze a corpus of your actual writing. Extract statistical fingerprints: vocabulary distribution, sentence length patterns, punctuation habits, recurring phrases, and systematically absent patterns.

2. **Register Classification** - Most people write in multiple distinct registers (professional email, personal message, analytical essay, social media). Classify and manage each register separately rather than collapsing them into one "style."

3. **Lexicon Branding** - Convert descriptive analysis into prescriptive rules. Each rule is executable at draft time: "use semicolons instead of em-dashes," "never open with a performative emotion," "latinate ratio above 20%."

4. **Fingerprint Detection** - Scan drafts for the specific stylistic moves that mark text as LLM-generated: staccato closes, hook-then-reveal openers, credential dumps, aphoristic wisdom, theatrical tricolons, and vocabulary that no human overuses the way models do.

5. **Self-Critique Pipeline** - Run the draft against all prescriptive rules, score it on multiple dimensions, and return specific fix suggestions before the text ships.

## Architecture

```
                                    +------------------+
                                    |  Voice Corpus    |
                                    |  (your writing)  |
                                    +--------+---------+
                                             |
                                             v
                                    +------------------+
                                    |  Voice Extractor |
                                    |  (statistical    |
                                    |   analysis)      |
                                    +--------+---------+
                                             |
                              +--------------+--------------+
                              |                             |
                              v                             v
                    +------------------+          +------------------+
                    | Register Library |          |  Lexicon Brand   |
                    | (multiple voice  |          |  (prescriptive   |
                    |  modes)          |          |   rules)         |
                    +--------+---------+          +--------+---------+
                              |                             |
                              +--------------+--------------+
                                             |
                                             v
                                    +------------------+
                                    |  Prompt          |
                                    |  Constraints     |
                                    |  (injected into  |
                                    |   LLM system     |
                                    |   prompt)        |
                                    +--------+---------+
                                             |
                                             v
                                    +------------------+
                                    |  LLM Generation  |
                                    |  (any provider)  |
                                    +--------+---------+
                                             |
                              +--------------+--------------+
                              |                             |
                              v                             v
                    +------------------+          +------------------+
                    |  Fingerprint     |          |  Self-Critique   |
                    |  Detector        |          |  (lexicon brand  |
                    |  (LLM tic scan)  |          |   validation)    |
                    +--------+---------+          +--------+---------+
                              |                             |
                              +--------------+--------------+
                                             |
                                             v
                                    +------------------+
                                    |  Fidelity Score  |
                                    |  + Fix Report    |
                                    +------------------+
```

## Quick Start

```bash
pip install ai-voice-transfer
```

```python
from voice_transfer import VoiceTransferPipeline, DraftRequest

# 1. Initialize with your writing samples
pipeline = VoiceTransferPipeline(name="my_voice")
profile = pipeline.load_corpus(
    samples=[
        "Your first writing sample goes here...",
        "Your second writing sample goes here...",
        "More samples produce better extraction.",
    ],
    name="My Voice",
    description="Professional emails and analytical writing",
)

# 2. See your extracted voice profile
print(profile.summary())

# 3. Activate the default lexicon brand (15 prescriptive modules)
pipeline.use_default_brand()

# 4. Build constraints for your LLM prompt
constraints = pipeline.build_constraints()
# Inject `constraints` into your LLM system prompt

# 5. Validate a draft (from any LLM or written by hand)
result = pipeline.validate("Your draft text here...")
print(result.summary())
# Score: 72.3/100
# HARD: Banned connector: 'moreover'
# FINGERPRINT: llm_vocab_delve - Found: "delve into the..."

# 6. Full pipeline with custom generator
def my_llm(system_prompt: str, user_prompt: str) -> str:
    # Call your LLM of choice here
    return "generated text..."

result = pipeline.run(
    DraftRequest(
        content_prompt="Write a brief analysis of...",
        target_register="analytical_essay",
        recipient="publication reader",
    ),
    generator=my_llm,
    max_iterations=2,  # retry if first draft fails validation
)
```

See `examples/quick_start.py` for a complete working example with sample data.

## Components

### Voice Extractor (`voice_transfer/extractor.py`)

Analyzes your writing corpus to extract:
- **Vocabulary profile**: latinate vs saxon ratio, type-token ratio, distinctive words, LLM-overrepresented transitions
- **Sentence profile**: length distribution, subordination depth, variation coefficient
- **Punctuation profile**: semicolons per 1000 words, em-dash ratio, comma splice frequency
- **Pattern extraction**: recurring phrases, opening formulas, closing formulas
- **Anti-pattern detection**: words and structures systematically absent from your writing

### Register Library (`voice_transfer/register_library.py`)

Manages multiple voice modes:
- Professional email, personal message, analytical essay, social media
- Register selection by recipient class, formality level, and context
- Register validation (does this draft match the selected register?)
- Best-match detection (which register does this text belong to?)

### Lexicon Brand (`voice_transfer/lexicon_brand.py`)

Fifteen prescriptive modules that convert voice description into executable rules:

| # | Module | What It Checks |
|---|--------|---------------|
| 1 | Base Register | Vocabulary origin consistency |
| 2 | Connector Discipline | Bans LLM-overrepresented transitions |
| 3 | Imagery Preferences | Bans abstract filler metaphors |
| 4 | Self-Reference | Flags excessive first-person density |
| 5 | Opening Formulas | Bans performative openers |
| 6 | Closing Formulas | Bans engagement bait closers |
| 7 | Punctuation Rules | Em-dash limits, exclamation control |
| 8 | Sentence Structure | Monotonous rhythm detection |
| 9 | Forbidden Vocabulary | LLM-fingerprint word ban list |
| 10 | Forbidden Structures | Structural pattern ban list |
| 11 | Signature Phrases | Checks presence/absence of voice markers |
| 12 | Humor/Tone | Bans generic internet humor |
| 13 | Cultural References | Bans generic quotation attribution |
| 14 | Formality Calibration | Detects casual/formal register mixing |
| 15 | Imperfection Preservation | Flags suspiciously uniform output |

### Fingerprint Detector (`voice_transfer/fingerprint_detector.py`)

Detects ten canonical LLM writing tics plus twenty supplementary vocabulary patterns. Full catalogue in `docs/fingerprints.md`.

### Voice Fidelity Scorer (`voice_transfer/scorer.py`)

Multi-dimensional scoring: vocabulary match, structure match, punctuation match, fingerprint absence, lexicon brand compliance. Returns a 0-100 score with human-readable verdicts.

## Academic Foundation

This framework draws on three lines of research:

1. **Register Analysis for Style Transfer** (arXiv 2505.00679, 2025). Documents that LLMs systematically apply their own trained style to source voices unless explicit register-preservation constraints are inserted in the prompt. The extractor and register library implement those constraints.

2. **Voice Cloning is Style Transfer** (arXiv 2605.16578, 2025). Confirms empirically that voice transfer in text follows the same mechanics as audio voice cloning: the target voice must be decomposed into measurable features (vocabulary, structure, rhythm) and reassembled through feature-level constraints rather than end-to-end imitation.

3. **Cultural Marker Erasure** (arXiv 2602.22145, February 2026). Documents 29% cultural marker erasure in LLM outputs and demonstrates that explicit preservation prompts reduce erasure significantly. The lexicon brand's Module 15 (Imperfection Preservation) implements this finding directly.

## Configuration

See `examples/config.yaml` for a complete configuration template.

## Requirements

- Python 3.9+
- PyYAML (for configuration files)
- No LLM API keys required (the framework produces constraints and validates output; the LLM call is yours)

## License

MIT License. See LICENSE file.

## Author

David Dabert

Built from the operational problem of making LLMs write in a specific human voice across seven distinct registers, validated against 30+ drafts over three months of production use.
