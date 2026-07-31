# Methodology: AI Voice Transfer

## Why Voice Transfer Is Hard

An LLM is a compression of its training data. When it generates text, it produces the most probable continuation given the prompt, which means it converges toward the statistical center of whatever genre the prompt implies. That center is not bad writing. It is *median* writing: grammatically correct, stylistically competent, and completely anonymous.

The anonymity is the problem. A human voice is not the median of anything. It is a specific, asymmetric, idiosyncratic distribution of choices: this word and not that word, this sentence length and not that one, this punctuation mark and not the other. The asymmetry is the identity. Remove it, and you get text that reads like it was written by a capable stranger.

Current approaches to voice matching fail because they operate at the wrong level of abstraction. "Write in a casual tone" is a direction, not a constraint. "Mimic this style" gives the model a target it cannot measure its distance from. "You are a witty, direct writer" asks the model to interpolate between its training examples of "witty" and "direct," which produces the model's *concept* of those qualities, not the person's *instantiation* of them.

## The Insight: Voice Is Measurable

Voice is not a feeling. It is a set of measurable features:

**Vocabulary distribution.** The ratio of Latinate to Saxon-origin words. The type-token ratio (lexical density). The specific words that appear in the top 1% of the person's writing but would not appear in a generic LLM output. The words that are systematically *absent* from their writing (often the strongest signal).

**Sentence structure.** Mean sentence length, standard deviation, the ratio of short to long sentences. Subordination depth (how many dependent clauses per sentence). The coefficient of variation across a document (low CV means monotonous rhythm, which is an LLM fingerprint).

**Punctuation habits.** Semicolons per 1000 words. Em-dashes per 1000 words. The ratio of dashes to semicolons (some writers use one, some use the other; almost no one uses both at the same rate). Comma splices (when intentional, they are a strong voice marker). Exclamation marks (their absence is often more diagnostic than their presence).

**Structural patterns.** How the person opens (thesis-first, scene-first, question-first). How they close (restatement, concrete proposal, trailing observation). Whether they use lists. Whether they use paragraph breaks as rhetorical tools.

**Anti-patterns.** What the person *never* does is often more informative than what they do. If a person never uses "moreover" or "furthermore" in 50,000 words of writing, the appearance of those words in a generated draft is a signal of model drift.

Once these features are quantified, they become constraints that can be injected into a prompt and validated against the output.

## The Five-Stage Pipeline

### Stage 1: Voice Extraction

The `VoiceCorpus` class ingests text samples and produces a `RegisterProfile`: a structured summary of the voice's measurable features.

The extraction is purely statistical. It does not attempt to "understand" the voice; it measures it. The resulting profile includes:

- Vocabulary: total words, unique words, type-token ratio, Latinate ratio, top-50 words, distinctive words, LLM-overrepresented transitions found
- Sentences: mean length, median, standard deviation, min/max, distribution across four length buckets, subordination markers per sentence
- Punctuation: semicolons, em-dashes, en-dashes, colons, exclamation marks, question marks, ellipses, parentheticals (all per 1000 words), comma splice candidates, dash-to-semicolon ratio
- Signature phrases: recurring bigrams and trigrams
- Opening and closing patterns: first and last sentences of each sample
- Anti-patterns: structures systematically absent from the corpus
- Absent transitions: LLM-overrepresented words that never appear

### Stage 2: Register Classification

Most people do not write in one voice. They write in several registers of the same voice. A professional email is not the same as a LinkedIn post is not the same as a personal message. The vocabulary shifts, the sentence length shifts, the punctuation shifts, but the underlying identity remains.

The `RegisterLibrary` manages these registers as separate configurations, each with its own markers (what signals this register is active), anti-markers (what signals a different register is leaking in), sample phrases, recipient classes, structural constraints, and vocabulary notes.

Register selection happens before drafting, not after. The pipeline asks: who is the recipient? What is the formality level? What is the context? Then it selects the register whose recipient class and formality level match, and injects that register's constraints into the prompt.

### Stage 3: Lexicon Branding

The gap between Stage 1 (descriptive) and the LLM prompt (prescriptive) is bridged by the `LexiconBrand`. Each of its fifteen modules converts a measured voice feature into an executable rule:

- "The source voice uses semicolons at 3.2 per 1000 words and em-dashes at 0.4 per 1000 words" becomes "Use semicolons for clause joining; avoid em-dashes."
- "The source voice never uses 'moreover,' 'furthermore,' 'additionally'" becomes "NEVER use these connectors: moreover, furthermore, additionally."
- "The source voice opens with concrete observations, never with 'In today's...'" becomes "Open with the thesis or a concrete fact. Never open with temporal framing."

Each module also carries a test function that can be run against a draft, returning specific violations with severity scores and fix suggestions. The self-critique runner iterates all modules and produces a scored report.

### Stage 4: Fingerprint Detection

Even with good constraints, LLMs drift. The fingerprint detector catches the drift by scanning for ten canonical LLM writing tics:

1. **Staccato three-word closes** ("Bold. Brave. Necessary.") - the single most recognizable LLM tic
2. **Hook-then-reveal openers** ("Compliance is the theatre. Enforcement is the physics.")
3. **"Here is what makes this different" formulas** - self-referential uniqueness claims
4. **"The [class] you would [verb] is that..." constructions** - tour-guide positioning
5. **Semicolon-list credential dumps** - density without argument
6. **Aphoristic closes** - fortune-cookie wisdom as conclusion
7. **Em-dash overuse** - parenthetical insertion every other sentence
8. **Theatrical tricolons** - three items for rhythm, not for meaning
9. **Generic-plausible specificity** - sounds specific, could describe anyone
10. **Mechanical parallelism** - "On one hand... on the other hand"

Plus twenty supplementary vocabulary patterns (delve, leverage, robust, holistic, nuanced, landscape, tapestry, beacon, foster, navigate, resonate, underscore, unpack, deep dive, and specific opener formulas).

### Stage 5: Validation and Scoring

The `VoiceFidelityScorer` combines all dimensions into a single 0-100 score with weighted components:

- **Vocabulary match** (weight 1.0): distance between draft and profile Latinate ratio, plus bonus for using distinctive words
- **Structure match** (weight 1.0): mean sentence length distance plus variation similarity
- **Punctuation match** (weight 0.8): semicolon and em-dash rate distance
- **Fingerprint score** (weight 1.5): penalty per detected LLM fingerprint, weighted by severity
- **Lexicon brand compliance** (weight 1.2): average module pass rate

The weights reflect a deliberate bias: fingerprint absence matters more than any single positive feature, because a single LLM fingerprint can destroy the illusion of human authorship while vocabulary or structure being slightly off merely makes the voice feel "less precise."

## What the Pipeline Does Not Do

The pipeline does not call an LLM. It produces constraints and validates output. The actual generation is the caller's responsibility, because different users will use different providers. The pipeline's value is in what surrounds the generation: the extraction that produces specific constraints, and the validation that catches the drift.

The pipeline does not guarantee voice fidelity. A score of 85 means the draft is recognizably in the right voice with minor issues. A score of 100 would indicate overfitting. The target is the range of 75-90, where the output is distinctively voiced without being a mechanical reproduction.

The pipeline does not replace human judgment. The self-critique report flags issues; the human decides which ones matter. Some violations are intentional (a professional writer might use "moreover" once in a specific context where it is the right word). The pipeline's job is to flag, not to override.

## Theoretical Grounding

Three academic references inform the framework's design:

**Register Analysis for Style Transfer** (arXiv 2505.00679, 2025) establishes that LLMs systematically apply their own trained style to source voices unless explicit register-preservation constraints are inserted in the prompt. This finding motivates the entire pipeline: without explicit constraints, the model will drift toward its training distribution regardless of how well it understands the target voice.

**Voice Cloning is Style Transfer** (arXiv 2605.16578, 2025) demonstrates empirically that text voice transfer follows the same mechanics as audio voice cloning. The target voice must be decomposed into measurable features and reassembled through feature-level constraints, not through end-to-end imitation. This motivates the statistical extraction approach over the "mimic this example" approach.

**Cultural Marker Erasure** (arXiv 2602.22145, February 2026) documents 29% erasure of cultural markers in LLM output and shows that explicit preservation prompts reduce the erasure rate significantly. This motivates Module 15 (Imperfection Preservation) and the anti-pattern detection: the things a voice *does not do* and the imperfections it *does have* are both part of the identity, and the model's default behavior is to erase both.

## Production Validation

This framework was built and validated in production use over three months, across seven distinct registers, against 30+ drafts for a single voice. The failure modes catalogued in the fingerprint detector are not theoretical; each one was observed empirically in generated output, flagged by the human writer, and codified as a detection pattern.

The most important finding from production use: **the extraction step matters more than the generation step.** A well-extracted profile with specific constraints injected into a vanilla prompt produces better voice fidelity than a poorly-extracted profile with an elaborate generation architecture. The constraint quality is the bottleneck, not the model quality.
