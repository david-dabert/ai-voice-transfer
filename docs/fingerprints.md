# LLM Fingerprint Catalogue

A complete catalogue of LLM writing tics detectable by the fingerprint scanner. Each entry includes the pattern, why it is a fingerprint, a real-world example, and the recommended fix.

## The Ten Canonical Fingerprints

### 1. Staccato Three-Word Close

**Pattern:** Three short sentences (1-3 words each) at the end of a text, used for rhetorical punch.

**Why it is a fingerprint:** Human writers occasionally use short sentences for emphasis. LLMs do it at the end of nearly every piece as a rhythmic closer. The pattern is so consistent across models and providers that readers now recognize it instantly.

**Example (LLM-generated):**
> The project delivered results. The team grew stronger. The mission continues.
>
> Simple. Powerful. Undeniable.

**Example (human):**
> The project delivered what the proposal promised, which was less than what the team needed and more than what the budget could sustain. We will revisit the scope in Q3.

**Fix:** Delete the staccato close entirely. End with the final substantive point or a concrete next step. If the text needs emphasis at the end, use a single sentence with real content.

**Severity:** 0.9 (HIGH)

---

### 2. Hook-Then-Reveal Opener

**Pattern:** Two sentences where the first names a surface phenomenon and the second names the "real" mechanism underneath. Format: "X is the theatre. Y is the physics."

**Why it is a fingerprint:** This construction creates an impression of insight through structure rather than through content. Human writers occasionally use it; LLMs use it as a default opening move across all genres.

**Example (LLM-generated):**
> Compliance is the theatre. Enforcement is the physics. What happens between the policy document and the warehouse floor is where the real story lives.

**Example (human):**
> Between the policy document and the warehouse floor, three compliance steps are supposed to happen. In practice, none of them do.

**Fix:** Open with the thesis directly. State what you mean without staging the insight through a two-part metaphor.

**Severity:** 0.8 (HIGH)

---

### 3. "Here Is What Makes This Different"

**Pattern:** A self-referential claim of uniqueness, typically "Here is what makes this different/unique/special/stand out."

**Why it is a fingerprint:** This is a meta-comment on the text itself rather than a substantive claim. It tells the reader "I am about to say something important" instead of saying the important thing. LLMs insert it as a transition to the "value proposition" section of any analytical text.

**Example (LLM-generated):**
> Many organizations track guest complaints. Here is what makes this approach different: it connects the complaint to the room number, the shift, and the resolution time.

**Example (human):**
> The tracking connects the complaint to the room, the shift, and the resolution time. Most systems do not.

**Fix:** Delete the formula entirely. Show the difference through evidence and let the reader conclude.

**Severity:** 0.9 (HIGH)

---

### 4. "The [Class] You Would [Verb] Is That..."

**Pattern:** "The pattern you would notice is that..." or "The thing you would expect is that..." Positions the reader as needing a guided tour.

**Why it is a fingerprint:** Human experts state what they see. LLMs position themselves as tour guides walking the reader through an experience. The construction implies the reader cannot see the pattern without help.

**Example (LLM-generated):**
> The trend you would observe across these data points is that smaller teams consistently outperform larger ones on cycle time.

**Example (human):**
> Smaller teams outperform larger ones on cycle time, consistently, across every data point in the set.

**Fix:** State the point directly. "The pattern is X" rather than "The pattern you would notice is that X."

**Severity:** 0.7 (HIGH)

---

### 5. Semicolon-List Credential Dump

**Pattern:** A list of adjectives or credentials separated by semicolons, deployed for density rather than argument. "Multilingual; cross-cultural; senior-tested; governance-ready."

**Why it is a fingerprint:** This construction compresses qualifications into a scannable list, which is an LLM optimization for token efficiency. Human writers integrate credentials into sentences where each one earns its place through context.

**Example (LLM-generated):**
> His profile is distinctive: multilingual; analytically rigorous; operationally tested; governance-aware; culturally fluent across four continents.

**Example (human):**
> He ran a fifteen-partner DFID consortium in Dakar for eighteen months and resolved three donors' confidence crises in six. He did it in French, English, and Wolof, in that order.

**Fix:** Integrate credentials into a sentence where each one is backed by a specific example. Or keep only the two most relevant and attach evidence.

**Severity:** 0.6 (MEDIUM)

---

### 6. Aphoristic Close

**Pattern:** A fortune-cookie wisdom sentence at the end that could apply to any topic. "And that's what real leadership looks like." "At the end of the day, it all comes down to people."

**Why it is a fingerprint:** LLMs are trained on text that often ends with summary wisdom, and they reproduce this as a closing reflex. Human writers more often end with a concrete next step, a restated thesis, or the final piece of evidence.

**Example (LLM-generated):**
> The future of hospitality belongs to those who listen. And that makes all the difference.

**Example (human):**
> The pilot costs one reconfigured floor for one season. If the demand I see nightly holds across the network, the decision to extend will be made on evidence.

**Fix:** Delete the aphorism. End with the final substantive point, a concrete proposal, or the restated thesis.

**Severity:** 0.7 (HIGH)

---

### 7. Em-Dash Overuse

**Pattern:** Parenthetical insertion using em-dashes (or double hyphens) at a rate above 3 per 1000 words. Above 5 per 1000 words is almost certainly LLM-generated.

**Why it is a fingerprint:** Em-dashes are versatile and LLMs lean on them heavily for mid-sentence qualification. Most human writers have a preferred parenthetical tool (semicolons, commas, parentheses, or em-dashes) and use it consistently. The rate is what distinguishes LLM from human: LLMs use em-dashes at 2-3x the rate of typical human writing.

**Example (LLM-generated):**
> The hotel -- which has operated for over a decade -- recently implemented a new system -- one that tracks complaints by room -- and the results have been -- in a word -- transformative.

**Example (human):**
> The hotel has operated for over a decade. It recently implemented a complaint-tracking system by room, and the results were immediate.

**Fix:** Replace most em-dashes with semicolons, commas, or restructure to eliminate the parenthetical.

**Severity:** 0.4-0.7 (varies by rate)

---

### 8. Theatrical Tricolon

**Pattern:** Three items listed for rhythm rather than meaning. The third item adds no information; it exists for cadence. "Smart, determined, and relentless."

**Why it is a fingerprint:** The rule of three is a real rhetorical device, but LLMs deploy it mechanically. The third item is often a near-synonym of the first or second, adding rhythm without meaning.

**Example (LLM-generated):**
> The approach was thoughtful, comprehensive, and transformative.

**Example (human):**
> The approach covered six countries and twelve partners, which was more than the budget assumed.

**Fix:** Keep only the two items that carry distinct meaning. If the third adds nothing, cut it.

**Severity:** 0.5 (MEDIUM)

---

### 9. Generic-Plausible Specificity

**Pattern:** Text that sounds specific but could apply to anyone or any organization. "With over a decade of experience," "a unique blend of," "proven track record of delivering results."

**Why it is a fingerprint:** LLMs generate plausible-sounding specifics that are actually generic because they lack access to the actual facts. Human writers with real knowledge provide falsifiable specifics: names, numbers, dates, outcomes.

**Example (LLM-generated):**
> With over a decade of experience in international development, she brings a unique blend of strategic thinking and operational execution to every engagement.

**Example (human):**
> She coordinated a fifteen-partner DFID consortium in Dakar from 2018 to 2019, recovered three donors' confidence in six months, and delivered the final report on time for the first time in the programme's history.

**Fix:** Replace every generic claim with a specific, falsifiable fact. Name the programme, the number, the outcome, the date.

**Severity:** 0.6 (MEDIUM)

---

### 10. Mechanical Parallelism

**Pattern:** "On one hand X. On the other hand Y." or equivalent paired constructions that stage balanced analysis through structure rather than through argument.

**Why it is a fingerprint:** Human analysis is rarely this balanced. Real arguments have asymmetric evidence: one side is stronger than the other, and the writer's job is to show why. LLMs default to symmetrical framing because it avoids taking a position.

**Example (LLM-generated):**
> On one hand, the policy ensures compliance. On the other hand, it creates administrative burden. Both perspectives have merit.

**Example (human):**
> The policy ensures compliance. The administrative burden it creates is the reason no one follows it.

**Fix:** State both positions without the mechanical frame. If one side is stronger, show it through the evidence rather than giving equal rhetorical weight to both.

**Severity:** 0.6 (MEDIUM)

---

## Supplementary Vocabulary Fingerprints

These words and phrases are overrepresented in LLM output relative to human writing at the same register level. Their presence in a draft is not automatically disqualifying, but each one increases the probability that a reader will flag the text as AI-generated.

| Word/Phrase | Severity | Why It Flags | Suggested Replacement |
|---|---|---|---|
| delve, delves, delving | 0.9 | Most overrepresented word in LLM output | examine, look at, explore |
| leverage (verb) | 0.6 | Corporate-LLM crossover | use, apply, draw on |
| robust | 0.5 | Vague positive adjective LLMs overuse | strong, solid, reliable |
| holistic | 0.6 | Abstract qualifier | complete, whole-system |
| nuanced | 0.5 | Self-congratulatory descriptor | (delete, or describe the actual nuance) |
| foster, fostering | 0.5 | LLM-preferred verb for "create/build" | build, create, support |
| navigate (metaphorical) | 0.6 | "Navigate the complex terrain of..." | handle, manage, work through |
| landscape (metaphorical) | 0.7 | "In today's hospitality landscape" | (name the actual domain) |
| tapestry | 0.8 | "A rich tapestry of experiences" | (use a concrete noun) |
| beacon | 0.7 | "A beacon of hope/innovation" | example, model |
| resonate | 0.5 | "This message resonates deeply" | (delete, or describe the effect) |
| underscore | 0.4 | "This underscores the importance of" | shows, confirms, proves |
| unpack (metaphorical) | 0.5 | "Let's unpack this concept" | explain, examine |
| deep dive | 0.6 | "A deep dive into the data" | detailed analysis, close look |
| cornerstone | 0.6 | "The cornerstone of our strategy" | foundation, basis |
| paradigm | 0.7 | "A paradigm shift" | (describe the actual change) |
| ecosystem | 0.6 | "The innovation ecosystem" | (name the actual network) |
| synergy | 0.7 | "Creating synergies between" | (describe the actual interaction) |
| streamline | 0.5 | "Streamline the process" | simplify, speed up, cut |
| pivotal | 0.5 | "A pivotal moment" | important, decisive |
| multifaceted | 0.6 | "A multifaceted approach" | (describe the actual facets) |
| comprehensive | 0.5 | "A comprehensive review" | full, complete, thorough |

## Supplementary Structural Fingerprints

| Pattern | Severity | Example | Fix |
|---|---|---|---|
| "In today's [X]..." opener | 0.7 | "In today's rapidly evolving market..." | Open with the thesis. |
| "In an era of/where..." opener | 0.7 | "In an era of digital transformation..." | State the thesis. |
| "Excited to/Thrilled to..." opener | 0.8 | "Excited to announce that we..." | State what happened. |
| "At the intersection of X and Y" | 0.8 | "At the intersection of tech and policy..." | Name the overlap concretely. |
| "As Einstein/Gandhi once said" | 0.6 | Generic quotation attribution | Use the person's actual references. |
| "As the saying goes" | 0.5 | Attribution to anonymous wisdom | Delete or cite specifically. |
| "Thoughts?" closer | 0.7 | Engagement bait at the end | End with substance. |
| "What do you think?" closer | 0.6 | Engagement bait | End with substance. |
| "Let that sink in." closer | 0.8 | Dramatic pause instruction | Delete. |
| "Spoiler alert:" | 0.5 | Generic internet humor | Remove or use the person's actual humor. |
| "Plot twist:" | 0.5 | Generic internet humor | Remove. |
| "Pro tip:" | 0.5 | Generic internet framing | Remove. |

## Using the Detector

```python
from voice_transfer.fingerprint_detector import FingerprintDetector

detector = FingerprintDetector()
report = detector.scan("Your text here...")

print(report.summary())

# Add custom patterns
detector.add_pattern(
    fingerprint_id=100,
    name="my_custom_tic",
    pattern=r"\bsynergy\b",
    severity=0.7,
    fix="Replace with a concrete description.",
    description="Corporate jargon fingerprint",
)
```

## Calibration Notes

The severity scores are calibrated from production use. A single high-severity fingerprint (0.7+) is enough to make a trained reader suspicious. Three or more medium-severity fingerprints (0.4-0.6) in a 500-word text produce the cumulative effect of "this reads like AI."

The detector errs on the side of flagging. Some flagged patterns are legitimate in specific contexts (a genuine use of "moreover" in formal academic writing, for instance). The human reviewer makes the final call. The detector's job is to surface the patterns; the writer's job is to decide which ones to fix.
