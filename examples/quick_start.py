#!/usr/bin/env python3
"""
Quick Start: AI Voice Transfer Pipeline

This example demonstrates the full pipeline:
1. Feed your writing samples into the voice extractor
2. Extract a statistical voice profile
3. Build a lexicon brand (prescriptive rules)
4. Validate a draft against your voice
5. Get a fidelity score with specific fix suggestions

No LLM API key required. The pipeline produces constraints
for your LLM of choice and validates the output.
"""

from voice_transfer.extractor import VoiceCorpus
from voice_transfer.register_library import (
    RegisterLibrary,
    Register,
    RegisterMarker,
    FormalityLevel,
    create_professional_email_register,
    create_analytical_essay_register,
)
from voice_transfer.lexicon_brand import LexiconBrand
from voice_transfer.fingerprint_detector import FingerprintDetector
from voice_transfer.pipeline import VoiceTransferPipeline, DraftRequest


def main():
    # ---------------------------------------------------------------
    # Step 1: Feed your writing samples
    # ---------------------------------------------------------------
    # In practice, these would be your actual emails, essays, reports.
    # The more samples, the better the extraction.

    samples = [
        # Sample 1: Analytical writing
        (
            "The demand for women-only shared accommodation at our "
            "Bordeaux property is real, repeated, and currently unmet. "
            "I watch women who have booked a mixed dormitory return to "
            "the desk and ask, often with visible discomfort, to be moved. "
            "I watch some of them offer to pay more for it. And I have "
            "watched a guest leave because we had nothing to offer her, "
            "toward a competitor a few minutes away who does. "
            "That last point is the heart of the matter. This is not a "
            "theoretical market. It is a booking we lose, on a product "
            "the market around us already offers and we do not."
        ),

        # Sample 2: Professional correspondence
        (
            "I present this as a receptionist, which is a modest post, "
            "and I hold no illusion about that. But it is precisely the "
            "floor from which this opportunity is visible, and analysing "
            "it is the work I was trained for. I would welcome the chance "
            "to contribute beyond my current post, and I offer this note "
            "as the first demonstration of how I think and what I notice."
        ),

        # Sample 3: Reflective writing
        (
            "I am not, by formation, a receptionist. My background is in "
            "governance and the recovery of programmes under pressure, "
            "between Europe and West Africa, with a humanitarian "
            "programme-coordination credential at the master's level "
            "and my own practice. I worked here, left to build that "
            "practice, and returned to the front desk by choice, because "
            "two daughters born in close succession made the international "
            "missions of my field impossible to reconcile with being "
            "present for my family."
        ),

        # Sample 4: Direct observation
        (
            "I want to be precise about the reasoning, because precision "
            "is what makes the case sound rather than sentimental. The "
            "security of women in shared spaces is a genuine concern, and "
            "a live one in the French market specifically. Meeting it is "
            "the right thing to do and the profitable thing to do at the "
            "same time, and the two are not in tension here."
        ),

        # Sample 5: Proposal
        (
            "A test built this way risks very little. The downside is one "
            "reconfigured floor for one season. The upside, if the pattern "
            "I see nightly holds across the network, is a defensible "
            "first-mover position in a category our competitors have "
            "entered piecemeal and no one has yet claimed by name."
        ),
    ]

    # ---------------------------------------------------------------
    # Step 2: Initialize the pipeline and extract the profile
    # ---------------------------------------------------------------

    pipeline = VoiceTransferPipeline(name="example_voice")
    profile = pipeline.load_corpus(
        samples=samples,
        name="Example Voice",
        description="Analytical-professional writing samples",
    )

    print("=" * 60)
    print("STEP 1: Voice Profile Extracted")
    print("=" * 60)
    print(profile.summary())
    print()

    # ---------------------------------------------------------------
    # Step 3: Set up register library
    # ---------------------------------------------------------------

    library = RegisterLibrary()

    # Add a custom analytical register
    analytical = create_analytical_essay_register(
        sample_phrases=[
            "This is not a theoretical market.",
            "The demand is real, repeated, and currently unmet.",
        ]
    )
    library.add_register(analytical, is_default=True)

    # Add professional email register
    professional = create_professional_email_register(
        forbidden_words=["delve", "leverage", "robust"],
        sample_phrases=[
            "I would welcome the chance to contribute.",
            "I offer this note as a demonstration.",
        ],
    )
    library.add_register(professional)

    pipeline.set_register_library(library)

    # ---------------------------------------------------------------
    # Step 4: Initialize lexicon brand with defaults
    # ---------------------------------------------------------------

    brand = pipeline.use_default_brand()

    print("=" * 60)
    print("STEP 2: Lexicon Brand Active")
    print("=" * 60)
    print(f"Modules loaded: {brand.module_count}")
    for name in brand.module_names:
        print(f"  - {name}")
    print()

    # ---------------------------------------------------------------
    # Step 5: Build constraints for LLM prompt injection
    # ---------------------------------------------------------------

    request = DraftRequest(
        content_prompt=(
            "Write a brief analysis of why hotel reception desks "
            "should track guest complaint patterns systematically."
        ),
        target_register="analytical_essay",
        recipient="publication reader",
        max_length=200,
    )

    constraints = pipeline.build_constraints(request, register_name="analytical_essay")

    print("=" * 60)
    print("STEP 3: Constraints for LLM Prompt")
    print("=" * 60)
    print(constraints[:1500])
    if len(constraints) > 1500:
        print(f"... ({len(constraints)} chars total)")
    print()

    # ---------------------------------------------------------------
    # Step 6: Validate a draft (simulating LLM output)
    # ---------------------------------------------------------------

    # This is what an LLM might produce WITHOUT voice constraints:
    bad_draft = (
        "In today's rapidly evolving hospitality landscape, "
        "the front desk serves as a pivotal nexus for guest "
        "experience management. By leveraging comprehensive "
        "complaint tracking systems, hotels can foster a more "
        "nuanced understanding of guest pain points. "
        "This holistic approach navigates the complex terrain "
        "of customer satisfaction, enabling robust data-driven "
        "decision-making that resonates across the organization. "
        "The key insight you would notice is that systematic "
        "tracking transforms reactive complaint handling into "
        "proactive service design. Bold. Necessary. Transformative."
    )

    print("=" * 60)
    print("STEP 4A: Validating BAD Draft (typical LLM output)")
    print("=" * 60)
    bad_result = pipeline.validate(bad_draft, register_name="analytical_essay")
    print(bad_result.summary())
    print()

    # This is what a voice-constrained output should look like:
    good_draft = (
        "The reception desk sees every complaint before management does. "
        "That is not a staffing observation; it is a data architecture "
        "problem. When a guest reports a broken lock at check-in, the "
        "night receptionist logs it in one system, housekeeping logs it "
        "in another, and maintenance resolves it without either record "
        "connecting to the guest's booking history. Three departments "
        "touched the same problem; none of them produced a pattern. "
        "A hotel that tracked complaints by room, by category, and by "
        "resolution time would see its repeat issues in the first month. "
        "The cost of the tracking is a spreadsheet. The cost of not "
        "tracking is the guest who books elsewhere next year and never "
        "tells you why."
    )

    print("=" * 60)
    print("STEP 4B: Validating GOOD Draft (voice-constrained output)")
    print("=" * 60)
    good_result = pipeline.validate(good_draft, register_name="analytical_essay")
    print(good_result.summary())
    print()

    # ---------------------------------------------------------------
    # Step 7: Standalone fingerprint scan
    # ---------------------------------------------------------------

    print("=" * 60)
    print("BONUS: Standalone Fingerprint Detection")
    print("=" * 60)
    detector = FingerprintDetector()
    fp_report = detector.scan(bad_draft)
    print(fp_report.summary())


if __name__ == "__main__":
    main()
