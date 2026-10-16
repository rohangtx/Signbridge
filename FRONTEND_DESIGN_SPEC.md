# SignBridge frontend — product design specification

## Product positioning
SignBridge is one product: an Indian Sign Language research workspace. V1–V4 are model-development milestones and must not appear as four equally complete products.

## Primary user journey
1. Open **Translate a sentence**.
2. Upload one continuous sign-language video.
3. Read filming guidance and model availability before submitting.
4. Generate an English draft when trained model artifacts are present.
5. Review the draft and understand its limitations.

## Secondary user journey
Use **Recognize one sign** only for isolated-sign baseline checks. Make it clear that a single-word classifier is not a sentence interpreter.

## Visual direction
- Dark navy / indigo top hero; off-white work surface; white bordered content panels.
- Teal is reserved for available states; amber for setup required; no decorative green “ready” claim unless the actual required artifact exists.
- Strong page titles, small uppercase eyebrow labels, numbered steps, readable explanatory text and consistent spacing.
- Main work area first; diagnostics stay separate from the user task.
- No emoji-only interface labels, giant grids of model-version cards, fake chat, fake predictions, false accuracy numbers, or claims that a source module is a trained model.

## Accessibility and clarity
- Keep controls labeled and describe accepted inputs.
- Explain the difference between isolated sign recognition and continuous sentence translation.
- Disable actions that cannot work yet and display the exact prerequisite.
- Display human-review caveats where output is shown.
- Preserve a useful, truthful empty state when no model artifacts exist.

## Before public release
- Validate keyboard navigation, screen-reader labels, high contrast and mobile layout in a real browser.
- Test with actual trained model artifacts and genuine ISL paired video/transcript data.
- Record real latency and task-appropriate evaluation metrics; do not advertise them until measured.
