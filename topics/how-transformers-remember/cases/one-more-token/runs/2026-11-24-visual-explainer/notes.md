# Visual Explainer trial · One more token

November 24, 2026. [Open the visual lesson](one-more-token.html) · [Editable figure source](one-more-token.src.html) · [Tool setup and replay](../../../../../../tools/visual-explainer/README.md)

## What ran

Codex followed Visual Explainer 0.15.0's `generate-web-diagram` instructions using the [same case](../../brief.md) and [field guide](../../../../assets/field-guide.md). The page uses the skill's paper layout, step player, flow diagram, paired bar charts, and byte breakdown. Codex authored the explanations, chart inputs, and prediction checks. The upstream Node renderer turned that source into a single HTML file with its chart graphics, controls, and styles embedded.

## What you can try

Press **Play** to follow one query from scores to a weighted sum. Predict the output for a zero query before opening the answer. Then compare the two token budgets using the shared bar scale. The last figure gives the byte size of an entry so you can reconstruct DeepSeek's average of 890 global bytes per token.

## Agent observations

- The two bar charts share a maximum of 4,096 MiB. This makes doubling the sequence length visible at a glance. The small single-head bar also has a numerical label.
- The upstream renderer reported three captioned figures, three data charts, and one flow diagram. Safari opened the result and displayed the step controls, prediction reveal, and final calculation. The optional automated layout check could not launch its headless browser in this session.
- The file opens directly from disk. The rendered page requests optional Google Fonts and uses browser fallbacks offline. This trial covered the HTML page workflow. Slide decks, video, code reviews, and other host integrations are untested here.

## Learner feedback

Pending. Which was easier to explain afterward: how the query changes the output, or how sharing heads changes memory? Which figure helped?
