# OpenMAIC trial · How Transformers Remember

Started November 23, 2026, and reviewed November 24. [Open the lesson](lesson.html) · [Try the calculator](calculator.html) · [Shared case](../../brief.md) · [Tool record](../../../../../../tools/openmaic/README.md)

## What ran

The source checkout was OpenMAIC 1.5.0 at commit `c09bb0e2a9f32c2ce330817767eda4697c635351`. The official `@openmaic/generation@0.3.34` SDK produced two scenes through `generateSceneContent`, using local Ollama `qwen3.5:35b` (model ID prefix `3460ffeede54`). The SDK supplied its simulation and quiz prompts and parsed the actual replies. Both calls used English instructions, temperature 0.2, seed 42, a 16,384-token context, and thinking disabled. The [simulator outline](scene-outline.json) and [quiz outline](quiz-outline.json) record the exact tool-specific inputs.

The full Docker Compose classroom was attempted. Docker denied socket access for stack startup in this session. The Compose configuration passed validation. This trial tested the lesson-generation SDK. Live teacher and student agents, speech, and classroom chat are still untested.

## Agent observations

- The first scene generated a working memory formula and a prediction exercise. It omitted the historical explanation and the requested attention calculation. The [worked guide](../../../../assets/field-guide.md) is agent-authored from primary sources. It fills those gaps and derives DeepSeek’s published figure.
- The generated preset buttons did nothing in a browser. Their internal names differed from their configuration, and their handler reset the selected values. The scene also initialized its observation element too late and emitted two non-English button labels. The learner-facing [calculator](calculator.html) fixes those defects, synchronizes the controls, and offers head counts that divide the 32 query heads. Git commit `f3a4093` preserves the unedited simulator at this run’s original `lesson.html` path.
- A second OpenMAIC scene produced four questions covering attention arithmetic, MLA, CSA2, and 890 bytes per token. The original answer key put every correct option first. The [lesson](lesson.html#quiz) varies option order, clarifies wording and two numerical distractors, and presents the explanations with the questions. The [original quiz](quiz.json) preserves the SDK result.
- The English walkthrough, reading interface, and quiz controls were assembled locally around those outputs. The classroom generator did not produce the walkthrough.
- After the learner reported Chinese text in the HTML, I found that the open browser preview still displayed the original Start and Reset captions. I refreshed that preview with the reviewed calculator, tested the controls there, and updated the generator to translate the SDK’s example captions during generation and replay.

## Learner feedback

**Language requirement:** “always use english only” (November 24, 2026). Recorded in the workspace instructions and preferences.

**Tool preference:** Pending. Which made the storage savings easier to explain in your own words — changing the calculator, working the arithmetic, or answering the quiz? Which part would you want from another topic?
