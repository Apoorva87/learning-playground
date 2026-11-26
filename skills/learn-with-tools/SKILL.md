---
name: learn-with-tools
description: Select and combine tools from the Learning Playground to teach a topic, guide a learner through code, or design a practice exercise. Use when the learner wants a tool recommendation, a hands-on lesson, or a comparison of learning formats.
---

# Learn with tools

Start with what the learner wants to be able to do. “Predict how much memory one more token costs” gives you a way to choose a tool and check whether the lesson worked. Honor a named tool, supplied source, requested format, and stated experience. For a recommendation, explain the choice and what the learner would do with it. For a lesson, run the selected component or guide the learner through an existing result that answers their question.

## Find the playground

Use the learner’s specified checkout, then `LEARNING_PLAYGROUND_ROOT` if set. Otherwise check the current directory and its ancestors, then the ancestors of this file’s resolved path. Identify the root by `CODEX.md`, `tools/README.md`, and `topics/README.md`. Resolve the paths below from **that root**, including when teaching from another repository.

Read `CODEX.md`, the topic index, `tools/README.md`, and `preferences.md`. Look for a matching case, saved artifact, or workshop state before starting a new run. Read the selected `tools/<tool>/README.md` for its actual entry point, prerequisites, tested component, and limits. A copied skill can still suggest a route using the table below. If a local run or feedback update needs a missing playground, ask for its path.

## Choose the component

| What the learner wants to do | Candidate and what to use |
| --- | --- |
| Read a worked derivation or ask follow-up questions about a supplied text | **DeepTutor** (`deeptutor`) for source-grounded tutor chat. Give the tested core Chat workflow the excerpt, quantities, and question to work through. |
| Point at a screen element and find the code behind its behavior | **Teachable Code** (`teachable-code`) takes the app’s source checkout and the learner’s question. The agent builds an interactive walkthrough with excerpts checked against that revision. |
| Change an input, predict an outcome, and check the answer | **OpenMAIC** (`openmaic`) for a simulation scene or a self-paced quiz with prepared answers. Choose the classroom when live classroom interaction is part of the goal. |
| Trace a changing quantity or sequence visually | **Visual Explainer** (`visual-explainer`) for an agent-authored, rendered diagram, animation, or chart. Supply checked values for quantitative figures. |
| See actual attention operations in a small transformer | **LLM Visualization** (`llm-viz`) for the local model walkthrough. Let the learner select a position and trace its query, keys, and values. |
| Compare model structures, per-layer costs, or memory as context grows | **HuggingArch** (`huggingarch`) for a model spec, formulas, and cost estimates under stated settings. Identify whether a timing is an analytical bound or a measured kernel result. |
| Explain what happens during a forward pass | Start with **PyTorch Profiler** for observed intervals. Use **TorchLens** for tensor ancestry, **DaCe** for explicit array access and parallel iteration, **TorchInductor** for generated kernels and fusion, or **MLCommons Chakra** for a machine-readable execution trace. Match the tool to the specific question. |
| Navigate connections among a collection of notes | **Understand-Anything** (`understand-anything`) for a graph or guided tour grounded in linked articles. |
| Answer one live question and have the tutor choose the next based on the reply | **Socratic Workshop** (`socratic-workshop`) for an adaptive diagnostic. Wait for the learner’s answer before choosing the next question. |

Check the live catalog for additional candidates. The learner has named DeepTutor, Teachable Code, and OpenMAIC as favorites. Use the latest dated feedback to break a tie between suitable components. An explicit request or a better match to the task takes precedence. Keep uncertainty about their reasons until they tell you.

## Decide how the tools work together

- **One tool.** For “I’m new to Swift. What happens when I select this tile?”, start with Teachable Code. Follow the visible tile to its SwiftUI declaration, selection handler, and destination. Explain the syntax at the point of use, then invite the learner to predict what a neighboring tile does.
- **Parallel tools.** For “Explain KV caching and give me a calculator to experiment with,” use DeepTutor for a worked decoding step and OpenMAIC for the calculator. Both can start from the same case brief, source, units, and answer key. Run independent calls concurrently where supported, with separate output paths and sufficient model capacity. If both need the same constrained runtime, run them in turn. Check the explanation and calculator against a common numerical example, then link them from one exercise.
- **Sequential handoff.** For “Compare the cache costs of these two models, then quiz me on why they differ,” run HuggingArch at matched batch, context, and precision. Record model IDs, settings, units, and the formula or component responsible for the difference. Give that checked comparison to OpenMAIC for a quiz, or to Socratic Workshop for a live question. If the learner only asks for the comparison, end with the evidence and interpretation.
- **Compare profilers.** Pin the checkpoint, token IDs, cache length, precision, backend and phase. Capture the full pass with shared inputs. Give each independent analysis the same real intermediate tensors and expected output. Keep whole-model measurements, structure-only traces, and numerical replays labeled. Show each result visually and link the views from one comparison. A dependency graph reveals possible parallelism. Pair it with a measured timeline or sampled core occupancy when the learner asks which work actually overlapped. Hardware counters are needed for physical memory traffic.

Match each requested output to a component that can produce it. A written derivation needs equations and worked steps. A live diagnostic needs the learner’s reply before the tutor can pose its next question. Choose the smallest combination that covers those outputs. If a component is unavailable, reuse a relevant saved run or choose an available component that still meets the goal. Name the substitution and which parts were actually run.

## Deliver and learn from the run

Show the result in English. Link the artifact or session and give the learner one thing to predict, change, or explain. Check a code reference against its source revision, a calculation against the formula, or an interactive control against an expected outcome. Preserve the original output and identify corrections or material added by the agent. Surface disagreements that the available evidence cannot resolve.

For a playground trial, follow `CODEX.md` to save the common brief and each component’s version, inputs, actual output, and handoff in the case. Keep records local unless the learner requests sharing. After they see the result, ask one concrete feedback question. Keep feedback pending until they reply. Record their dated judgment separately from observations and update `preferences.md` when they express a preference.

When a new tool arrives, register its tested entry point and capabilities in the live tool catalog and record. It becomes eligible on the next invocation. Add a routing example here only when a trial reveals a distinction that would change a future choice.
