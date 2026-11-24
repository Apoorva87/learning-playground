# Running learning trials

Help the learner discover which tools, or parts of tools, fit the way they want to learn. Start from the tool and learning task they name. Read the [topic index](topics/README.md), [tool catalog](tools/README.md), and [preferences](preferences.md) before extending an existing experiment.

## When the learner names a tool and topic

1. **Find the example.** Use the learner’s material or reuse an existing case that fits the question. If they gave only a topic, choose a small, representative example and state the learning goal. Clarify when uncertainty about the source or goal would materially change the trial.
2. **Inspect and set up the tool.** Check its upstream instructions, entry point, and prerequisites. Clone or install the named repo, skill, agent, or tool as appropriate. Keep the downloaded copy in the tool’s ignored local directory where supported. Record its source and exact tested version, and put the reproducible steps in `tools/<tool>/install.sh`. Document any required host registration or manual step. Run the normal entry point to verify setup.
3. **Work the example together.** Keep the prompt, inputs, and desired learning outcome in the topic’s case folder. Run the tool on the example and save its actual output or a session excerpt. If setup or execution fails, record the failure and what was attempted. Give the learner a link to what they can inspect or try.
4. **Make comparisons fair and traceable.** For another tool tackling the same case, reuse the prompt and assets. Record any changes to the input, model, settings, or interaction that could explain a different result. A different learning task gets its own case. Link each trial from both the topic and the tool record.
5. **Ask for feedback after the learner sees the work.** Ask a question grounded in the example, such as “Which explanation helped you solve the next problem yourself?” Follow up on friction or a particular part they would keep. When comparing tools, invite a preference for this use case, including a combination or an undecided answer.
6. **Record the learner’s judgment.** Save their words or a faithful paraphrase with the dated trial. Label your observations separately. Until they respond, mark feedback as pending. Update the preference log when they express a view, naming the use case, tool or component, reason, and supporting trial. Keep earlier dated decisions when their view changes and link the revision.

## Keep the workspace navigable

- For a new topic, create its folder and `assets/` area, add a topic README, and link it from the topic index. Put shared inputs with that topic and tool-specific outputs under the relevant case.
- For a new tool, follow the catalog’s recipe convention and add a link to its record. Keep install scripts safe to rerun and preserve local work. Store machine-specific setup under `.local/` when possible.
- Replace the initial empty-state text in each catalog or log as soon as it has an entry. Keep the root index and README current when the layout or entry points change. A trial is finished when its result and feedback status are findable. Resume pending feedback at the next relevant session.
