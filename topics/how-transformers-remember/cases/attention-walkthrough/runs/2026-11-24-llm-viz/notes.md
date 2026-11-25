# Running LLM Visualization locally

November 24, 2026 · [Play this walkthrough on GitHub Pages](https://apoorva87.github.io/learning-playground/tools/llm-viz/play/) · [Local walkthrough](http://127.0.0.1:3002/llm) · [Setup and restart](../../../../../../tools/llm-viz/README.md)

## What ran

Cloned `bbycroft/llm-viz` at `9da93742382f1bf36c020c38a1ace454e82c4490`. Yarn’s dependency fetch failed on DNS for its registry. npm installed the dependencies. The upstream development command started Next.js 16.3.0 on the loopback address. The walkthrough, the sorting model’s JSON weights and the WebAssembly binary all returned HTTP 200.

## Agent observations

In a local Chromium session the overview showed the six input tokens C, B, A, B, B, C and the target ABBBCC. Choosing **Self Attention** changed the chapter and showed its Continue control. The 3D canvas had an active WebGL 2 context. React reported invalid paragraph nesting in the upstream chapter markup in development mode. The chapter still opened.

This is a companion to the KV-cache case. The diagram explains how attention uses a query to weight past values. To connect it to decoding, imagine adding one letter. Which keys and values for earlier positions could you keep? The [worked guide](../../../../assets/field-guide.md) follows that question through to the memory calculation. This trial covered the bundled nano model and its attention chapter.

## Learner feedback

Pending. Does seeing the Q, K and V vectors in the model make the cache easier to explain?
