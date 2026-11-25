# LLM Visualization

Explore how a small GPT model predicts its next token while sorting six letters. Its guided chapters cover embeddings, attention, the feed-forward network and output probabilities. Larger model layouts are also available. [Upstream at the tested commit](https://github.com/bbycroft/llm-viz/tree/9da93742382f1bf36c020c38a1ace454e82c4490)

## Run it locally

From the playground root:

```sh
./tools/llm-viz/start.sh
```

The launcher installs missing dependencies on first use, then starts the upstream Next.js development server on this computer. Keep the terminal open, and use Ctrl+C there to stop it. [Open the walkthrough](http://127.0.0.1:3002/llm)

To prepare the checkout without starting the server, run `./install.sh llm-viz`. You need Git, Node.js with npm, an internet connection for the first install, and a browser with WebGL 2 for the model view. We tested on an Apple silicon Mac with Node 25.8.1, npm 11.11.1 and Next.js 16.3.0. The downloaded source and installed packages stay under `.local/tools/llm-viz/`.

The repository ships Yarn 4.1.0 and a Yarn lockfile. Yarn’s registry hostname could not be resolved by the trial process. The playground recipe uses npm with legacy peer dependency resolution, which installed successfully. This uses npm’s dependency resolution and keeps the upstream Yarn lockfile intact. The small model weights and WebAssembly runtime are included in the checkout.

Choose **Self Attention** from the walkthrough contents, then press **Continue**. Look for the query from the current position, the keys it compares against, and the values those comparisons weight. [Local trial and observations](../../topics/how-transformers-remember/cases/attention-walkthrough/runs/2026-11-24-llm-viz/notes.md)
