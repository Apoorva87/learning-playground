# Teachable Code

Tell the agent which repository you want to understand, what you already know and a question the lesson should answer. For an app, the skill guides the agent in writing a 2D walkthrough of what happens to the data. Its builder checks each cited line against a pinned checkout and packages the result as an interactive HTML file. The skill also has modes for 3D training and tensor diagrams, research reproduction, and exploring potential code changes. [Upstream skill at the tested revision](https://github.com/ywEdAi/teachable-code/tree/e74598d4b0d2a13cf67182ec897ad8f308894495)

## Try it on Prism

Open the [screen explorer](../../topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/index.html) to start from an element you recognize, or use the [guided map](../../topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/walkthrough.html) to follow the native tool’s sequence and inspect files. The screen explorer is a playground addition built from the same validated excerpts. [Run notes](../../topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/notes.md)

## Set up and rebuild

From the playground root, run `./install.sh teachable-code` with Git, Python 3 and Node.js. The recipe installs the pinned upstream checkout under `.local/tools/teachable-code/source` and checks the builder’s CLI. An agent can use `SKILL.md` directly there. Reading the saved HTML takes only a browser.

To rebuild this trial, use a clean checkout of the Prism revision from the [case brief](../../topics/inside-prism/cases/swiftui-elements/brief.md). The example below uses the original local repository as its source. Run the clone command on first setup.

```sh
tool=.local/tools/teachable-code
run=topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code
git clone --no-hardlinks --no-checkout ~/experiments/my_iptv/app "$tool/prism"
git -C "$tool/prism" checkout --detach 960198d0b364f60eecd9cad591b2873b4cf51e58
python3 "$run/build.py" "$tool/prism"
```

The build reads the saved matcher output. To regenerate the evidence, use the [Swift commands in the run notes](../../topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/notes.md).
