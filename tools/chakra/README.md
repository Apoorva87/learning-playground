# MLCommons Chakra

[Tool guide](index.html) · [Interactive Qwen study](../../topics/qwen-forward-pass/chakra/index.html)

Chakra defines a protocol-buffer execution trace format and provides tools for reading PyTorch host traces, joining Kineto traces, converting workloads, visualizing graphs and generating simulation inputs. The Qwen study uses its official reader, linker invocation, converter, schema bindings, message framing and jsonizer. It also distinguishes the standard converter’s ordering edges from a small tensor-identity projection.

## Tested installation

The trial uses MLCommons’ `mlc-chakra` 1.0.0 from [commit 065445b5](https://github.com/mlcommons/chakra/tree/065445b5d48ffb8796c2dbf4f0540cc608551f8d), CPython 3.13 on Apple Silicon, Git and uv. The package builds its Python protobuf bindings during installation. From the workspace root run:

```sh
./install.sh chakra
.local/tools/chakra/source/.venv/bin/chakra_trace_link --help
.local/tools/chakra/source/.venv/bin/chakra_converter --help
.local/tools/chakra/source/.venv/bin/chakra_jsonizer --help
```

For trace capture use the shared PyTorch 2.11.0 environment described in the [Qwen case](../../topics/qwen-forward-pass/README.md). The Chakra installation is separate from the PyTorch checkpoint environment. The checked-in evidence opens without either environment. [Run notes](../../topics/qwen-forward-pass/chakra/notes.md) · [MLCommons user guide](https://github.com/mlcommons/chakra/blob/065445b5d48ffb8796c2dbf4f0540cc608551f8d/USER_GUIDE.md)
