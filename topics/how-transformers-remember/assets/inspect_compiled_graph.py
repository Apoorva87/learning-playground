"""Compare tensor dependencies with the CPU kernel that TorchInductor generates.

Run with PyTorch 2.11 and a C++ compiler. For a fresh compiler report, run
from a scratch directory and supply the full path to this script:

    TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR="$(mktemp -d)" python /path/to/inspect_compiled_graph.py

Open the reported debug directory's output_code.py to inspect loads and stores.
"""

import torch
from torch._dynamo.backends.registry import lookup_backend


def inspect_graph(graph_module, example_inputs):
    print("Node | Depends on | Shape | Type | Logical bytes")
    for node in graph_module.graph.nodes:
        tensor = node.meta.get("example_value")
        if isinstance(tensor, torch.Tensor):
            parents = ", ".join(parent.name for parent in node.all_input_nodes)
            print(node.name, parents or "input", tuple(tensor.shape), tensor.dtype,
                  tensor.numel() * tensor.element_size(), sep=" | ")
    return lookup_backend("inductor")(graph_module, example_inputs)


def two_branches(x):
    return (x + 1) * (x - 1)


if __name__ == "__main__":
    x = torch.linspace(-2, 2, 4096, dtype=torch.float32, device="cpu")
    compiled = torch.compile(two_branches, backend=inspect_graph, fullgraph=True)
    torch.testing.assert_close(compiled(x), two_branches(x))
    print("The compiled result matches the eager calculation.")
