"""Bundle the three captures for an offline browser. Source excerpts come from the saved compiler files."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def runner_calls(code):
    tree = ast.parse(code)
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    function = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    lines = []
    for stmt in function.body:
        if not isinstance(stmt, (ast.Assign, ast.Expr)):
            continue
        expr = stmt.value
        if isinstance(expr, ast.Call) and ast.unparse(expr.func).startswith(("cpp_fused_", "extern_kernels.", "torch.ops.aten.")):
            lines.append(ast.get_source_segment(code, stmt).strip())
    return lines


def excerpts(folder, case):
    code = (folder / "output_code.py").read_text()
    pre = (folder / "ir_pre_fusion.txt").read_text()
    post = (folder / "ir_post_fusion.txt").read_text()
    ir_rows = [l for l in post.splitlines() if ".unmet_dependencies =" in l or ".writes =" in l]
    fx = (folder / "dynamo_fx.txt").read_text()
    input_graph = "\n".join(l.strip() for l in fx.splitlines() if l.startswith("        ") and ': "' in l)
    cpp = code.split("extern \"C\"", 1)[1].split("'''", 1)[0].strip()
    if case == "attention":
        selections = [ir_rows[:4], ir_rows[4:]]
    else:
        selections = [ir_rows[i:i + 2] for i in range(0, len(ir_rows), 2)]
    calls = runner_calls(code)
    return dict(ir_identical=pre == post, runner=calls, ir=["\n".join(rows) for rows in selections],
                graph=input_graph, cpp='extern "C" ' + cpp)


def build():
    data = {}
    for case in ("attention", "mlp-decode", "mlp-prefill"):
        folder = HERE / "evidence" / case
        report = json.loads((folder / "run.json").read_text())
        report.update(excerpts(folder, case))
        assert len(report["runner"]) == len(report["ir"]) == len(report["wrapper_calls"])
        data[case] = report
    (HERE / "data.js").write_text("window.INDUCTOR_CAPTURE = " + json.dumps(data, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    build()
