class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 1, 5120]", arg1_1: "bf16[17408, 5120]", arg2_1: "bf16[17408, 5120]", arg3_1: "bf16[5120, 17408]"):
        # File: topics/qwen-forward-pass/torchinductor/capture.py:48 in mlp, code: return linear(torch.nn.functional.silu(linear(x, gate)) * linear(x, up), down)
        permute: "bf16[5120, 17408]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view: "bf16[1, 5120]" = torch.ops.aten.view.default(arg0_1, [1, 5120])
        mm: "bf16[1, 17408]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 1, 17408]" = torch.ops.aten.view.default(mm, [1, 1, 17408]);  mm = None
        convert_element_type_2: "f32[1, 1, 17408]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        neg: "f32[1, 1, 17408]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[1, 1, 17408]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[1, 1, 17408]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1, 17408]" = torch.ops.aten.div.Tensor(convert_element_type_2, add);  convert_element_type_2 = add = None
        convert_element_type_3: "bf16[1, 1, 17408]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        permute_1: "bf16[5120, 17408]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_2: "bf16[1, 5120]" = torch.ops.aten.view.default(arg0_1, [1, 5120]);  arg0_1 = None
        mm_1: "bf16[1, 17408]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "bf16[1, 1, 17408]" = torch.ops.aten.view.default(mm_1, [1, 1, 17408]);  mm_1 = None
        mul: "bf16[1, 1, 17408]" = torch.ops.aten.mul.Tensor(convert_element_type_3, view_3);  convert_element_type_3 = view_3 = None
        permute_2: "bf16[17408, 5120]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view_4: "bf16[1, 17408]" = torch.ops.aten.view.default(mul, [1, 17408]);  mul = None
        mm_2: "bf16[1, 5120]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[1, 1, 5120]" = torch.ops.aten.view.default(mm_2, [1, 1, 5120]);  mm_2 = None
        return (view_5,)
