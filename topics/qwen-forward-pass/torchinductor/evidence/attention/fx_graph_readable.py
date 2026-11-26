class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 4, 17, 256]", arg1_1: "f32[1, 4, 17, 256]", arg2_1: "f32[1, 24, 1, 256]"):
        # File: topics/qwen-forward-pass/assets/workload.py:16 in forward, code: keys = k.repeat_interleave(6, dim=1)
        unsqueeze: "f32[1, 4, 1, 17, 256]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand: "f32[1, 4, 6, 17, 256]" = torch.ops.aten.expand.default(unsqueeze, [1, 4, 6, 17, 256]);  unsqueeze = None
        clone: "f32[1, 4, 6, 17, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[1, 24, 17, 256]" = torch.ops.aten.view.default(clone, [1, 24, 17, 256]);  clone = None

        # File: topics/qwen-forward-pass/assets/workload.py:17 in forward, code: values = v.repeat_interleave(6, dim=1)
        unsqueeze_1: "f32[1, 4, 1, 17, 256]" = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        expand_1: "f32[1, 4, 6, 17, 256]" = torch.ops.aten.expand.default(unsqueeze_1, [1, 4, 6, 17, 256]);  unsqueeze_1 = None
        clone_1: "f32[1, 4, 6, 17, 256]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f32[1, 24, 17, 256]" = torch.ops.aten.view.default(clone_1, [1, 24, 17, 256]);  clone_1 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_for_cpu_default = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg2_1, view, view_1, scale = 0.0625);  arg2_1 = view = view_1 = None
        getitem: "f32[1, 24, 1, 256]" = _scaled_dot_product_flash_attention_for_cpu_default[0];  _scaled_dot_product_flash_attention_for_cpu_default = None
        return (getitem,)
