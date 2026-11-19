"""Grouped expert down projection with FP32 outputs and upstream gradients."""

import torch
import torch.nn.functional as F


def _grouped_mm(a, b, boundaries, *, reduction=False):
    from quack.gemm_interface import gemm

    layout = {"cu_seqlens_k" if reduction else "cu_seqlens_m": boundaries}
    return gemm(a, b, out_dtype=torch.float32, tuned=False, **layout)


class _ExpertDownProjection(torch.autograd.Function):
    """Use GPU-resident ragged metadata for forward, dX and dW.

    Quack accepts BF16 operands and produces FP32 output. Split an FP32
    upstream gradient into three BF16 residual components, and accumulate
    their grouped products in FP32. This preserves its precision instead of
    rounding the entire upstream gradient to BF16 before backpropagation.
    The fixed three-component loop is independent of the expert count.
    """

    @staticmethod
    def forward(ctx, inputs, weight, offsets):
        if inputs.is_cuda:
            if inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
                raise ValueError("Grouped CUDA down projection requires BF16 operands")
            boundaries = torch.cat((offsets.new_zeros(1), offsets))
            ctx.save_for_backward(inputs, weight, boundaries)
            return _grouped_mm(inputs, weight.transpose(1, 2), boundaries)

        # Small CPU correctness oracle; the CUDA path never reads offsets on CPU.
        ctx.boundaries = [0] + offsets.tolist()
        ctx.save_for_backward(inputs, weight, offsets)
        pieces = []
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            for expert, (start, end) in enumerate(zip(ctx.boundaries, ctx.boundaries[1:])):
                pieces.append(F.linear(inputs[start:end].float(), weight[expert].float()))
        return torch.cat(pieces)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, boundaries = ctx.saved_tensors
        if inputs.is_cuda:
            dx = dw = None
            residual = grad_output.float()
            for _ in range(3):
                component = residual.to(torch.bfloat16)
                residual = residual - component.float()
                if ctx.needs_input_grad[0]:
                    contribution = _grouped_mm(component, weight, boundaries)
                    dx = contribution if dx is None else dx.add_(contribution)
                    del contribution
                if ctx.needs_input_grad[1]:
                    contribution = _grouped_mm(component.T, inputs, boundaries, reduction=True)
                    dw = contribution if dw is None else dw.add_(contribution)
                    del contribution
            return (
                dx.to(inputs.dtype) if dx is not None else None,
                dw.to(weight.dtype) if dw is not None else None,
                None,
            )

        dx = torch.empty_like(inputs) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            for expert, (start, end) in enumerate(zip(ctx.boundaries, ctx.boundaries[1:])):
                if dx is not None:
                    dx[start:end] = (grad_output[start:end].float() @ weight[expert].float()).to(inputs.dtype)
                if dw is not None:
                    dw[expert] = (grad_output[start:end].float().T @ inputs[start:end].float()).to(weight.dtype)
        return dx, dw, None
