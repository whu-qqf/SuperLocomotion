"""Optional network acceleration; physical geometry and returns stay in float32."""

import torch

from instinctlab_supermimic import networks


def configure_network_execution(runner, precision="float32", compile_mode="eager"):
    if precision not in ("float32", "tf32", "bf16"):
        raise ValueError(f"Unknown network precision: {precision}")
    if compile_mode not in ("eager", "default", "reduce-overhead"):
        raise ValueError(f"Unknown compilation mode: {compile_mode}")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 network acceleration requires a CUDA device with native BF16 support.")
    networks.AMP_DTYPE = torch.bfloat16 if precision == "bf16" else None
    torch.backends.cuda.matmul.allow_tf32 = precision != "float32"
    torch.backends.cudnn.allow_tf32 = precision != "float32"
    if compile_mode != "eager":
        # Compile the dense kernels, preserving the original module objects,
        # parameter identities and checkpoint keys. FK/raycast and variable
        # episode masks remain eager. Dynamic batches avoid per-mask graphs.
        for net in (runner.physics.model.net, runner.actor.actor, runner.critic.net):
            net.forward = torch.compile(net.forward, mode=compile_mode, dynamic=True)
    runner.execution = dict(precision=precision, compile_mode=compile_mode)
    return runner.execution


def optimizer_step(loss, module, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite native supervised gradient; optimizer not stepped.")
    optimizer.step()
    return float(norm)
