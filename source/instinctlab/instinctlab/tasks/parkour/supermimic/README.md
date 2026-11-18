# Native Hiking world-model locomotion

The maintained workflow is the single physical world model with a depth actor,
TGI-aligned policy updates and SGS sampling. See
[the Hiking guide](../../../../../../docs/hiking.md) for installation, training,
evaluation, result interpretation and the repository layout.

Training enters through `native_latent_training.py`; evaluation through
`native_latent_evaluation.py`. Shared physics, terrain, reward and replay helpers support diagnostics and checkpoint compatibility. Historical teacher,
dual-model and distillation entrypoints are archived locally and are not the
recommended workflow.
