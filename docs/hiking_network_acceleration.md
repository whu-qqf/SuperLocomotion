# Hiking network execution acceleration

The latent training entry accepts `--precision float32|tf32|bf16` and
`--compile_mode eager|default|reduce-overhead`. Defaults remain float32/eager;
resume inherits the saved execution settings unless explicitly overridden.

BF16 autocasts only the world-model, Actor, Critic and depth encoder/decoder
network kernels. Outputs return to float32 before state integration, geometry,
reward and return computations. This is separate from `--amp`, which enables
the adversarial motion prior and changes the learning objective.

Compilation targets the network forward methods and preserves module parameter
identities and state-dict names. It does not compile the entire simulator,
raycast or imagination loop. Compile modes are opt-in: small-network benchmarks
on this workload did not demonstrate an end-to-end gain, and CUDA Graph speedup
must not be assumed from selecting a compilation mode.

`--policy_microbatch` may be overridden on resume after restoring and validating
the checkpoint. The effective actor batch, horizon, losses and optimizer update
counts remain unchanged. Different chunk sizes can change sampled noise and
floating-point accumulation order, so they do not promise bitwise-identical runs.
Execution settings are written to config.json, metrics.jsonl and checkpoints.
Use latest.pt to resume with replay and SGS state; model_N.pt is weights only.

Validation artifacts: `logs/hiking_supermimic/network_acceleration_20261117/`.

The selected run uses BF16/eager with policy microbatch 2048 and resumes
iteration 600 from `sgs_tgi_20261117/hiking_train_2048/latest.pt` into
`sgs_tgi_accel_20261117/hiking_train_2048`. The 32-environment diagnostic
retained actor batch 2048, horizon 32 and 20 Critic updates. Warmed full-update
measurements were approximately 10.1 s for float32/microbatch 512 and 2.5 s for
BF16/microbatch 2048, with 11.9 GB peak allocated memory in the latter diagnostic.
The original training shared the GPU during early benchmark cases, so these
numbers establish a promising configuration, not a controlled speed ratio.
Formal training additionally performs 8 larger model updates, replay sampling
and real collection; measure its metrics.jsonl for actual iteration throughput.

The 40 selected regression tests include a CUDA BF16 controller export test.
The first diagnostic exposed an encoder precision mismatch in the export check;
controller encoder inference now uses the same network precision helper as
training, and exact equality after save/load is tested. All modified-file hooks
pass. Repository-wide hooks were run and retain unrelated existing failures;
formatting changes outside this task were restored.

Formal 2048-environment continuation verified through iteration 620:
baseline iterations 510–535 averaged 12.33 s per iteration;
accelerated iterations 603–620 averaged 9.16 s.
The policy portion decreased from 8.87 s to
3.13 s. Model training is slower under the
current BF16 path and offsets part of that improvement. An isolated large-batch
depth reconstruction benchmark also showed BF16 slower than FP32 on this GPU;
the selected configuration is retained because actual total iteration time
improved. Further model optimization requires a full-update comparison.
All recorded model, Actor and task-depth gradient norms were finite. Each
iteration retained 8 model, 1 Actor, 20 Critic and 9 depth optimizer updates.
Timing artifacts include training_performance.json and reconstruction_benchmark.log.
