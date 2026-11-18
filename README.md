# SuperLocomotion

Depth-based terrain locomotion for Unitree G1, built on InstinctLab's Hiking
scene with SuperMimic world-model learning, TGI-aligned imagined policy updates
and Success-Guided Sampling (SGS).

This repository contains the current reference-free Hiking task. It does not
include the upstream shadowing, BeyondMimic, InstantTrack or motion-imitation
tasks. The deployed controller uses proprioceptive history and depth; the
physical world model additionally queries privileged terrain during training.

- [Installation, training and evaluation](docs/hiking.md)
- [Final 2,048-environment configuration](configs/hiking/tgi_sgs_2048.json)
- [14,000-iteration training results](docs/results/hiking_14000.json)

Train with `scripts/instinct_rl/train_hiking_native_latent.py`; evaluate or
record video with `scripts/instinct_rl/evaluate_hiking_native_latent.py`.
The maintained route uses BF16 network execution and no adversarial motion prior
or imitation objective. Checkpoints, datasets, logs and environments stay local.

The code builds on [InstinctLab](https://github.com/project-instinct/InstinctLab)
and [SuperMimic](https://github.com/HybridRobotics/supermimic), with design ideas
from Terrain-Grounded Imagination. It is an experimental implementation, not a
complete reproduction of TGI. Gait quality and difficult-terrain success remain
areas for further work.

Original attribution and third-party notices are retained. The inherited
[CC BY-NC 4.0 license](LICENSE) applies to the project; individual components
retain their own license notices.
