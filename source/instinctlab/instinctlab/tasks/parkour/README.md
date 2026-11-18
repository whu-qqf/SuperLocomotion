# Hiking terrain locomotion

The maintained learner uses a physical world model and a depth actor, without
motion imitation or an adversarial motion prior. The task keeps its inherited
`Instinct-Parkour-Target-Amp-G1-Newton-v0` identifier for checkpoint compatibility;
the training entry removes reference motion inputs before creating the scene.

See [the Hiking guide](../../../../../docs/hiking.md) for the supported training,
evaluation and video commands.
