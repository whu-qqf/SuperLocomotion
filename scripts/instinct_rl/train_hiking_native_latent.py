"""Train native Hiking using one SuperMimic physical/visual world model."""

import faulthandler

from instinctlab.tasks.parkour.supermimic.native_latent_training import main

if __name__ == "__main__":
    faulthandler.enable()
    main()
