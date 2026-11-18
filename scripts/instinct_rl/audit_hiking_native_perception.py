"""Check native Hiking camera/time/replay and the shared perception model."""

import faulthandler

from instinctlab.tasks.parkour.supermimic.native_perception_audit import main

if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
