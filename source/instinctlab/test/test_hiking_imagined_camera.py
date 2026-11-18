"""Imagined ray buffers stay bounded when terminal masks shrink batches."""

from types import SimpleNamespace

from instinctlab.tasks.parkour.supermimic.native_imagined_camera import NativeImaginedDepthCamera


def test_terminal_batch_sizes_do_not_accumulate_ray_buffer_caches():
    camera = object.__new__(NativeImaginedDepthCamera)
    camera.device = "cpu"
    camera.image_shape, camera.num_rays = (2, 3), 6
    camera.cfg = SimpleNamespace(data_types=["distance_to_image_plane"], noise_pipeline={})
    camera.buffers = {}
    for batch in (32, 31, 29, 28, 16, 8, 3, 32):
        buffers = camera._allocate(batch)
        assert list(camera.buffers) == [batch]
        assert camera.buffers[batch] is buffers
        assert buffers.depth.shape == (batch, 2, 3, 1)
