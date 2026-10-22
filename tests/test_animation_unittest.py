import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from PIL import Image

from tokenbloom.animation import interpolate_camera_path, local_orbit_camera_path, save_gif


class CameraPathTests(unittest.TestCase):
    def test_interpolation_keeps_endpoints_and_produces_rigid_transforms(self):
        start = torch.eye(4)
        end = torch.eye(4)
        end[:3, 3] = torch.tensor((2.0, -1.0, 0.5))
        # A 90 degree rotation around the vertical axis.
        end[:3, :3] = torch.tensor(
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
        )

        path = interpolate_camera_path(start, end, num_frames=5)

        self.assertEqual(tuple(path.shape), (5, 4, 4))
        self.assertTrue(torch.allclose(path[0], start))
        self.assertTrue(torch.allclose(path[-1], end))
        self.assertTrue(torch.allclose(path[:, 3], torch.tensor((0.0, 0.0, 0.0, 1.0)).expand(5, -1)))
        identity = torch.eye(3).expand(5, -1, -1)
        self.assertTrue(torch.allclose(path[:, :3, :3].transpose(-1, -2) @ path[:, :3, :3], identity, atol=1e-5))

    def test_rejects_invalid_frame_count(self):
        with self.assertRaises(ValueError):
            interpolate_camera_path(torch.eye(4), torch.eye(4), num_frames=1)

    def test_local_orbit_starts_at_target_pose_and_keeps_its_orientation(self):
        target = torch.eye(4)
        target[:3, 3] = torch.tensor((1.0, 2.0, 3.0))

        path = local_orbit_camera_path(target, num_frames=12, radius=0.1)

        self.assertEqual(tuple(path.shape), (12, 4, 4))
        self.assertTrue(torch.allclose(path[0], target))
        self.assertTrue(torch.allclose(path[:, :3, :3], target[:3, :3].expand(12, -1, -1)))
        self.assertGreater(float((path[1:, :3, 3] - target[:3, 3]).norm(dim=-1).max()), 0.0)

    def test_two_cycle_local_orbit_returns_to_target_halfway(self):
        target = torch.eye(4)
        path = local_orbit_camera_path(target, num_frames=12, radius=0.1, cycles=2)
        self.assertTrue(torch.allclose(path[0], target))
        self.assertTrue(torch.allclose(path[6], target, atol=1e-6))

    def test_save_gif_writes_an_animated_asset(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "novel_view.gif"
            save_gif(
                [Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")],
                output,
                fps=8,
            )
            with Image.open(output) as result:
                self.assertEqual(result.n_frames, 2)
                self.assertEqual(result.size, (4, 4))


if __name__ == "__main__":
    unittest.main()
