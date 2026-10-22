import unittest

import torch


class CameraValidationTests(unittest.TestCase):
    def test_camera_validation_rejects_zero_focal_length(self) -> None:
        from tokenbloom.cameras import validate_camera_tensors

        intrinsics = torch.eye(3).unsqueeze(0)
        intrinsics[0, 0, 0] = 0.0
        pose = torch.eye(4).unsqueeze(0)
        with self.assertRaisesRegex(ValueError, "focal"):
            validate_camera_tensors(intrinsics, pose)

    def test_camera_validation_rejects_non_finite_pose(self) -> None:
        from tokenbloom.cameras import validate_camera_tensors

        intrinsics = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        pose = torch.eye(4).unsqueeze(0)
        pose[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_camera_tensors(intrinsics, pose)


if __name__ == "__main__":
    unittest.main()
