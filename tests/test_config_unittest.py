import unittest
from pathlib import Path


class InferenceConfigTests(unittest.TestCase):
    def test_default_config_is_the_supported_14_point_model(self) -> None:
        from tokenbloom.config import DEFAULT_CONFIG

        self.assertEqual(DEFAULT_CONFIG.num_input_views, 2)
        self.assertEqual(DEFAULT_CONFIG.initial_tokens, 512)
        self.assertEqual(DEFAULT_CONFIG.error_sample_count, 14)
        self.assertEqual(DEFAULT_CONFIG.point_knn, 16)
        self.assertEqual(DEFAULT_CONFIG.point_blocks, 2)

    def test_weight_layout_paths_are_relative_to_package_root(self) -> None:
        from tokenbloom.config import DEFAULT_CONFIG

        self.assertEqual(
            DEFAULT_CONFIG.dino_weights_relpath,
            Path("weights/dinov2/dinov2_vitb14_pretrain.pth"),
        )


if __name__ == "__main__":
    unittest.main()
