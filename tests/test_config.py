from pathlib import Path


def test_default_config_is_the_supported_14_point_model():
    from tokenbloom.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.num_input_views == 2
    assert DEFAULT_CONFIG.initial_tokens == 512
    assert DEFAULT_CONFIG.error_sample_count == 14
    assert DEFAULT_CONFIG.point_knn == 16
    assert DEFAULT_CONFIG.point_blocks == 2


def test_weight_layout_paths_are_relative_to_package_root():
    from tokenbloom.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.dino_weights_relpath == Path(
        "weights/dinov2/dinov2_vitb14_pretrain.pth"
    )
