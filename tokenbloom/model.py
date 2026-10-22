"""Trainable ATSplat-style Stage 1 built on TokenGS rendering and losses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .encoder import ATSplatEncoder
from .anchor_xyz_pos_encoder import AnchorXYZPosEncoder
from .enc_dec import DecoderBlock

from .error_feedback import (
    FullResolutionTokenProjectionErrorFeedback,
    FullResolutionTokenProjectionFeedbackResult,
)
from .input_types import (
    EncoderLatent,
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    Reconstruction,
)
from .base_model import TokenGS


from typing import Any as Options



@dataclass
class ATSplatStage1Reconstruction(Reconstruction):
    """Stage-1 Gaussians, predicted uncertainty, and optional error feedback."""

    token_split_scores: tuple[torch.Tensor, ...]
    gaussian_uncertainties: tuple[torch.Tensor, ...]
    context_uncertainty_error_sums: tuple[torch.Tensor, ...]
    context_uncertainty_element_counts: tuple[int, ...]
    gaussian_error_features: tuple[torch.Tensor, ...]
    gaussian_error_visibility: tuple[torch.Tensor, ...]
    gaussian_error_mean_magnitudes: tuple[torch.Tensor, ...]
    gaussian_visible_ratios: tuple[torch.Tensor, ...]
    gaussian_max_error_indices: tuple[torch.Tensor, ...]
    fullres_point_error_mean_magnitudes: tuple[torch.Tensor, ...]
    fullres_point_valid_ratios: tuple[torch.Tensor, ...]
    fullres_view_valid_ratios: tuple[torch.Tensor, ...]
    fullres_token_error_mean_magnitudes: tuple[torch.Tensor, ...]
    multiscale_rgb_error_magnitudes: tuple[torch.Tensor, ...]
    multiscale_feature_error_magnitudes: tuple[torch.Tensor, ...]
    multiscale_gaussian_error_means: tuple[torch.Tensor, ...]
    multiscale_gaussian_error_stds: tuple[torch.Tensor, ...]
    multiscale_gaussian_visible_ratios: tuple[torch.Tensor, ...]
    gaussian_error_heatmap: torch.Tensor | None  # [B, V_input, 1, H, W]
    intermediate_gaussians: tuple[torch.Tensor, ...]
    intermediate_gaussian_valid_masks: tuple[torch.Tensor, ...]
    intermediate_token_positions: tuple[torch.Tensor, ...]
    intermediate_token_radii: tuple[torch.Tensor, ...]
    intermediate_token_valid_masks: tuple[torch.Tensor, ...]
    selected_parent_indices: tuple[torch.Tensor, ...]
    token_feature_snapshots: dict[str, torch.Tensor]
    final_token_features: torch.Tensor  # [B, final_tokens, C]
    final_token_positions: torch.Tensor  # [B, final_tokens, 3]
    final_token_radii: torch.Tensor  # [B, final_tokens, 1]
    final_token_valid_mask: torch.Tensor  # [B, final_tokens]
    final_position_bias: torch.Tensor  # [B, final_gaussians, 3]
    gaussian_valid_mask: torch.Tensor  # [B, final_gaussians]


@dataclass
class _DifixFullresFeedbackViews:
    """Detached context/target observations for one ATE feedback stage."""

    rendered_rgb: torch.Tensor
    reference_images: torch.Tensor
    reference_features: torch.Tensor
    decoder_input: ModelInputDecoder
    pseudo_images: torch.Tensor
    pseudo_features: torch.Tensor




class ATSplatStage1(TokenGS):
    """Depth or hybrid-initialized /16 anchors with TokenGS rendering losses.

    The inherited renderer and ``compute_loss`` make RGB, SSIM, visibility,
    mask, opacity, and optional LPIPS supervision identical to TokenGS.  Only
    the image encoder and the Gaussian center parameterization differ.
    """

    def __init__(self, opt: Options):
        if opt.enc_embed_dim != 768 or opt.token_dim != 768:
            raise ValueError("atsplat_stage1 requires enc_embed_dim=token_dim=768")
        if opt.enc_num_heads != 12:
            raise ValueError("atsplat_stage1 requires enc_num_heads=12")
        if opt.dec_depth != 2:
            raise ValueError(
                "atsplat_stage1 uses one anchor SA-FFN block followed by one CA-SA-FFN layer"
            )
        if opt.enc_depth != 12:
            raise ValueError("atsplat_stage1 requires the 12-layer ATsplat coarse encoder")
        if opt.use_latent_bottleneck:
            raise ValueError("atsplat_stage1 does not use the TokenGS latent bottleneck")
        if opt.time_embedding or opt.num_dynamic_gs_tokens:
            raise ValueError("atsplat_stage1 currently supports static Gaussian tokens only")
        if opt.use_ttt_for_eval:
            raise ValueError("atsplat_stage1 does not yet support TokenGS test-time tuning")
        if opt.mean_of_grads != "none":
            raise ValueError("atsplat_stage1 statistical ATE requires mean_of_grads='none'")
        if (
            opt.ate_split_ratios != (0.5, 0.5, 0.25)
            or opt.ate_refine_depths != (2, 2, 2)
        ):
            raise ValueError(
                "atsplat_stage1 requires split ratios (0.5, 0.5, 0.25) "
                "and refine depths (2, 2, 2)"
            )

        super().__init__(opt)
        self.training_step: int | None = None
        self.log_gaussian_error_heatmap = False
        self.log_token_feature_numerics = False
        # ATsplat supplies its own image tokens; inherited TokenGS image
        # embedding and encoder modules are not part of this baseline.
        self.gs_tokens.requires_grad_(False)
        self.patch_embed = nn.Identity()
        self.patch_plucker_embed = nn.Identity()
        self.enc_dec_backbone.encoder = nn.Identity()
        self.enc_dec_backbone.encoder_norm = nn.Identity()
        # ATSplat performs the selected-layer fusion inside its image encoder;
        # discard the inherited TokenGS readout norms so DDP sees no unused params.
        if hasattr(self.enc_dec_backbone, "multiscale_norms"):
            self.enc_dec_backbone.multiscale_norms = nn.ModuleList()
        self.enc_dec_backbone.kv_proj = nn.Identity()
        self.enc_dec_backbone.k_proj_norm = nn.Identity()
        first_decoder_layer = self.enc_dec_backbone.decoder_blocks[0]
        # The first inherited decoder layer is used only as the initial anchor
        # Transformer block (SA + FFN). Its CA branch is intentionally unused.
        first_decoder_layer.gs_cross_attn.requires_grad_(False)
        first_decoder_layer.gs_cross_attn_scale.requires_grad_(False)
        # The second inherited layer is the initial CA/SA/FFN decoder. Give
        # its CA the same anchor-to-ray geometric bias as later ATE decoders,
        # so it can use the depth-predicted initial mu and radius.
        initial_decoder_layer = self.enc_dec_backbone.decoder_blocks[1]
        initial_decoder_layer.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            opt.token_dim,
            opt.enc_num_heads,
            qkv_bias=True,
            q_norm=True,
            geometric_bias=True,
            geo_sigma0=opt.cross_attn_geo_sigma0,
            geo_bias_min=opt.cross_attn_geo_bias_min,
            geo_bandwidth2_min=opt.cross_attn_geo_bandwidth2_min,
            geo_gamma_init=opt.cross_attn_geo_gamma_init,
        )

        self.image_encoder = ATSplatEncoder(
            image_size=opt.img_size,
            num_input_views=opt.num_input_views,
            dim=opt.enc_embed_dim,
            num_heads=opt.enc_num_heads,
            coarse_depth=opt.enc_depth,
            fine_depth=2,
            mlp_ratio=opt.mlp_ratio,
            dino_weights=opt.dino_weights,
                depth_min=opt.znear,
                depth_max=opt.zfar,
                token_radius_init=opt.token_radius_init,
                token_radius_head_init_std=opt.token_radius_head_init_std,
                token_radius_epsilon=opt.token_radius_epsilon,
                use_multiscale_anchor_fusion=opt.use_multiscale_encoder,
            multiscale_layers=opt.multiscale_encoder_layers,
        )
        token_dim = opt.token_dim
        num_heads = opt.enc_num_heads
        if opt.use_hybrid_anchor_init:
            self.prior_anchor_features = nn.Parameter(
                opt.gs_token_std * torch.randn(opt.num_prior_anchors, token_dim)
            )
            prior_positions = torch.empty(opt.num_prior_anchors, 3)
            # iLRM RE10K normalization aligns the mean camera forward direction
            # with +Z. Start priors in a compact, visible normalized-space box;
            # the coordinates remain fully learnable after initialization.
            nn.init.uniform_(prior_positions[:, :2], -1.0, 1.0)
            prior_z_min = max(opt.znear, 0.5)
            prior_z_max = max(2.0, prior_z_min + 1.0)
            nn.init.uniform_(prior_positions[:, 2], prior_z_min, prior_z_max)
            self.prior_anchor_positions = nn.Parameter(prior_positions)
            self.prior_anchor_raw_radii = nn.Parameter(
                torch.full(
                    (opt.num_prior_anchors, 1),
                    math.log(math.expm1(opt.token_radius_init - opt.token_radius_epsilon)),
                )
            )
        self.anchor_xyz_pos_encoder = (
            AnchorXYZPosEncoder(
                dim=token_dim,
                num_freqs=opt.anchor_xyz_pe_num_freqs,
                radius=10.0,
            )
            if opt.use_anchor_xyz_pe
            else None
        )
        # Match TokenGS: normalize each fused image token before K/V projection.
        self.fine_kv_input_norm = nn.LayerNorm(token_dim)
        self.fine_kv_proj = nn.Linear(token_dim, token_dim * 2)
        self.fine_k_norm = nn.LayerNorm(token_dim // num_heads)
        self.gaussian_head_norm = nn.LayerNorm(token_dim)
        # After the initial CA/SA/FFN block, update the depth-initialized
        # token centre mu and raw radius with the same 3D+radius residual
        # parameterization used by the ATE split/update path.
        self.initial_token_spatial_updater = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 4),
        )
        self.uncertainty_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, token_dim),
                    nn.GELU(),
                    nn.Linear(token_dim, 1),
                )
                for _ in opt.ate_split_ratios
            ]
        )
        # The selected release uses projection14 context feedback, not the
        # historical child-Gaussian VJP assignment path.
        self.gaussian_error_back_assignment = None
        self.multiscale_error_back_assignment = None
        self.gaussian_error_encoder = None
        self.error_fusions = nn.ModuleList()
        self.fullres_gaussian_token_error_feedback = None
        self.fullres_proxy_gaussian_error_feedback = None
        self.difix_target_teacher = None
        if opt.use_ate_error_feedback:
            self.multiscale_error_back_assignment = MultiScaleGaussianErrorBackAssignment(
                self.gs,
                downscale=opt.error_feature_downscale,
            )
            self.gaussian_error_encoder = nn.Sequential(
                nn.Linear(
                    self.multiscale_error_back_assignment.error_channels + 3,
                    256,
                ),
                nn.GELU(),
                nn.Linear(256, 256),
            )
            self.error_fusions = nn.ModuleList(
                [nn.Linear(512, token_dim) for _ in opt.ate_split_ratios]
            )
            for fusion in self.error_fusions:
                nn.init.zeros_(fusion.weight)
                nn.init.zeros_(fusion.bias)
        if opt.use_fullres_gaussian_token_error_feedback:
            attribution_mode = getattr(
                opt, "fullres_error_attribution_mode", "projection14"
            )
            if attribution_mode == "projection14":
                self.fullres_gaussian_token_error_feedback = (
                    FullResolutionTokenProjectionErrorFeedback(
                        token_dim=token_dim,
                        num_gaussians_per_token=opt.num_gaussians_per_token,
                        separate_prior_cue=opt.use_difix_target_error_feedback,
                        knn_samples=opt.fullres_error_knn_samples,
                        point_transformer_blocks=(
                            opt.fullres_error_point_transformer_blocks
                        ),
                        point_attention_channels=(
                            opt.fullres_error_point_attention_channels
                        ),
                        eps=opt.token_radius_epsilon,
                    )
                )
            else:
                num_proxies = (
                    opt.num_gaussians_per_token
                    if attribution_mode == "child_vjp"
                    else opt.fullres_error_proxy_count
                )
                aggregation_mode = (
                    "mlp" if attribution_mode == "proxy_mlp" else "direct"
                )
                self.fullres_proxy_gaussian_error_feedback = (
                    TokenProxyGaussianErrorFeedback(
                        renderer=self.gs,
                        token_dim=token_dim,
                        num_gaussians_per_token=opt.num_gaussians_per_token,
                        num_proxies=num_proxies,
                        aggregation_mode=aggregation_mode,
                        view_weight_mode=opt.fullres_error_proxy_view_weight,
                        aggregate_views_before_mlp=(
                            opt.fullres_error_proxy_aggregate_views_before_mlp
                        ),
                        use_rgb_error=opt.fullres_error_proxy_use_rgb_error,
                        progressive_feature_levels=(
                            opt.fullres_error_proxy_progressive_feature_levels
                        ),
                        vjp_channel_chunk_size=opt.fullres_error_vjp_chunk_size,
                        eps=opt.token_radius_epsilon,
                    )
                )
        if getattr(opt, "use_difix_target_error_feedback", False):
            self.difix_target_teacher = DifixTargetTeacher(
                model_path=opt.difix_model_path,
                prompt=opt.difix_prompt,
                timestep=opt.difix_timestep,
            )
        self.token_spatial_expanders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, token_dim),
                    nn.GELU(),
                    nn.Linear(
                        token_dim, opt.ate_children_per_parent * 4
                    ),
                )
                for _ in opt.ate_split_ratios
            ]
        )
        self.token_feature_expanders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, token_dim),
                    nn.GELU(),
                    nn.Linear(
                        token_dim, opt.ate_children_per_parent * token_dim
                    ),
                )
                for _ in opt.ate_split_ratios
            ]
        )
        self.token_unsplit_spatial_updaters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, token_dim),
                    nn.GELU(),
                    nn.Linear(token_dim, 4),
                )
                for _ in opt.ate_split_ratios
            ]
        )
        # Keep the hidden projection expressive while starting both learned
        # split residuals one order of magnitude smaller.
        split_hidden_init_std = 0.02
        split_output_init_std = 0.002
        for expander in (
            self.initial_token_spatial_updater,
            *self.token_spatial_expanders,
            *self.token_feature_expanders,
            *self.token_unsplit_spatial_updaters,
        ):
            linear_layers = [
                layer for layer in expander if isinstance(layer, nn.Linear)
            ]
            for layer_index, layer in enumerate(linear_layers):
                init_std = (
                    split_output_init_std
                    if layer_index == len(linear_layers) - 1
                    else split_hidden_init_std
                )
                nn.init.trunc_normal_(layer.weight, std=init_std)
                nn.init.zeros_(layer.bias)
        # Keep the existing delta-mu initialization, while making the new
        # delta-rho channels start at exact inheritance.
        for expander in self.token_spatial_expanders:
            output = expander[-1]
            with torch.no_grad():
                output.weight.view(
                    opt.ate_children_per_parent, 4, token_dim
                )[:, 3].zero_()
                output.bias.view(opt.ate_children_per_parent, 4)[:, 3].zero_()
        for updater in (
            self.initial_token_spatial_updater,
            *self.token_unsplit_spatial_updaters,
        ):
            with torch.no_grad():
                updater[-1].weight[3].zero_()
                updater[-1].bias[3].zero_()
        refine_init_values = (
            opt.dec_init_values
            if opt.dec_init_values is not None
            else 5e-3 * opt.gs_token_std
        )
        self.ate_refine_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderBlock(
                            token_dim,
                            num_heads,
                            opt.mlp_ratio,
                            qkv_bias=True,
                            ffn_bias=True,
                            init_values=refine_init_values,
                            qk_norm=True,
                            cross_attn_geometric_bias=True,
                            cross_attn_geo_sigma0=opt.cross_attn_geo_sigma0,
                            cross_attn_geo_bias_min=opt.cross_attn_geo_bias_min,
                            cross_attn_geo_bandwidth2_min=(
                                opt.cross_attn_geo_bandwidth2_min
                            ),
                            cross_attn_geo_gamma_init=(
                                opt.cross_attn_geo_gamma_init
                            ),
                        )
                        for _ in range(depth)
                    ]
                )
                for depth in opt.ate_refine_depths
            ]
        )
    def set_training_step(self, step: int) -> None:
        """Stamp the zero-based optimizer step for scheduled operations."""
        step = int(step)
        self.training_step = step

    def set_error_heatmap_logging(self, enabled: bool) -> None:
        """Request an error-magnitude render for the current forward only."""
        self.log_gaussian_error_heatmap = bool(enabled)

    def set_numerics_logging(self, enabled: bool) -> None:
        """Request detailed token-feature snapshots for the current forward."""
        self.log_token_feature_numerics = bool(enabled)

    @staticmethod
    def _raw_images(encoder_input: ModelInputEncoder) -> torch.Tensor:
        if encoder_input.images_rgb_unnormalized is not None:
            return encoder_input.images_rgb_unnormalized
        mean = encoder_input.images_rgb.new_tensor((0.485, 0.456, 0.406))[
            None, None, :, None, None
        ]
        std = encoder_input.images_rgb.new_tensor((0.229, 0.224, 0.225))[
            None, None, :, None, None
        ]
        return encoder_input.images_rgb * std + mean

    def _encode_patch_tokens(self, encoder_input: ModelInputEncoder) -> dict:
        encoded = self.image_encoder(
            images=self._raw_images(encoder_input),
            intrinsics=encoder_input.intrinsics_input,
            c2w=encoder_input.cam_to_world_input,
        )
        return self._apply_hybrid_anchor_initialization(encoded)

    @staticmethod
    def _uniform_anchor_indices(
        total_anchors: int,
        anchors_to_keep: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select depth anchors uniformly without privileging one input view."""
        if not 0 < anchors_to_keep <= total_anchors:
            raise ValueError(
                f"anchors_to_keep must be in [1,{total_anchors}], got {anchors_to_keep}"
            )
        return torch.div(
            torch.arange(anchors_to_keep, device=device) * total_anchors,
            anchors_to_keep,
            rounding_mode="floor",
        )

    def _apply_hybrid_anchor_initialization(self, encoded: dict) -> dict:
        """Combine uniformly sampled depth anchors with learnable prior anchors."""
        if not self.opt.use_hybrid_anchor_init:
            return encoded

        depth_features = encoded["initial_token_features"]
        depth_positions = encoded["initial_token_positions"]
        depth_valid_mask = encoded["initial_token_valid_mask"]
        depth_raw_radii = encoded.get("initial_token_raw_radii")
        if depth_features.ndim != 3 or depth_features.shape[-1] != self.opt.token_dim:
            raise ValueError(
                "initial_token_features must have shape [B,N,token_dim], got "
                f"{tuple(depth_features.shape)}"
            )
        if depth_positions.shape != (*depth_features.shape[:2], 3):
            raise ValueError("initial depth feature/position shapes do not align")
        if depth_valid_mask.shape != depth_features.shape[:2]:
            raise ValueError("initial depth feature/valid-mask shapes do not align")
        if depth_raw_radii is not None and depth_raw_radii.shape != (
            *depth_features.shape[:2],
            1,
        ):
            raise ValueError("initial depth radius/feature shapes do not align")

        num_depth_anchors = self.opt.num_gs_tokens - self.opt.num_prior_anchors
        if depth_features.shape[1] < num_depth_anchors:
            raise ValueError(
                f"encoder produced {depth_features.shape[1]} depth anchors, but hybrid "
                f"initialization needs {num_depth_anchors}"
            )
        depth_indices = self._uniform_anchor_indices(
            depth_features.shape[1], num_depth_anchors, depth_features.device
        )
        selected_features = depth_features.index_select(1, depth_indices)
        selected_positions = depth_positions.index_select(1, depth_indices)
        selected_valid_mask = depth_valid_mask.index_select(1, depth_indices)
        selected_raw_radii = (
            None
            if depth_raw_radii is None
            else depth_raw_radii.index_select(1, depth_indices)
        )

        batch_size = depth_features.shape[0]
        prior_features = self.prior_anchor_features.to(depth_features).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        prior_positions = self.prior_anchor_positions.to(depth_positions).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        prior_valid_mask = torch.ones(
            batch_size,
            self.opt.num_prior_anchors,
            dtype=torch.bool,
            device=depth_valid_mask.device,
        )

        encoded = dict(encoded)
        encoded["initial_token_features"] = torch.cat(
            (selected_features, prior_features), dim=1
        )
        encoded["initial_token_positions"] = torch.cat(
            (selected_positions, prior_positions), dim=1
        )
        encoded["initial_token_valid_mask"] = torch.cat(
            (selected_valid_mask, prior_valid_mask), dim=1
        )
        if selected_raw_radii is not None:
            prior_raw_radii = self.prior_anchor_raw_radii.to(
                selected_raw_radii
            ).unsqueeze(0).expand(batch_size, -1, -1)
            encoded["initial_token_raw_radii"] = torch.cat(
                (selected_raw_radii, prior_raw_radii), dim=1
            )
        # Keep the generic token-position alias consistent for diagnostics.
        encoded["token_positions"] = encoded["initial_token_positions"]
        return encoded

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        """Expose /8 global features as the ATE cross-attention K/V."""
        encoded = self._encode_patch_tokens(encoder_input)
        keys, values = self._fine_features_to_kv(encoded["fine_ca_features"])
        return EncoderLatent(keys=keys, values=values)

    def _fine_features_to_kv(self, fine_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fine_features = self.fine_kv_input_norm(fine_features)
        fine_keys, fine_values = rearrange(
            self.fine_kv_proj(fine_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        fine_keys = self.fine_k_norm(fine_keys)
        return fine_keys, fine_values

    def _activate_token_radii(self, raw_radii: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_radii) + self.opt.token_radius_epsilon

    @staticmethod
    def _update_token_mu_and_raw_radius(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        raw_radii: torch.Tensor,
        valid_mask: torch.Tensor,
        updater: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a token-feature-conditioned (delta-mu, delta-raw-radius)."""
        if positions.shape != (*tokens.shape[:2], 3):
            raise ValueError("token positions must have shape [B,N,3]")
        if raw_radii.shape != (*tokens.shape[:2], 1):
            raise ValueError("token raw radii must have shape [B,N,1]")
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError("token valid mask must have shape [B,N]")
        residuals = updater(tokens)
        if residuals.shape != (*tokens.shape[:2], 4):
            raise ValueError("token geometry updater must output [B,N,4]")
        mask = valid_mask.unsqueeze(-1)
        updated_positions = torch.where(
            mask,
            positions + residuals[..., :3].to(positions.dtype),
            positions,
        )
        updated_raw_radii = torch.where(
            mask,
            raw_radii + residuals[..., 3:4].to(raw_radii.dtype),
            raw_radii,
        )
        return updated_positions, updated_raw_radii

    def _decode_tokens_to_gaussians(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        raw_radii: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_radii.shape != (*positions.shape[:2], 1):
            raise ValueError(
                "raw_radii must have shape [B,N,1] matching token positions"
            )
        gaussians = self.activation_head(self.gaussian_head_norm(tokens))
        gaussians_per_token = self.opt.num_gaussians_per_token
        gaussian_valid_mask = repeat(
            valid_mask, "b n -> b (n p)", p=gaussians_per_token
        )
        reference_positions = repeat(positions, "b n c -> b (n p) c", p=gaussians_per_token)
        radii = self._activate_token_radii(raw_radii)
        gaussian_radii = repeat(
            radii, "b n 1 -> b (n p) 1", p=gaussians_per_token
        )
        position_bias = gaussian_radii * gaussians[..., :3]
        gaussians = gaussians.clone()
        gaussians[..., :3] = reference_positions + position_bias
        gaussian_valid = gaussian_valid_mask.unsqueeze(-1)
        gaussians[..., 3:4] = (
            gaussians[..., 3:4] * gaussian_valid.to(gaussians.dtype)
        )
        identity_rotation = gaussians.new_tensor((1.0, 0.0, 0.0, 0.0))
        gaussians[..., 7:11] = torch.where(
            gaussian_valid, gaussians[..., 7:11], identity_rotation
        )
        return gaussians, position_bias

    def _gaussian_valid_mask(self, token_valid_mask: torch.Tensor) -> torch.Tensor:
        return repeat(
            token_valid_mask,
            "b n -> b (n p)",
            p=self.opt.num_gaussians_per_token,
        )

    @staticmethod
    def _gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=gather_index)

    def expand_tokens(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        raw_radii: torch.Tensor,
        valid_mask: torch.Tensor,
        split_scores: torch.Tensor,
        ratio: float,
        spatial_expander: nn.Module,
        feature_expander: nn.Module,
        unsplit_spatial_updater: nn.Module,
        feature_debug: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split tokens while updating center and raw radius in parallel."""
        if raw_radii.shape != (*positions.shape[:2], 1):
            raise ValueError(
                "raw_radii must have shape [B,N,1] matching token positions"
            )
        batch, _, dim = tokens.shape
        children_per_parent = self.opt.ate_children_per_parent
        expanded_tokens_per_scene: list[torch.Tensor] = []
        expanded_positions_per_scene: list[torch.Tensor] = []
        expanded_raw_radii_per_scene: list[torch.Tensor] = []
        selected_indices_per_scene: list[torch.Tensor] = []
        selected_features_per_scene: list[torch.Tensor] = []
        selected_positions_per_scene: list[torch.Tensor] = []
        position_residuals_per_scene: list[torch.Tensor] = []
        child_positions_per_scene: list[torch.Tensor] = []
        feature_residuals_per_scene: list[torch.Tensor] = []
        child_features_per_scene: list[torch.Tensor] = []
        unsplit_position_residuals_per_scene: list[torch.Tensor] = []
        updated_unsplit_positions_per_scene: list[torch.Tensor] = []

        for scene_index in range(batch):
            valid_indices = torch.nonzero(valid_mask[scene_index], as_tuple=False).squeeze(-1)
            num_valid = int(valid_indices.numel())
            num_selected = max(1, min(num_valid, math.ceil(num_valid * ratio)))
            valid_scores = split_scores[scene_index, valid_indices, 0].detach()
            selected_local = valid_scores.topk(num_selected, largest=True).indices
            selected_indices = valid_indices[selected_local]
            selected_flag = torch.zeros(num_valid, dtype=torch.bool, device=tokens.device)
            selected_flag[selected_local] = True
            unselected_indices = valid_indices[~selected_flag]

            selected_tokens = tokens[scene_index, selected_indices]
            selected_positions = positions[scene_index, selected_indices]
            selected_raw_radii = raw_radii[scene_index, selected_indices]
            unselected_tokens = tokens[scene_index, unselected_indices]
            unselected_positions = positions[scene_index, unselected_indices]
            unselected_raw_radii = raw_radii[scene_index, unselected_indices]
            spatial_residuals = spatial_expander(selected_tokens).view(
                num_selected, children_per_parent, 4
            )
            position_residuals = spatial_residuals[..., :3]
            radius_residuals = spatial_residuals[..., 3:4]
            # Position and feature children are predicted by independent heads
            # from the same parent feature. Predicted XYZ is not used to
            # condition the child feature head.
            feature_residuals = feature_expander(selected_tokens).view(
                num_selected, children_per_parent, dim
            )
            child_tokens = selected_tokens.unsqueeze(1) + feature_residuals
            child_positions = (
                selected_positions.unsqueeze(1)
                + position_residuals.to(selected_positions.dtype)
            )
            unsplit_spatial_residuals = unsplit_spatial_updater(
                unselected_tokens
            )
            unsplit_position_residuals = unsplit_spatial_residuals[..., :3]
            unsplit_radius_residuals = unsplit_spatial_residuals[..., 3:4]
            updated_unsplit_positions = (
                unselected_positions
                + unsplit_position_residuals.to(unselected_positions.dtype)
            )
            child_raw_radii = (
                selected_raw_radii.unsqueeze(1)
                + radius_residuals.to(selected_raw_radii.dtype)
            )
            updated_unsplit_raw_radii = (
                unselected_raw_radii
                + unsplit_radius_residuals.to(unselected_raw_radii.dtype)
            )

            scene_tokens = torch.cat(
                (unselected_tokens, child_tokens.flatten(0, 1)), dim=0
            )
            scene_positions = torch.cat(
                (updated_unsplit_positions, child_positions.flatten(0, 1)), dim=0
            )
            scene_raw_radii = torch.cat(
                (updated_unsplit_raw_radii, child_raw_radii.flatten(0, 1)), dim=0
            )
            expanded_tokens_per_scene.append(scene_tokens)
            expanded_positions_per_scene.append(scene_positions)
            expanded_raw_radii_per_scene.append(scene_raw_radii)
            selected_indices_per_scene.append(selected_indices)
            if feature_debug is not None:
                selected_features_per_scene.append(selected_tokens.detach())
                selected_positions_per_scene.append(selected_positions.detach())
                position_residuals_per_scene.append(
                    position_residuals.detach().flatten(0, 1)
                )
                child_positions_per_scene.append(
                    child_positions.detach().flatten(0, 1)
                )
                feature_residuals_per_scene.append(
                    feature_residuals.detach().flatten(0, 1)
                )
                child_features_per_scene.append(child_tokens.detach().flatten(0, 1))
                unsplit_position_residuals_per_scene.append(
                    unsplit_position_residuals.detach()
                )
                updated_unsplit_positions_per_scene.append(
                    updated_unsplit_positions.detach()
                )

        if feature_debug is not None:
            feature_debug["selected_parent_features"] = torch.cat(
                selected_features_per_scene, dim=0
            )
            feature_debug["selected_parent_positions"] = torch.cat(
                selected_positions_per_scene, dim=0
            )
            feature_debug["child_position_deltas"] = torch.cat(
                position_residuals_per_scene, dim=0
            )
            feature_debug["child_positions"] = torch.cat(
                child_positions_per_scene, dim=0
            )
            feature_debug["child_feature_deltas"] = torch.cat(
                feature_residuals_per_scene, dim=0
            )
            feature_debug["child_features"] = torch.cat(
                child_features_per_scene, dim=0
            )
            feature_debug["unsplit_position_deltas"] = torch.cat(
                unsplit_position_residuals_per_scene, dim=0
            )
            feature_debug["updated_unsplit_positions"] = torch.cat(
                updated_unsplit_positions_per_scene, dim=0
            )

        max_tokens = max(value.shape[0] for value in expanded_tokens_per_scene)
        max_selected = max(value.shape[0] for value in selected_indices_per_scene)
        expanded_tokens = torch.stack([
            F.pad(value, (0, 0, 0, max_tokens - value.shape[0]))
            for value in expanded_tokens_per_scene
        ])
        expanded_positions = torch.stack([
            F.pad(value, (0, 0, 0, max_tokens - value.shape[0]))
            for value in expanded_positions_per_scene
        ])
        expanded_raw_radii = torch.stack([
            F.pad(value, (0, 0, 0, max_tokens - value.shape[0]))
            for value in expanded_raw_radii_per_scene
        ])
        expanded_counts = torch.tensor(
            [value.shape[0] for value in expanded_tokens_per_scene],
            dtype=torch.long,
            device=tokens.device,
        )
        expanded_valid_mask = torch.arange(max_tokens, device=tokens.device)[None] < expanded_counts[:, None]
        selected_indices = torch.stack([
            F.pad(value, (0, max_selected - value.shape[0]), value=-1)
            for value in selected_indices_per_scene
        ])
        return (
            expanded_tokens,
            expanded_positions,
            expanded_raw_radii,
            expanded_valid_mask,
            selected_indices,
        )

    def _predict_token_uncertainty(
        self,
        stage_index: int,
        tokens: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one non-negative scalar per token and repeat it per Gaussian."""
        token_uncertainty = F.softplus(
            self.uncertainty_heads[stage_index](tokens.detach())
        )
        token_uncertainty = token_uncertainty * token_valid_mask.unsqueeze(-1).to(
            token_uncertainty.dtype
        )
        gaussian_uncertainty = repeat(
            token_uncertainty,
            "b n 1 -> b (n k) 1",
            k=self.opt.num_gaussians_per_token,
        )
        return token_uncertainty, gaussian_uncertainty

    def _context_uncertainty_error(
        self,
        stage_gaussians: torch.Tensor,
        gaussian_uncertainty: torch.Tensor,
        gt_images: torch.Tensor,
        decoder_input: ModelInputDecoder,
        background_color: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Render RGB+uncertainty on context views and return detached-target L1 sum."""
        rgb_features = self.gs.color_parameters_to_rgb(
            stage_gaussians[..., 11:14].detach().float()
        )
        rendered = self.gs.render_features(
            stage_gaussians.detach(),
            torch.cat((rgb_features, gaussian_uncertainty), dim=-1),
            decoder_input.cam_view.detach(),
            decoder_input.intrinsics.detach(),
            background=torch.cat(
                (
                    background_color.detach().float(),
                    background_color.new_zeros(1).float(),
                )
            ),
        )
        predicted_uncertainty = rendered[:, :, 3:4]
        with torch.no_grad():
            target_uncertainty = dssim_error_map(
                rendered[:, :, :3].detach(), gt_images.detach()
            )
        absolute_error = (predicted_uncertainty - target_uncertainty).abs()
        return absolute_error.sum(), absolute_error.numel()

    def _encode_backassigned_error(
        self,
        tokens: torch.Tensor,
        token_positions: torch.Tensor,
        stage_gaussians: torch.Tensor,
        gaussian_error: torch.Tensor,
    ) -> torch.Tensor:
        """Encode grouped Gaussian observations and pool them to each token."""
        gaussians_per_token = self.opt.num_gaussians_per_token
        expected_gaussians = tokens.shape[1] * gaussians_per_token
        if stage_gaussians.shape[1] != expected_gaussians:
            raise ValueError("token/Gaussian count mismatch in error back-assignment")
        error_channels = self.multiscale_error_back_assignment.error_channels
        if gaussian_error.shape != (
            tokens.shape[0],
            expected_gaussians,
            error_channels,
        ):
            raise ValueError(
                "gaussian_error must have shape "
                f"[{tokens.shape[0]}, {expected_gaussians}, {error_channels}]"
            )
        grouped_error = gaussian_error.reshape(
            tokens.shape[0],
            tokens.shape[1],
            gaussians_per_token,
            error_channels,
        )
        gaussian_xyz = stage_gaussians[..., :3].reshape(
            tokens.shape[0], tokens.shape[1], gaussians_per_token, 3
        )
        relative_xyz = gaussian_xyz.detach() - token_positions.detach().unsqueeze(-2)
        encoder_input = torch.cat((grouped_error.detach(), relative_xyz), dim=-1)
        encoded = self.gaussian_error_encoder(encoder_input)
        return torch.cat(
            (encoded.mean(dim=2), encoded.amax(dim=2)), dim=-1
        )

    def _apply_fullres_error_feedback(
        self,
        token_features: torch.Tensor,
        token_positions: torch.Tensor,
        token_raw_radii: torch.Tensor,
        token_valid_mask: torch.Tensor,
        stage_gaussians: torch.Tensor,
        error_gt_images: torch.Tensor,
        error_decoder_input: ModelInputDecoder,
        background_color: torch.Tensor,
        target_features: torch.Tensor | None,
        rendered_rgb: torch.Tensor | None = None,
        context_view_count: int | None = None,
    ) -> FullResolutionTokenProjectionFeedbackResult:
        """Use rendered view errors to update token content without changing mu/r."""
        if self.fullres_gaussian_token_error_feedback is None:
            raise RuntimeError("full-resolution error feedback is not initialized")
        if rendered_rgb is None:
            rendered_rgb = self._render_fullres_error_views(
                stage_gaussians,
                error_decoder_input,
                background_color,
            )
        return self.fullres_gaussian_token_error_feedback(
            token_features,
            token_positions,
            self._activate_token_radii(token_raw_radii),
            token_valid_mask,
            stage_gaussians,
            rendered_rgb,
            error_gt_images,
            error_decoder_input.cam_view,
            error_decoder_input.intrinsics,
            target_features=target_features,
            context_view_count=context_view_count,
        )

    def _apply_fullres_proxy_error_feedback(
        self,
        token_features: torch.Tensor,
        token_positions: torch.Tensor,
        token_raw_radii: torch.Tensor,
        token_valid_mask: torch.Tensor,
        stage_gaussians: torch.Tensor,
        error_gt_images: torch.Tensor,
        error_decoder_input: ModelInputDecoder,
        background_color: torch.Tensor,
        target_features: torch.Tensor | None,
        rendered_rgb: torch.Tensor | None = None,
        error_feature_levels: int = 3,
    ) -> TokenProxyErrorFeedbackResult:
        """Use H3D token proxies only for full-resolution error attribution."""
        if self.fullres_proxy_gaussian_error_feedback is None:
            raise RuntimeError("full-resolution proxy feedback is not initialized")
        if rendered_rgb is None:
            rendered_rgb = self._render_fullres_error_views(
                stage_gaussians,
                error_decoder_input,
                background_color,
            )
        return self.fullres_proxy_gaussian_error_feedback(
            token_features,
            token_positions,
            self._activate_token_radii(token_raw_radii),
            token_valid_mask,
            stage_gaussians,
            rendered_rgb,
            error_gt_images,
            error_decoder_input.cam_view,
            error_decoder_input.intrinsics,
            target_features=target_features,
            error_feature_levels=error_feature_levels,
        )

    def _render_fullres_error_views(
        self,
        stage_gaussians: torch.Tensor,
        decoder_input: ModelInputDecoder,
        background_color: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input.cam_view is None or decoder_input.intrinsics is None:
            raise ValueError("error-feedback cameras and intrinsics are required")
        rgb_features = self.gs.color_parameters_to_rgb(
            stage_gaussians[..., 11:14].detach().float()
        )
        return self.gs.render_features(
            stage_gaussians.detach(),
            rgb_features,
            decoder_input.cam_view.detach(),
            decoder_input.intrinsics.detach(),
            background=background_color.detach().float(),
        ).detach()

    def _select_difix_target_decoder_input(
        self,
        decoder_input: ModelInputDecoder,
    ) -> ModelInputDecoder:
        if decoder_input.cam_view is None or decoder_input.intrinsics is None:
            raise ValueError("Difix target cameras and intrinsics are required")
        first_target = self.opt.num_input_views if self.opt.use_input_supervision else 0
        cam_view = decoder_input.cam_view[:, first_target:]
        intrinsics = decoder_input.intrinsics[:, first_target:]
        if cam_view.shape[1] == 0:
            raise ValueError("Difix target camera slice is empty")
        if intrinsics.shape[:2] != cam_view.shape[:2]:
            raise ValueError("Difix target cameras and intrinsics must align")
        max_views = min(self.opt.difix_target_max_views, cam_view.shape[1])
        if self.training and cam_view.shape[1] > max_views:
            random_scores = torch.rand(
                cam_view.shape[0],
                cam_view.shape[1],
                device=cam_view.device,
            )
            selected = random_scores.topk(max_views, dim=1).indices
            selected = selected.sort(dim=1).values

            def gather_views(values: torch.Tensor) -> torch.Tensor:
                index = selected.reshape(
                    selected.shape[0],
                    selected.shape[1],
                    *([1] * (values.ndim - 2)),
                ).expand(-1, -1, *values.shape[2:])
                return torch.gather(values, dim=1, index=index)

            cam_view = gather_views(cam_view)
            intrinsics = gather_views(intrinsics)
        else:
            cam_view = cam_view[:, :max_views]
            intrinsics = intrinsics[:, :max_views]
        return ModelInputDecoder(cam_view=cam_view, intrinsics=intrinsics)

    def _prepare_difix_fullres_feedback_views(
        self,
        *,
        stage_gaussians: torch.Tensor,
        context_images: torch.Tensor,
        context_decoder_input: ModelInputDecoder,
        target_decoder_input: ModelInputDecoder,
        background_color: torch.Tensor,
        context_features: torch.Tensor,
        cached_pseudo_images: torch.Tensor | None,
        cached_pseudo_features: torch.Tensor | None,
    ) -> _DifixFullresFeedbackViews:
        if self.difix_target_teacher is None:
            raise RuntimeError("Difix target teacher is not initialized")
        if self.fullres_gaussian_token_error_feedback is None:
            raise RuntimeError("full-resolution error feedback is not initialized")
        if (
            context_decoder_input.cam_view is None
            or context_decoder_input.intrinsics is None
            or target_decoder_input.cam_view is None
            or target_decoder_input.intrinsics is None
        ):
            raise ValueError("context and target feedback cameras are required")
        combined_decoder = ModelInputDecoder(
            cam_view=torch.cat(
                (context_decoder_input.cam_view, target_decoder_input.cam_view),
                dim=1,
            ),
            intrinsics=torch.cat(
                (context_decoder_input.intrinsics, target_decoder_input.intrinsics),
                dim=1,
            ),
        )
        rendered_rgb = self._render_fullres_error_views(
            stage_gaussians,
            combined_decoder,
            background_color,
        )
        context_views = context_images.shape[1]
        if context_views != context_decoder_input.cam_view.shape[1]:
            raise ValueError("context images and feedback cameras must align")
        target_rendered_rgb = rendered_rgb[:, context_views:]
        if cached_pseudo_images is None or cached_pseudo_features is None:
            if not (cached_pseudo_images is None and cached_pseudo_features is None):
                raise ValueError("Difix pseudo image and feature caches must be paired")
            pseudo_images = self.difix_target_teacher.repair(target_rendered_rgb)
            pseudo_features = (
                self.fullres_gaussian_token_error_feedback.feature_extractor.extract_features(
                    pseudo_images.detach()
                ).detach()
            )
        else:
            pseudo_images = cached_pseudo_images
            pseudo_features = cached_pseudo_features
        if pseudo_images.shape != target_rendered_rgb.shape:
            raise ValueError("Difix pseudo images must match target render shape")
        if context_features.shape[:2] != context_images.shape[:2]:
            raise ValueError("context images and features must align")
        if pseudo_features.shape[0:2] != target_rendered_rgb.shape[0:2]:
            raise ValueError("Difix pseudo features must align with target views")
        if pseudo_features.shape[2:] != context_features.shape[2:]:
            raise ValueError("context and Difix pseudo feature shapes must match")
        return _DifixFullresFeedbackViews(
            rendered_rgb=rendered_rgb,
            reference_images=torch.cat((context_images, pseudo_images), dim=1),
            reference_features=torch.cat(
                (context_features, pseudo_features), dim=1
            ),
            decoder_input=combined_decoder,
            pseudo_images=pseudo_images.detach(),
            pseudo_features=pseudo_features.detach(),
        )

    def forward_decoder_from_patch_tokens(
        self,
        encoded: dict,
        decoder_input: ModelInputDecoder,
        error_decoder_input: ModelInputDecoder,
        error_gt_images: torch.Tensor,
        background_color: torch.Tensor,
        target_error_decoder_input: ModelInputDecoder | None = None,
    ) -> tuple:
        """Run initial anchor PE/SA/FFN, then CA-PE-SA-FFN decoder layers and ATE."""
        fine_keys, fine_values = self._fine_features_to_kv(encoded["fine_ca_features"])
        fine_rays = encoded["fine_ca_rays"]
        if fine_rays.shape != (fine_keys.shape[0], fine_keys.shape[2], 6):
            raise ValueError(
                "fine patch rays must align one-to-one with cross-attention keys"
            )
        gs_tokens = encoded["initial_token_features"]
        token_positions = encoded["initial_token_positions"]
        token_valid_mask = encoded["initial_token_valid_mask"]
        token_raw_radii = encoded["initial_token_raw_radii"]
        if token_raw_radii.shape != (*gs_tokens.shape[:2], 1):
            raise ValueError(
                "initial_token_raw_radii must have shape [B,N,1] aligned with "
                "initial token features"
            )
        anchor_transformer_layer, decoder_layer = self.enc_dec_backbone.decoder_blocks

        token_feature_snapshots: dict[str, torch.Tensor] = {}

        def snapshot(name: str, value: torch.Tensor) -> None:
            if self.log_token_feature_numerics:
                token_feature_snapshots[name] = value.detach()

        def run_decoder(
            layer: DecoderBlock,
            name: str,
            tokens: torch.Tensor,
            use_geometric_bias: bool = False,
        ) -> torch.Tensor:
            xyz_pos = None
            if self.anchor_xyz_pos_encoder is not None:
                xyz_pos = self.anchor_xyz_pos_encoder(token_positions).to(
                    tokens.dtype
                )
                xyz_pos = xyz_pos * token_valid_mask.unsqueeze(-1).to(xyz_pos.dtype)
                snapshot(f"{name}_xyz_positional_encoding_before_sa", xyz_pos)
            if not self.log_token_feature_numerics:
                return layer(
                    gs_tokens=tokens,
                    keys=fine_keys,
                    values=fine_values,
                    token_mask=token_valid_mask,
                    self_attn_position=xyz_pos,
                    cross_attn_anchor_positions=(
                        token_positions if use_geometric_bias else None
                    ),
                    cross_attn_anchor_radii=(
                        self._activate_token_radii(token_raw_radii)
                        if use_geometric_bias
                        else None
                    ),
                    cross_attn_image_rays=(
                        fine_rays if use_geometric_bias else None
                    ),
                )
            output, intermediates = layer(
                gs_tokens=tokens,
                keys=fine_keys,
                values=fine_values,
                token_mask=token_valid_mask,
                self_attn_position=xyz_pos,
                cross_attn_anchor_positions=(
                    token_positions if use_geometric_bias else None
                ),
                cross_attn_anchor_radii=(
                    self._activate_token_radii(token_raw_radii)
                    if use_geometric_bias
                    else None
                ),
                cross_attn_image_rays=(fine_rays if use_geometric_bias else None),
                return_intermediates=True,
            )
            for suffix, feature in intermediates.items():
                snapshot(f"{name}_{suffix}", feature)
            return output

        def run_anchor_transformer(
            layer: DecoderBlock,
            tokens: torch.Tensor,
            xyz_pos: torch.Tensor | None,
        ) -> torch.Tensor:
            """Run only the SA and FFN branches of a decoder block."""
            mask = token_valid_mask.unsqueeze(-1).to(tokens.dtype)
            tokens = tokens + layer.gs_self_attn_scale(
                layer.gs_self_attn(
                    tokens,
                    token_mask=token_valid_mask,
                    self_attn_position=xyz_pos,
                )
            )
            tokens = tokens * mask
            snapshot("anchor_initial_transformer_after_sa", tokens)
            tokens = tokens + layer.mlp_scale(layer.mlp(tokens))
            tokens = tokens * mask
            snapshot("anchor_initial_transformer_after_ffn", tokens)
            return tokens

        snapshot("encoder_coarse_features", encoded["coarse_features"])
        snapshot("encoder_fine_branch_features", encoded["fine_branch_features"])
        snapshot("encoder_fine_fused_features", encoded["fine_features"])
        snapshot("encoder_raw_log_depths", encoded["raw_log_depths"])
        snapshot("encoder_positive_depths", encoded["fine_depths"])
        snapshot("decoder_fine_keys", fine_keys)
        snapshot("decoder_fine_values", fine_values)
        snapshot("anchor_initial_features", gs_tokens)
        anchor_xyz_pos = None
        if self.anchor_xyz_pos_encoder is not None:
            anchor_xyz_pos = self.anchor_xyz_pos_encoder(token_positions).to(
                gs_tokens.dtype
            )
            anchor_xyz_pos = anchor_xyz_pos * token_valid_mask.unsqueeze(-1).to(
                anchor_xyz_pos.dtype
            )
            snapshot("anchor_absolute_xyz_positional_encoding", anchor_xyz_pos)
        gs_tokens = run_anchor_transformer(
            anchor_transformer_layer, gs_tokens, anchor_xyz_pos
        )
        gs_tokens = run_decoder(
            decoder_layer,
            "anchor_initial_decoder",
            gs_tokens,
            use_geometric_bias=True,
        )
        token_positions, token_raw_radii = self._update_token_mu_and_raw_radius(
            gs_tokens,
            token_positions,
            token_raw_radii,
            token_valid_mask,
            self.initial_token_spatial_updater,
        )
        snapshot(
            "anchor_initial_updated_positions",
            token_positions,
        )
        snapshot(
            "anchor_initial_updated_radius",
            self._activate_token_radii(token_raw_radii),
        )
        token_split_scores: list[torch.Tensor] = []
        gaussian_uncertainties: list[torch.Tensor] = []
        context_uncertainty_error_sums: list[torch.Tensor] = []
        context_uncertainty_element_counts: list[int] = []
        token_error_features: list[torch.Tensor] = []
        gaussian_error_features: list[torch.Tensor] = []
        gaussian_error_visibility: list[torch.Tensor] = []
        gaussian_error_mean_magnitudes: list[torch.Tensor] = []
        gaussian_visible_ratios: list[torch.Tensor] = []
        gaussian_max_error_indices: list[torch.Tensor] = []
        fullres_point_error_mean_magnitudes: list[torch.Tensor] = []
        fullres_point_valid_ratios: list[torch.Tensor] = []
        fullres_view_valid_ratios: list[torch.Tensor] = []
        fullres_token_error_mean_magnitudes: list[torch.Tensor] = []
        multiscale_rgb_error_magnitudes: list[torch.Tensor] = []
        multiscale_feature_error_magnitudes: list[torch.Tensor] = []
        multiscale_gaussian_error_means: list[torch.Tensor] = []
        multiscale_gaussian_error_stds: list[torch.Tensor] = []
        multiscale_gaussian_visible_ratios: list[torch.Tensor] = []
        gaussian_error_heatmap: torch.Tensor | None = None
        intermediate_gaussians: list[torch.Tensor] = []
        intermediate_gaussian_valid_masks: list[torch.Tensor] = []
        intermediate_token_positions: list[torch.Tensor] = []
        intermediate_token_radii: list[torch.Tensor] = []
        intermediate_token_valid_masks: list[torch.Tensor] = []
        selected_indices: list[torch.Tensor] = []

        if self.opt.use_ate_split:
            gt_error_features = None
            fullres_gt_error_features = None
            difix_pseudo_images = None
            difix_pseudo_features = None
            if self.opt.use_ate_error_feedback:
                assert self.multiscale_error_back_assignment is not None
                gt_error_features = (
                    self.multiscale_error_back_assignment.extract_features(
                        error_gt_images
                    )
                )
            if self.opt.use_fullres_gaussian_token_error_feedback:
                fullres_feedback_module = (
                    self.fullres_gaussian_token_error_feedback
                    if self.fullres_gaussian_token_error_feedback is not None
                    else self.fullres_proxy_gaussian_error_feedback
                )
                assert fullres_feedback_module is not None
                fullres_gt_error_features = (
                    fullres_feedback_module.feature_extractor.extract_features(
                        error_gt_images
                    )
                )

            def record_intermediate(
                stage_index: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                nonlocal gaussian_error_heatmap
                snapshot(f"ate_{stage_index}_pre_split_features", gs_tokens)
                if self.log_token_feature_numerics:
                    with torch.no_grad():
                        snapshot(
                            f"ate_{stage_index}_gaussian_head_input_features",
                            self.gaussian_head_norm(gs_tokens.detach()),
                        )
                stage_gaussians, position_bias = self._decode_tokens_to_gaussians(
                    gs_tokens, token_positions, token_raw_radii, token_valid_mask
                )
                gaussian_valid_mask = self._gaussian_valid_mask(token_valid_mask)
                split_score, gaussian_uncertainty = self._predict_token_uncertainty(
                    stage_index, gs_tokens, token_valid_mask
                )
                token_split_scores.append(split_score)
                gaussian_uncertainties.append(gaussian_uncertainty)
                if (
                    self.opt.lambda_uncertainty > 0
                    and not self.opt.use_input_supervision
                ):
                    context_error_sum, context_element_count = (
                        self._context_uncertainty_error(
                            stage_gaussians,
                            gaussian_uncertainty,
                            error_gt_images,
                            error_decoder_input,
                            background_color,
                        )
                    )
                else:
                    context_error_sum = gaussian_uncertainty.sum() * 0
                    context_element_count = 0
                context_uncertainty_error_sums.append(context_error_sum)
                context_uncertainty_element_counts.append(context_element_count)

                if self.opt.use_ate_error_feedback:
                    assert self.multiscale_error_back_assignment is not None
                    assignment = self.gaussian_error_back_assignment(
                        stage_gaussians,
                        error_gt_images,
                        error_decoder_input.cam_view,
                        error_decoder_input.intrinsics,
                        background_color,
                        gaussian_valid_mask,
                    )
                    multiscale_assignment = self.multiscale_error_back_assignment(
                        stage_gaussians,
                        assignment.rendered_rgb,
                        error_gt_images,
                        error_decoder_input.cam_view,
                        error_decoder_input.intrinsics,
                        gaussian_valid_mask,
                        gt_features=gt_error_features,
                    )
                    error_feat = self._encode_backassigned_error(
                        gs_tokens,
                        token_positions,
                        stage_gaussians,
                        multiscale_assignment.fused_error,
                    )
                    token_error_features.append(error_feat)
                    gaussian_error_features.append(assignment.fused_error)
                    gaussian_error_visibility.append(assignment.fused_visibility)
                    gaussian_error_mean_magnitudes.append(assignment.mean_error_magnitude)
                    gaussian_visible_ratios.append(assignment.visible_gaussian_ratio)
                    gaussian_max_error_indices.append(assignment.max_error_gaussian_index)
                    multiscale_rgb_error_magnitudes.append(
                        multiscale_assignment.rgb_error_magnitude
                    )
                    multiscale_feature_error_magnitudes.append(
                        multiscale_assignment.feature_error_magnitude
                    )
                    multiscale_gaussian_error_means.append(
                        multiscale_assignment.gaussian_error_feature_mean
                    )
                    multiscale_gaussian_error_stds.append(
                        multiscale_assignment.gaussian_error_feature_std
                    )
                    multiscale_gaussian_visible_ratios.append(
                        multiscale_assignment.visible_gaussian_ratio
                    )
                if (
                    self.opt.use_ate_error_feedback
                    and stage_index == len(self.opt.ate_split_ratios) - 1
                    and self.log_gaussian_error_heatmap
                ):
                    error_magnitude = torch.linalg.vector_norm(
                        assignment.fused_error.float(), dim=-1, keepdim=True
                    )
                    with torch.no_grad():
                        gaussian_error_heatmap = self.gs.render_features(
                            stage_gaussians,
                            error_magnitude,
                            error_decoder_input.cam_view,
                            error_decoder_input.intrinsics,
                        )
                intermediate_gaussians.append(stage_gaussians)
                intermediate_gaussian_valid_masks.append(gaussian_valid_mask)
                intermediate_token_positions.append(token_positions)
                intermediate_token_radii.append(
                    self._activate_token_radii(token_raw_radii)
                )
                intermediate_token_valid_masks.append(token_valid_mask)
                return (
                    split_score,
                    stage_gaussians,
                    position_bias,
                    gaussian_valid_mask,
                )

            def inject_error(stage_index: int) -> None:
                nonlocal gs_tokens
                delta = self.error_fusions[stage_index](token_error_features[-1])
                gs_tokens = gs_tokens + delta
                gs_tokens = gs_tokens * token_valid_mask.unsqueeze(-1).to(gs_tokens.dtype)

            # Each stage predicts uncertainty, splits the highest-scoring
            # parents, updates positions, and refines the expanded tokens.
            for stage_index, split_ratio in enumerate(self.opt.ate_split_ratios):
                (
                    split_score,
                    stage_gaussians,
                    position_bias,
                    gaussian_valid_mask,
                ) = record_intermediate(stage_index)
                if self.opt.use_ate_error_feedback:
                    inject_error(stage_index)
                if self.opt.use_fullres_gaussian_token_error_feedback:
                    if self.fullres_proxy_gaussian_error_feedback is not None:
                        feedback = self._apply_fullres_proxy_error_feedback(
                            gs_tokens,
                            token_positions,
                            token_raw_radii,
                            token_valid_mask,
                            stage_gaussians,
                            error_gt_images,
                            error_decoder_input,
                            background_color,
                            fullres_gt_error_features,
                            error_feature_levels=(
                                max(1, 3 - stage_index)
                                if self.opt.fullres_error_proxy_progressive_feature_levels
                                else 3
                            ),
                        )
                    elif self.opt.use_difix_target_error_feedback:
                        if target_error_decoder_input is None:
                            raise ValueError(
                                "Difix target feedback requires target cameras"
                            )
                        assert fullres_gt_error_features is not None
                        difix_views = self._prepare_difix_fullres_feedback_views(
                            stage_gaussians=stage_gaussians,
                            context_images=error_gt_images,
                            context_decoder_input=error_decoder_input,
                            target_decoder_input=target_error_decoder_input,
                            background_color=background_color,
                            context_features=fullres_gt_error_features,
                            cached_pseudo_images=difix_pseudo_images,
                            cached_pseudo_features=difix_pseudo_features,
                        )
                        difix_pseudo_images = difix_views.pseudo_images
                        difix_pseudo_features = difix_views.pseudo_features
                        feedback = self._apply_fullres_error_feedback(
                            gs_tokens,
                            token_positions,
                            token_raw_radii,
                            token_valid_mask,
                            stage_gaussians,
                            difix_views.reference_images,
                            difix_views.decoder_input,
                            background_color,
                            difix_views.reference_features,
                            rendered_rgb=difix_views.rendered_rgb,
                            context_view_count=error_gt_images.shape[1],
                        )
                    else:
                        feedback = self._apply_fullres_error_feedback(
                            gs_tokens,
                            token_positions,
                            token_raw_radii,
                            token_valid_mask,
                            stage_gaussians,
                            error_gt_images,
                            error_decoder_input,
                            background_color,
                            fullres_gt_error_features,
                        )
                    gs_tokens = feedback.updated_token_features
                    if self.fullres_proxy_gaussian_error_feedback is not None:
                        proxy_visible = feedback.proxy_visibility > self.opt.token_radius_epsilon
                        proxy_count = proxy_visible.sum().clamp_min(1)
                        proxy_error_magnitude = torch.linalg.vector_norm(
                            feedback.proxy_error.float(), dim=-1
                        )
                        fullres_point_error_mean_magnitudes.append(
                            (
                                proxy_error_magnitude
                                * proxy_visible.to(proxy_error_magnitude.dtype)
                            ).sum() / proxy_count
                        )
                        valid_tokens = token_valid_mask.sum().clamp_min(1)
                        fullres_point_valid_ratios.append(
                            proxy_visible.sum().float()
                            / (
                                valid_tokens.float()
                                * proxy_visible.shape[1]
                                * proxy_visible.shape[3]
                            )
                        )
                        visible_views = feedback.proxy_visibility.sum(dim=3) > self.opt.token_radius_epsilon
                        fullres_view_valid_ratios.append(
                            visible_views.sum().float()
                            / (valid_tokens.float() * visible_views.shape[1])
                        )
                        token_error_magnitude = torch.linalg.vector_norm(
                            feedback.token_error.float(), dim=-1
                        )
                        fullres_token_error_mean_magnitudes.append(
                            (
                                token_error_magnitude
                                * token_valid_mask.to(token_error_magnitude.dtype)
                            ).sum() / valid_tokens
                        )
                    else:
                        fullres_point_error_mean_magnitudes.append(
                            feedback.point_error_mean_magnitude
                        )
                        fullres_point_valid_ratios.append(
                            feedback.point_valid_ratio
                        )
                        fullres_view_valid_ratios.append(
                            feedback.view_valid_ratio
                        )
                        fullres_token_error_mean_magnitudes.append(
                            feedback.token_error_mean_magnitude
                        )
                    snapshot(
                        f"ate_{stage_index}_after_fullres_error_feedback",
                        gs_tokens,
                    )
                split_feature_debug = (
                    {} if self.log_token_feature_numerics else None
                )
                (
                    gs_tokens,
                    token_positions,
                    token_raw_radii,
                    token_valid_mask,
                    selected,
                ) = self.expand_tokens(
                    gs_tokens,
                    token_positions,
                    token_raw_radii,
                    token_valid_mask,
                    split_score,
                    split_ratio,
                    self.token_spatial_expanders[stage_index],
                    self.token_feature_expanders[stage_index],
                    self.token_unsplit_spatial_updaters[stage_index],
                    feature_debug=split_feature_debug,
                )
                if split_feature_debug is not None:
                    for debug_name, debug_feature in split_feature_debug.items():
                        snapshot(
                            f"ate_{stage_index}_expander_{debug_name}",
                            debug_feature,
                        )
                selected_indices.append(selected)
                snapshot(f"ate_{stage_index}_post_split_features", gs_tokens)
                snapshot(
                    f"ate_{stage_index}_post_split_radius",
                    self._activate_token_radii(token_raw_radii),
                )
                for layer_index, layer in enumerate(self.ate_refine_blocks[stage_index]):
                    gs_tokens = run_decoder(
                        layer,
                        f"ate_{stage_index}_refine_{layer_index}",
                        gs_tokens,
                        use_geometric_bias=True,
                    )

        snapshot("final_pre_gaussian_head_features", gs_tokens)
        if self.log_token_feature_numerics:
            with torch.no_grad():
                snapshot(
                    "final_gaussian_head_input_features",
                    self.gaussian_head_norm(gs_tokens.detach()),
                )
        gaussians, final_position_bias = self._decode_tokens_to_gaussians(
            gs_tokens, token_positions, token_raw_radii, token_valid_mask
        )
        return (
            gaussians,
            final_position_bias,
            tuple(token_split_scores),
            tuple(gaussian_uncertainties),
            tuple(context_uncertainty_error_sums),
            tuple(context_uncertainty_element_counts),
            tuple(gaussian_error_features),
            tuple(gaussian_error_visibility),
            tuple(gaussian_error_mean_magnitudes),
            tuple(gaussian_visible_ratios),
            tuple(gaussian_max_error_indices),
            tuple(fullres_point_error_mean_magnitudes),
            tuple(fullres_point_valid_ratios),
            tuple(fullres_view_valid_ratios),
            tuple(fullres_token_error_mean_magnitudes),
            tuple(multiscale_rgb_error_magnitudes),
            tuple(multiscale_feature_error_magnitudes),
            tuple(multiscale_gaussian_error_means),
            tuple(multiscale_gaussian_error_stds),
            tuple(multiscale_gaussian_visible_ratios),
            gaussian_error_heatmap,
            tuple(intermediate_gaussians),
            tuple(intermediate_gaussian_valid_masks),
            tuple(intermediate_token_positions),
            tuple(intermediate_token_radii),
            tuple(intermediate_token_valid_masks),
            tuple(selected_indices),
            token_feature_snapshots,
            gs_tokens,
            token_positions,
            self._activate_token_radii(token_raw_radii),
            token_valid_mask,
        )

    @staticmethod
    def select_uncertain_tokens(split_scores: torch.Tensor, ratio: float) -> torch.Tensor:
        """Return Top-rho token indices for the future ATE split branch."""
        num_tokens = split_scores.shape[1]
        k = max(1, min(num_tokens, math.ceil(num_tokens * ratio)))
        return split_scores.squeeze(-1).topk(k, dim=1, largest=True).indices

    def forward_reconstruction(self, model_input: ModelInput) -> ATSplatStage1Reconstruction:
        encoded = self._encode_patch_tokens(model_input.encoder)
        error_gt_images = self._raw_images(model_input.encoder)
        error_decoder_input = ModelInputDecoder(
            cam_view=torch.inverse(model_input.encoder.cam_to_world_input).transpose(-2, -1),
            intrinsics=model_input.encoder.intrinsics_input,
        )
        target_error_decoder_input = None
        if self.opt.use_difix_target_error_feedback:
            target_error_decoder_input = self._select_difix_target_decoder_input(
                model_input.decoder
            )
        background_color = self._background_color(
            encoded["initial_token_features"].dtype,
            encoded["initial_token_features"].device,
        )
        (
            gaussians,
            final_position_bias,
            token_split_scores,
            gaussian_uncertainties,
            context_uncertainty_error_sums,
            context_uncertainty_element_counts,
            gaussian_error_features,
            gaussian_error_visibility,
            gaussian_error_mean_magnitudes,
            gaussian_visible_ratios,
            gaussian_max_error_indices,
            fullres_point_error_mean_magnitudes,
            fullres_point_valid_ratios,
            fullres_view_valid_ratios,
            fullres_token_error_mean_magnitudes,
            multiscale_rgb_error_magnitudes,
            multiscale_feature_error_magnitudes,
            multiscale_gaussian_error_means,
            multiscale_gaussian_error_stds,
            multiscale_gaussian_visible_ratios,
            gaussian_error_heatmap,
            intermediate_gaussians,
            intermediate_gaussian_valid_masks,
            intermediate_token_positions,
            intermediate_token_radii,
            intermediate_token_valid_masks,
            selected_parent_indices,
            token_feature_snapshots,
            final_token_features,
            final_token_positions,
            final_token_radii,
            final_token_valid_mask,
        ) = self.forward_decoder_from_patch_tokens(
            encoded,
            model_input.decoder,
            error_decoder_input,
            error_gt_images,
            background_color,
            target_error_decoder_input,
        )
        base = self._reconstruction_from_gaussians(gaussians)
        return ATSplatStage1Reconstruction(
            gaussians=base.gaussians,
            background_color=base.background_color,
            token_split_scores=token_split_scores,
            gaussian_uncertainties=gaussian_uncertainties,
            context_uncertainty_error_sums=context_uncertainty_error_sums,
            context_uncertainty_element_counts=context_uncertainty_element_counts,
            gaussian_error_features=gaussian_error_features,
            gaussian_error_visibility=gaussian_error_visibility,
            gaussian_error_mean_magnitudes=gaussian_error_mean_magnitudes,
            gaussian_visible_ratios=gaussian_visible_ratios,
            gaussian_max_error_indices=gaussian_max_error_indices,
            fullres_point_error_mean_magnitudes=(
                fullres_point_error_mean_magnitudes
            ),
            fullres_point_valid_ratios=fullres_point_valid_ratios,
            fullres_view_valid_ratios=fullres_view_valid_ratios,
            fullres_token_error_mean_magnitudes=(
                fullres_token_error_mean_magnitudes
            ),
            multiscale_rgb_error_magnitudes=multiscale_rgb_error_magnitudes,
            multiscale_feature_error_magnitudes=multiscale_feature_error_magnitudes,
            multiscale_gaussian_error_means=multiscale_gaussian_error_means,
            multiscale_gaussian_error_stds=multiscale_gaussian_error_stds,
            multiscale_gaussian_visible_ratios=multiscale_gaussian_visible_ratios,
            gaussian_error_heatmap=gaussian_error_heatmap,
            intermediate_gaussians=intermediate_gaussians,
            intermediate_gaussian_valid_masks=intermediate_gaussian_valid_masks,
            intermediate_token_positions=intermediate_token_positions,
            intermediate_token_radii=intermediate_token_radii,
            intermediate_token_valid_masks=intermediate_token_valid_masks,
            selected_parent_indices=selected_parent_indices,
            token_feature_snapshots=token_feature_snapshots,
            final_token_features=final_token_features,
            final_token_positions=final_token_positions,
            final_token_radii=final_token_radii,
            final_token_valid_mask=final_token_valid_mask,
            final_position_bias=final_position_bias,
            gaussian_valid_mask=self._gaussian_valid_mask(final_token_valid_mask),
        )


    def _intermediate_rgb_loss(
        self,
        predicted_rgb: torch.Tensor,
        supervision: ModelSupervision,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt_rgb = supervision.images_output
        gt_rgb = (
            gt_rgb * supervision.masks_output
            + torch.ones_like(gt_rgb) * (1 - supervision.masks_output)
        )
        if self.opt.rgb_loss_type == "l1":
            rgb_loss = F.l1_loss(predicted_rgb, gt_rgb)
        else:
            rgb_loss = F.mse_loss(predicted_rgb, gt_rgb)
        loss = self.opt.lambda_rgb * rgb_loss
        ssim_loss = predicted_rgb.new_zeros(())
        if self.opt.lambda_intermediate_ssim > 0:
            height, width = predicted_rgb.shape[-2:]
            ssim = fused_ssim(
                predicted_rgb.reshape(-1, 3, height, width),
                gt_rgb.reshape(-1, 3, height, width),
            )
            ssim_loss = (1 - ssim) / 2
            loss = loss + self.opt.lambda_intermediate_ssim * ssim_loss
        return loss, rgb_loss, ssim_loss

    def _uncertainty_loss(
        self,
        reconstruction: ATSplatStage1Reconstruction,
        stage_index: int,
        stage_rgb: torch.Tensor,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> torch.Tensor:
        """Supervise rendered uncertainty on all decoder supervision views."""
        stage_gaussians = reconstruction.intermediate_gaussians[stage_index]
        predicted_uncertainty = self.gs.render_features(
            stage_gaussians.detach(),
            reconstruction.gaussian_uncertainties[stage_index],
            decoder_input.cam_view.detach(),
            decoder_input.intrinsics.detach(),
            background=stage_gaussians.new_zeros(1),
        )
        gt_rgb = (
            supervision.images_output * supervision.masks_output
            + torch.ones_like(supervision.images_output)
            * (1 - supervision.masks_output)
        )
        with torch.no_grad():
            target_uncertainty = dssim_error_map(
                stage_rgb.detach(), gt_rgb.detach()
            )
        target_error = (predicted_uncertainty - target_uncertainty).abs()
        error_sum = (
            reconstruction.context_uncertainty_error_sums[stage_index]
            + target_error.sum()
        )
        element_count = (
            reconstruction.context_uncertainty_element_counts[stage_index]
            + target_error.numel()
        )
        return error_sum / max(1, element_count)

    def _position_bias_regularization(
        self,
        reconstruction: ATSplatStage1Reconstruction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final Gaussian biases and their masked magnitude statistics."""
        position_bias = reconstruction.final_position_bias
        if position_bias.shape[:2] != reconstruction.gaussian_valid_mask.shape:
            raise ValueError(
                "position bias and Gaussian valid mask shapes do not match: "
                f"bias={tuple(position_bias.shape)}, "
                f"mask={tuple(reconstruction.gaussian_valid_mask.shape)}"
            )

        valid = reconstruction.gaussian_valid_mask
        valid_float = valid.to(torch.float32)
        squared_norm = position_bias.float().square().sum(dim=-1)
        valid_count = valid_float.sum().clamp_min(1.0)
        loss_position_bias = (squared_norm * valid_float).sum() / valid_count
        bias_norm = torch.sqrt(squared_norm)
        mean_position_bias_norm = (bias_norm * valid_float).sum() / valid_count
        max_position_bias_norm = bias_norm.masked_fill(~valid, 0.0).amax()
        weighted_position_bias = self.opt.lambda_position_bias * loss_position_bias
        return (
            position_bias,
            loss_position_bias,
            weighted_position_bias,
            mean_position_bias_norm,
            max_position_bias_norm,
        )

    def compute_loss(
        self,
        reconstruction: Reconstruction,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> dict:
        """Combine final TokenGS loss with per-stage intermediate rendering losses."""
        if not isinstance(reconstruction, ATSplatStage1Reconstruction):
            raise TypeError("ATSplatStage1 requires ATSplatStage1Reconstruction for ATE training")
        pgsr_view_pairs: PGSRViewPairs | None = None
        training_step = 0 if self.training_step is None else self.training_step
        if (
            self.opt.pgsr_enabled
            and training_step >= int(self.opt.pgsr_weight_from_step)
        ):
            pgsr_view_pairs = select_pgsr_view_pairs(
                decoder_input.cam_view,
                training=self.training,
                max_neighbors=int(self.opt.pgsr_multi_view_num),
                max_angle_degrees=float(self.opt.pgsr_multi_view_max_angle),
                min_distance=float(self.opt.pgsr_multi_view_min_distance),
                max_distance=float(self.opt.pgsr_multi_view_max_distance),
            )
        results = super().compute_loss(
            reconstruction,
            decoder_input,
            supervision,
            pgsr_view_pairs=pgsr_view_pairs,
        )
        results["loss_final"] = results["loss"].detach()
        stage_count = len(reconstruction.intermediate_gaussians)
        if not (
            stage_count
            == len(reconstruction.token_split_scores)
            == len(reconstruction.gaussian_uncertainties)
            == len(reconstruction.context_uncertainty_error_sums)
            == len(reconstruction.context_uncertainty_element_counts)
            == len(reconstruction.selected_parent_indices)
            == len(reconstruction.intermediate_gaussian_valid_masks)
            == len(reconstruction.intermediate_token_positions)
            == len(reconstruction.intermediate_token_radii)
            == len(reconstruction.intermediate_token_valid_masks)
        ):
            raise ValueError("ATE stage outputs must have equal lengths")
