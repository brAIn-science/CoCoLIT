DEFAULT_VAE_ARGS = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "num_channels": [64, 128, 256],
    "num_res_blocks": [2, 2, 2],
    "norm_num_groups": 32,
    "norm_eps": 0.000001,
    "attention_levels": [False, False, False],
    "with_encoder_nonlocal_attn": False,
    "with_decoder_nonlocal_attn": False,
    "use_checkpointing": False,
    "use_convtranspose": False,
    "norm_float16": True,
    "num_splits": 2,
    "dim_split": 1
}

DEFAULT_DIFFUSION_ARGS = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "num_res_blocks": 2,
    "num_channels": [256, 256, 256],
    "attention_levels": [False, True, True],
    "norm_num_groups": 32,
    "norm_eps": 1.0e-6,
    "resblock_updown": True,
    "num_head_channels": [0, 256, 256],
    "transformer_num_layers": 1,
    "num_class_embeds": None,
    "upcast_attention": True,
    "use_flash_attention": False    
} 

DEFAULT_CONTROLNET_ARGS = {
    "spatial_dims": 3,
    "in_channels": 4,
    "num_res_blocks": 2,
    "num_channels": [256, 256, 256],
    "attention_levels": [False, True, True],
    "norm_num_groups": 32,
    "norm_eps": 1.0e-6,
    "resblock_updown": True,
    "num_head_channels": [0, 256, 256],
    "transformer_num_layers": 1,
    "num_class_embeds": None,
    "upcast_attention": True,
    "use_flash_attention": False,
    "conditioning_embedding_in_channels": 4,
    "conditioning_embedding_num_channels": [256]   
}

DEFAULT_DIFFSCHED_ARGS = {
    "num_train_timesteps": 1000,
    "schedule": "scaled_linear_beta",
    "beta_start": 0.0015,
    "beta_end": 0.0205
}

DEFAULT_ZSCORES_PARAMS = {
    'suvr_mean': 0.370730756043226,
    'suvr_std':  0.667580822498126,
    'smri_mean': -49.757356053158766,
    'smri_std': 24.68868751354585,
}