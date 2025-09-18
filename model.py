import tensorflow as tf
from tensorflow.keras import layers, models, losses

def build_pulse_motion_unet_with_latent(T, num_features=2):
    encoder = build_encoder(T, num_features)

    inputs = encoder.input
    skip1, skip2, bottleneck = encoder(inputs)

    # Attention masks
    latent_vec = layers.GlobalAveragePooling1D()(bottleneck)
    pulse_mask = layers.Dense(128, activation="sigmoid", name="pulse_mask")(latent_vec)
    motion_mask = layers.Lambda(lambda z: 1.0 - z, name="motion_mask")(pulse_mask)

    # Reshape masks to match bottleneck shape
    pulse_mask_reshaped = layers.Reshape((1, 128))(pulse_mask)
    motion_mask_reshaped = layers.Reshape((1, 128))(motion_mask)

    pulse_bottleneck = layers.Multiply(name="pulse_latent")([bottleneck, pulse_mask_reshaped])
    motion_bottleneck = layers.Multiply(name="motion_latent")([bottleneck, motion_mask_reshaped])

    # Decoders
    pulse_decoder = build_decoder("pulse_pred", T, num_features)
    motion_decoder = build_decoder("motion_pred", T, num_features)

    pulse_out = pulse_decoder([skip1, skip2, pulse_bottleneck])
    motion_out = motion_decoder([skip1, skip2, motion_bottleneck])

    return models.Model(inputs, [pulse_out, motion_out, pulse_bottleneck, motion_bottleneck], 
                        name="PulseMotionUNet")

