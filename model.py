import tensorflow as tf
from tensorflow.keras import layers, models, losses

def build_denoising_model(T, num_features=2, latent_dim=64):
    """
    Model: noisy_video → pulse_pred, motion_pred
    Args:
        T: time steps
        num_features: input features per dot (x,y)
        latent_dim: embedding dimension
    """
    inputs = layers.Input(shape=(T, num_features))

    # Shared encoder (temporal CNN)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    latent = layers.Dense(latent_dim, activation="relu")(x)  # (latent_dim,)

    # Attention masks (pulse vs motion)
    pulse_mask = layers.Dense(latent_dim, activation="sigmoid", name="pulse_mask")(latent)
    motion_mask = layers.Lambda(lambda z: 1.0 - z, name="motion_mask")(pulse_mask)

    # Apply masks
    pulse_feat = layers.Multiply()([latent, pulse_mask])
    motion_feat = layers.Multiply()([latent, motion_mask])

    # Decoders
    pulse_pred = layers.Dense(T * num_features, name="pulse_pred")(pulse_feat)
    pulse_pred = layers.Reshape((T, num_features))(pulse_pred)

    motion_pred = layers.Dense(T * num_features, name="motion_pred")(motion_feat)
    motion_pred = layers.Reshape((T, num_features))(motion_pred)

    return models.Model(inputs, [pulse_pred, motion_pred], name="PulseMotionNet")

# Example
model = build_denoising_model(T=100)
model.summary()
