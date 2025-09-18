def orthogonality_loss(pulse_latent, motion_latent):
    # Flatten temporal dimension
    pulse_flat = tf.reduce_mean(pulse_latent, axis=1)   # (batch, 128)
    motion_flat = tf.reduce_mean(motion_latent, axis=1) # (batch, 128)
    
    dot = tf.reduce_sum(pulse_flat * motion_flat, axis=-1)
    return tf.reduce_mean(tf.square(dot))  # want dot → 0
