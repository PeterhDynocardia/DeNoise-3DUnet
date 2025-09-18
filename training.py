# Build base model
base_model = build_pulse_motion_unet_with_latent(T=128)

# Wrap in trainer
trainer = PulseMotionTrainer(base_model, lambda_ortho=0.1)
trainer.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss_fn=lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8)
)

# Train
history = trainer.fit(
    noisy_videos,
    (pulse_videos, motion_videos),
    epochs=50,
    batch_size=32,
    validation_split=0.1
)
