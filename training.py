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

# Save comparison videos for sanity check
i = 0
pulse_true = pulse_videos[i]     # (T, N, 2)
motion_true = motion_videos[i]

pulse_pred, motion_pred, _, _ = base_model.predict(noisy_videos[i][None, ...])
pulse_pred = pulse_pred[0]
motion_pred = motion_pred[0]

os.makedirs("results", exist_ok=True)

# Save side-by-side videos
save_comparison_video(pulse_true, pulse_pred, "results/pulse_compare.avi", title="Pulse")
save_comparison_video(motion_true, motion_pred, "results/motion_compare.avi", title="Motion")
