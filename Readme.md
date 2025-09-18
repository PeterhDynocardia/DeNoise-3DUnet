### High Level Summary:
Encoder extracts features with multi-scale context.

Dual decoders reconstruct pulse and motion using skip connections for fine detail.

Attention mask ensures disentanglement between signal and noise.

### Model: 
```Training data = triplets [noisy_video, pulse_video, motion_video].

Input = noisy_video (dot trajectories, shape (N, T, 2) after preprocessing).

Outputs =

pulse_pred (denoised, pulse-related sequence)

motion_pred (artifact/noise sequence)

Training objective =

Reconstruction losses: RMSE(pulse_pred, pulse_video) + RMSE(motion_pred, motion_video).

Separation loss: cross-attention / contrastive loss to enforce orthogonality (pulse vs noise disentangled).
```
