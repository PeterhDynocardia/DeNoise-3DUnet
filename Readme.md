📌 Overview

This repository implements Denoise-3D-UNET, a deep learning model for video denoising and heart-rate extraction.

Input: noisy video represented as sparse dot trajectories (T, N, 2)

Outputs:

pulse_pred: denoised physiological signal (pulse-related dynamics)

motion_pred: noise/artifact signal (motion, illumination, jitter)

The model is based on a temporal U-Net encoder–decoder with dual attention masks:

pulse_mask highlights regions carrying heart-rate information.

inverse (motion) mask captures noise patterns.

Training is supervised with paired data [noisy_video, pulse_video, motion_video] and includes an orthogonality regularizer to disentangle signal and noise in the latent space.

🏗 Model Architecture

Encoder: Temporal CNN (1D convolutions along time) with downsampling.

Latent attention masks:

pulse_latent = bottleneck × pulse_mask

motion_latent = bottleneck × (1 − pulse_mask)

Dual Decoders: U-Net style upsampling with skip connections to reconstruct:

pulse_pred (denoised pulse trajectory)

motion_pred (artifact trajectory)

Regularizer: Orthogonality loss enforces pulse vs motion separation.

⚙️ Installation
git clone https://github.com/yourusername/Denoise-3D-UNET.git
cd Denoise-3D-UNET
pip install -r requirements.txt


Dependencies

Python 3.9+

TensorFlow 2.11+

NumPy, OpenCV, Matplotlib

📂 Data Preparation

Training requires three aligned folders containing CSV files:
```
data/
 ├── noise/
 │    ├── sample1_noise.csv
 │    ├── sample2_noise.csv
 ├── pulse/
 │    ├── sample1_pulse.csv
 │    ├── sample2_pulse.csv
 ├── mixed/
 │    ├── sample1_mixed.csv
 │    ├── sample2_mixed.csv
```

Each *_noise.csv, *_pulse.csv, *_mixed.csv must share the same prefix (sample1, sample2, …).

Each CSV stores trajectories (T, N, 2) for all dots over time.

📊 Visualization

We provide utilities to save predicted videos as .avi for qualitative comparison.
```
from utils import save_comparison_video

# After prediction
save_comparison_video(pulse_true, pulse_pred, "results/pulse_compare.avi", title="Pulse")
save_comparison_video(motion_true, motion_pred, "results/motion_compare.avi", title="Motion")
```

Left: ground truth trajectory

Right: predicted trajectory

Green = ground truth, Red = prediction
