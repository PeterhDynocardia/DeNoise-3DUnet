import cv2
import numpy as np
import os

def save_dot_video(dot_trajs, filename, H=128, W=128, fps=30):
    """
    Save dot trajectories as a video (scatter plot of dots).
    
    Args:
        dot_trajs: np.array, shape (T, N, 2) = time, dots, (x,y)
        filename: str, output AVI file path
        H, W: frame size
        fps: frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (W, H))

    T, N, _ = dot_trajs.shape
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        for n in range(N):
            x, y = dot_trajs[t, n]
            if 0 <= int(x) < W and 0 <= int(y) < H:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        writer.write(frame)

    writer.release()
    print(f"[Saved] {filename}")
  
def save_comparison_video(true_trajs, pred_trajs, filename, H=128, W=128, fps=30, title="Pulse"):
    """
    Save side-by-side video comparing true vs predicted dot trajectories.
    
    Args:
        true_trajs: np.array, (T, N, 2)
        pred_trajs: np.array, (T, N, 2)
        filename: str, output AVI file path
        H, W: frame size
        fps: frames per second
        title: str, label overlay (e.g., "Pulse" or "Motion")
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (2*W, H))  # doubled width

    T, N, _ = true_trajs.shape
    for t in range(T):
        frame_true = np.zeros((H, W, 3), dtype=np.uint8)
        frame_pred = np.zeros((H, W, 3), dtype=np.uint8)

        # Draw true
        for n in range(N):
            x, y = true_trajs[t, n]
            if 0 <= int(x) < W and 0 <= int(y) < H:
                cv2.circle(frame_true, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Draw pred
        for n in range(N):
            x, y = pred_trajs[t, n]
            if 0 <= int(x) < W and 0 <= int(y) < H:
                cv2.circle(frame_pred, (int(x), int(y)), 2, (0, 0, 255), -1)

        # Concatenate
        frame = np.concatenate([frame_true, frame_pred], axis=1)

        # Add labels
        cv2.putText(frame, f"{title} TRUE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{title} PRED", (W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"[Saved] {filename}")
