# tvl1_gpu.py — GPU-only TV-L1 with imageio-ffmpeg I/O (no CPU fallback)
import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio  # v2 API (uses imageio-ffmpeg)
import imageio_ffmpeg

REQUIRE_GPU = True  # stop if CUDA TV-L1 isn't available

def check_cuda_support():
    print("🔍 Checking CUDA availability in OpenCV...")
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"💻 CUDA-enabled GPU(s) detected: {count}")
        if count == 0:
            raise RuntimeError("No CUDA device found")
    except Exception as e:
        raise RuntimeError(f"Could not query CUDA device count: {e}")

    if not hasattr(cv2.cuda, "OpticalFlowDual_TVL1"):
        raise RuntimeError("cv2.cuda.OpticalFlowDual_TVL1 not in this OpenCV build")
    print("✅ cv2.cuda.OpticalFlowDual_TVL1 is available.")
    try:
        # Log ffmpeg binary that imageio-ffmpeg will use
        print(f"🎬 imageio-ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception:
        pass

def iter_frames_bgr(video_path):
    """
    Generator of BGR uint8 frames using imageio v2's ffmpeg reader.
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        for frame_rgb in reader:       # RGB uint8
            yield frame_rgb[:, :, ::-1]  # → BGR
    finally:
        reader.close()

class VideoWriterBGR:
    """
    Stream BGR frames to MP4 (H.264) using imageio v2's ffmpeg writer.
    """
    def __init__(self, out_path, fps, size_hw):
        h, w = size_hw
        # Do NOT pass pix_fmt (v2 writer will choke on it). Defaults are fine.
        self._writer = imageio.get_writer(
            out_path, format="ffmpeg", mode="I", fps=fps, codec="libx264"
        )
        self._open = True

    def write(self, frame_bgr):
        if self._open:
            self._writer.append_data(frame_bgr[:, :, ::-1])  # BGR → RGB

    def close(self):
        if self._open:
            self._writer.close()
            self._open = False

def tvl1_gpu_create():
    try:
        return cv2.cuda.OpticalFlowDual_TVL1.create()
    except Exception as e:
        raise RuntimeError(f"CUDA TV-L1 creation failed: {e}")

def tvl1_gpu_calc(tvl1, prev_gray, gray, scratch=None):
    """
    Run TV-L1 on GPU. Reuse GpuMats in 'scratch' to reduce allocations.
    """
    try:
        if scratch is None:
            scratch = {}
        gpu_prev = scratch.get("prev") or cv2.cuda_GpuMat()
        gpu_gray = scratch.get("gray") or cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        gpu_gray.upload(gray)
        flow_gpu = tvl1.calc(gpu_prev, gpu_gray, None)
        flow = flow_gpu.download()
        scratch["prev"], scratch["gray"] = gpu_prev, gpu_gray
        return flow
    except Exception as e:
        raise RuntimeError(f"CUDA TV-L1 failed: {e}")

def flow_to_color(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)     # [0,180]
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compute_flow_from_videos(input_dir, output_dir, fps=10):
    print(f"📂 Input dir: {input_dir}")
    print(f"💾 Output dir: {output_dir}")

    if REQUIRE_GPU:
        check_cuda_support()

    tvl1 = tvl1_gpu_create()
    scratch = {}

    splits = ['train', 'train_augmented', 'val', 'test']
    for split in splits:
        split_input = os.path.join(input_dir, split)
        split_output = os.path.join(output_dir, split)
        if not os.path.exists(split_input):
            print(f"⚠️ Skipping missing split: {split_input}")
            continue

        for cls in sorted(os.listdir(split_input)):
            class_input = os.path.join(split_input, cls)
            if not os.path.isdir(class_input):
                continue
            class_output = os.path.join(split_output, cls)
            os.makedirs(class_output, exist_ok=True)

            videos = [f for f in os.listdir(class_input) if f.lower().endswith(".mp4")]
            for vid in tqdm(sorted(videos), desc=f"{split}/{cls}", unit="video"):
                in_path  = os.path.join(class_input, vid)
                out_path = os.path.join(class_output, vid[:-4] + "_flow_vis.mp4")

                frames = iter_frames_bgr(in_path)
                try:
                    first = next(frames, None)
                except StopIteration:
                    first = None
                if first is None:
                    print(f"❌ Cannot read {vid}")
                    continue

                h, w = first.shape[:2]
                writer = VideoWriterBGR(out_path, fps=fps, size_hw=(h, w))
                prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

                for frame in frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = tvl1_gpu_calc(tvl1, prev_gray, gray, scratch=scratch)  # GPU only
                    writer.write(flow_to_color(flow))
                    prev_gray = gray

                writer.close()

    print("✅ All videos processed.")

if __name__ == "__main__":
    input_dir = "/weka/kanpur/data_cardiovision/valve_anatomy_classification/segmented_video_combined_32"
    output_dir = "/weka/kanpur/data_cardiovision/valve_anatomy_classification/TVL1_combined_32"
    compute_flow_from_videos(input_dir, output_dir, fps=10)
