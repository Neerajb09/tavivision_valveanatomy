import os
import cv2
import numpy as np
from ultralytics import YOLO

class VideoSegmenter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path)

    def process_video(self, input_video_path, output_video_path):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        video_filename = os.path.basename(input_video_path)
        video_name, video_ext = os.path.splitext(video_filename)

        if os.path.exists(output_video_path):
            print(f"[SKIP] '{output_video_path}' already exists. Skipping...")
            return output_video_path

        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height//2))

        print(f"[INFO] Processing '{video_filename}' ({frame_count} frames)...")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, task="segment", save=False, verbose=False)[0]

            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                masked_frame = np.zeros_like(frame, dtype=np.uint8)

                for mask in masks:
                    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)

                    for c in range(3):
                        masked_frame[:, :, c] = np.where(binary_mask == 1, frame[:, :, c], masked_frame[:, :, c])

                output_frame = masked_frame
            else:
                output_frame = np.zeros_like(frame)

            height, width, channels = output_frame.shape
            # Calculate the new width (half)
            new_height = height // 2
            # Crop horizontally: keep only the left half
            cropped_frame = output_frame[:new_height, :, :]
            out.write(cropped_frame)
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"   → Processed {frame_idx}/{frame_count} frames")

        cap.release()
        out.release()
        print(f"[✅] Saved segmented video to: {output_video_path}")
        
        # return output_video_path

# Example usage:
# model_path = '/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/Models/yolov8_segmentation_run8_weights_best.pt'
# segmenter = VideoSegmenter(model_path)
# input_video_path = '/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/inference_pipeline/temp_output_removed_crosshair.mp4'
# output_video_path = '/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/inference_pipeline/temp_output_Segmented.mp4'
# segmenter.process_video(input_video_path, output_video_path)
