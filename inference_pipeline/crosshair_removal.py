import os
import cv2
import numpy as np
from ultralytics import YOLO

class CrossHairProcessor:
    def __init__(self):
        pass

    def remove_pink_blue_lines(self, frame):
        try:
            upper_half = frame[:self.height // 2, :]
            hsv_upper_half = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)

            lower_pink = np.array([150, 30, 50])
            upper_pink = np.array([179, 255, 255])
            pink_mask = cv2.inRange(hsv_upper_half, lower_pink, upper_pink)

            lower_blue = np.array([40, 50, 50])
            upper_blue = np.array([80, 255, 255])
            blue_mask = cv2.inRange(hsv_upper_half, lower_blue, upper_blue)

            combined_mask_upper = cv2.bitwise_or(pink_mask, blue_mask)
            kernel = np.ones((3, 3), np.uint8)
            expanded_mask_upper = cv2.dilate(combined_mask_upper, kernel, iterations=4)

            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            mask[:self.height // 2, :] = expanded_mask_upper

            return cv2.inpaint(frame, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        except Exception as e:
            print(f"Error removing pink/blue lines: {e}")
            return frame

    def process_video(self, input_video_path, output_video_path):
        # try:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.width, self.height))

        print(f"Processing video: {input_video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.remove_pink_blue_lines(frame)

            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Video processing complete. Saved to {output_video_path}")

        # except Exception as e:
        #     print(f"❌ Error processing video {input_video_path}: {e}")

# Example usage:
# processor = CrossHairProcessor()
# input_video_path = "temp_output_SOV.mp4"
# output_video_path = "temp_output_removed_crosshair.mp4"
# processor.process_video(input_video_path, output_video_path)
