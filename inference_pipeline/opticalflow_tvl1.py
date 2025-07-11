import cv2
import numpy as np
import os

class OpticalFlowTVL1Processor:
    def __init__(self):
        # CPU version, works with opencv-contrib-python
        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    @staticmethod
    def flow_to_color(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = angle / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_video_path}")

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Cannot read the first frame from video.")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        height, width = prev_frame.shape[:2]
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow on CPU directly (no GpuMat, no upload)
            flow = self.tvl1.calc(prev_gray, curr_gray, None)

            flow_vis = self.flow_to_color(flow)
            out.write(flow_vis)

            prev_gray = curr_gray

        cap.release()
        out.release()
        print(f"Optical flow video saved at: {output_video_path}")

# if __name__ == "__main__":
#     input_path = "/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/inference_pipeline/temp_output_Segmented.mp4"         # Replace with your actual input path
#     output_path = "/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/inference_pipeline/temp.mp4" # Replace with your desired output path

#     processor = OpticalFlowTVL1Processor()
#     processor.process_video(input_path, output_path)
