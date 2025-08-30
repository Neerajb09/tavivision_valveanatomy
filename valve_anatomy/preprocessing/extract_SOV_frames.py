import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

class SovFrameDetector:
    def __init__(self ):
        self.yolo_model_path = "/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/models/trained_yolo.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing YOLO model on {self.device}")
        self.model = self.load_yolo_model(self.yolo_model_path)
        print("YOLO model loaded successfully.")

    def pink_line(self, frame):
        frame = frame[500:-100, :]
        lower_pink = np.array([125, 30, 50])
        upper_pink = np.array([180, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        pink_pixel_count = np.sum(mask, axis=1)
        max_row = np.argmax(pink_pixel_count)
        return max_row

    def load_yolo_model(self, model_path):
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        return YOLO(model_path).to(self.device)

    def run_yolo_inference(self, image):
        lower_half_img = image[image.shape[0] // 2:, :, :]
        temp_path = "temp_yolo_input.png"
        cv2.imwrite(temp_path, lower_half_img)

        with torch.no_grad():
            results = self.model(temp_path, save=False, verbose=False, classes=[1])

        bboxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        sov_bbox, highest_conf = None, -1
        for bbox, cls_id, conf in zip(bboxes, class_ids, confidences):
            if cls_id == 1 and conf > highest_conf:
                sov_bbox, highest_conf = bbox, conf

        return sov_bbox

    def process_video(self, input_video_path, output_video_path):
        if not os.path.exists(input_video_path):
            print(f"Input video does not exist: {input_video_path}")
            return

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Unable to open input video: {input_video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frames = []
        seen_y_values = set()
        sov_bbox = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}...")

            if sov_bbox is None:
                sov_bbox = self.run_yolo_inference(frame)
                if sov_bbox is not None:
                    sov_bbox = [int(coord) for coord in sov_bbox]

            if sov_bbox:
                x1, y1, x2, y2 = sov_bbox
                y1 += frame_height // 2
                y2 += frame_height // 2

                y_pink = self.pink_line(frame) + 500
                if y1 <= y_pink <= y2:
                    if y_pink not in seen_y_values:
                        seen_y_values.add(y_pink)
                        frames.append((y_pink, frame.copy()))

        print(f"Unique pink-line frames selected: {len(frames)}")

        for _, sorted_frame in sorted(frames, key=lambda x: x[0]):
            out.write(sorted_frame)

        cap.release()
        out.release()
        print(f"Finished writing output video to: {output_video_path}")


# input_video = "/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/final_dataset/Tricuspid/Tricuspid - 3mensio Screen Recording (1).mp4"
# output_video = "/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/inference_pipeline/temp_output_SOV.mp4"

# detector = SovFrameDetector()
# detector.process_video(input_video, output_video)
