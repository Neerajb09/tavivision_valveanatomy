import os
import uuid
import cv2
from extract_sov_frame import SovFrameDetector
from crosshair_removal import CrossHairProcessor
from sov_roi_segmentation import VideoSegmenter
from valve_classification import HierarchicalDualStreamInference

class InferencePipeline:
    def __init__(self, stage1_model, stage2_model, yolo_model_path, device='cuda:5'):
        self.device = device
        self.inference_model = HierarchicalDualStreamInference(stage1_path=stage1_model, stage2_path=stage2_model, device=device)
        self.sov_detector = SovFrameDetector()
        self.crosshair_processor = CrossHairProcessor()
        self.segmenter = VideoSegmenter(yolo_model_path)

    def run(self, input_video_path):
        uid = str(uuid.uuid4())[:8]
        sov_path = f'temp_output_SOV_{uid}.mp4'
        no_crosshair_path = f'temp_output_no_crosshair_{uid}.mp4'
        segmented_path = f'temp_output_segmented_{uid}.mp4'

        try:
            # Step 1: Extract SOV frames
            self.sov_detector.process_video(input_video_path, sov_path)

            # Step 2: Remove crosshair
            self.crosshair_processor.process_video(sov_path, no_crosshair_path)

            # Step 3: Segment SOV region
            self.segmenter.process_video(no_crosshair_path, segmented_path)

            # Step 4: Run classification
            result = self.inference_model.predict(segmented_path)
            return result

        finally:
            # Clean up temporary files
            for f in [sov_path, no_crosshair_path, segmented_path]:
                if os.path.exists(f):
                    os.remove(f)
                

# # ==== Example Usage ====
# if __name__ == "__main__":
#     stage1_path = "/nuvodata/User_data/neeraj/tavivision/valve_anatomy/weights/dual_stream_level1_iaff_random_opticaltvl1_final.pth"
#     stage2_path = "/nuvodata/User_data/neeraj/tavivision/valve_anatomy/weights/dual_stream_level2_iaff_opticaltvl1_final_full.pth"
#     yolo_model_path = "/weka/kanpur/data_cardiovision/valve_anatomy_classification/models/yolov8_segmentation_run8_weights_best.pt"
#     input_video = "/nuvodata/User_data/neeraj/WhatsApp Video 2025-06-30 at 2.01.39 PM.mp4"

#     pipeline = InferencePipeline(stage1_path, stage2_path, yolo_model_path, device='cuda:6')
#     classification = pipeline.run(input_video)
#     print(f"Predicted Valve Class: {classification}")
