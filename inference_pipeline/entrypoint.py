from extract_sov_frame import SovFrameDetector
from crosshair_removal import CrossHairProcessor
from sov_roi_segmentation import VideoSegmenter
# from segment_ROI import VideoSegmenter
from valve_classification import HierarchicalDualStreamInference
import os
import cv2


# extract SOV only frames 
class InferencePipeline:
    def __init__(self, stage1_model, stage2_model,yolo_model_path, device='cuda:6'):
        
        self.device = device

        # Initialize the inference model
        self.inference_model = HierarchicalDualStreamInference(stage1_path=stage1_model,stage2_path=stage2_model, device = device)

        # Initialize the SOV frame detector
        self.sov_detector = SovFrameDetector()

        # Initialize the crosshair remover
        self.crosshair_processor = CrossHairProcessor()

        # Initialize the video segmenter
        self.segmenter = VideoSegmenter(yolo_model_path)

    def run(self, input_video_path):
        # # Step 1: Extract SOV frames
        sov_output_path = 'temp_output_SOV.mp4'
        self.sov_detector.process_video(input_video_path, sov_output_path)

        # # Step 2: Remove crosshair lines
        crosshair_output_path = 'temp_output_removed_crosshair.mp4'
        self.crosshair_processor.process_video(sov_output_path, crosshair_output_path)

        # # Step 3: Segment the ROI
        segment_output_path = 'temp_output_Segmented.mp4'
        self.segmenter.process_video(crosshair_output_path, segment_output_path)

        # # Step 4: Classify valve anatomy
        classification_result = self.inference_model.predict(segment_output_path)
        os.remove(sov_output_path)
        os.remove(crosshair_output_path)
        os.remove(segment_output_path)
        
        return classification_result
    
# remove crosshair
# segment the ROI
# valve anatomy classification
stage1_path = "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level1_iaff.pth"
stage2_path = "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level2_iaff.pth"
yolo_model_path= '/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/models/yolov8_segmentation_run8_weights_best.pt'
InferencePipeline_instance = InferencePipeline(stage1_model=stage1_path,stage2_model=stage2_path,yolo_model_path=yolo_model_path, device='cuda:6') 
InferencePipeline_instance.run("/mnt/nvme_disk2/User_data/nb57077k/WhatsApp Video 2025-06-30 at 2.01.39 PM.mp4")
