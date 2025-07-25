from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from entrypoint import InferencePipeline

app = FastAPI()

# Load models once
pipeline = InferencePipeline(
    stage1_model= "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level1_iaff.pth",
    stage2_model= "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level2_iaff.pth",
    yolo_model_path= "/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/models/yolov8_segmentation_run8_weights_best.pt",
    device="cuda:0"
)

@app.get("/")
def root():
    return {"message": "Valve Anatomy Inference API"}

@app.post("/predict/")
async def predict(video: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    temp_input_path = f"temp_upload_{uid}.mp4"
    
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Run inference
        result = pipeline.run(temp_input_path)
        return JSONResponse(content={"classification": result})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
