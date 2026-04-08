import os
import urllib.request

URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
OUT_DIR = "models"
OUT_NAME = "pose_landmarker_full.task"

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_NAME)
urllib.request.urlretrieve(URL, out_path)
print("saved:", os.path.abspath(out_path))
