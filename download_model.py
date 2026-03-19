import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
out = "pose_landmarker_full.task"
urllib.request.urlretrieve(url, out)
print("saved:", out)
