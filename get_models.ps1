# Windows PowerShell: download ONNX models into ./models
param()
$ErrorActionPreference = "Stop"
if (!(Test-Path -Path "models")) { New-Item -ItemType Directory -Path "models" | Out-Null }
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "models\face_recognition_sface_2021dec.onnx"
Write-Host "Models downloaded to .\models"
