# face-lock-api

FastAPI service for **real-time face recognition** using OpenCV YuNet (detection) and SFace (recognition), backed by **Firebase Realtime Database**.

## Folder layout
```
face-lock-api/
├─ main.py
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ models/
   ├─ face_detection_yunet_2023mar.onnx      (add this)
   └─ face_recognition_sface_2021dec.onnx    (add this)
```

> The two `.onnx` model files are **not included** in this zip. Download them with the scripts below before committing or deploying.

## Download models (Windows PowerShell)
```powershell
mkdir models -ea ignore
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

## Download models (Linux/macOS)
```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

---

## Deploy on Koyeb (free)

1. Push this repo to GitHub.
2. In Koyeb: **Create Web Service → GitHub** → pick your repo (Dockerfile auto-detected).
3. **Secrets**: create `FIREBASE_CREDENTIALS` with your full service-account JSON.
4. **Env vars**:
   - `FIREBASE_DB_URL = https://face-1338e-default-rtdb.firebaseio.com/`
   - Add from secret → `FIREBASE_CREDENTIALS`
   - (optional) `SIM_MEASURE=cosine`, `COS_THRESHOLD=0.36`
5. Deploy → test:
```bash
curl https://<your-app>.koyeb.app/
curl https://<your-app>.koyeb.app/engine
curl https://<your-app>.koyeb.app/users
```

---

## API quick reference

- **POST /register**  (multipart) → fields: `username`, `image=@me.jpg`
- **GET /users**
- **GET /users/{userId}`**
- **PATCH /users/{userId}**  body: `{"username":"New Name"}`
- **DELETE /users/{userId}`**
- **POST /authenticate**  (multipart) → field: `image=@capture.jpg`

ESP32-CAM target:
```cpp
const char* API_URL = "https://<your-app>.koyeb.app/authenticate";
```

---

## Local dev (optional)

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Remember: download the models into ./models first
uvicorn main:app --reload
```
