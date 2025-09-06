from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os, io, json, time, base64, uuid, pathlib
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

# ========================= Firebase Admin =========================
import firebase_admin
from firebase_admin import credentials, db

APP_NAME = "Face Recognition API (YuNet + SFace + Firebase)"

# ---- Tunables via env ----
SIM_MEASURE = os.getenv("SIM_MEASURE", "cosine").lower()  # "cosine" or "l2"
COS_THRESHOLD = float(os.getenv("COS_THRESHOLD", "0.36")) # SFace good start
L2_THRESHOLD  = float(os.getenv("L2_THRESHOLD",  "1.10"))
DET_INPUT_SIZE = int(os.getenv("DET_INPUT_SIZE", "320"))  # YuNet input

# Quality gates
MIN_FACE_AREA_RATIO_REG  = 0.04   # >= 4% of original frame
MIN_FACE_AREA_RATIO_AUTH = 0.02   # >= 2% for ESP32-CAM
MIN_LAPLACIAN_VAR        = 70.0   # blur threshold

# Image storage
MAX_JPEG_SIDE = 640
JPEG_QUALITY  = 85

# ========================= Models (local only) =========================
MODELS_DIR = pathlib.Path("./models")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

def ensure_models():
    if not YUNET_PATH.exists() or not SFACE_PATH.exists():
        raise RuntimeError("ONNX models missing. Put them under ./models/")

def create_det_rec():
    det = cv2.FaceDetectorYN_create(
        YUNET_PATH.as_posix(), "", (DET_INPUT_SIZE, DET_INPUT_SIZE),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    rec = cv2.FaceRecognizerSF_create(SFACE_PATH.as_posix(), "")
    return det, rec

# ========================= Firebase helpers =========================
def init_firebase():
    if firebase_admin._apps:
        return
    creds_json = os.getenv("FIREBASE_CREDENTIALS")
    db_url = os.getenv("FIREBASE_DB_URL")
    if not creds_json or not db_url:
        raise RuntimeError("Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL environment variables.")
    creds = credentials.Certificate(json.loads(creds_json))
    firebase_admin.initialize_app(creds, {"databaseURL": db_url})

def users_ref():
    return db.reference("/users")

# ========================= CV helpers =========================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def read_upload_as_bgr(upload: UploadFile) -> np.ndarray:
    """Decode any common image. Prefer PIL, then fallback to OpenCV.
       Raises HTTP 415 for unsupported/invalid files instead of 500.
    """
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) Try Pillow (handles jpg/png/webp if codecs are available)
    try:
        img = Image.open(io.BytesIO(data))
        return pil_to_bgr(img)
    except UnidentifiedImageError:
        pass
    except Exception:
        # if PIL fails for other reasons, still try OpenCV
        pass

    # 2) Fallback to OpenCV decode
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=415, detail="Unsupported or corrupted image format")
    return bgr

def auto_enhance(bgr: np.ndarray) -> np.ndarray:
    img = cv2.convertScaleAbs(bgr, alpha=1.05, beta=8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def lap_var(bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def bgr_to_b64jpeg(bgr: np.ndarray, max_side=MAX_JPEG_SIDE, quality=JPEG_QUALITY) -> str:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side/max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("utf-8")

def face_area_ratio_from_box(box: np.ndarray, shape) -> float:
    # YuNet box = [x, y, w, h, landmarks..., score]
    h_img, w_img = shape[:2]
    w, h = float(box[2]), float(box[3])
    return max(0.0, (w*h) / float(w_img*h_img))

def get_face_embedding(det, rec, bgr: np.ndarray):
    det.setInputSize((bgr.shape[1], bgr.shape[0]))
    ok, faces = det.detect(bgr)
    if not ok or faces is None or len(faces) == 0:
        return None, None
    faces = faces[np.argsort(-(faces[:,2]*faces[:,3]))]  # largest area
    box = faces[0]
    aligned = rec.alignCrop(bgr, box)
    feat = rec.feature(aligned)  # 128-D vector
    return feat.astype("float32"), box

def compare(a: np.ndarray, b: np.ndarray) -> float:
    if SIM_MEASURE == "cosine":
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a, b))
    return float(np.linalg.norm(a - b))  # L2

def is_match(score: float) -> bool:
    if SIM_MEASURE == "cosine":
        return score >= COS_THRESHOLD
    return score <= L2_THRESHOLD

def _embedding_from_rec(rec: dict) -> np.ndarray:
    """Make sure embedding always becomes a float32 np.array."""
    v = rec.get("embedding", [])
    if isinstance(v, dict):
        # Firebase can return dict like {"0":0.1,"1":0.2,...}; sort numeric keys
        v = [v[k] for k in sorted(v.keys(), key=lambda x: int(x))]
    return np.array(v, dtype="float32")

# ========================= FastAPI app =========================
app = FastAPI(title=APP_NAME, description="Register (phone) + Authenticate (ESP32-CAM)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DETECTOR = None
RECOGNIZER = None

@app.on_event("startup")
def _startup():
    init_firebase()
    ensure_models()
    global DETECTOR, RECOGNIZER
    DETECTOR, RECOGNIZER = create_det_rec()

@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "detector": "YuNet",
        "recognizer": "SFace",
        "sim": SIM_MEASURE,
        "threshold": COS_THRESHOLD if SIM_MEASURE=="cosine" else L2_THRESHOLD
    }

@app.get("/engine")
def engine():
    return {
        "detector": "YuNet (ONNX)",
        "recognizer": "SFace (ONNX)",
        "det_input": DET_INPUT_SIZE,
        "similarity": SIM_MEASURE,
        "threshold": COS_THRESHOLD if SIM_MEASURE=="cosine" else L2_THRESHOLD
    }

# ---------------- Registration (mobile) ----------------
@app.post("/register")
def register(username: str = Form(...), image: UploadFile = File(...)):
    username = (username or "").strip()
    if not username:
        raise HTTPException(400, "username required")

    bgr = auto_enhance(read_upload_as_bgr(image))
    if lap_var(bgr) < MIN_LAPLACIAN_VAR:
        raise HTTPException(422, "Image too blurry. Try more light / hold steady.")

    feat, box = get_face_embedding(DETECTOR, RECOGNIZER, bgr)
    if feat is None:
        raise HTTPException(422, "No face detected. Center your face and try again.")
    if face_area_ratio_from_box(box, bgr.shape) < MIN_FACE_AREA_RATIO_REG:
        raise HTTPException(422, "Face too small. Move closer to the camera.")

    user_id = uuid.uuid4().hex[:12]
    rec = {
        "userId": user_id,
        "username": username,
        "image_b64": bgr_to_b64jpeg(bgr),  # immutable
        "embedding": feat.tolist(),        # single vector
        "createdAt": int(time.time()),
        "updatedAt": int(time.time()),
    }
    users_ref().child(user_id).set(rec)
    return {"ok": True, "userId": user_id}

# ---------------- Management ----------------
@app.get("/users")
def list_users():
    """Return ALL users with username + image (and timestamps)."""
    data = users_ref().get() or {}
    out = []
    for u in data.values():
        out.append({
            "userId":   u.get("userId"),
            "username": u.get("username"),
            "image_b64": u.get("image_b64"),
            "createdAt": u.get("createdAt"),
            "updatedAt": u.get("updatedAt"),
        })
    # keep a stable order (by createdAt if present)
    out.sort(key=lambda x: x.get("createdAt") or 0)
    return {"count": len(out), "users": out}

@app.get("/users/{user_id}")
def get_user(user_id: str, include_image: bool = True):
    rec = users_ref().child(user_id).get()
    if not rec:
        raise HTTPException(404, "User not found")
    if not include_image:
        rec.pop("image_b64", None)
    rec.pop("embedding", None)  # don't leak vector
    return rec

@app.patch("/users/{user_id}")
def rename_user(user_id: str, payload: Dict[str, Any] = Body(...)):
    new_name = (payload.get("username") or "").strip()
    if not new_name:
        raise HTTPException(400, "username required")
    ref = users_ref().child(user_id)
    if not ref.get():
        raise HTTPException(404, "User not found")
    ref.update({"username": new_name, "updatedAt": int(time.time())})
    return {"ok": True}

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    ref = users_ref().child(user_id)
    if not ref.get():
        raise HTTPException(404, "User not found")
    ref.delete()
    return {"ok": True}

# ---------------- Authentication (ESP32-CAM) ----------------
@app.post("/authenticate")
def authenticate(image: UploadFile = File(...)):
    bgr = auto_enhance(read_upload_as_bgr(image))
    feat, box = get_face_embedding(DETECTOR, RECOGNIZER, bgr)
    thr = COS_THRESHOLD if SIM_MEASURE=="cosine" else L2_THRESHOLD

    if feat is None:
        return {"authenticated": False, "message": "No face. More light / closer.", "score": 0.0, "threshold": thr}

    if face_area_ratio_from_box(box, bgr.shape) < MIN_FACE_AREA_RATIO_AUTH:
        return {"authenticated": False, "message": "Face too small. Move closer.", "score": 0.0, "threshold": thr}

    data = users_ref().get() or {}
    if not data:
        return {"authenticated": False, "message": "No users enrolled.", "score": 0.0, "threshold": thr}

    best_user, best_score = None, (-1e9 if SIM_MEASURE=="cosine" else 1e9)
    for rec in data.values():
        v = _embedding_from_rec(rec)
        if v.size == 0:
            continue
        sc = compare(feat, v)
        if SIM_MEASURE == "cosine":
            if sc > best_score:
                best_score, best_user = sc, rec
        else:
            if sc < best_score:
                best_score, best_user = sc, rec

    ok = (best_user is not None) and ( (best_score >= COS_THRESHOLD) if SIM_MEASURE=="cosine" else (best_score <= L2_THRESHOLD) )

    return {
        "authenticated": ok,
        "username": best_user.get("username") if (ok and best_user) else None,
        "score": round(float(best_score), 4),
        "threshold": thr,
        "message": "Access granted!" if ok else "Access denied."
    }
