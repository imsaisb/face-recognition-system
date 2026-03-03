import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlalchemy as sa
from datetime import datetime
import av  # pip install av if missing

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
STAFF_FOLDER = 'staff'
THRESHOLD = 0.80
CONFIDENCE = 0.90

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load authenticator config
with open('config.yaml', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# SQLite for attendance
engine = sa.create_engine('sqlite:///attendance.db')
metadata = sa.MetaData()
attendance = sa.Table('attendance', metadata,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('staff_id', sa.String),
    sa.Column('action', sa.String),
    sa.Column('timestamp', sa.DateTime),
)
metadata.create_all(engine)

# Models
@st.cache_resource
def load_models():
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    return mtcnn, resnet

mtcnn, resnet = load_models()

# Helpers
def extract_embedding(img_pil):
    faces = mtcnn(img_pil)
    if faces is not None:
        return resnet(faces.unsqueeze(0)).detach().cpu().numpy()
    return None

@st.cache_data
def load_staff_database():
    database = {}
    for staff_id in os.listdir(STAFF_FOLDER):
        path = os.path.join(STAFF_FOLDER, staff_id)
        if os.path.isdir(path):
            embeddings = []
            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = Image.open(os.path.join(path, file))
                    emb = extract_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
            if embeddings:
                database[staff_id] = np.mean(embeddings, axis=0)
    return database

def record_attendance(staff_id, action):
    timestamp = datetime.now()
    with engine.connect() as conn:
        ins = attendance.insert().values(staff_id=staff_id, action=action, timestamp=timestamp)
        conn.execute(ins)
        conn.commit()
    return timestamp

# ────────────────────────────────────────────────
# VIDEO PROCESSOR (fixed for av.VideoFrame return)
# ────────────────────────────────────────────────
class FaceProcessor(VideoProcessorBase):
    def __init__(self, database, threshold, action):
        self.database = database
        self.threshold = threshold
        self.action = action
        self.match_staff_id = None
        self.captured_frame = None
        self.error = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            boxes, probs = mtcnn.detect(img_pil)

            if boxes is not None and len(boxes) > 0 and probs[0] > CONFIDENCE:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                face_crop = img_pil.crop((x1, y1, x2, y2))
                emb = extract_embedding(face_crop)

                if emb is not None:
                    max_sim = 0
                    best_id = None
                    for sid, w_emb in self.database.items():
                        sim = cosine_similarity(emb, w_emb.reshape(1, -1))[0][0]
                        if sim > max_sim:
                            max_sim = sim
                            best_id = sid

                    if max_sim > self.threshold:
                        self.match_staff_id = best_id
                        self.captured_frame = img.copy()
                        label = "PASS"
                        color = (0, 255, 0)
                    else:
                        label = "NOT VERIFIED"
                        color = (0, 0, 255)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            print(f"Processing error: {e}")
            # Return original frame on error
            return frame

# ────────────────────────────────────────────────
# APP
# ────────────────────────────────────────────────
st.set_page_config(page_title="FACE RECOGNITION SYSTEM – PDRM NEGERI SEMBILAN", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom, #001f3f, #000814); color: white; }
    .stButton>button { background-color: #0066cc; color: white; border: none; border-radius: 8px; padding: 12px; font-weight: bold; font-size: 1.1em; }
    h1, h2, h3 { color: #00aaff; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Login
authenticator.login(
    location='main',
    fields={
        'Form name': 'LOG MASUK',
        'Username': 'ID Pengguna',
        'Password': 'Kata Laluan',
        'Login': 'LOG IN'
    }
)

authentication_status = st.session_state.get('authentication_status', False)

if authentication_status:
    name = st.session_state.get('name', 'Pengguna')
    username = st.session_state.get('username', 'unknown')

    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if st.session_state.page == 'main':
        st.markdown("""
        <div style="text-align:center; background: linear-gradient(to right, #001f3f, #003366); padding: 20px; border-radius: 10px;">
            <h1 style="color: white;">FACE RECOGNITION SYSTEM</h1>
            <h3 style="color: #00aaff;">“THE MANAGEMENT COPS”</h3>
            <p style="color: #ffcc00;">POLIS NEGERI SEMBILAN</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align:center;'>Selamat Datang, {name} ({username})</h2>", unsafe_allow_html=True)

        database = load_staff_database()
        if not database:
            st.error("Tiada data kakitangan di folder 'staff/'.")
            st.stop()

        action = st.radio("Pilih Tindakan", ["TIME IN (Masuk)", "TIME OUT (Keluar)"], horizontal=True)
        action_key = "IN" if "Masuk" in action else "OUT"

        if st.button("IMBAS SEKARANG", type="primary", use_container_width=True):
            st.session_state.page = 'scan'
            st.session_state.action_key = action_key
            st.rerun()

        authenticator.logout('Log Keluar', location='main')

    elif st.session_state.page == 'scan':
        st.markdown("<h2 style='color:#00aaff; text-align:center;'>Pengimbasan Wajah</h2>", unsafe_allow_html=True)
        st.info("Sila poskan wajah di hadapan kamera...")

        database = load_staff_database()
        action_key = st.session_state.action_key

        ctx = webrtc_streamer(
            key="scan_page",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] }),
            video_processor_factory=lambda: FaceProcessor(database, THRESHOLD, action_key),
            media_stream_constraints={
                "video": {"facingMode": "user"},
                "audio": False
            },
            async_processing=True
        )

        if ctx.video_processor:
            if ctx.video_processor.match_staff_id:
                staff_id = ctx.video_processor.match_staff_id
                ts = record_attendance(staff_id, action_key)
                st.success(f"**PASS** – {action_key} direkod pada {ts.strftime('%H:%M HRS')}")
                if ctx.video_processor.captured_frame is not None:
                    st.image(ctx.video_processor.captured_frame, channels="BGR")

            else:
                st.error("**NOT VERIFIED** – Wajah tidak dikenali")

            if ctx.video_processor.captured_frame is not None:
                st.image(ctx.video_processor.captured_frame, channels="BGR")

        if st.button("Kembali ke Menu Utama"):
            st.session_state.page = 'main'
            st.rerun()

else:
    if authentication_status is False:
        st.error('ID Pengguna atau Kata Laluan salah')
    else:
        st.warning('Sila log masuk terlebih dahulu')