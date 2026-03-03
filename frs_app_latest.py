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
import av
import pandas as pd

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
STAFF_FOLDER = 'staff'
THRESHOLD = 0.80
CONFIDENCE = 0.90

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Images (using your GitHub raw links)
BACKGROUND_IMAGE = "https://raw.githubusercontent.com/imsaisb/face-recognition-system/main/img/background.jpeg"
BANNER_IMAGE = "https://raw.githubusercontent.com/imsaisb/face-recognition-system/main/img/logo pdrm.jpg"

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

def has_open_clock_in(staff_id):
    with engine.connect() as conn:
        query = sa.select(attendance).where(
            attendance.c.staff_id == staff_id,
            attendance.c.action == 'IN'
        ).order_by(attendance.c.timestamp.desc())
        in_records = conn.execute(query).fetchall()

        if not in_records:
            return False

        latest_in = in_records[0]
        out_query = sa.select(attendance).where(
            attendance.c.staff_id == staff_id,
            attendance.c.action == 'OUT',
            attendance.c.timestamp > latest_in.timestamp
        )
        has_out = conn.execute(out_query).fetchone() is not None

        return not has_out

# ────────────────────────────────────────────────
# VIDEO PROCESSOR
# ────────────────────────────────────────────────
class FaceProcessor(VideoProcessorBase):
    def __init__(self, database, threshold, action):
        self.database = database
        self.threshold = threshold
        self.action = action
        self.match_staff_id = None
        self.captured_frame = None

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
            return frame

# ────────────────────────────────────────────────
# APP
# ────────────────────────────────────────────────
st.set_page_config(page_title="FACE RECOGNITION SYSTEM – PDRM NEGERI SEMBILAN", layout="wide")

# Background + dark overlay for readability
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.60);  /* Medium-dark overlay - adjust 0.60 to make darker/lighter */
        z-index: -1;
    }}
    .login-box {{
        background: rgba(0, 0, 0, 0.80);  /* Strong dark box for login text */
        border-radius: 20px;
        padding: 40px 30px;
        margin: 50px auto;
        max-width: 560px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 170, 255, 0.35);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        text-align: center;
    }}
    .glow-title {{
        text-shadow: 0 0 18px #00aaff, 0 0 35px #00aaff;
        margin-bottom: 10px;
    }}
    .subtitle {{
        color: #ffcc00;
        font-weight: bold;
        margin: 12px 0;
    }}
    .description {{
        color: #f0f0f0;
        margin: 20px 0 30px 0;
        line-height: 1.6;
    }}
    .pass-result {{
        background: #004d00;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 20px #00ff00;
    }}
    .not-verified-result {{
        background: #990000;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 20px #ff0000;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #0066cc, #0099ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        box-shadow: 0 0 20px rgba(0, 170, 255, 0.6) !important;
    }}
    h1, h2, h3 {{ color: #000000; text-align: center; }}  /* ← Changed to black as requested */
</style>
""", unsafe_allow_html=True)

# ────── LOGIN SCREEN ──────
if not st.session_state.get('authentication_status', False):
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="glow-title">FACE RECOGNITION SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">IPK NEGERI SEMBILAN</h3>', unsafe_allow_html=True)
    st.markdown('<p class="description">Pengguna perlu Log In dan masukkan kata laluan untuk menggunakan sistem</p>', unsafe_allow_html=True)

    authenticator.login(
        location='main',
        fields={
            'Form name': 'LOG MASUK',
            'Username': 'ID Pengguna',
            'Password': 'Kata Laluan',
            'Login': 'LOG IN'
        }
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('authentication_status', False) is False:
        st.error('ID Pengguna atau Kata Laluan salah')
    elif st.session_state.get('authentication_status') is None:
        st.warning('Sila masukkan ID Pengguna dan Kata Laluan')

else:
    name = st.session_state.get('name', 'Pengguna')
    username = st.session_state.get('username', 'unknown')

    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if st.session_state.page == 'main':
        # Banner header
        st.image(BANNER_IMAGE, width=1200)

        st.markdown(f"<h2 style='color:#000000; text-align:center;'>Selamat Datang, {name} ({username})</h2>", unsafe_allow_html=True)

        database = load_staff_database()
        if not database:
            st.error("Tiada data kakitangan di folder 'staff/'.")
            st.stop()

        action = st.radio("Pilih Tindakan", ["TIME IN (Masuk)", "TIME OUT (Keluar)"], horizontal=True)
        action_key = "IN" if "Masuk" in action else "OUT"

        col1, col2 = st.columns(2)
        with col1:
            if st.button("IMBAS SEKARANG", type="primary", use_container_width=True):
                st.session_state.page = 'scan'
                st.session_state.action_key = action_key
                st.rerun()

        with col2:
            if st.button("Senarai Clock In / Out", type="secondary", use_container_width=True):
                st.session_state.page = 'logs'
                st.rerun()

        authenticator.logout('Log Keluar', location='main')

    elif st.session_state.page == 'scan':
        title = "TIME IN (Masuk)" if st.session_state.action_key == "IN" else "TIME OUT (Keluar)"
        st.markdown(f"<h2 style='color:#000000; text-align:center;'>{title}</h2>", unsafe_allow_html=True)

        st.markdown("""
        <div class="scan-frame">
            <h3 style="color:#00aaff;">Sistem Pengimbas Wajah</h3>
            <p style="color:#ffffff;">SCANNING...</p>
        </div>
        """, unsafe_allow_html=True)

        database = load_staff_database()
        action_key = st.session_state.action_key

        ctx = webrtc_streamer(
            key="scan_page",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] }),
            video_processor_factory=lambda: FaceProcessor(database, THRESHOLD, action_key),
            media_stream_constraints={
                "video": True,  # Auto / let user choose via SELECT DEVICE
                "audio": False
            },
            async_processing=True
        )

        if ctx.video_processor:
            if ctx.video_processor.match_staff_id:
                staff_id = ctx.video_processor.match_staff_id

                if action_key == "OUT":
                    if not has_open_clock_in(staff_id):
                        st.error("**TIDAK DIBENARKAN** – Tiada rekod TIME IN terbuka.")
                    else:
                        ts = record_attendance(staff_id, action_key)
                        st.markdown(f"""
                        <div class="pass-result">
                            <h2>PASS</h2>
                            <p style="font-size:1.4em;">{title} direkod pada {ts.strftime('%H:%M HRS')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if ctx.video_processor.captured_frame is not None:
                            st.image(ctx.video_processor.captured_frame, width=500)
                else:
                    ts = record_attendance(staff_id, action_key)
                    st.markdown(f"""
                    <div class="pass-result">
                        <h2>PASS</h2>
                        <p style="font-size:1.4em;">{title} direkod pada {ts.strftime('%H:%M HRS')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if ctx.video_processor.captured_frame is not None:
                        st.image(ctx.video_processor.captured_frame, width=500)

            else:
                st.markdown("""
                <div class="not-verified-result">
                    <h2>NOT VERIFIED</h2>
                    <p>Wajah tidak dikenali dalam sistem</p>
                </div>
                """, unsafe_allow_html=True)

            if ctx.video_processor.captured_frame is not None:
                st.image(ctx.video_processor.captured_frame, width=500)

        if st.button("Kembali ke Menu Utama"):
            st.session_state.page = 'main'
            st.rerun()

    elif st.session_state.page == 'logs':
        st.markdown("<h2 style='color:#000000; text-align:center;'>Senarai Clock In & Clock Out</h2>", unsafe_allow_html=True)

        with engine.connect() as conn:
            df = pd.read_sql("SELECT staff_id, action, timestamp FROM attendance ORDER BY timestamp DESC", conn)

        if df.empty:
            st.info("Tiada rekod lagi dalam sistem.")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['Tarikh & Masa'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(
                df[['staff_id', 'action', 'Tarikh & Masa']].rename(columns={
                    'staff_id': 'ID Kakitangan',
                    'action': 'Tindakan'
                }),
                use_container_width=True
            )

            csv = df[['staff_id', 'action', 'Tarikh & Masa']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Muat Turun Rekod (CSV)",
                data=csv,
                file_name="rekod_clock_in_out.csv",
                mime="text/csv"
            )

        if st.button("Kembali ke Menu Utama"):
            st.session_state.page = 'main'
            st.rerun()

        authenticator.logout('Log Keluar', location='main')