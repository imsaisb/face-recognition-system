import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

st.set_page_config(page_title="Test Pilih Kamera", layout="centered")

st.title("Test Pilih Kamera")
st.markdown("Pilih kamera yang ingin digunakan. Pastikan webcam luar sudah disambungkan.")

# Dropdown pilihan kamera
camera_choice = st.selectbox(
    "Pilih Kamera",
    options=[
        "Kamera Bawaan Laptop (Built-in)",
        "Webcam Luar / USB",
        "Auto (biarkan browser pilih)"
    ],
    index=1  # default ke webcam luar
)

# If-else untuk tentukan facingMode
if camera_choice == "Kamera Bawaan Laptop (Built-in)":
    facing_mode = "user"
    st.caption("Sedang menggunakan: **Kamera Bawaan (user)**")
elif camera_choice == "Webcam Luar / USB":
    facing_mode = "environment"
    st.caption("Sedang menggunakan: **Webcam Luar (environment)**")
else:
    facing_mode = None
    st.caption("Mod Auto: Browser akan memilih kamera yang tersedia")

# Jalankan kamera dengan pilihan facingMode
webrtc_streamer(
    key="test_camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={
        "video": {
            "facingMode": facing_mode,  # ← dikawal oleh if-else
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    video_html_attrs={
        "style": {"width": "100%", "maxWidth": "640px", "borderRadius": "12px"}
    }
)

st.info("Jika kamera tidak muncul:")
st.markdown("- Tutup semua aplikasi lain yang menggunakan kamera (Zoom, Teams, Camera app, tab browser lain)")
st.markdown("- Refresh halaman (F5)")
st.markdown("- Pastikan webcam luar sudah disambungkan dan dikesan di Device Manager")
st.markdown("- Jika masih guna kamera bawaan → disable kamera bawaan di Device Manager → restart laptop")