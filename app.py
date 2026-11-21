import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import os
import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt, detrend
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.green_values = []
        self.timestamps = []
        self.start_time = time.perf_counter()
        self.duration = 30

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        elapsed = time.perf_counter() - self.start_time

        if elapsed <= self.duration:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(100, 100)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                x1 = x + int(0.15 * w)
                x2 = x + int(0.85 * w)
                y1 = y + int(0.40 * h)
                y2 = y + int(0.70 * h)

                roi = img[y1:y2, x1:x2]

                if roi.size > 0:
                    self.green_values.append(float(np.mean(roi[:, :, 1])))
                    self.timestamps.append(elapsed)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    img, "SIGNAL RECORDING",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            cv2.putText(
                img, f"Time: {int(elapsed)} / 30 seconds",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

        else:
            cv2.putText(
                img, "SCAN COMPLETE",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                img,
                "FACE DETECTED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

         


st.markdown("""
<style>
.stApp {
    background: #0B1424;
    color: #FFFFFF;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3, p, label {
    color: #FFFFFF !important;
}
.metric-card {
    background: #15243A;
    border: 1px solid #263850;
    border-radius: 14px;
    padding: 22px;
    min-height: 125px;
}
.metric-label {
    color: #9DB4D0;
    font-size: 13px;
    letter-spacing: 1px;
}
.metric-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}
.metric-note {
    color: #91A6C4;
    font-size: 12px;
}
.panel {
    background: #15243A;
    border: 1px solid #263850;
    border-radius: 14px;
    padding: 24px;
    margin-top: 18px;
}
.status {
    color: #54D6A0;
    font-weight: bold;
    font-size: 13px;
}
.footer {
    color: #91A6C4;
    font-size: 12px;
    text-align: center;
    padding: 25px;
}
div.stButton > button {
    background: #1769AA;
    color: white;
    border: none;
    border-radius: 10px;
    height: 52px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background: #238BD7;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# ❤️ HEARTVISION AI
### INTELLIGENT PULSE MONITORING PLATFORM

<div class="status">● SYSTEM ONLINE &nbsp; | &nbsp; EXPERIMENTAL PROTOTYPE</div>
""", unsafe_allow_html=True)

st.write(" ")

left, right = st.columns(2)

with left:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">HEART RATE ESTIMATE</div>
        <div class="metric-value">-- BPM</div>
        <div class="metric-note">Awaiting measurement</div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">CAMERA STATUS</div>
        <div class="metric-value">STANDBY</div>
        <div class="metric-note">Camera not connected</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <h3>◉ PULSE SIGNAL ANALYSIS</h3>
    <p style="color:#91A6C4">
    Your processed pulse waveform will appear here
    once a scan has been completed.
    </p>
</div>
""", unsafe_allow_html=True)

if "scan_started" not in st.session_state:
    st.session_state.scan_started = False

if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

if st.button("▶ START NEW SCAN", disabled=st.session_state.scan_started):
    st.session_state.scan_started = True
    st.rerun()

if st.session_state.scan_started:

    st.subheader("Live Camera Scan")

    ctx = webrtc_streamer(
        key="heartvision-scan",
        video_processor_factory=FaceDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        media_toggle_controls=False,
        async_processing=True,
    )

    timer_box = st.empty()
    processor = None
    wait_start = time.perf_counter()

    while True:
        processor = ctx.video_processor

        if processor is not None:
            elapsed = time.perf_counter() - processor.start_time
            remaining = max(0, 30 - int(elapsed))
            timer_box.info(f"Recording scan: {remaining} seconds remaining")

            if elapsed >= 30:
                break

        else:
            timer_box.info("Waiting for camera to start...")

        if time.perf_counter() - wait_start > 60:
            break

        time.sleep(1)

    if processor is None:
        st.session_state.scan_result = {
            "error": "Camera did not start. Please check camera permission and try again."
        }

    else:
        signal = np.array(processor.green_values, dtype=float)
        t = np.array(processor.timestamps, dtype=float)

        if len(signal) >= 100:
            duration_recorded = t[-1] - t[0]
            fs = (len(t) - 1) / duration_recorded

            if duration_recorded >= 20 and fs > 6.5:
                uniform_t = np.linspace(t[0], t[-1], len(t))
                uniform_signal = np.interp(uniform_t, t, signal)
                clean_signal = detrend(uniform_signal)

                nyquist = 0.5 * fs
                low = 0.7 / nyquist
                high = 3.0 / nyquist

                b, a = butter(3, [low, high], btype="band")
                filtered_signal = filtfilt(b, a, clean_signal)

                fft_values = np.abs(np.fft.rfft(filtered_signal))
                frequencies = np.fft.rfftfreq(len(filtered_signal), d=1/fs)

                valid = (frequencies >= 0.7) & (frequencies <= 3.0)

                if np.any(valid):
                    peak_frequency = frequencies[valid][
                        np.argmax(fft_values[valid])
                    ]
                    heart_rate = peak_frequency * 60

                    graph = pd.DataFrame({
                        "Time": uniform_t - uniform_t[0],
                        "Signal": filtered_signal
                    })

                    st.session_state.scan_result = {
                        "bpm": heart_rate,
                        "samples": len(signal),
                        "graph": graph
                    }
                else:
                    st.session_state.scan_result = {
                        "error": "Could not identify a clear pulse signal."
                    }
            else:
                st.session_state.scan_result = {
                    "error": "Recording quality or duration is insufficient."
                }
        else:
            st.session_state.scan_result = {
                "error": "Not enough face samples. Please try again."
            }

    st.session_state.scan_started = False
    st.rerun()

if st.session_state.scan_result is not None:
    result = st.session_state.scan_result

    if "error" in result:
        st.warning(result["error"])
    else:
        st.metric("Estimated Heart Rate", f"{result['bpm']:.1f} BPM")
        st.write("Total samples collected:", result["samples"])
        st.subheader("30-Second Pulse Signal")
        st.line_chart(result["graph"], x="Time", y="Signal")
st.markdown("""
<div class="footer">
HEARTVISION AI - EDUCATIONAL AI PROJECT<br>
Experimental pulse estimation - Not intended for medical diagnosis
</div>
""", unsafe_allow_html=True)