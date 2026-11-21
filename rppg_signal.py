import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt, detrend
# HEARTVISION AI - rPPG Signal Extraction

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face detection model not loaded.")
    raise SystemExit

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

green_values = []
timestamps = []

start_time = time.perf_counter()
duration = 30

print("HEARTVISION AI started!")
print("Keep your face still and look at the camera.")
print("Recording facial colour signal for 30 seconds...")

try:
    while time.perf_counter() - start_time < duration:

        success, frame = camera.read()

        if not success:
            print("Camera frame could not be read.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(100, 100)
        )

        if len(faces) > 0:
            # Select the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Select a skin region around the cheeks
            x1 = x + int(0.15 * w)
            x2 = x + int(0.85 * w)
            y1 = y + int(0.40 * h)
            y2 = y + int(0.70 * h)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                green_mean = np.mean(roi[:, :, 1])

                green_values.append(green_mean)
                timestamps.append(time.perf_counter() - start_time)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )

                cv2.putText(
                    frame, "SIGNAL RECORDING",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

        cv2.imshow("HEARTVISION AI - Signal Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Recording stopped by user.")
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Recording finished.")
print("Total samples collected:", len(green_values))

# HEART RATE CALCULATION

if len(green_values) >= 100 and len(timestamps) == len(green_values):

    signal = np.array(green_values, dtype=float)
    t = np.array(timestamps, dtype=float)

    # Calculate actual sampling frequency
    duration_recorded = t[-1] - t[0]
    fs = (len(t) - 1) / duration_recorded

    if duration_recorded >= 20 and fs > 6.5:

        # Make timestamps evenly spaced
        uniform_t = np.linspace(t[0], t[-1], len(t))
        uniform_signal = np.interp(uniform_t, t, signal)

        # Remove slow changes in the signal
        clean_signal = detrend(uniform_signal)

        # Filter the pulse-related frequency range
        nyquist = 0.5 * fs
        low = 0.7 / nyquist
        high = 3.0 / nyquist

        b, a = butter(3, [low, high], btype="band")
        filtered_signal = filtfilt(b, a, clean_signal)

        # Calculate frequency using FFT
        fft_values = np.abs(np.fft.rfft(filtered_signal))
        frequencies = np.fft.rfftfreq(
            len(filtered_signal), d=1/fs
        )

        # Consider pulse rates from 42 to 180 BPM
        valid = (frequencies >= 0.7) & (frequencies <= 3.0)

        if np.any(valid):
            valid_frequencies = frequencies[valid]
            valid_fft = fft_values[valid]

            peak_frequency = valid_frequencies[np.argmax(valid_fft)]
            heart_rate = peak_frequency * 60

            print("\n------------------------------")
            print("       HEARTVISION AI")
            print("------------------------------")
            print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
            print("------------------------------")

            # Save graph data for the website
            graph_data = np.column_stack(
                (uniform_t - uniform_t[0], filtered_signal)
            )
            np.savetxt(
                "pulse_signal.csv",
                graph_data,
                delimiter=",",
                header="Time,Signal",
                comments=""
            )

        else:
            print("Could not identify a clear pulse signal.")

    else:
        print("Recording quality or duration is insufficient.")

else:
    print("Not enough face samples. Please try again.")