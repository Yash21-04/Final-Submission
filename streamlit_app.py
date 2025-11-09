import streamlit as st
import cv2
import numpy as np
import time
import threading
import os
from tensorflow.keras.models import load_model
from imutils import face_utils
import dlib
from scipy.spatial import distance
from collections import deque

st.set_page_config(page_title='Drowsiness Demo — Dual Models', layout='wide')
st.title('Driver Drowsiness — Streamlit Frontend (Two Models & Live Graphs)')

st.sidebar.header('Settings')
mode = st.sidebar.selectbox('Mode', ['Compare Models (CNNs)', 'EAR (fast) + CNN'])
use_camera = st.sidebar.checkbox('Use webcam (cv2.VideoCapture)', value=True)
video_file = st.sidebar.file_uploader('Or upload video file (.mp4)', type=['mp4','avi','mov'])

predictor_file = st.sidebar.file_uploader('Upload dlib shape_predictor_68_face_landmarks.dat', type=['dat'])
modelA_file = st.sidebar.file_uploader('Upload model A (eye_model_A.h5)', type=['h5'])
modelB_file = st.sidebar.file_uploader('Upload model B (eye_model_B.h5)', type=['h5'])

start_button = st.sidebar.button('Start')
stop_button = st.sidebar.button('Stop')

placeholder = st.empty()
col1, col2 = st.columns([2,1])

# session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'modelA' not in st.session_state:
    st.session_state.modelA = None
if 'modelB' not in st.session_state:
    st.session_state.modelB = None
if 'predictor_path' not in st.session_state:
    st.session_state.predictor_path = None
if 'histA' not in st.session_state:
    st.session_state.histA = deque(maxlen=200)
if 'histB' not in st.session_state:
    st.session_state.histB = deque(maxlen=200)
if 'histEAR' not in st.session_state:
    st.session_state.histEAR = deque(maxlen=200)

# Save uploaded files
os.makedirs('uploads', exist_ok=True)
if predictor_file is not None:
    ppath = os.path.join('uploads', predictor_file.name)
    with open(ppath, 'wb') as f:
        f.write(predictor_file.getbuffer())
    st.session_state.predictor_path = ppath

if modelA_file is not None:
    mpath = os.path.join('uploads', modelA_file.name)
    with open(mpath, 'wb') as f:
        f.write(modelA_file.getbuffer())
    st.session_state.modelA = load_model(mpath)

if modelB_file is not None:
    mpath = os.path.join('uploads', modelB_file.name)
    with open(mpath, 'wb') as f:
        f.write(modelB_file.getbuffer())
    st.session_state.modelB = load_model(mpath)

# helpers
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def crop_eye(frame, eye_pts):
    x1 = np.min(eye_pts[:,0]) - 5
    y1 = np.min(eye_pts[:,1]) - 5
    x2 = np.max(eye_pts[:,0]) + 5
    y2 = np.max(eye_pts[:,1]) + 5
    x1, y1 = max(0,int(x1)), max(0,int(y1))
    x2, y2 = int(x2), int(y2)
    crop = frame[y1:y2, x1:x2]
    try:
        crop = cv2.resize(crop, (64,64))
    except:
        return None
    return crop

# processing thread
def run_pipeline(cap, mode, modelA, modelB, predictor_path):
    if predictor_path is None:
        st.error('Dlib predictor required. Upload it in the sidebar.')
        st.session_state.running = False
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    ear_thresh = 0.25
    ear_consec = 18
    ear_closed_count = 0

    cnn_thresh = 0.6
    cnn_consec = 18
    cnnA_closed_count = 0
    cnnB_closed_count = 0

    img_placeholder = placeholder.empty()
    chartA = col2.empty()
    chartB = col2.empty()
    status = st.empty()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        probA, probB, probEAR = None, None, None
        for rect in rects:
            shape = face_utils.shape_to_np(predictor(gray, rect))
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_display, [leftHull], -1, (0,255,0), 1)
            cv2.drawContours(frame_display, [rightHull], -1, (0,255,0), 1)

            if mode == 'EAR (fast) + CNN':
                # EAR
                l_ear = eye_aspect_ratio(leftEye)
                r_ear = eye_aspect_ratio(rightEye)
                ear = (l_ear + r_ear) / 2.0
                probEAR = max(0.0, min(1.0, 1.0 - (ear / 0.4)))  # heuristically convert to "closed probability"
                st.session_state.histEAR.append(probEAR)
                if ear < ear_thresh:
                    ear_closed_count += 1
                    if ear_closed_count >= ear_consec:
                        cv2.putText(frame_display, 'DROWSINESS ALERT (EAR)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                else:
                    ear_closed_count = 0

            # CNN models (if provided)
            le = crop_eye(frame, leftEye)
            re = crop_eye(frame, rightEye)
            if le is not None and re is not None:
                if modelA is not None:
                    pA_left = modelA.predict(np.expand_dims(le.astype('float32')/255.0,0))[0][0]
                    pA_right = modelA.predict(np.expand_dims(re.astype('float32')/255.0,0))[0][0]
                    probA = 1.0 - ((pA_left + pA_right)/2.0)
                    st.session_state.histA.append(probA)
                    if probA > cnn_thresh:
                        cnnA_closed_count += 1
                        if cnnA_closed_count >= cnn_consec:
                            cv2.putText(frame_display, 'DROWSINESS ALERT (Model A)', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
                    else:
                        cnnA_closed_count = 0
                if modelB is not None:
                    pB_left = modelB.predict(np.expand_dims(le.astype('float32')/255.0,0))[0][0]
                    pB_right = modelB.predict(np.expand_dims(re.astype('float32')/255.0,0))[0][0]
                    probB = 1.0 - ((pB_left + pB_right)/2.0)
                    st.session_state.histB.append(probB)
                    if probB > cnn_thresh:
                        cnnB_closed_count += 1
                        if cnnB_closed_count >= cnn_consec:
                            cv2.putText(frame_display, 'DROWSINESS ALERT (Model B)', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,128,255), 2)
                    else:
                        cnnB_closed_count = 0

        # display frame
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        img_placeholder.image(frame_display, channels='RGB')

        # update charts (show last N values)
        if len(st.session_state.histA) > 0:
            chartA.line_chart({'Model A (closed_prob)': list(st.session_state.histA)})
        if len(st.session_state.histB) > 0:
            chartB.line_chart({'Model B (closed_prob)': list(st.session_state.histB)})
        if len(st.session_state.histEAR) > 0:
            # show EAR converted probability alongside (in small chart)
            col2.line_chart({'EAR (heuristic closed_prob)': list(st.session_state.histEAR)})

        status.text(f"Mode: {mode} — Running — ModelA: {'loaded' if modelA is not None else 'none'} — ModelB: {'loaded' if modelB is not None else 'none'}")
        time.sleep(0.02)

    status.text('Stopped')
    img_placeholder.empty()
    col2.empty()

# Start/Stop
if start_button:
    if st.session_state.running:
        st.warning('Already running')
    else:
        st.session_state.running = True
        if use_camera:
            cap = cv2.VideoCapture(0)
        else:
            if video_file is None:
                st.error('No video uploaded and webcam not selected')
                st.session_state.running = False
                cap = None
            else:
                vf_path = os.path.join('uploads', video_file.name)
                with open(vf_path, 'wb') as f:
                    f.write(video_file.getbuffer())
                cap = cv2.VideoCapture(vf_path)
        st.session_state.cap = cap
        t = threading.Thread(target=run_pipeline, args=(cap, mode, st.session_state.modelA, st.session_state.modelB, st.session_state.predictor_path))
        t.start()

if stop_button:
    if not st.session_state.running:
        st.warning('Not running')
    else:
        st.session_state.running = False
        if st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except:
                pass
        st.success('Stopped')

st.markdown('---')
st.write('Tips: Upload two different trained models to compare their closed-eye probability outputs in real-time. Use the charts on the right to compare performance.')
st.write('Run this app locally with: `streamlit run streamlit_app.py`')
