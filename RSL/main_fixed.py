#!/usr/bin/env python3
# ---------------------------------------------------------------
# SIGN LANGUAGE DETECTION — MODERN MODULAR SINGLE-FILE VERSION
# ---------------------------------------------------------------

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import mediapipe as mp


# ===============================================================
# CONFIG CLASS (modifies application behavior)
# ===============================================================
@dataclass
class AppConfig:
    data_path: str = os.path.join(os.path.dirname(__file__), "..", "..", "MP_Data")
    model_path: str = os.path.join(os.path.dirname(__file__), "..", "model.h5")
    model_weights_path: str = os.path.join(os.path.dirname(__file__), "..", "model_weights.h5")

    # Full Mediapipe Holistic feature vector: 1662 values
    feature_vector_length: int = 1662

    # Default values
    sequence_length: int = 30
    sequences_per_sign: int = 30
    training_epochs: int = 2000

    # Drawing palette (Material Design)
    palette: dict = None

    def __post_init__(self):
        self.palette = {
            "face": (66, 133, 244),
            "pose": (52, 168, 83),
            "left_hand": (251, 188, 5),
            "right_hand": (234, 67, 53),
            "prob_bg": (33, 150, 243),
        }


# ===============================================================
# MEDIAPIPE HANDLER
# ===============================================================
class MediapipeHandler:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, image, model):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = model.process(rgb)
        rgb.flags.writeable = True
        out_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return out_image, results

    def draw_landmarks(self, image, results, palette):
        # Face
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=palette["face"], thickness=1, circle_radius=1),
            )
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=palette["pose"], thickness=2, circle_radius=2),
            )
        # Left Hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=palette["left_hand"], thickness=2, circle_radius=2),
            )
        # Right Hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=palette["right_hand"], thickness=2, circle_radius=2),
            )

    @staticmethod
    def extract_keypoints(results):
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[f.x, f.y, f.z] for f in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[h.x, h.y, h.z] for h in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[h.x, h.y, h.z] for h in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])


# ===============================================================
# DATASET HANDLING
# ===============================================================
class DatasetManager:
    def __init__(self, cfg: AppConfig, signs: List[str]):
        self.cfg = cfg
        self.signs = signs
        self.mp_handler = MediapipeHandler()
        if not os.path.exists(cfg.data_path):
            os.mkdir(cfg.data_path)

    def collect(self):
        cap = cv2.VideoCapture(0)
        with self.mp_handler.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
            for sign in self.signs:
                for seq in range(self.cfg.sequences_per_sign):
                    for frame_num in range(self.cfg.sequence_length):
                        ret, frame = cap.read()
                        if not ret: continue
                        image, results = self.mp_handler.detect(frame, model)
                        self.mp_handler.draw_landmarks(image, results, self.cfg.palette)
                        if frame_num == 0:
                            cv2.putText(image, f"Start: {sign} (seq {seq})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            cv2.imshow("Collecting Data", image)
                            cv2.waitKey(1000)
                        keypoints = self.mp_handler.extract_keypoints(results)
                        seq_path = os.path.join(self.cfg.data_path, sign, str(seq))
                        os.makedirs(seq_path, exist_ok=True)
                        np.save(os.path.join(seq_path, f"{frame_num}.npy"), keypoints)
                        cv2.imshow("Collecting Data", image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
        cap.release()
        cv2.destroyAllWindows()
        return True

    def load_dataset(self):
        sequences, labels = [], []
        label_map = {label: i for i, label in enumerate(self.signs)}
        for sign in self.signs:
            sign_dir = os.path.join(self.cfg.data_path, sign)
            if not os.path.exists(sign_dir): continue
            
            for seq in os.listdir(sign_dir):
                window = []
                valid_sequence = True
                for frame_num in range(self.cfg.sequence_length):
                    file_path = os.path.join(sign_dir, str(seq), f"{frame_num}.npy")
                    if not os.path.exists(file_path):
                        valid_sequence = False
                        break
                    res = np.load(file_path)
                    window.append(res)
                
                if valid_sequence:
                    sequences.append(window)
                    labels.append(label_map[sign])
        
        if not sequences:
            return np.array([]), np.array([])
            
        return np.array(sequences), to_categorical(labels).astype(int)


# ===============================================================
# MODEL CREATION + TRAINING
# ===============================================================
class ModelHandler:
    def __init__(self, cfg: AppConfig, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = None

    def build_lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.cfg.sequence_length, self.cfg.feature_vector_length)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        self.model.fit(X_train, y_train, epochs=self.cfg.training_epochs)
        predictions = np.argmax(self.model.predict(X_test), axis=1)
        real = np.argmax(y_test, axis=1)
        print("Accuracy:", accuracy_score(real, predictions))

    def save(self):
        self.model.save(self.cfg.model_path)
        self.model.save_weights(self.cfg.model_weights_path)

    def load(self):
        self.model = tf.keras.models.load_model(self.cfg.model_path)


# ===============================================================
# INFERENCE (webcam or video)
# ===============================================================
class InferenceEngine:
    def __init__(self, cfg: AppConfig, signs: List[str], model):
        self.cfg = cfg
        self.signs = signs
        self.model = model
        self.mp_handler = MediapipeHandler()

    def run(self, video_source=0):
        sequence, sentence = [], []
        threshold = 0.8
        cap = cv2.VideoCapture(video_source)
        with self.mp_handler.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                image, results = self.mp_handler.detect(frame, model)
                self.mp_handler.draw_landmarks(image, results, self.cfg.palette)
                keypoints = self.mp_handler.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-self.cfg.sequence_length:]
                if len(sequence) == self.cfg.sequence_length:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))
                    if res[0][np.argmax(res[0])] > threshold:
                        if len(sentence) > 0:
                            if self.signs[np.argmax(res[0])] != sentence[-1]:
                                sentence.append(self.signs[np.argmax(res[0])])
                        else:
                            sentence.append(self.signs[np.argmax(res[0])])
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Live Detection', image)
                if cv2.waitKey(10) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()


# ===============================================================
# MAIN CONSOLE APPLICATION
# ===============================================================
def main():
    print("\n=== SIGN LANGUAGE DETECTION SYSTEM ===\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("GPU disabled. Running on CPU.")
    
    cfg = AppConfig()
    
    # 1. Detect available signs from data directory
    if not os.path.exists(cfg.data_path):
        print(f"Error: Data directory '{cfg.data_path}' not found.")
        print("Please run 'train_gestures.py' first to collect data and train the model.")
        return

    signs = sorted([d for d in os.listdir(cfg.data_path) if os.path.isdir(os.path.join(cfg.data_path, d))])
    
    if not signs:
        print(f"Error: No gesture data found in '{cfg.data_path}'.")
        print("Please run 'train_gestures.py' to add gestures.")
        return

    print(f"Detected gestures: {', '.join(signs)}")
    print(f"Frames per sequence: {cfg.sequence_length}")
    
    # 2. Check for model
    if not os.path.exists(cfg.model_path):
        print(f"Error: Model file '{cfg.model_path}' not found.")
        print("Please run 'train_gestures.py' to train the model.")
        return

    model_handler = ModelHandler(cfg, len(signs))
    
    print("\nMODE: Live Detection (Webcam)")
    try:
        model_handler.load()
        engine = InferenceEngine(cfg, signs, model_handler.model)
        engine.run(0)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to a mismatch between the current signs and the trained model.")
        print("Please re-run 'train_gestures.py' to update the model.")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()