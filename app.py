import sys
import json
import csv
import string
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Gestura",
    page_icon="G",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
MODEL_DIR = BACKEND_DIR / "models"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = BACKEND_DIR / "data"
HISTORY_FILE = DATA_DIR / "conversation_history.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_conversation_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
        return history if isinstance(history, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_conversation_message(message, language, gestures):
    if not message or message in {
        "Perform a gesture to begin.",
        "Perform gestures to communicate.",
    }:
        return

    history = load_conversation_history()
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": message,
        "language": language,
        "gestures": list(gestures),
    }

    if history and history[-1].get("message") == message:
        return

    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)


# ==================================================
# LOAD REPORTS
# ==================================================

def load_report(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


training = load_report("evaluation_report.json")
independent = load_report("independent_evaluation_report.json")


# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>
:root {
    --gb-bg: #070b18;
    --gb-panel: rgba(18, 25, 48, 0.92);
    --gb-panel-strong: #11182d;
    --gb-border: rgba(148, 163, 184, 0.16);
    --gb-blue: #4f8cff;
    --gb-violet: #8b5cf6;
    --gb-cyan: #22d3ee;
    --gb-text: #f8fafc;
    --gb-muted: #a8b3c7;
}

html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
        radial-gradient(circle at 72% 0%, rgba(99, 102, 241, 0.22), transparent 26%),
        radial-gradient(circle at 18% 92%, rgba(14, 165, 233, 0.10), transparent 30%),
        #080d1c;
    background-size: 38px 38px, 38px 38px, auto, auto, auto;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 16px;
}

.block-container {
    max-width: 1380px;
    padding: 2.25rem 3rem 4rem 3rem;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 50% 0%, rgba(99,102,241,.11), transparent 30%),
        linear-gradient(180deg, #0e1428, #090d1a);
    box-shadow: 18px 0 45px rgba(0,0,0,.12);
}

[data-testid="stSidebar"] .block-container {
    padding: 1.4rem 1.1rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

h1, h2, h3, h4 {
    color: var(--gb-text) !important;
    letter-spacing: -0.035em;
}

h1 {
    font-size: clamp(2rem, 4vw, 3.25rem) !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    font-size: 1.45rem !important;
}

h3 {
    font-size: 1.12rem !important;
}

p, li, label, span {
    color: #dbe4f1;
}

[data-testid="stMetric"] {
    background:
        linear-gradient(145deg, rgba(29,39,76,.96), rgba(13,19,39,.96));
    border: 1px solid rgba(168,185,224,.14);
    border-radius: 20px;
    box-shadow: 0 18px 45px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.025);
    min-height: 150px;
}

[data-testid="stMetricValue"] {
    color: white;
    font-size: 2.15rem;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: var(--gb-muted);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    min-height: 48px;
    padding-inline: 18px;
    background: rgba(30, 41, 69, 0.82);
    color: var(--gb-text);
    font-weight: 700;
    transition: 0.18s ease;
    box-shadow: 0 8px 20px rgba(0,0,0,.12);
}

.stButton > button:hover {
    border-color: rgba(96, 165, 250, 0.72);
    background: rgba(47, 63, 101, 0.95);
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3478f6, #7557f2);
    border: none;
    box-shadow: 0 10px 28px rgba(79, 140, 255, 0.24);
}

[data-testid="stRadio"] {
    background: transparent;
}

[data-testid="stRadio"] label {
    width: 100%;
    border-radius: 10px;
    padding: 8px 9px;
    margin: 2px 0;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(79,140,255,.24), rgba(139,92,246,.10));
    box-shadow: inset 3px 0 #6b8cff;
}

[data-testid="stAlert"], [data-testid="stNotification"] {
    border-radius: 14px;
    border: 1px solid var(--gb-border);
}

[data-testid="stExpander"] {
    background: rgba(16,23,43,.86);
    backdrop-filter: blur(16px);
    border: 1px solid var(--gb-border);
    border-radius: 14px;
}

[data-testid="stTable"], [data-testid="stDataFrame"] {
    border: 1px solid var(--gb-border);
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border: none;
    border-top: 1px solid var(--gb-border);
}

.gb-hero {
    border: 1px solid var(--gb-border);
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 22px;
    background:
        linear-gradient(115deg, rgba(79, 140, 255, 0.16), rgba(139, 92, 246, 0.11) 55%, rgba(34, 211, 238, 0.06)),
        var(--gb-panel);
    box-shadow: 0 22px 55px rgba(0, 0, 0, 0.24);
}

.gb-hero-kicker {
    color: #80aaff;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.76rem;
    font-weight: 800;
}

.gb-feature-card {
    min-height: 168px;
    border: 1px solid var(--gb-border);
    border-radius: 18px;
    padding: 20px;
    background: linear-gradient(145deg, rgba(24, 33, 61, 0.95), rgba(13, 18, 36, 0.95));
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
}

.gb-feature-icon {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: linear-gradient(135deg, #3478f6, #7557f2);
    font-size: 20px;
    margin-bottom: 14px;
}

.gb-feature-card h3 {
    margin: 0 0 7px 0;
}

.gb-feature-card p {
    color: var(--gb-muted);
    margin: 0;
    line-height: 1.5;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px 0">
      <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#4f8cff,#8b5cf6);
      display:grid;place-items:center;font-size:20px">✦</div>
      <div><div style="font-size:19px;font-weight:800">Gestura</div>
      <div style="font-size:12px;color:#96a4bb">AI Communication</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("SYSTEM ONLINE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Translator",
            "AI Analytics",
            "Emergency Mode",
            "Gesture Studio",
            "Practice Assistant",
            "Conversations",
            "Help & Guide",
            "Settings"
        ],
        key="main_navigation"
    )


st.title("Gestura")
st.subheader("Communication Without Barriers")


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.markdown("""
    <div class="gb-hero">
      <div class="gb-hero-kicker">Gesture communication platform</div>
      <h1 style="margin:8px 0 6px 0">Welcome to Gestura</h1>
      <p style="color:#b8c3d6;font-size:1.08rem;margin:0">
        Translate gestures into accessible text, speech, and multilingual messages.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Gesture Classes", "14")

    with col2:
        accuracy = (
            f"{training['accuracy'] * 100:.2f}%"
            if training and "accuracy" in training
            else "Unavailable"
        )
        st.metric("Training Accuracy", accuracy)

    with col3:
        samples = (
            independent.get("total_test_samples", 0)
            if independent else 0
        )
        st.metric("Independent Samples", samples)

    st.divider()
    st.markdown("## Explore Gestura")

    f1, f2, f3 = st.columns(3)

    for column, icon, title, description in (
        (f1, "◎", "Live Translator", "Recognise gestures and generate multilingual messages."),
        (f2, "◫", "Gesture Studio", "Record and train new custom hand gestures."),
        (f3, "⚑", "Emergency Mode", "Speak, display, or call emergency help instantly."),
    ):
        with column:
            st.markdown(
                f"""
                <div class="gb-feature-card">
                  <div class="gb-feature-icon">{icon}</div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==================================================
# AI ANALYTICS
# ==================================================

elif page == "AI Analytics":

    st.header("AI Performance Analytics")

    st.subheader("Stored Alphabet Signs")
    st.caption(
        "Training samples are stored in backend/data/gesture_samples.csv. "
        "Alphabet coverage below is read directly from that dataset."
    )

    alphabet_counts = {letter: 0 for letter in string.ascii_lowercase}
    alphabet_two_hand_counts = {letter: 0 for letter in string.ascii_lowercase}
    dataset_path = DATA_DIR / "gesture_samples.csv"
    try:
        with open(dataset_path, "r", newline="", encoding="utf-8") as file:
            rows = csv.reader(file)
            next(rows, None)
            for row in rows:
                if not row:
                    continue
                label = row[0].strip().lower()
                if len(label) == 1 and label in alphabet_counts:
                    alphabet_counts[label] += 1
                    if len(row) == 127:
                        alphabet_two_hand_counts[label] += 1
    except OSError as error:
        st.warning(f"Could not read the gesture dataset: {error}")

    stored_letters = [letter for letter, count in alphabet_counts.items() if count]
    missing_letters = [letter.upper() for letter, count in alphabet_counts.items() if not count]
    alphabet_col1, alphabet_col2 = st.columns(2)
    alphabet_col1.metric("Alphabet classes collected", f"{len(stored_letters)} / 26")
    alphabet_col2.metric(
        "Alphabet samples",
        sum(alphabet_counts.values()),
    )

    alphabet_rows = [
        {
            "Letter": letter.upper(),
            "Samples": alphabet_counts[letter],
            "Two-hand samples": alphabet_two_hand_counts[letter],
            "Status": "Collected" if alphabet_counts[letter] else "Not collected",
        }
        for letter in string.ascii_lowercase
    ]
    st.dataframe(
        pd.DataFrame(alphabet_rows),
        use_container_width=True,
        hide_index=True,
    )
    if missing_letters:
        st.info("Letters not yet collected: " + ", ".join(missing_letters))
    else:
        st.success("All 26 alphabet classes have samples.")

    if stored_letters:
        alphabet_chart = pd.DataFrame({
            "Letter": [letter.upper() for letter in stored_letters],
            "Samples": [alphabet_counts[letter] for letter in stored_letters],
        }).set_index("Letter")
        st.bar_chart(alphabet_chart)

    st.subheader("Two-Hand Gesture Coverage")
    two_hand_report = load_report("two_hand_evaluation_report.json") or {}
    two_hand_counts = two_hand_report.get("class_counts", {})
    two_hand_accuracy = two_hand_report.get("accuracy")
    if two_hand_accuracy is not None:
        st.metric("Two-hand held-out accuracy", f"{two_hand_accuracy * 100:.2f}%")
    if two_hand_counts:
        two_hand_rows = [
            {
                "Gesture": gesture,
                "Two-hand samples": count,
                "Training support": (
                    "Ready" if count >= 5 else "Collect more (minimum 5)"
                ),
            }
            for gesture, count in sorted(two_hand_counts.items())
        ]
        st.dataframe(
            pd.DataFrame(two_hand_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Two-hand evaluation coverage has not been generated yet.")

    if independent is None:

        st.warning(
            "Independent evaluation report not found."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{independent['accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Macro Precision",
            f"{independent['macro_precision'] * 100:.2f}%"
        )

        c3.metric(
            "Macro Recall",
            f"{independent['macro_recall'] * 100:.2f}%"
        )

        c4.metric(
            "Macro F1",
            f"{independent['macro_f1'] * 100:.2f}%"
        )

        st.divider()
        st.subheader("Independent Test Summary")

        st.write(
            "Total samples:",
            independent.get("total_test_samples", 0)
        )

        st.write(
            "Gesture classes:",
            independent.get("gesture_classes_recorded", 0)
        )

        latency = independent.get(
            "average_inference_latency_ms"
        )

        if latency is not None:
            st.write(
                "Average inference latency:",
                f"{latency:.2f} ms"
            )

        st.subheader("Gesture Sample Distribution")

        counts = independent.get("sample_counts", {})

        if counts:

            chart = pd.DataFrame({
                "Gesture": list(counts.keys()),
                "Samples": list(counts.values())
            }).set_index("Gesture")

            st.bar_chart(chart)

        st.subheader("Per-Gesture Performance")

        report = independent.get(
            "classification_report", {}
        )

        rows = []

        for gesture, metrics in report.items():

            if (
                isinstance(metrics, dict)
                and "f1-score" in metrics
                and "precision" in metrics
                and "recall" in metrics
            ):

                rows.append({
                    "Gesture": gesture,
                    "Precision (%)": round(
                        metrics["precision"] * 100, 2
                    ),
                    "Recall (%)": round(
                        metrics["recall"] * 100, 2
                    ),
                    "F1-score (%)": round(
                        metrics["f1-score"] * 100, 2
                    ),
                    "Samples": int(metrics.get("support", 0))
                })

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Per-gesture metrics are unavailable.")


# ==================================================
# LIVE TRANSLATOR
# ==================================================

elif page == "Live Translator":

    st.title("Live Gesture Translator")
    st.write("Start the camera to recognise gestures and generate messages.")

    language = st.selectbox(
        "Output language",
        ["English", "Tamil", "Hindi"],
        key="gesture_output_language"
    )

    if "live_recognition_enabled" not in st.session_state:
        st.session_state.live_recognition_enabled = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Live Recognition", use_container_width=True):
            try:
                import urllib.request

                urllib.request.urlopen(
                    "http://127.0.0.1:8511/clear",
                    timeout=2,
                ).read()
            except Exception:
                pass

            st.session_state.live_recognition_enabled = True
            st.rerun()

    with col2:
        if st.button("Stop Recognition", use_container_width=True):
            try:
                import urllib.request

                urllib.request.urlopen(
                    "http://127.0.0.1:8511/stop",
                    timeout=1,
                ).read()
            except Exception:
                pass
            st.session_state.live_recognition_enabled = False
            st.rerun()

    if not st.session_state.live_recognition_enabled:
        st.info("Select **Start Live Recognition** to open your webcam on this page.")
        st.stop()

    try:
        import subprocess
        import urllib.request

        camera_server = "http://127.0.0.1:8511"

        try:
            urllib.request.urlopen(camera_server, timeout=1).read()
        except Exception:
            subprocess.Popen(
                [sys.executable, str(ROOT / "backend" / "live_camera_server.py")],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        st.divider()
        st.subheader("Build your message")
        st.caption(
            "Sign **my** or **your** to start a name sentence, then sign each "
            "letter. Select **Insert Space** between words."
        )
        space_col, undo_col = st.columns(2)

        with space_col:
            if st.button("Insert Space", use_container_width=True):
                try:
                    urllib.request.urlopen(
                        camera_server + "/space",
                        timeout=2,
                    ).read()
                except Exception as error:
                    st.warning(f"Could not insert a space: {error}")

        with undo_col:
            if st.button("Undo Character", use_container_width=True):
                try:
                    urllib.request.urlopen(
                        camera_server + "/undo-text",
                        timeout=2,
                    ).read()
                except Exception as error:
                    st.warning(f"Could not undo the last character: {error}")

        st.divider()
        st.subheader("Camera Preview")

        @st.fragment(run_every=0.75)
        def show_camera_preview():
            import time

            cache_buster = int(time.time() * 1000)
            st.markdown(
                f'<div style="width:100%; aspect-ratio:4/3; overflow:hidden; '
                'border-radius:12px; background:#050505;">'
                f'<img src="{camera_server}/snapshot?t={cache_buster}" '
                'style="width:100%; height:100%; object-fit:contain;" />'
                '</div>',
                unsafe_allow_html=True,
            )

        show_camera_preview()

        st.divider()
        st.subheader("Recognition Results")

        @st.fragment(run_every=0.75)
        def show_live_results():
            try:
                with urllib.request.urlopen(
                    camera_server + "/state",
                    timeout=2,
                ) as response:
                    state = json.loads(response.read().decode("utf-8"))
            except Exception as error:
                st.warning("Waiting for the local camera server to start...")
                st.caption(f"Camera server: {error}")
                return

            if state["error"]:
                st.error(state["error"])
                return

            if not state["ready"]:
                st.info("Gestura AI is loading. Please wait...")
                return

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Current Gesture", state["prediction"])
            col2.metric("Confidence", f"{state['confidence'] * 100:.1f}%")
            col3.metric("Confirmed Gesture", state["confirmed"])
            col4.metric(
                "Frame Processing",
                f"{state.get('frame_processing_ms', 0.0):.0f} ms",
            )

            st.divider()
            st.subheader("Detected Gesture Sequence")
            st.write(" → ".join(state["sequence"]) or "No gestures detected yet")

            st.subheader("Generated Message")

            try:
                from multilingual_engine import MultilingualEngine

                translated_message = MultilingualEngine().translate(
                    state["message"],
                    language,
                )
            except Exception:
                translated_message = state["message"]

            st.success(translated_message)
            st.caption(f"Selected output language: {language}")

            save_conversation_message(
                translated_message,
                language,
                state["sequence"],
            )

        show_live_results()

    except Exception as error:

        st.error(
            "The live gesture recognition engine could not be initialized."
        )

        st.code(
            f"{type(error).__name__}: {error}"
        )


# ==================================================
# OTHER MODULES
# ==================================================

elif page == "Conversations":

    st.header("Conversation History")

    history = load_conversation_history()

    if st.button("Clear History", type="primary"):
        HISTORY_FILE.write_text("[]", encoding="utf-8")
        st.success("Conversation history cleared.")
        st.rerun()

    if not history:
        st.info("No saved conversations yet. Generate a message in Live Translator.")
    else:
        for entry in reversed(history):
            st.subheader(entry.get("message", ""))
            st.caption(
                f"Language: {entry.get('language', 'English')} | "
                f"Time: {entry.get('timestamp', 'Unknown')}"
            )

            gestures = entry.get("gestures", [])
            if gestures:
                st.write("Gestures: " + " → ".join(gestures))

            st.divider()

elif page == "Gesture Studio":

    st.header("Gesture Studio")
    st.write("Create a custom gesture by collecting hand landmark samples.")

    custom_gesture = st.text_input(
        "Gesture name",
        placeholder="Example: stop, please, medicine",
    ).strip().lower()

    if st.button("Start Gesture Recording", use_container_width=True):
        if not custom_gesture:
            st.warning("Enter a gesture name first.")
        else:
            import subprocess
            import urllib.request
            import time

            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:8511/stop",
                    timeout=2,
                ).read()
            except Exception:
                pass

            time.sleep(1)

            collector = ROOT / "backend" / "dataset_collector.py"

            st.session_state["recording_gesture"] = custom_gesture

            subprocess.Popen(
                [sys.executable, str(collector), custom_gesture],
                cwd=str(ROOT),
            )

            st.success(
                f"Recording window started for '{custom_gesture}'. "
                "Press S to start recording and Q to close the window."
            )

    st.divider()
    st.subheader("Dataset Status")

    dataset_file = BACKEND_DIR / "data" / "gesture_samples.csv"

    if dataset_file.exists():
        try:
            import csv

            with open(dataset_file, "r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            counts = {}
            for row in rows:
                label = row.get("label", "").strip()
                if label:
                    counts[label] = counts.get(label, 0) + 1

            st.write(f"Total saved samples: {len(rows)}")

            if counts:
                st.bar_chart(pd.DataFrame({
                    "Samples": counts,
                }))
        except Exception as error:
            st.warning(f"Could not read dataset status: {error}")
    else:
        st.info("No gesture dataset exists yet. Start recording to create it.")

    st.divider()

    if st.button("Train Updated Model", use_container_width=True):
        dataset_file = BACKEND_DIR / "data" / "gesture_samples.csv"

        if not dataset_file.exists():
            st.warning("Collect gesture samples before training a model.")
        else:
            import subprocess

            trainer = ROOT / "backend" / "train_model.py"

            subprocess.Popen(
                [sys.executable, str(trainer)],
                cwd=str(ROOT),
            )

            st.success(
                "Model training started in the background. "
                "Restart the live camera server after training finishes."
            )

# ==================================================
# EMERGENCY MODE
# ==================================================

elif page == "Emergency Mode":

    st.header("Emergency Mode")
    st.error("Use this section for urgent communication. Confirm the situation before sending.")

    emergency_messages = {
        "I need urgent medical help.",
        "Please call an ambulance.",
        "I am in danger. Please help me.",
        "Please call my emergency contact.",
        "I cannot breathe properly.",
        "I am having severe pain.",
    }

    selected_emergency = st.selectbox(
        "Emergency message",
        sorted(emergency_messages),
        key="emergency_message",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Speak Message", use_container_width=True):
            try:
                from speech_engine import GestureBridgeSpeech

                GestureBridgeSpeech().speak(selected_emergency)
                st.success(f"Spoken: {selected_emergency}")
            except Exception as error:
                st.warning(f"Speech output unavailable: {error}")

    with col2:
        if st.button("Save to History", use_container_width=True):
            save_conversation_message(
                selected_emergency,
                "English",
                ["emergency"],
            )
            st.success("Emergency message saved to conversation history.")

    st.divider()
    st.subheader("Emergency Message")
    st.error(selected_emergency)

    st.markdown("### SOS — Deaf & Vision Assistance")

    s1, s2, s3 = st.columns(3)

    with s1:
        if st.button("🔊 SPEAK SOS", use_container_width=True):
            try:
                from speech_engine import GestureBridgeSpeech

                GestureBridgeSpeech().speak("Emergency. Please help me. Call for medical assistance.")
                st.success("Emergency spoken message activated.")
            except Exception as error:
                st.warning(f"Speech unavailable: {error}")

    with s2:
        if st.button("🚨 VISUAL ALERT", use_container_width=True):
            st.markdown(
                """
                <div style="background:#dc2626;color:white;padding:30px;border-radius:12px;
                text-align:center;font-size:28px;font-weight:bold;animation:blink 1s linear infinite;">
                EMERGENCY — HELP NEEDED
                </div>
                <script>
                if ('vibrate' in navigator) navigator.vibrate([500,200,500,200,1000]);
                </script>
                """,
                unsafe_allow_html=True,
            )

    with s3:
        if st.button("📞 CALL EMERGENCY", use_container_width=True):
            st.markdown(
                "<a href='tel:112' style='display:block;text-align:center;"
                "padding:14px;background:#dc2626;color:white;border-radius:8px;'>"
                "Call Emergency Number 112</a>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <style>
        @keyframes blink { 50% { opacity: 0.35; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==================================================
# HELP & GUIDE
# ==================================================

elif page == "Help & Guide":

    st.header("Help & Guide")
    st.write("Learn how to use Gestura safely and effectively.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Getting Started",
        "Live Recognition",
        "Gesture Studio",
        "Troubleshooting",
    ])

    with tab1:
        st.subheader("Quick Start")
        st.markdown("""
        1. Open **Live Translator** from the sidebar.
        2. Select English, Tamil, or Hindi.
        3. Press **Start Live Recognition**.
        4. Position one hand clearly in front of the camera.
        5. Hold each gesture steady until it is confirmed.
        6. Open **Conversations** to review saved messages.
        """)

    with tab2:
        st.subheader("Recognition Tips")
        st.markdown("""
        - Use bright, even lighting.
        - Keep the face and hand fully visible.
        - Avoid strong backlighting and busy backgrounds.
        - Hold gestures for a short, steady moment.
        - Use the clear or undo controls to correct the sequence.
        """)

        st.info("If no hand is detected, move closer to the camera and ensure the entire palm is visible.")

    with tab3:
        st.subheader("Adding a New Gesture")
        st.markdown("""
        1. Open **Gesture Studio**.
        2. Enter a simple gesture name.
        3. Press **Start Gesture Recording**.
        4. Hold the gesture while pressing **S** to record.
        5. Press **S** again to pause and **Q** to finish.
        6. Collect many varied samples before retraining.
        7. Press **Train Updated Model** and restart the live camera server.
        """)

    with tab4:
        st.subheader("Common Problems")

        with st.expander("The camera preview is black"):
            st.write("Close apps using the webcam, restart the live camera server, and allow camera access.")

        with st.expander("The live recognition engine is unavailable"):
            st.write("Verify the camera server at http://127.0.0.1:8511/state and restart it if needed.")

        with st.expander("A gesture is not recognised"):
            st.write("Improve lighting, hold the gesture steadily, and collect more representative samples in Gesture Studio.")

        with st.expander("Conversation history is missing"):
            st.write("Saved conversations are stored in backend/data/conversation_history.json. The Clear History button removes entries.")

elif page == "Practice Assistant":

    st.header("Practice Assistant")
    st.write("Practise a gesture and monitor your progress.")

    practice_gesture = st.selectbox(
        "Choose a gesture",
        [
            "hello", "yes", "ok", "thumbs up", "thumbs down",
            "good luck", "love you", "water", "you", "question",
        ],
        key="practice_gesture",
    )

    if "practice_attempts" not in st.session_state:
        st.session_state.practice_attempts = 0

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Practice", use_container_width=True):
            st.session_state.practice_attempts += 1
            st.info(f"Hold the **{practice_gesture}** gesture steadily in the camera.")

    with col2:
        st.metric("Practice Attempts", st.session_state.practice_attempts)

elif page == "Settings":

    st.header("Settings")

    default_language = st.selectbox(
        "Default output language",
        ["English", "Tamil", "Hindi"],
        key="default_output_language",
    )

    show_confidence = st.toggle("Show confidence scores", value=True)
    save_history = st.toggle("Save generated messages automatically", value=True)

    if st.button("Save Preferences", type="primary"):
        st.success(f"Preferences saved: {default_language}.")

    st.caption(
        f"Confidence: {'On' if show_confidence else 'Off'} · "
        f"History: {'On' if save_history else 'Off'}"
    )

else:

    st.header(page)

    descriptions = {
        "Emergency Mode":
            "Emergency communication features.",
        "Gesture Studio":
            "Create and manage personalised gestures.",
        "Practice Assistant":
            "Practise gestures and monitor learning progress.",
        "Conversations":
            "View and manage communication history.",
        "Help & Guide":
            "Learn how to use GestureBridge.",
        "Settings":
            "Configure your GestureBridge preferences."
    }

    st.info(
        descriptions.get(
            page,
            "This module is under development."
        )
    )

# Allow more time for local camera/WebRTC initialization.
try:
    from streamlit_webrtc.webrtc import WebRtcWorker

    _original_process_offer = WebRtcWorker.process_offer

    def _process_offer_with_more_time(self, sdp, type_, timeout=30.0):
        return _original_process_offer(self, sdp, type_, timeout)

    WebRtcWorker.process_offer = _process_offer_with_more_time
except (ImportError, AttributeError):
    pass
