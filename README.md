# Gestura – AI-Powered Intelligent Multilingual Communication Platform

Gestura is an AI-powered gesture communication platform designed to make communication more accessible by converting recognised hand gestures into meaningful text. It combines computer vision, machine learning, and context-aware interpretation to support more natural communication.

## Live Demo

**Streamlit Community Cloud:** [View Gestura on Streamlit](https://share.streamlit.io/user/van-1105)

## Project Overview

Gestura captures hand movements through a webcam, identifies hand landmarks, recognises gestures, and interprets gesture sequences to generate meaningful text.

The platform aims to bridge communication gaps between sign-language users and people unfamiliar with sign language.

## Key Features

- **Real-Time Gesture Recognition:** Detects and recognises hand gestures using webcam input.
- **Hand Landmark Detection:** Extracts 21 hand landmarks using MediaPipe.
- **Context-Aware Interpretation:** Uses gesture sequences and contextual rules to generate meaningful phrases.
- **Alphabet Recognition:** Supports recognition of hand signs representing letters.
- **Multi-Hand Support:** Designed to process gestures involving up to two hands.
- **Text-Based Communication:** Converts recognised gestures into readable text.
- **AI Analytics:** Provides insights into gesture recognition and model performance.
- **Offline Recognition:** Supports local gesture recognition without requiring continuous internet access.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Machine Learning | Random Forest Classifier |
| User Interface | Streamlit |
| Data Processing | NumPy, Pandas |
| Model Storage | Pickle |

## How It Works

1. **Camera Input:** Captures live hand gestures through a webcam.
2. **Landmark Extraction:** MediaPipe identifies 21 key points on each hand.
3. **Feature Processing:** Extracts landmark coordinates for gesture classification.
4. **Gesture Recognition:** The trained Random Forest model predicts the corresponding gesture.
5. **Context Interpretation:** Processes recognised gestures and their sequence to generate meaningful phrases.
6. **Text Output:** Displays the interpreted communication in the application.

## Model Evaluation

The project was evaluated using an independently collected test set.

| Evaluation Metric | Result |
|---|---:|
| Gesture Samples | 1,810 |
| Gesture Classes | 14 |
| Independent Test Accuracy | 86% |

*These results represent the tested gesture classes and evaluation conditions. Performance may vary with different users, lighting conditions, camera positions, and hand movements.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/van-1105/Gestura-AI-Communication.git
```

### 2. Navigate to the Project

```bash
cd Gestura-AI-Communication
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Allow webcam access when prompted or when the camera window opens.

## Project Structure

The project is organised into modules for the application interface, gesture recognition, models, data, and supporting resources.

```text
Gestura/
│
├── app.py
├── backend/
│   ├── models/
│   ├── data/
│   └── realtime_recognition.py
│
├── frontend/
├── database/
├── documentation/
├── requirements.txt
└── README.md
```

*The structure above is a simplified overview. Actual folders and files may vary by project version.*

## Applications

- Communication assistance for deaf and hard-of-hearing individuals.
- Sign-language learning and practice.
- Accessibility support in educational environments.
- Potential communication assistance in public service centres and workplaces.

## Current Limitations

- Recognition is limited to the gestures included in the trained dataset.
- Accuracy can be affected by lighting, camera quality, and hand positioning.
- Gesture interpretation depends on the supported gesture sequences and contextual rules.
- Gestura is a prototype and is not a replacement for qualified sign-language interpreters.

## Future Enhancements

- Expand the gesture dataset with more signs and users.
- Improve recognition of dynamic gestures and continuous signing.
- Enhance multilingual text and speech output.
- Improve context-aware interpretation and gesture correction.
- Extend accessibility support across different communication environments.

## Project Information

**Project:** Gestura – AI-Powered Intelligent Multilingual Communication Platform

**Domain:** Artificial Intelligence, Machine Learning and Computer Vision

**Developed using:** Python, OpenCV, MediaPipe, Random Forest and Streamlit

**GitHub Repository:** [Gestura – GitHub](https://github.com/van-1105/Gestura-AI-Communication)

**Streamlit Profile:** [View deployed applications](https://share.streamlit.io/user/van-1105)

---

*Gestura aims to make communication more accessible through intelligent gesture recognition and interpretation.*

*Gestura aims to make communication more accessible through intelligent gesture recognition and interpretation.*
