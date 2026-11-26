
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = "backend/models/vosk-model-small-en-in-0.4"

audio_queue = queue.Queue()


def find_microphone():
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        if (
            "Microphone Array" in device["name"]
            and device["max_input_channels"] > 0
        ):
            return index, device

    # Fallback to Windows default input device
    default_input = sd.default.device[0]

    if default_input is not None and default_input >= 0:
        device = sd.query_devices(default_input)

        if device["max_input_channels"] > 0:
            return default_input, device

    raise RuntimeError("No suitable microphone found.")


def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)

    audio_queue.put(bytes(indata))


def start_speech_recognition():

    print("Loading offline speech model...")

    model = Model(MODEL_PATH)

    mic_index, mic = find_microphone()

    sample_rate = int(mic["default_samplerate"])
    channels = 1

    print("\nSelected microphone:", mic["name"])
    print("Device index:", mic_index)
    print("Sample rate:", sample_rate)

    recognizer = KaldiRecognizer(model, sample_rate)

    print("\nGestureBridge Speech Recognition Ready!")
    print("Speak normally into your laptop.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.RawInputStream(
            device=mic_index,
            samplerate=sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=channels,
            callback=audio_callback
        ):

            print("Listening...\n")

            while True:
                audio_data = audio_queue.get()

                if recognizer.AcceptWaveform(audio_data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")

                    if text:
                        print("Recognised:", text)

    except KeyboardInterrupt:
        print("\nSpeech recognition stopped.")

    except Exception as error:
        print("Speech recognition error:", error)


if __name__ == "__main__":
    start_speech_recognition()
