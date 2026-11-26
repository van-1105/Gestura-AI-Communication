
import sounddevice as sd
import numpy as np

MIC_INDEX = 13
SAMPLE_RATE = 48000

print("Using laptop built-in microphone...")

try:
    audio = sd.rec(
        int(5 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=MIC_INDEX
    )

    print("Speak now for 5 seconds...")
    sd.wait()

    level = np.abs(audio.astype(np.float32)).mean()

    print("Audio level:", round(float(level), 2))

    if level > 10:
        print("BUILT-IN MICROPHONE WORKING!")
    else:
        print("Audio level is low. Check microphone permissions.")

except Exception as error:
    print("Microphone error:", error)
