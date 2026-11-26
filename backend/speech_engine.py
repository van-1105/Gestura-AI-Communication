
import pyttsx3


class GestureBridgeSpeech:

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 155)
        self.engine.setProperty("volume", 1.0)

    def speak(self, message):
        if message:
            self.engine.say(message)
            self.engine.runAndWait()


if __name__ == "__main__":
    speaker = GestureBridgeSpeech()

    speaker.speak(
        "Hello! Welcome to GestureBridge. "
        "Are you okay?"
    )

    print("Speech output completed.")
