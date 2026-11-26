class GestureIntentEngine:

    def __init__(self):
        self.single_gesture_meanings = {
            "hello": "Hello!",
            "greetings": "Hello!",
            "yes": "Yes.",
            "ok": "Okay.",
            "thumbs up": "I agree!",
            "thumbs down": "I disagree.",
            "good luck": "Good luck!",
            "love you": "I love you.",
            "water": "I would like some water.",
            "you": "You.",
            "question": "I have a question.",
            "pointing left": "Please look to the left.",
            "pointing right": "Please look to the right.",
            "my": "My",
            "your": "Your",
            "name": "name",
            "please": "Please.",
        }

    def interpret(self, sequence):

        if not sequence:
            return "Perform a gesture to begin."

        # Convert labels to lowercase
        words = [str(x).strip().lower() for x in sequence]

        # Remove consecutive duplicate gestures
        cleaned = []
        for word in words:
            if not cleaned or cleaned[-1] != word:
                cleaned.append(word)

        # Context-aware question interpretation
        if (
            "you" in cleaned
            and "ok" in cleaned
            and "question" in cleaned
        ):
            return "Are you okay?"

        if "you" in cleaned and "ok" in cleaned:
            return "Are you okay?"

        # Other sequence interpretations
        if "hello" in cleaned or "greetings" in cleaned:
            if "you" in cleaned:
                return "Hello! How are you?"

            return "Hello!"

        if "water" in cleaned:
            return "I would like some water."

        if "love you" in cleaned:
            return "I love you."

        if "good luck" in cleaned:
            return "I wish you good luck."

        if "thumbs up" in cleaned:
            return "I agree!"

        if "thumbs down" in cleaned:
            return "I disagree."

        if "yes" in cleaned:
            return "Yes."

        if "question" in cleaned:
            return "I have a question."

        # Fall back to individual gesture meanings
        meanings = [
            self.single_gesture_meanings[word]
            for word in cleaned
            if word in self.single_gesture_meanings
        ]

        if meanings:
            return " ".join(meanings)

        return "Message could not be interpreted yet."


if __name__ == "__main__":

    engine = GestureIntentEngine()

    test_sequences = [
        ["you", "ok", "question"],
        ["hello", "you"],
        ["water"],
        ["thumbs up"],
        ["thumbs down"],
        ["good luck"]
    ]

    print("\nGESTUREBRIDGE INTENT ENGINE TEST\n")

    for sequence in test_sequences:
        result = engine.interpret(sequence)

        print("Gestures:", " + ".join(sequence))
        print("Message:", result)
        print("-" * 40)
