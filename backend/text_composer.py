class GestureTextComposer:
    """Build text from alphabet signs and optional word-prefix signs."""

    def __init__(self):
        self.text = ""

    def apply_gesture(self, gesture):
        label = str(gesture).strip().lower()

        if len(label) == 1 and label.isalpha():
            self.text += label.upper()
            return self.text

        if label in {"my", "your"}:
            if self.text and not self.text.endswith(" "):
                self.text += " "
            pronoun = "My" if label == "my" else "Your"
            self.text += f"{pronoun} name is "
            return self.text

        if label == "name":
            if self.text.rstrip().lower().endswith("name is"):
                return self.text
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += "name is "
            return self.text

        return None

    def insert_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        return self.text

    def backspace(self):
        self.text = self.text[:-1]
        return self.text

    def clear(self):
        self.text = ""
        return self.text