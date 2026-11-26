
class MultilingualEngine:

    def __init__(self):

        self.translations = {
            "Are you okay?": {
                "English": "Are you okay?",
                "Tamil": "நீங்கள் நலமாக இருக்கிறீர்களா?",
                "Hindi": "क्या आप ठीक हैं?"
            },

            "Hello! How are you?": {
                "English": "Hello! How are you?",
                "Tamil": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
                "Hindi": "नमस्ते! आप कैसे हैं?"
            },

            "I would like some water.": {
                "English": "I would like some water.",
                "Tamil": "எனக்கு கொஞ்சம் தண்ணீர் வேண்டும்.",
                "Hindi": "मुझे थोड़ा पानी चाहिए।"
            },

            "I agree!": {
                "English": "I agree!",
                "Tamil": "நான் ஒப்புக்கொள்கிறேன்!",
                "Hindi": "मैं सहमत हूँ!"
            },

            "I disagree.": {
                "English": "I disagree.",
                "Tamil": "நான் உடன்படவில்லை.",
                "Hindi": "मैं सहमत नहीं हूँ।"
            },

            "I wish you good luck.": {
                "English": "I wish you good luck.",
                "Tamil": "உங்களுக்கு நல்வாழ்த்துகள்!",
                "Hindi": "आपको शुभकामनाएँ!"
            }
        }

    def translate(self, message, language="English"):

        if language not in ["English", "Tamil", "Hindi"]:
            return "Unsupported language."

        result = self.translations.get(message)

        if result:
            return result[language]

        return message if language == "English" else (
            "Translation not available for this phrase."
        )


if __name__ == "__main__":

    engine = MultilingualEngine()

    test_message = "Are you okay?"

    for language in ["English", "Tamil", "Hindi"]:

        translated = engine.translate(
            test_message,
            language
        )

        print(language, ":", translated)

