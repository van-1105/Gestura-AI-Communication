import unittest

from text_composer import GestureTextComposer


class GestureTextComposerTests(unittest.TestCase):
    def setUp(self):
        self.composer = GestureTextComposer()

    def test_letters_combine_into_word_until_space(self):
        for letter in "sam":
            self.composer.apply_gesture(letter)

        self.assertEqual(self.composer.text, "SAM")
        self.composer.insert_space()
        for letter in "lee":
            self.composer.apply_gesture(letter)

        self.assertEqual(self.composer.text, "SAM LEE")

    def test_my_and_your_prefixes_create_name_sentences(self):
        self.composer.apply_gesture("my")
        for letter in "ana":
            self.composer.apply_gesture(letter)
        self.assertEqual(self.composer.text, "My name is ANA")

        self.composer.clear()
        self.composer.apply_gesture("your")
        for letter in "ana":
            self.composer.apply_gesture(letter)
        self.assertEqual(self.composer.text, "Your name is ANA")

    def test_name_sign_does_not_duplicate_prefix(self):
        self.composer.apply_gesture("my")
        self.composer.apply_gesture("name")
        self.assertEqual(self.composer.text, "My name is ")

    def test_backspace_removes_last_character(self):
        self.composer.apply_gesture("a")
        self.composer.apply_gesture("b")
        self.assertEqual(self.composer.backspace(), "A")


if __name__ == "__main__":
    unittest.main()