import unittest

from gesture_sequence import GestureSequenceDetector


class GestureConfirmationTests(unittest.TestCase):
    def test_low_confidence_predictions_are_not_confirmed(self):
        detector = GestureSequenceDetector(
            history_size=10,
            minimum_votes=7,
            confidence_threshold=0.55,
        )

        events = [
            detector.update("your", 0.36, hand_present=True)
            for _ in range(20)
        ]

        self.assertTrue(all(event is None for event in events))

    def test_stable_confident_sign_is_confirmed(self):
        detector = GestureSequenceDetector(
            history_size=10,
            minimum_votes=7,
            confidence_threshold=0.55,
        )

        events = [
            detector.update("hello", 0.92, hand_present=True)
            for _ in range(10)
        ]

        self.assertEqual(events[-1], "hello")
        self.assertEqual(sum(event is not None for event in events), 1)


if __name__ == "__main__":
    unittest.main()