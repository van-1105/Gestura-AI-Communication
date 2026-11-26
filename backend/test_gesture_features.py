import unittest

from gesture_features import (
    FEATURES_FOR_TWO_HANDS,
    FEATURES_PER_HAND,
    normalize_hands,
)


def make_hand(offset_x=0.0):
    return [
        [offset_x + index / 20, index / 40, index / 60]
        for index in range(21)
    ]


class GestureFeatureTests(unittest.TestCase):
    def test_single_hand_is_zero_padded_for_two_hand_model(self):
        features = normalize_hands([make_hand()])

        self.assertEqual(len(features), FEATURES_FOR_TWO_HANDS)
        self.assertEqual(features[FEATURES_PER_HAND:], [0.0] * FEATURES_PER_HAND)

    def test_two_hands_produce_combined_features(self):
        features = normalize_hands([make_hand(), make_hand(2.0)])

        self.assertEqual(len(features), FEATURES_FOR_TWO_HANDS)
        self.assertNotEqual(features[FEATURES_PER_HAND:], [0.0] * FEATURES_PER_HAND)

    def test_legacy_single_hand_shape_remains_available(self):
        features = normalize_hands([make_hand()], pad_to_two=False)

        self.assertEqual(len(features), FEATURES_PER_HAND)

    def test_invalid_hand_is_ignored(self):
        self.assertIsNone(normalize_hands([[ [0.0, 0.0, 0.0] ]]))


if __name__ == "__main__":
    unittest.main()