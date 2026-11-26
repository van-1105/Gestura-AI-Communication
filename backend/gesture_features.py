LANDMARKS_PER_HAND = 21
VALUES_PER_POINT = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * VALUES_PER_POINT
FEATURES_FOR_TWO_HANDS = FEATURES_PER_HAND * 2


def normalize_hands(hands, pad_to_two=True):
    """Normalize up to two hands using the dataset's wrist-relative scaling."""
    normalized_hands = []
    for hand in list(hands or [])[:2]:
        points = [
            [
                float(point.x if hasattr(point, "x") else point[0]),
                float(point.y if hasattr(point, "y") else point[1]),
                float(point.z if hasattr(point, "z") else point[2]),
            ]
            for point in hand
        ]
        if len(points) != LANDMARKS_PER_HAND:
            continue
        normalized_hands.append(points)

    if not normalized_hands:
        return None

    wrist = normalized_hands[0][0]
    relative = [
        [value - wrist[axis] for axis, value in enumerate(point)]
        for hand in normalized_hands
        for point in hand
    ]
    scale = max(abs(value) for point in relative for value in point)
    if scale <= 0:
        return None

    features = [
        value / scale
        for point in relative
        for value in point
    ]

    if pad_to_two and len(normalized_hands) == 1:
        features.extend([0.0] * FEATURES_PER_HAND)

    return features