import numpy as np

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize 3D hand landmarks:
    - Translate so wrist (landmark 0) is at origin
    - Scale based on distance from wrist to middle finger MCP (landmark 9)
    - Flatten to 1D vector of length 63

    Args:
        landmarks (np.ndarray): (21,3) array of 3D landmarks

    Returns:
        np.ndarray: Flattened normalized landmarks (63,)

    """
    if landmarks.shape != (21, 3):
        raise ValueError("Expected landmarks shape (21,3), got {}".format(landmarks.shape))

    # Translation: subtract wrist coordinates
    wrist = landmarks[0]
    translated = landmarks - wrist

    # Scale: distance wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(translated[9])
    if scale < 1e-6:
        scale = 1.0  # avoid division by zero, no scaling

    normalized = translated / scale
    return normalized.flatten()