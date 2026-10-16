import numpy as np

FEATURE_DIM = 351  # pose 99 + hands (body-relative) 126 + hands (wrist-relative) 126


def normalize_sequence(seq):
    """(T, 258) raw MediaPipe landmarks -> (T, 351) body-normalized features.

    Raw layout per frame: pose 33*(x,y,z,vis)=132 | left hand 21*3=63 | right hand 21*3=63.
    Everything is re-centred on the shoulder midpoint and divided by shoulder width,
    so position in frame and distance from camera stop mattering.
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    out = np.zeros((T, FEATURE_DIM), np.float32)
    for t in range(T):
        pose = seq[t, :132].reshape(33, 4)[:, :3]
        if not pose.any():
            continue  # no body detected -> stays a zero (masked) frame
        ls, rs = pose[11], pose[12]
        center = (ls + rs) / 2
        scale = float(np.linalg.norm((ls - rs)[:2]))
        if scale < 1e-3:
            scale = 1.0
        parts = [((pose - center) / scale).ravel()]
        hands = (seq[t, 132:195].reshape(21, 3), seq[t, 195:258].reshape(21, 3))
        for hand in hands:
            parts.append(((hand - center) / scale).ravel() if hand.any() else np.zeros(63, np.float32))
        for hand in hands:
            parts.append(((hand - hand[0]) / scale).ravel() if hand.any() else np.zeros(63, np.float32))
        out[t] = np.concatenate(parts)
    return out


def pooled_features(seq):
    """(T, 258) raw landmarks -> (702,) mean+std of normalized frames. Used by the baseline.
    A single frame (1, 258) works too: the std half is simply zeros."""
    n = normalize_sequence(seq)
    return np.concatenate([n.mean(axis=0), n.std(axis=0)]).astype(np.float32)