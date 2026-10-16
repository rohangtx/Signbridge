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


def augment_sequence(seq, jitter=True, scale=True, rotate=True, speed=True, drop_frame=True):
    """Augment a normalized (T, 351) landmark sequence.
    Coordinates are arranged as 117 3D points (x, y, z).
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    points = seq.reshape(T, 117, 3).copy()
    active = np.abs(points).sum(axis=(1, 2)) > 1e-4
    if not np.any(active):
        return seq

    if scale:
        s = np.random.uniform(0.92, 1.08)
        points[active] *= s

    if rotate:
        theta = np.radians(np.random.uniform(-8.0, 8.0))
        c, s = np.cos(theta), np.sin(theta)
        x = points[:, :, 0] * c - points[:, :, 1] * s
        y = points[:, :, 0] * s + points[:, :, 1] * c
        points[:, :, 0] = x
        points[:, :, 1] = y

    if jitter:
        noise = np.random.normal(0.0, 0.005, points.shape).astype(np.float32)
        noise[~active] = 0.0
        points += noise

    res = points.reshape(T, 351).astype(np.float32)

    if speed and T > 4 and np.sum(active) > 4:
        factor = np.random.uniform(0.85, 1.15)
        orig_indices = np.arange(T)
        new_indices = np.clip(np.linspace(0, (T - 1) * factor, T), 0, T - 1)
        res = np.array([np.interp(new_indices, orig_indices, res[:, d]) for d in range(351)]).T.astype(np.float32)

    if drop_frame and np.sum(active) > 6:
        active_indices = np.where(active)[0]
        n_drop = np.random.randint(1, min(3, len(active_indices)))
        drop_idx = np.random.choice(active_indices, size=n_drop, replace=False)
        res[drop_idx] = 0.0

    return res