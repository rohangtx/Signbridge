import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

FALLBACK_GROUP = 'unknown::unknown'


def split_data(X, y, groups, test_size=.2, random_state=42, attempts=50):
    y = np.asarray(y)
    has_real_groups = (
        groups is not None
        and len(set(groups)) > 1
        and any(g != FALLBACK_GROUP for g in groups)
    )

    if has_real_groups:
        all_classes = set(y.tolist())
        last = None
        for i in range(max(1, attempts)):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=random_state + i)
            a, b = next(splitter.split(X, y, groups))
            last = (a, b)
            if set(y[a].tolist()) == all_classes:   # every class present in train
                return a, b, 'signer-aware group split (category-based for some sources)'
        print(f"WARNING: no group split in {attempts} attempts put every class in "
              f"training; using the last attempt anyway — some classes may be "
              f"missing from train.")
        a, b = last
        return a, b, 'signer-aware group split (unbalanced after retries, category-based for some sources)'

    counts = {c: int((y == c).sum()) for c in set(y.tolist())}
    too_small = [c for c, n in counts.items() if n < 2]

    if too_small:
        print(f"WARNING: classes with fewer than 2 samples can't be stratified "
              f"and will be excluded from the split evaluation: {too_small}.")
        keep = np.array([c not in too_small for c in y])
        idx = np.arange(len(y))[keep]
        a_rel, b_rel = train_test_split(np.arange(len(idx)), test_size=test_size,
                                         random_state=random_state, stratify=y[keep])
        singleton_idx = np.arange(len(y))[~keep]
        a = np.concatenate([idx[a_rel], singleton_idx])
        b = idx[b_rel]
        return a, b, 'stratified split (singleton classes train-only, excluded from test)'

    a, b = train_test_split(np.arange(len(y)), test_size=test_size,
                             random_state=random_state, stratify=y)
    return a, b, 'stratified sample split (no usable signer metadata)'