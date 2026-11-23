"""
test_pairs.py — test pairs with a known answer, for E1a and E1b

Makes a downscaled or rotated copy of one of your frames together with the exact 3x3 transform
that maps the original onto the copy, and scores your matches against it. Detection, description
and matching stay yours.

    from test_pairs import downscaled_copy, rotated_copy, correct_matches

    small, T = downscaled_copy(gray, 2.0)            # or: rotated, T = rotated_copy(gray, 30)
    xy_a, desc_a = my_sift(gray)                     # your detector + descriptor
    xy_b, desc_b = my_sift(small)
    matches = my_ratio_test(desc_a, desc_b)          # (K, 2) index pairs
    ok = correct_matches(xy_a, xy_b, matches, T)     # boolean per match
    print(ok.sum(), ok.mean())                       # correct matches, precision

Use the same frame, the same factors and angles, and the same matcher settings for every
version you compare in E1b, so that only the change you made differs between rows.
"""
import numpy as np
import cv2

FACTORS = (1.5, 2.0, 3.0)
ANGLES = (15, 30, 90)


def downscaled_copy(gray, factor):
    """`gray` reduced by `factor` with an area filter.

    Returns (small, T): T is the 3x3 matrix taking a point (x, y, 1) of `gray` to `small`, with
    pixel centres at integer coordinates. The scale in T is the one actually applied after the
    size is rounded to whole pixels.
    """
    h, w = gray.shape[:2]
    nw, nh = int(round(w / factor)), int(round(h / factor))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = nw / w, nh / h
    # pixel centres sit at integer coordinates, so the copy is aligned at the outer edges of the
    # grids (-0.5 and w - 0.5), as in the lecture on resizing: x' = s (x + 0.5) - 0.5
    T = np.array([[sx, 0.0, 0.5 * sx - 0.5],
                  [0.0, sy, 0.5 * sy - 0.5],
                  [0.0, 0.0, 1.0]])
    return small, T


def rotated_copy(gray, degrees):
    """`gray` rotated by `degrees` (counter-clockwise on screen) about its centre, same size.

    Returns (rotated, T): T is the 3x3 matrix taking a point of `gray` to `rotated`.
    Corners rotated out of the frame are lost and filled with black, so key-points found near
    the black border of `rotated` have no partner in `gray`; that is expected.
    """
    h, w = gray.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    rotated = cv2.warpAffine(gray, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    T = np.vstack([A, [0.0, 0.0, 1.0]])
    return rotated, T


def correct_matches(xy_a, xy_b, matches, T, tol=2.0):
    """Which of your matches agree with the known transform T (from the functions above).

    xy_a, xy_b : (N, 2), (M, 2) key-point positions (x, y) in the original and in the copy
    matches    : (K, 2) integer index pairs (i into xy_a, j into xy_b) from your matcher
    T          : 3x3 transform from downscaled_copy or rotated_copy
    tol        : allowed distance in pixels of the copy

    Returns a boolean array of length K. Report both the number of correct matches (ok.sum())
    and the precision (ok.mean()): some changes raise one and lower the other.
    """
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    if len(matches) == 0:
        return np.zeros(0, bool)
    p = np.asarray(xy_a, dtype=np.float64)[matches[:, 0]]
    q = (np.c_[p, np.ones(len(p))] @ np.asarray(T, dtype=np.float64).T)
    q = q[:, :2] / q[:, 2:3]
    return np.linalg.norm(q - np.asarray(xy_b, dtype=np.float64)[matches[:, 1]], axis=1) < tol


if __name__ == "__main__":
    # Self-check of THIS FILE only: a reference detector/descriptor and a throw-away matcher are
    # used here to confirm that the transforms and the scoring are right. Do not use them in E1b.
    rng = np.random.default_rng(0)
    img = np.full((480, 640), 120, np.uint8)
    for _ in range(150):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 440))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 60)), y + int(rng.integers(8, 60))), int(rng.integers(20, 235)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    ref = cv2.SIFT_create()

    def feats(g):
        k, d = ref.detectAndCompute(g, None)
        return np.array([p.pt for p in k]), d.astype(np.float32)

    def match(da, db, r=0.8):
        d = np.linalg.norm(da[:, None] - db[None], axis=2)
        o = np.argsort(d, 1)[:, :2]
        i = np.arange(len(d))
        keep = d[i, o[:, 0]] < r * d[i, o[:, 1]]
        return np.c_[i[keep], o[keep, 0]]

    xa, da = feats(img)
    for name, (copy, T) in [(f"downscaled {f}x", downscaled_copy(img, f)) for f in FACTORS] + \
                           [(f"rotated {a} deg", rotated_copy(img, a)) for a in ANGLES]:
        xb, db = feats(copy)
        ok = correct_matches(xa, xb, match(da, db), T)
        print(f"self-check {name:16s}: {ok.sum():4d} correct of {len(ok):4d} ({ok.mean():.0%})")
    print("A clear majority correct on every line means the transforms are right.")
