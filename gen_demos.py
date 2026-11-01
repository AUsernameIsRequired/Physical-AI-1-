"""エキスパート(IK)デモ生成: 入力 (x, y, 色one-hot) → 出力 28次元キーポーズ"""
import sys
import numpy as np
from common import *

def main(n=600, seed=1, out="data/demos.npz"):
    import os
    os.makedirs("data", exist_ok=True)
    env = Env(seed=seed)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    tries = 0
    while len(X) < n:
        tries += 1
        xy = (rng.uniform(*WS_X), rng.uniform(*WS_Y))
        c = COLORS[rng.integers(3)]
        y = env.expert_keyposes(xy, c)
        if y is None:
            continue
        onehot = np.eye(3)[COLORS.index(c)]
        X.append([xy[0], xy[1], *onehot])
        Y.append(y)
    np.savez(out, X=np.array(X), Y=np.array(Y))
    print(f"demos: {len(X)} (tries {tries}) -> {out}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
