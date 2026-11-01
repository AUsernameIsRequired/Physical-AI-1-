"""成功率評価: エキスパート(IK) vs 学習ポリシー(知覚→MLP→実行)"""
import sys, json, pickle
import numpy as np
from common import *

def run_episode(env, policy=None):
    env.reset_arm()
    env.spawn_cubes()
    est = env.perceive()
    perc_err = [math.dist(est[c], p.getBasePositionAndOrientation(env.cubes[c])[0][:2])
                for c in est if c in env.cubes]
    for c in COLORS:
        if c not in est:
            continue
        if policy is None:
            kp = env.expert_keyposes(est[c], c)
        else:
            kp = policy.predict([[est[c][0], est[c][1], *np.eye(3)[COLORS.index(c)]]])[0]
        if kp is None:
            continue
        env.execute(kp)
    env.go_home()
    for _ in range(60):
        env.step()
    return [env.is_sorted(c) for c in COLORS], perc_err

if __name__ == "__main__":
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    policy = pickle.load(open("data/policy.pkl", "rb"))
    out = {}
    for name, pol in [("expert_IK", None), ("learned_MLP", policy)]:
        env = Env(seed=123)  # 同一シードで同一配置を比較
        ok, pe = [], []
        for e in range(n_ep):
            r, perr = run_episode(env, pol)
            ok += r; pe += perr
            print(f"{name} ep{e}: {r}", flush=True)
        out[name] = {"cube_success": float(np.mean(ok)), "n_cubes": len(ok),
                     "episode_all3": None,
                     "perception_err_mm": float(np.mean(pe) * 1000)}
        print(name, out[name], flush=True)
        p.disconnect()
    json.dump(out, open("data/eval_results.json", "w"), indent=1)
