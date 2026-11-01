"""模倣学習(Behavior Cloning): MLPでキーポーズを回帰。データ量別の誤差も出力"""
import json, pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

d = np.load("data/demos.npz")
X, Y = d["X"], d["Y"]
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)

def fit(n):
    m = make_pipeline(StandardScaler(),
                      MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu", solver="adam",
                                   learning_rate_init=1e-3, max_iter=3000, tol=1e-9,
                                   n_iter_no_change=60, random_state=0))
    m.fit(Xtr[:n], Ytr[:n])
    err = np.abs(m.predict(Xte) - Yte)
    return m, float(np.degrees(err.mean())), float(np.degrees(err.max()))

res = {}
for n in [25, 50, 100, 200, len(Xtr)]:
    m, mae, mx = fit(n)
    res[n] = {"mae_deg": mae, "max_deg": mx}
    print(f"n={n:4d}  関節角MAE={mae:.3f} deg  max={mx:.2f} deg")
pickle.dump(m, open("data/policy.pkl", "wb"))
json.dump(res, open("data/train_results.json", "w"), indent=1)
