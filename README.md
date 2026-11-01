# 色別仕分けピック&プレース(PyBullet + Franka Panda)

机上にランダム配置された赤・緑・青のキューブ3個を、ロボットアーム(Franka Panda)が色に対応するパッドに動かす。
実行時は仮想カメラの画像から位置を推定し、模倣学習で得たMLPがキーポーズを出力して動かす。

---

## Files Overview

### 0. 共通モジュール — `common.py`

- 環境(Panda・机・色パッド・キューブ)、知覚、IK、動作実行をまとめたモジュール。

---

### 1. デモ生成 — `gen_demos.py`

- ランダムな (x, y, 色) に対し、IKでキーポーズを計算して「お手本データ」を作る。
- 物理シミュレーションは進めず、IKの計算だけなので数秒で終わる。
- 引数: エピソード数
- 出力: `data/demos.npz`(入力 X は5次元、出力 Y は 4ポーズ×7関節=28次元)

```bash
python gen_demos.py 800
```

---

### 2. 模倣学習 — `train.py`

- `data/demos.npz` から MLP(隠れ層 256×256、scikit-learn)を学習し、(x, y, 色) → 28次元のキーポーズを回帰する。
- デモ数ごとのテスト誤差も出力する。
- 出力: `data/policy.pkl`、`data/train_results.json`

```bash
python train.py
```

---

### 3. 評価 — `evaluate.py`

- ランダム配置のエピソードで、エキスパート(IK)と学習ポリシーの成功率を同じ配置で比較する。
- 流れ: 配置 → カメラで認識 → キーポーズを決定 → 実行。成功は「キューブが同色パッド上に静止」。
- 引数: エピソード数
- 出力: `data/eval_results.json`

```bash
python evaluate.py 50
```

---

## Setup and Run

1. 環境構築する。

```bash
pip install -r requirements.txt
```

2. 順番に実行する。

2-1. 初回のみ
```bash
python gen_demos.py 800
python train.py
```

   `data/policy.pkl` が生成される。

2-2. 推論実行
```bash
python evaluate.py 50
```

3. モデル(アセット):

   * Franka Panda(`franka_panda/panda.urdf`)、机、床は `pybullet_data` パッケージ同梱のもの。
   1. の `pybullet` でインストールされる。

---

## 結果(50エピソード=キューブ150個、同一配置で比較)

| 手法 | 仕分け成功 |
|---|---|
| エキスパート(IK) | 150/150 |
| 学習ポリシー(MLP) | 150/150 |

- テスト誤差(関節角)は、デモ25件で MAE 1.41°、640件で 0.14°(`data/train_results.json`)。
- 認識(色セグメンテーション)の位置誤差は平均約 3.6 mm。

---

## 制約

- シミュレーションのみ。実機は未検証。
- 認識は色の閾値処理で対応している。AIを使っているのは行動ポリシーの部分のみ。
- キューブの向き(yaw)は固定している。
- これらの制約を外すと難化するため、制約とした。

---

## References

- [PyBullet](https://pybullet.org/)
