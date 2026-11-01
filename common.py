"""共通モジュール: PyBullet上のFranka Panda色別仕分け環境・知覚・IK・動作生成"""
import math
import numpy as np
import pybullet as p
import pybullet_data

COLORS = ["red", "green", "blue"]
RGBA = {"red": [0.9, 0.1, 0.1, 1], "green": [0.1, 0.8, 0.2, 1], "blue": [0.15, 0.3, 0.95, 1]}
# 色ごとの仕分け先パッド中心(x, y)
PADS = {"red": (0.38, -0.40), "green": (0.62, -0.40), "blue": (0.50, 0.42)}
PAD_HALF = 0.075
CUBE = 0.04  # 一辺[m]
WS_X = (0.40, 0.62)  # 物体の出現範囲
WS_Y = (-0.20, 0.20)
Z_TABLE = 0.0
Z_GRASP = Z_TABLE + CUBE / 2 + 0.005   # EE(grasptarget)の把持高さ
Z_HOVER = 0.26
Z_PLACE = Z_TABLE + CUBE + 0.03
ARM = list(range(7))
FINGERS = [9, 10]
EE = 11
HOME = [0.0, -0.5, 0.0, -2.3, 0.0, 1.85, 0.785]
DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])


class Env:
    def __init__(self, gui=False, seed=0):
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        self.rng = np.random.default_rng(seed)
        self.plane = p.loadURDF("plane.urdf", [0, 0, -0.65])
        self.table = p.loadURDF("table/table.urdf", [0.5, 0, -0.625], useFixedBase=True)
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        for f in FINGERS:
            p.changeDynamics(self.robot, f, lateralFriction=2.0, spinningFriction=0.5)
        self.pads = {}
        for c in COLORS:
            x, y = PADS[c]
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[PAD_HALF, PAD_HALF, 0.003],
                                     rgbaColor=[*RGBA[c][:3], 0.55])
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[PAD_HALF, PAD_HALF, 0.003])
            self.pads[c] = p.createMultiBody(0, cs, vs, [x, y, 0.003])
        self.cubes = {}
        self.reset_arm()
        self._hist = None  # 録画用コールバック

    # ---------- 基本操作 ----------
    def reset_arm(self, q=None):
        q = HOME if q is None else q
        for i, v in zip(ARM, q):
            p.resetJointState(self.robot, i, v)
        for f in FINGERS:
            p.resetJointState(self.robot, f, 0.04)

    def spawn_cubes(self, min_sep=0.09):
        for b in self.cubes.values():
            p.removeBody(b)
        self.cubes = {}
        pts = []
        while len(pts) < 3:
            q = (self.rng.uniform(*WS_X), self.rng.uniform(*WS_Y))
            if all(math.dist(q, r) > min_sep for r in pts):
                pts.append(q)
        order = list(self.rng.permutation(3))
        for c_i, (x, y) in zip(order, pts):
            c = COLORS[c_i]
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE / 2] * 3)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE / 2] * 3, rgbaColor=RGBA[c])
            b = p.createMultiBody(0.05, cs, vs, [x, y, CUBE / 2 + 0.001])
            p.changeDynamics(b, -1, lateralFriction=1.5, spinningFriction=0.3)
            self.cubes[c] = b
        for _ in range(60):
            self.step()

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            if self._hist is not None:
                self._hist()

    # ---------- 知覚(仮想カメラ + 色セグメンテーション) ----------
    def perceive(self):
        """真上カメラのRGB-D画像から各色キューブのワールド座標(x,y)を推定"""
        W = H = 320
        eye, tgt = [0.5, 0, 1.0], [0.5, 0, 0]
        view = p.computeViewMatrix(eye, tgt, [0, 1, 0])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.05, 2.0)
        _, _, rgb, depth, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(int)
        depth = np.reshape(depth, (H, W))
        V = np.array(view).reshape(4, 4).T
        P = np.array(proj).reshape(4, 4).T
        inv = np.linalg.inv(P @ V)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        masks = {"red": (r > 150) & (g < 90) & (b < 90),
                 "green": (g > 120) & (r < 90) & (b < 110),
                 "blue": (b > 170) & (r < 90) & (g < 110)}
        out = {}
        for c, m in masks.items():
            # パッド(半透明・低位置)を除外するため、深度で高さのあるものだけ使う
            ys, xs = np.nonzero(m)
            if len(xs) < 5:
                continue
            keep = []
            for u, v in zip(xs, ys):
                ndc = np.array([2 * u / W - 1, 1 - 2 * v / H, 2 * depth[v, u] - 1, 1.0])
                w = inv @ ndc
                w = w[:3] / w[3]
                if w[2] > 0.02:
                    keep.append(w)
            if len(keep) < 5:
                continue
            k = np.array(keep)
            out[c] = (float(k[:, 0].mean()), float(k[:, 1].mean()))
        return out

    # ---------- IK ----------
    def ik(self, x, y, z):
        """決定的なIK(毎回HOMEから解く)。関節角7次元を返す。失敗時None"""
        self.reset_arm()
        q = HOME
        for _ in range(3):
            q = p.calculateInverseKinematics(self.robot, EE, [x, y, z], DOWN,
                                             lowerLimits=[-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8],
                                             upperLimits=[2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8],
                                             jointRanges=[5.6, 3.4, 5.6, 2.9, 5.6, 3.7, 5.6],
                                             restPoses=HOME, maxNumIterations=200,
                                             residualThreshold=1e-5)[:7]
            for i, v in zip(ARM, q):
                p.resetJointState(self.robot, i, v)
        pos = p.getLinkState(self.robot, EE)[0]
        err = math.dist(pos, (x, y, z))
        self.reset_arm()
        return (list(q), err)

    def keypose_targets(self, xy, color):
        """(把持xy, 色) → 4つのキーポーズのEE目標位置"""
        gx, gy = xy
        px, py = PADS[color]
        return [(gx, gy, Z_HOVER), (gx, gy, Z_GRASP), (px, py, Z_HOVER), (px, py, Z_PLACE)]

    def expert_keyposes(self, xy, color):
        """IKによるエキスパート。28次元(4姿勢×7関節)。IK誤差>5mmならNone"""
        qs = []
        for (x, y, z) in self.keypose_targets(xy, color):
            q, err = self.ik(x, y, z)
            if err > 0.005:
                return None
            qs.append(q)
        return np.array(qs).reshape(-1)

    # ---------- 動作実行 ----------
    def _gripper(self, width, steps=60, force=80):
        for _ in range(steps):
            for f in FINGERS:
                p.setJointMotorControl2(self.robot, f, p.POSITION_CONTROL, width, force=force)
            self.step()

    def _move(self, q_goal, speed=1.6, hold_grip=None, force=87):
        q0 = np.array([p.getJointState(self.robot, i)[0] for i in ARM])
        q1 = np.array(q_goal)
        T = max(int(np.abs(q1 - q0).max() / speed * 240), 40)
        for t in range(1, T + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * t / T)
            q = q0 + (q1 - q0) * s
            p.setJointMotorControlArray(self.robot, ARM, p.POSITION_CONTROL, list(q),
                                        forces=[force] * 7, positionGains=[0.08] * 7)
            if hold_grip is not None:
                for f in FINGERS:
                    p.setJointMotorControl2(self.robot, f, p.POSITION_CONTROL, hold_grip, force=80)
            self.step()

    def execute(self, keyposes):
        """4キーポーズで1個のキューブを把持→移動→配置"""
        kp = np.array(keyposes).reshape(4, 7)
        self._gripper(0.04, 20)
        self._move(kp[0])
        self._move(kp[1], speed=0.8)
        self._gripper(0.0, 80)                     # 把持
        self._move(kp[0], speed=0.8, hold_grip=0.0)  # 持ち上げ
        self._move(kp[2], hold_grip=0.0)
        self._move(kp[3], speed=0.8, hold_grip=0.0)
        self._gripper(0.04, 50)                    # 解放
        self._move(kp[2], speed=0.8)

    def go_home(self):
        self._move(HOME)

    def is_sorted(self, color):
        pos, _ = p.getBasePositionAndOrientation(self.cubes[color])
        px, py = PADS[color]
        return abs(pos[0] - px) < PAD_HALF and abs(pos[1] - py) < PAD_HALF and pos[2] < 0.06
