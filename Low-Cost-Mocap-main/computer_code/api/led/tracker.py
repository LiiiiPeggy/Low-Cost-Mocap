"""灯板三维几何中心的卡尔曼跟踪。

状态 [x, y, z, vx, vy, vz, ax, ay, az]，量测 [x, y, z]。
沿用原作者 KalmanFilter 的“位置 + 速度/加速度”状态思路，但：
* 只有一个目标，不再保留无人机编号；
* 传播间隔来自真实同步帧时间戳，而不是固定 0.1 s 或 60 Hz 假设；
* 连续漏检只做短时预测，超过时间/次数上限明确返回 lost。
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class LedKalmanTracker:
    def __init__(self, *, max_missed: int = 8,
                 max_predict_age_s: float = 0.6,
                 process_noise: float = 1e-4,
                 measurement_noise: float = 1e-4):
        self.max_missed = int(max_missed)
        self.max_predict_age_s = float(max_predict_age_s)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.kf = self._new_filter()
        self.reset()

    def _new_filter(self):
        kf = cv2.KalmanFilter(9, 3)
        kf.measurementMatrix = np.zeros((3, 9), np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
        kf.measurementMatrix[2, 2] = 1
        kf.processNoiseCov = np.eye(9, dtype=np.float32) * self.process_noise
        kf.measurementNoiseCov = (np.eye(3, dtype=np.float32)
                                  * self.measurement_noise)
        kf.errorCovPost = np.eye(9, dtype=np.float32) * 0.1
        return kf

    def _set_transition(self, dt: float) -> None:
        dt = float(np.clip(dt, 1e-3, 0.5))
        A = np.eye(9, dtype=np.float32)
        A[:3, 3:6] = dt * np.eye(3, dtype=np.float32)
        A[3:6, 6:9] = dt * np.eye(3, dtype=np.float32)
        A[:3, 6:9] = (0.5 * dt * dt) * np.eye(3, dtype=np.float32)
        self.kf.transitionMatrix = A

    def reset(self) -> None:
        self.initialized = False
        self.missed = 0
        self.last_measure_stamp: float | None = None
        self.last_state_stamp: float | None = None
        self.kf.statePost = np.zeros((9, 1), np.float32)
        self.kf.statePre = np.zeros((9, 1), np.float32)
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 0.1

    def update(self, position, stamp: float) -> dict:
        pos = np.asarray(position, np.float32).reshape(3)
        stamp = float(stamp)
        reacquired = False
        if (not self.initialized or self.last_measure_stamp is None
                or stamp - self.last_measure_stamp > self.max_predict_age_s):
            self.reset()
            state = np.zeros((9, 1), np.float32)
            state[:3, 0] = pos
            self.kf.statePost = state
            self.kf.statePre = state.copy()
            self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 0.1
            self.initialized = True
            self.last_measure_stamp = stamp
            self.last_state_stamp = stamp
            reacquired = True
            return {"state": "tracking", "position": pos.tolist(),
                    "velocity": [0.0, 0.0, 0.0], "dt_s": 0.0,
                    "missed": 0, "reacquired": reacquired}

        dt = stamp - (self.last_state_stamp or stamp)
        self._set_transition(dt)
        self.kf.predict()
        self.kf.correct(pos.reshape(3, 1))
        self.last_measure_stamp = stamp
        self.last_state_stamp = stamp
        self.missed = 0
        state = self.kf.statePost.reshape(-1)
        return {"state": "tracking",
                "position": state[:3].astype(float).tolist(),
                "velocity": state[3:6].astype(float).tolist(),
                "dt_s": round(float(dt), 6), "missed": 0,
                "reacquired": reacquired}

    def predict(self, stamp: float) -> dict:
        if not self.initialized or self.last_measure_stamp is None:
            return {"state": "idle", "position": None, "velocity": None}
        stamp = float(stamp)
        age = stamp - self.last_measure_stamp
        if age > self.max_predict_age_s or self.missed >= self.max_missed:
            self.reset()
            return {"state": "lost", "position": None, "velocity": None,
                    "age_s": round(float(age), 4), "missed": self.max_missed}
        dt = stamp - (self.last_state_stamp or stamp)
        self._set_transition(dt)
        pred = self.kf.predict()
        self.kf.statePost = pred.copy()
        self.last_state_stamp = stamp
        self.missed += 1
        state = pred.reshape(-1)
        return {"state": "predicted",
                "position": state[:3].astype(float).tolist(),
                "velocity": state[3:6].astype(float).tolist(),
                "age_s": round(float(age), 4), "missed": self.missed,
                "dt_s": round(float(dt), 6)}


__all__ = ["LedKalmanTracker"]
