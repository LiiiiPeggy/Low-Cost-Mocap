"""红外 LED 检测 / 匹配 / 三角化 / 跟踪的针对性合成测试。

运行（在 computer_code/api 下）::

    python -m led.selftest

合成图只验证算法链路与坐标/单位，不代表实机精度。
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

from led.detector import default_params, detect_leds, validate_params
from led.geometry import match_and_triangulate
from led.tracker import LedKalmanTracker

RESULT = []


def check(name: str, ok: bool, detail="") -> bool:
    detail = "" if detail == "" else str(detail)
    RESULT.append((name, bool(ok), detail))
    print(("  [PASS] " if ok else "  [FAIL] ") + name
          + ((" —— " + detail) if detail else ""))
    return ok


def spots(centers, sigma=3.0, size=(720, 1280), amp=255, sat_radius=0.0):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros((h, w), np.float64)
    for x, y in centers:
        field = np.maximum(field, amp * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)))
        if sat_radius:
            m = (xx - x) ** 2 + (yy - y) ** 2 <= sat_radius ** 2
            field[m] = 255
    return np.clip(field, 0, 255).astype(np.uint8)


def look_at(C, target):
    C = np.asarray(C, np.float64)
    target = np.asarray(target, np.float64)
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def project(P, K, D, R, t):
    rvec = cv2.Rodrigues(np.asarray(R, np.float64))[0]
    out, _ = cv2.projectPoints(np.asarray(P, np.float64),
                               rvec, np.asarray(t, np.float64).reshape(3, 1),
                               np.asarray(K, np.float64),
                               np.asarray(D, np.float64).reshape(-1))
    return out.reshape(-1, 2)


def test_detector():
    print("\n[1] 单相机 LED 检测 / 粘连分离 / 过曝失败")
    p = default_params()
    p.update({"threshold": 120, "min_area": 3, "max_area": 600,
              "min_peak_distance": 6.0, "blur_kernel": 3})

    img = spots([(400.4, 300.3), (430.0, 300.0), (460.0, 300.0)])
    det = detect_leds(img, p)
    check("三个分离光斑检测到 3 个", det["count"] == 3, det["count"])
    if det["count"] == 3:
        got = sorted([(d["x"], d["y"]) for d in det["leds"]])
        want = [(400.4, 300.3), (430.0, 300.0), (460.0, 300.0)]
        err = max(math.hypot(g[0] - w[0], g[1] - w[1])
                  for g, w in zip(got, want))
        check("亚像素中心误差 < 0.5px", err < 0.5, "max=%.3fpx" % err)

    p2 = dict(p); p2["min_peak_distance"] = 3.0
    touch = spots([(400.0, 300.0), (408.0, 300.0)], sigma=3.0)
    d2 = detect_leds(touch, p2)
    check("两个粘连光斑被拆成 2 个", d2["count"] == 2,
          "count=%d split=%s" % (d2["count"],
                                 [d.get("split") for d in d2["leds"]]))

    merged = spots([(400.0, 300.0), (410.0, 300.0), (420.0, 300.0)],
                   sigma=3.0)
    d3 = detect_leds(merged, p2)
    check("三个粘连光斑至少拆出 2 个", d3["count"] >= 2,
          "count=%d" % d3["count"])

    sat = np.zeros((720, 1280), np.uint8)
    sat[280:340, 380:480] = 255
    d4 = detect_leds(sat, p)
    check("完全饱和的大亮区不伪造三点",
          d4["count"] == 0 and d4["status"] == "unresolved"
          and d4["unresolved_count"] >= 1,
          "count=%d status=%s" % (d4["count"], d4["status"]))

    single_sat = np.zeros((720, 1280), np.uint8)
    single_sat[290:310, 390:410] = 255
    d5 = detect_leds(single_sat, p)
    check("单个大面积饱和亮区给过曝提示",
          d5["count"] == 1 and bool(d5["overexposed"])
          and any("饱和" in w for w in d5["warnings"]),
          d5["warnings"])

    # 细长灯带/地面反光：形状过滤后不应成为 LED 候选，但原始连通域仍保留
    refl = np.zeros((720, 1280), np.uint8)
    refl[280:288, 300:700] = 255
    d6 = detect_leds(refl, p)
    check("细长反光被过滤且保留原始候选",
          d6["count"] == 0 and d6["raw_count"] >= 1
          and any("细长" in r.get("reason", "") for r in d6["rejected"]),
          "count=%d raw=%d" % (d6["count"], d6["raw_count"]))

    # ROI：只检测 ROI 内的光斑，但坐标仍是整图坐标
    two = spots([(300.0, 250.0), (340.0, 250.0)])
    d7 = detect_leds(two, p, roi=(250, 200, 140, 120))
    check("ROI 内检测坐标加回整图偏移",
          d7["count"] == 2
          and all(280 <= d["x"] <= 360 and 230 <= d["y"] <= 280
                  for d in d7["leds"])
          and d7.get("roi") == [250, 200, 140, 120],
          [(round(d["x"], 1), round(d["y"], 1)) for d in d7["leds"]])
    d8 = detect_leds(two, p, roi=(500, 400, 120, 100))
    check("ROI 外光斑不参与检测", d8["count"] == 0, d8["count"])

    # partial=True 只覆盖 min_area 时，不应该拿缺省的 max_area 去比较
    try:
        validate_params({"min_area": 50}, partial=True)
        partial_ok = True
    except ValueError:
        partial_ok = False
    check("单相机覆盖参数可只给 min_area", partial_ok)

    # 实机 matched=0 的直接原因复现：max_area 被调到 10 px² 时，真实 LED
    # （这里用 sigma=5 的光斑，面积约 37 px²）会被当成“过大且无法分离”拒绝。
    large_leds = spots([(400.0, 300.0), (440.0, 300.0), (420.0, 330.0)],
                       sigma=5.0)
    old = dict(p)
    old.update({"min_area": 1, "max_area": 10, "max_candidates": 12})
    d_old = detect_leds(large_leds, old)
    check("旧 max_area=10 会拒绝真实 LED（实机 matched=0 原因）",
          d_old["count"] == 0 and d_old["unresolved_count"] >= 3,
          "count=%d unresolved=%d" % (d_old["count"],
                                      d_old["unresolved_count"]))
    d_new = detect_leds(large_leds, default_params())
    check("新默认 max_area=800 能保留真实 LED",
          d_new["count"] == 3, d_new["count"])


def _rig():
    K = np.array([[910.0, 0.0, 643.5], [0.0, 908.0, 358.2], [0.0, 0.0, 1.0]])
    D = np.array([-0.08, 0.03, 0.0002, -0.0001, 0.0])
    R0, t0 = look_at([-1.05, -0.85, 1.20], [0.0, 0.0, 0.20])
    R1, t1 = look_at([0.98, -0.78, 1.28], [0.0, 0.0, 0.20])
    return K, D, R0, t0, R1, t1


def _triangle(center=(0.0, 0.0, 0.60), yaw=0.0):
    c = np.asarray(center, np.float64)
    e1 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    e2 = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    r = 0.033 / math.sqrt(3.0)
    return np.array([c + r * (math.cos(a) * e1 + math.sin(a) * e2)
                     for a in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)])


def _dets(points2d):
    return {"leds": [{"x": float(x), "y": float(y)} for x, y in points2d]}


def test_geometry():
    print("\n[2] 极线匹配 / 三角化 / 33 mm 等边三角形")
    K, D, R0, t0, R1, t1 = _rig()
    world = _triangle()
    p0 = project(world, K, D, R0, t0)
    p1 = project(world, K, D, R1, t1)
    calib = dict(K0=K, D0=D, K1=K, D1=D, R0=R0, t0=t0, R1=R1, t1=t1,
                 image_size=(1280, 720))
    params = default_params()
    params.update({"epipolar_tol_px": 3.0, "reprojection_tol_px": 3.0,
                   "triangle_tol_mm": 3.0})
    res = match_and_triangulate(_dets(p0), _dets(p1), calib, params)
    check("三点匹配成功", res.get("ok") is True, res.get("reason"))
    if res.get("ok"):
        center = np.asarray(res["raw_center"])
        err = float(np.linalg.norm(center - np.asarray(world).mean(axis=0)))
        check("三角化世界中心误差 < 1mm", err < 0.001,
              "%.4f mm" % (err * 1000))
        check("三条边均为 33 mm（±0.5mm）",
              max(abs(v - 33.0) for v in res["edges_mm"]) < 0.5,
              res["edges_mm"])
        check("重投影 < 0.2px", res["mean_reprojection_px"] < 0.2,
              res["mean_reprojection_px"])
        check("三个 LED 标签是临时编号",
              res.get("labels_are_temporary") is True)
        check("两台相机三点绕向一致",
              res.get("orientation_consistent") is True,
              res.get("orientation_consistent"))

    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.25, p0.shape)
    p0n = p0 + noise
    p1n = p1 + rng.normal(0, 0.25, p1.shape)
    res_n = match_and_triangulate(_dets(p0n), _dets(p1n), calib, params)
    check("带 0.25px 噪声仍能匹配", res_n.get("ok") is True, res_n.get("reason"))
    if res_n.get("ok"):
        err = float(np.linalg.norm(np.asarray(res_n["raw_center"])
                                   - np.asarray(world).mean(axis=0)))
        check("噪声下中心误差 < 3mm", err < 0.003, "%.3f mm" % (err * 1000))

    # 加一个无关亮点，仍应挑出正确三角形
    sp0 = np.vstack([p0n, [[100.0, 600.0]]])
    sp1 = np.vstack([p1n, [[900.0, 500.0]]])
    res_sp = match_and_triangulate(_dets(sp0), _dets(sp1), calib, params)
    check("有无关亮点时仍选出三点三角形", res_sp.get("ok") is True,
          res_sp.get("reason"))

    # 多组反射点：每台相机加 6 个随机点，仍应挑出真实三角形
    rng2 = np.random.default_rng(11)
    extra0 = rng2.uniform([50, 50], [1200, 650], (6, 2))
    extra1 = rng2.uniform([50, 50], [1200, 650], (6, 2))
    res_many = match_and_triangulate(
        _dets(np.vstack([p0n, extra0])),
        _dets(np.vstack([p1n, extra1])), calib, params)
    check("多个反射候选下仍能选出 33 mm 三角形",
          res_many.get("ok") is True
          and max(abs(v - 33.0) for v in res_many["edges_mm"]) < 0.5,
          None if not res_many.get("ok") else res_many["edges_mm"])

    # 少一个 LED
    res_missing = match_and_triangulate(_dets(p0[:2]), _dets(p1[:2]),
                                        calib, params)
    check("少于 3 个检测点时拒绝定位",
          res_missing.get("ok") is False
          and res_missing.get("status") == "insufficient_detections",
          res_missing.get("status"))

    # 破坏三角形几何
    bad_world = np.array(world)
    bad_world[2] += np.array([0.0, 0.0, 0.03])
    b0 = project(bad_world, K, D, R0, t0)
    b1 = project(bad_world, K, D, R1, t1)
    res_bad = match_and_triangulate(_dets(b0), _dets(b1), calib, params)
    check("33 mm 几何不一致时不输出有效中心",
          res_bad.get("ok") is False
          and res_bad.get("status") == "geometry_mismatch",
          res_bad.get("status"))


def test_tracker():
    print("\n[3] 三维卡尔曼跟踪 / 短时预测 / 丢失")
    tr = LedKalmanTracker()
    t = 100.0
    last = None
    for i in range(30):
        last = tr.update([0.03 * i, 0.0, 1.0], t + 0.033 * i)
    check("匀速运动速度估计接近真值",
          abs(last["velocity"][0] - 0.03 / 0.033) < 0.06,
          "vx=%.4f" % last["velocity"][0])
    check("滤波后位置接近测量",
          abs(last["position"][0] - 0.03 * 29) < 0.01,
          "x=%.4f" % last["position"][0])

    states = [tr.predict(t + 0.033 * 30 + 0.033 * i)["state"]
              for i in range(1, 11)]
    check("短时漏检先预测、超时后丢失",
          states[:8] == ["predicted"] * 8 and states[8] == "lost",
          states)


def test_service_pipeline():
    """用假相机/假同步/假标定验证后台服务：去重、跟踪、叠加图。"""
    print("\n[4] 后台服务：同步帧去重 / 三维状态 / 叠加图")
    tmp = tempfile.mkdtemp(prefix="led_service_test_")
    old_env = os.environ.get("MOCAP_CALIB_DIR")
    os.environ["MOCAP_CALIB_DIR"] = tmp
    try:
        import led.service as LS
        from camera_manager import Frame
        from sync import PairRecord

        K, D, R0, t0, R1, t1 = _rig()
        world = _triangle()
        p0 = project(world, K, D, R0, t0)
        p1 = project(world, K, D, R1, t1)
        img0 = spots([(x, y) for x, y in p0], sigma=2.2)
        img1 = spots([(x, y) for x, y in p1], sigma=2.2)

        def frame(img, seq):
            return Frame(gray=img, seq=seq, mf_ts=seq * 333333,
                         host_ts=time.time(), mono_ts=time.perf_counter())

        class Worker:
            state = "streaming"
            device_name = "fake"
            format_info = "NV12"
            requested = {"width": 1280, "height": 720}

            def __init__(self, f):
                self.f = f

            def latest(self):
                return self.f

        class Manager:
            def __init__(self):
                self.workers = {0: Worker(frame(img0, 1)),
                                1: Worker(frame(img1, 2))}

            def worker(self, i):
                return self.workers.get(int(i))

        class Sync:
            max_dt_ms = 20.0

            def __init__(self):
                self.pair = None

            def recent_pairs(self, count=1):
                return [self.pair] if self.pair is not None else []

        class CalibSvc:
            def __init__(self, snap):
                self.snap = snap

            def localization_snapshot(self):
                return copy.deepcopy(self.snap)

        snapshot = {
            "intrinsics": {
                "0": {"K": K.tolist(), "D": D.tolist(),
                      "image_size": [1280, 720], "num_views": 8},
                "1": {"K": K.tolist(), "D": D.tolist(),
                      "image_size": [1280, 720], "num_views": 8},
            },
            "extrinsic": {
                "orientation_confirmed": True,
                "cameras": {
                    "0": {"R": R0.tolist(), "t": t0.tolist()},
                    "1": {"R": R1.tolist(), "t": t1.tolist()},
                },
                "board": {}, "world_frame": {},
                "source": "selftest", "created": "test",
            },
            "cameras": [0, 1],
            "image_size": [1280, 720],
        }
        mgr, syn, cal = Manager(), Sync(), CalibSvc(snapshot)
        orig = (LS.manager, LS.sync_service, LS.calibration_service,
                LS.DEFAULT_PARAMS_PATH)
        LS.manager = lambda: mgr
        LS.sync_service = lambda: syn
        LS.calibration_service = lambda: cal
        LS.DEFAULT_PARAMS_PATH = os.path.join(tmp, "calibration",
                                               "led_params.json")
        svc = None
        try:
            svc = LS.LedService()
            svc.start_detection()
            svc.start_locate()
            # 先启动服务，再给一组“新鲜”的同步帧，符合实际 recent_pairs 行为
            syn.pair = PairRecord(pair_seq=1, dt_ms=0.3,
                                  frame_a=mgr.workers[0].f,
                                  frame_b=mgr.workers[1].f,
                                  created_mono=time.perf_counter())
            state = None
            for _ in range(80):
                time.sleep(0.02)
                state = svc.state()
                if state.get("status") == "tracking":
                    break
            check("后台服务进入 tracking 状态",
                  state is not None and state.get("status") == "tracking",
                  None if state is None else
                  "%s | pair=%s processed=%s ms=%s" % (
                      state.get("reason"), state.get("pair"),
                      state.get("processed_pairs"), state.get("processing_ms")))
            if state and state.get("status") == "tracking":
                center = np.asarray(state["filtered_center"])
                err = float(np.linalg.norm(center - world.mean(axis=0)))
                check("服务输出中心接近真值 (<3mm)", err < 0.003,
                      "%.3f mm" % (err * 1000))
                check("服务输出三条边 33 mm",
                      max(abs(v - 33.0) for v in state["edges_mm"]) < 0.5,
                      state["edges_mm"])
                check("LED 标签标记为临时编号",
                      state.get("labels_are_temporary") is True)
                check("服务状态包含 ROI / 动态跟踪信息",
                      isinstance(state.get("roi"), dict)
                      and "tracking" in state["roi"]
                      and state.get("detections", {}).get("0", {}).get("raw_count") is not None,
                      state.get("roi"))
                svc._roi_tracking = True
                svc._roi_track["0"] = [600, 500, 40, 30]
                roi_track, mode_track = svc._roi_for(0, (720, 1280))
                check("动态 ROI 向外扩 80px 且裁剪到图像内",
                      mode_track == "track" and roi_track == [520, 420, 200, 190],
                      "%s %s" % (mode_track, roi_track))
                svc._roi_force_full["0"] = True
                _, mode_full = svc._roi_for(0, (720, 1280))
                check("匹配失败后允许下一帧全图重搜", mode_full == "full_search",
                      mode_full)
                svc._clear_track_rois()
                before = state["processed_pairs"]
                time.sleep(0.3)
                after = svc.state()["processed_pairs"]
                check("同一 pair_seq 不重复处理", after == before,
                      "%s -> %s" % (before, after))
                syn.pair = PairRecord(pair_seq=2, dt_ms=0.3,
                                      frame_a=mgr.workers[0].f,
                                      frame_b=mgr.workers[1].f,
                                      created_mono=time.perf_counter())
                time.sleep(0.25)
                after2 = svc.state()["processed_pairs"]
                check("新的 pair_seq 被处理", after2 > after,
                      "%s -> %s" % (after, after2))
                data, info = svc.overlay(0, view="detect")
                check("二维叠加图可生成",
                      len(data) > 1000 and info.get("count") is not None,
                      "bytes=%d count=%s" % (len(data), info.get("count")))
                data_z, info_z = svc.overlay(0, view="detect", zoom=2.0)
                check("局部放大叠加图可生成",
                      len(data_z) > 1000 and info_z.get("zoom") == 2.0,
                      "bytes=%d zoom=%s" % (len(data_z), info_z.get("zoom")))
                dbg = svc.save_debug()
                check("实机调试帧保存（原始图/二值图/JSON）",
                      os.path.isdir(dbg["abs_path"])
                      and "detections.json" in dbg["files"]
                      and any(f.endswith("_raw.png") for f in dbg["files"]),
                      "%s files=%s" % (dbg.get("path"), len(dbg.get("files", []))))
                upd = svc.update_params({
                    "params": {"threshold": 190},
                    "overrides": {"0": {"split_blobs": False},
                                  "1": {"min_area": 6}},
                    "rois": {"0": [0.1, 0.1, 0.6, 0.7], "1": None},
                    "roi_tracking": False,
                })
                check("参数/ROI/单相机分离覆盖可保存",
                      upd["params"]["threshold"] == 190
                      and upd["overrides"]["0"].get("split_blobs") is False
                      and upd["overrides"]["1"].get("min_area") == 6
                      and upd["rois"]["0"] == [0.1, 0.1, 0.6, 0.7]
                      and upd["roi_tracking"] is False,
                      upd.get("path"))
                upd_clear = svc.update_params({"overrides": {}})
                check("显式空 overrides 可清空相机覆盖",
                      upd_clear["overrides"] == {"0": {}, "1": {}},
                      upd_clear["overrides"])
                svc.reset_params()
        finally:
            if svc is not None:
                svc.stop_detection()
            (LS.manager, LS.sync_service, LS.calibration_service,
             LS.DEFAULT_PARAMS_PATH) = orig
    finally:
        if old_env is None:
            os.environ.pop("MOCAP_CALIB_DIR", None)
        else:
            os.environ["MOCAP_CALIB_DIR"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_detector()
    test_geometry()
    test_tracker()
    test_service_pipeline()
    print("\n" + "=" * 62)
    passed = sum(1 for _, ok, _ in RESULT if ok)
    failed = [(n, d) for n, ok, d in RESULT if not ok]
    print("LED 自检：%d/%d 通过" % (passed, len(RESULT)))
    for n, d in failed:
        print("  FAILED: %s —— %s" % (n, d))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
