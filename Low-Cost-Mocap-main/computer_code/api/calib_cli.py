"""标定命令行工具：采集 / 重检测 / 内参 / 外参 / 验证 / 保存，全流程可在命令行完成。

在 ``Low-Cost-Mocap-main/computer_code`` 目录下运行::

    python api/calib_cli.py list
    python api/calib_cli.py new --cols 9 --rows 6 --square-mm 25 --cameras 0 1
    python api/calib_cli.py capture-single --camera 0 --count 20 --interval 1.5
    python api/calib_cli.py capture-stereo --count 15 --interval 2.0
    python api/calib_cli.py detect --session captures/calibration/20261113_120000
    python api/calib_cli.py intrinsic --session <dir>
    python api/calib_cli.py stereo    --session <dir>
    python api/calib_cli.py validate  --session <dir> --holdout 0.3
    python api/calib_cli.py save      --session <dir>
    python api/calib_cli.py info

``intrinsic / stereo / validate / save / detect`` **不需要相机在线**：它们只读
会话目录里的原始 PNG 和 samples.json，可以换一台机器离线重算。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2  # noqa: E402

import calibration.routes  # noqa: E402,F401  （确保包可导入，同时注册路由模块）
from calibration import board as B  # noqa: E402
from calibration.dataset import Session  # noqa: E402
from calibration.service import DEFAULT_RESULT_PATH, DEFAULT_SESSION_ROOT, service  # noqa: E402


def _die(msg: str, code: int = 1):
    print("错误：%s" % msg, file=sys.stderr)
    raise SystemExit(code)


def _session(root: str) -> Session:
    if not root:
        _die("需要 --session <会话目录>")
    if not os.path.isdir(root):
        _die("会话目录不存在：%s" % root)
    return Session.load(root)


def _print_intrinsic(res: dict) -> None:
    for cam, d in res.items():
        print("\n相机 %s  样本 %d  分辨率 %s"
              % (cam, d["num_views"], "x".join(str(v) for v in d["image_size"])))
        print("  fx=%.4f  fy=%.4f  cx=%.4f  cy=%.4f"
              % (d["focal_px"][0], d["focal_px"][1],
                 d["principal_point_px"][0], d["principal_point_px"][1]))
        print("  D = [%s]" % ", ".join("%.6f" % v for v in d["D"]))
        print("  RMS = %.4f px（mean %.4f / median %.4f / p95 %.4f / max %.4f）"
              % (d["rms_px"], d["reprojection"]["mean_px"], d["reprojection"]["median_px"],
                 d["reprojection"]["p95_px"], d["reprojection"]["max_px"]))
        cov = d.get("radius_coverage") or {}
        if cov:
            print("  角点半径覆盖 = %.0f%%（画面半对角）" % (100 * cov.get("ratio", 0)))
        for w in d.get("warnings", []):
            print("  ! %s" % w)
        bad = (d.get("outlier_suggestion") or {}).get("exclude") or []
        if bad:
            print("  建议剔除：%s（阈值 %.3fpx）"
                  % ("，".join("%s=%.3fpx" % (b["sample"], b["rms_px"]) for b in bad),
                     d["outlier_suggestion"]["threshold_px"]))
        worst = sorted(d.get("per_view", []), key=lambda p: -p["rms_px"])[:5]
        if worst:
            print("  误差最大的样本：%s"
                  % "，".join("%s=%.3fpx" % (p["sample"], p["rms_px"]) for p in worst))


def _print_stereo(res: dict) -> None:
    print("\n外参（固定内参）  帧对 %d" % res["num_pairs"])
    print("  约定：%s   T 单位 %s" % (res["convention"], res["T_unit"]))
    print("  R_10 =")
    for row in res["R_10"]:
        print("    [%9.6f %9.6f %9.6f]" % tuple(row))
    print("  T_10 = [%.6f, %.6f, %.6f] m" % tuple(res["T_10"]))
    print("  基线 = %.3f mm" % (res["baseline_m"] * 1000))
    print("  相对旋转 = %.4f°  轴 %s"
          % (res["checks"]["rotation_angle_deg"], res["checks"]["rotation_axis"]))
    print("  旋转正交误差 = %.2e   det = %.9f"
          % (res["checks"]["orthogonality_error"], res["checks"]["determinant"]))
    print("  双目 RMS = %.4f px" % res["rms_px"])
    print("  极线误差 median %.4f / mean %.4f / p95 %.4f / max %.4f px"
          % (res["epipolar"]["median_px"], res["epipolar"]["mean_px"],
             res["epipolar"]["p95_px"], res["epipolar"]["max_px"]))
    print("  三角化重投影 median %.4f / p95 %.4f px"
          % (res["triangulation"]["median_px"], res["triangulation"]["p95_px"]))
    kd = res.get("known_distance") or {}
    if kd:
        print("  已知间距误差 mean %.4fmm  中位 %.4fmm  最大 %.4fmm  相对 %.3f%%"
              % (kd["mean_error_mm"], kd["median_error_mm"],
                 kd["max_abs_error_mm"], kd["mean_relative_error_pct"]))
    print("  三角化点落在两相机前方 = %.1f%%" % (100 * res["checks"]["in_front_ratio"]))
    if res.get("flip_b"):
        print("  [注意] 相机 1 角点顺序被整体翻转 180°（自动择向结果）")
    for w in res.get("warnings", []):
        print("  ! %s" % w)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_list(args) -> int:
    root = args.root or DEFAULT_SESSION_ROOT
    if not os.path.isdir(root):
        print("没有标定目录：%s" % root)
        return 0
    rows = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isfile(os.path.join(p, "session.json")):
            continue
        try:
            s = Session.load(p)
            st = s.stats()
            rows.append((name, "%dx%d" % (s.spec.cols, s.spec.rows), s.spec.square_mm,
                         st["single_valid"], st["stereo_valid"], s.meta.get("updated", "")))
        except Exception as exc:
            rows.append((name, "?", "", {}, {}, "读取失败：%s" % exc))
    if not rows:
        print("目录 %s 下没有标定会话" % root)
        return 0
    print("%-22s %-8s %-8s %-16s %-8s %s"
          % ("会话", "内角点", "方格mm", "单相机有效", "帧对", "更新时间"))
    for r in rows:
        print("%-22s %-8s %-8s %-16s %-8s %s"
              % (r[0], r[1], r[2], json.dumps(r[3], ensure_ascii=False), r[4], r[5]))
    return 0


def _print_rig(rig: dict) -> None:
    print("\n外参（固定棋盘格 + solvePnP）  全局 flip180=%s" % rig["flip180"])
    print("  世界系：原点=图案中心，X=12 格长边，Y=9 格短边，Z=向上（右手）")
    for key in ("0", "1"):
        p = rig["cameras"][key]
        c = p["camera_centre_world"]
        print("  相机 %s：位置 [%.4f, %.4f, %.4f] m  离地 %.3f m  距板心 %.3f m"
              % (key, c[0], c[1], c[2], p["height_above_floor_m"],
                 p["distance_to_pattern_centre_m"]))
        print("           重投影 %.4f px  max %.4f px  对应=%s  帧=%s"
              % (p["reprojection"]["rms_px"], p["reprojection"]["max_px"],
                 p["correspondence"],
                 len((rig.get("frames", {}).get(key) or []))))
        print("           光轴(世界) [%.3f, %.3f, %.3f]  det(R)=%.9f"
              % (p["optical_axis_world"][0], p["optical_axis_world"][1],
                 p["optical_axis_world"][2], p["rotation"]["determinant"]))
    rel = rig["relative"]
    print("  基线 = %.3f mm  相对旋转 = %.4f°"
          % (rel["baseline_m"] * 1000, rel["rotation_angle_deg"]))
    print("  X_cam1 = R_10 @ X_cam0 + T_10   T_10 = [%.6f, %.6f, %.6f] m"
          % tuple(rel["T_10"]))
    cons = rig["consistency"]
    print("  三角化自洽（对已知世界点）：rms %.4f mm  max %.4f mm"
          % (cons["rms_m"] * 1000, cons["max_m"] * 1000))
    for k, st in rig.get("stability", {}).items():
        print("  相机 %s 帧间抖动：中心标准差 %.4f mm  最大偏离 %.4f mm"
              % (k, st["centre_std_m"] * 1000, st["centre_range_m"] * 1000))
    for w in rig.get("warnings", []):
        print("  ! %s" % w)
    print("  · %s" % rig.get("caveat", ""))


def _print_rig_validation(val: dict) -> None:
    agg = val["aggregate"]
    print("\n独立验证：%d 个位置/高度" % agg["placements"])
    print("  重投影   mean %.3f px   max %.3f px"
          % (agg["reprojection_rms_px"]["mean"], agg["reprojection_rms_px"]["max"]))
    print("  极线     mean %.3f px" % agg["epipolar_median_px"]["mean"])
    sp = agg["spacing_abs_error_mm"]
    print("  网格间距（已知 %.1f mm）平均绝对误差 %.3f mm  p95 %.3f mm  max %.3f mm"
          % (sp["expected_spacing_mm"], sp["mean"], sp["p95"], sp["max"]))
    print("  板心高度范围 %.3f ~ %.3f m（说明覆盖了不同高度）"
          % tuple(agg["centre_z_range_m"]))
    for p in val["per_placement"]:
        print("   %-8s 重投影 %.3fpx  间距误差 %.3fmm  平面度 %.3fmm  板心 z=%.3f"
              % (p["id"], p["reprojection_rms_px"],
                 p["spacing_mean_abs_error_mm"], p["planarity_rms_mm"],
                 p["centre_world_m"][2]))
    for c in val.get("cross_checks", []):
        print("  跨位置距离 %d-%d：期望 %.1fmm 实测 %.1fmm 误差 %.1fmm"
              % (c["i"], c["j"], c["expected_mm"], c["measured_mm"], c["error_mm"]))
    for n in val.get("notes", []):
        print("  · %s" % n)


def cmd_new(args) -> int:
    st = service().open_session(cols=args.cols, rows=args.rows, square_mm=args.square_mm,
                               cameras=args.cameras, session_id=args.id,
                               pixel_format=args.format, note=args.note,
                               surface_height_m=args.surface_height_mm / 1000.0,
                               root=args.root)
    s = st["session"]
    print("已创建会话：%s" % s["root"])
    wf = s["board"]["world_frame"]
    print("  标定板 %dx%d 内角点（%d×%d 方格），方格 %g mm，图像 %dx%d"
          % (s["board"]["cols"], s["board"]["rows"],
             s["board"]["squares"][0], s["board"]["squares"][1],
             s["board"]["square_mm"], s["image_size"][0], s["image_size"][1]))
    print("  世界系：%s；板面高度 %.1f mm（地面 Z=%.3f m）"
          % (wf["origin"], wf["surface_height_m"] * 1000, wf["floor_z_m"]))
    print("  共 %d 个角点，X∈[%.3f, %.3f] m，Y∈[%.3f, %.3f] m"
          % (s["board"]["inner_corners"],
             -wf["board_extent_x_m"] / 2 + s["board"]["square_m"],
             wf["board_extent_x_m"] / 2 - s["board"]["square_m"],
             -wf["board_extent_y_m"] / 2 + s["board"]["square_m"],
             wf["board_extent_y_m"] / 2 - s["board"]["square_m"]))
    return 0


def _ensure_cameras(indices) -> None:
    from camera_manager import manager
    m = manager()
    missing = [i for i in indices if m.worker(i) is None]
    if missing:
        _die("相机 %s 未启动。请先启动预览服务（python api/preview_server.py），"
             "本工具不会自己打开相机" % missing)
    not_ready = [i for i in indices
                 if m.worker(i) is not None and m.worker(i).state != "streaming"]
    if not_ready:
        _die("相机 %s 还没进入 streaming 状态" % not_ready)


def cmd_capture_single(args) -> int:
    svc = service()
    if svc.session is None:
        svc.reload_session(args.session)
    _ensure_cameras([args.camera])
    ok = 0
    for k in range(args.count):
        try:
            r = svc.capture_single(args.camera, force=args.force, phase=args.phase)
        except Exception as exc:
            print("第 %d 次采集失败：%s" % (k + 1, exc), file=sys.stderr)
            if not args.keep_going:
                break
            time.sleep(args.interval)
            continue
        print("[%2d/%2d] %s" % (k + 1, args.count, r["message"]))
        ok += 1 if r["ok"] else 0
        if k + 1 < args.count:
            time.sleep(args.interval)
    print("\n本次有效 %d / %d 张。移动标定板换姿态后再次运行本命令。" % (ok, args.count))
    return 0


def cmd_capture_stereo(args) -> int:
    svc = service()
    if svc.session is None:
        svc.reload_session(args.session)
    _ensure_cameras(svc.session.cameras[:2])
    ok = 0
    for k in range(args.count):
        try:
            r = svc.capture_stereo(force=args.force)
        except Exception as exc:
            print("第 %d 对采集失败：%s" % (k + 1, exc), file=sys.stderr)
            if not args.keep_going:
                break
            time.sleep(args.interval)
            continue
        print("[%2d/%2d] %s  Δt=%s ms" % (k + 1, args.count, r["message"], r.get("dt_ms")))
        ok += 1 if r["ok"] else 0
        if k + 1 < args.count:
            time.sleep(args.interval)
    print("\n本次有效 %d / %d 对。" % (ok, args.count))
    return 0


def cmd_detect(args) -> int:
    """对已落盘的图重新检测（换了检测参数或想复核时用），并刷新有效性判定。"""
    sess = _session(args.session)
    spec = sess.spec
    svc = service()
    svc._session = sess          # 复用服务里的判重/阈值逻辑

    ok_single = ok_stereo = 0
    for s in sess.samples:
        if s.kind == "single":
            cam = next(iter(s.images))
            img = cv2.imread(os.path.join(sess.img_dir, s.images[cam]["file"]),
                             cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            det = B.detect(img, spec)
            s.detections[cam] = det.to_dict()
            s.status = "valid" if det.ok else "invalid"
            s.reason = "" if det.ok else "角点检测失败：%s" % det.reason
            if det.ok:
                s.signature = B.pose_signature(det, spec)
                ok_single += 1
            cv2.imwrite(os.path.join(sess.overlay_dir, s.images[cam]["file"]),
                        B.draw_overlay(img, det, spec, label_step=3))
        else:
            keys = sorted(s.images)
            if len(keys) < 2:
                continue
            a = cv2.imread(os.path.join(sess.img_dir, s.images[keys[0]]["file"]),
                           cv2.IMREAD_UNCHANGED)
            b = cv2.imread(os.path.join(sess.img_dir, s.images[keys[1]]["file"]),
                           cv2.IMREAD_UNCHANGED)
            if a is None or b is None:
                continue
            da, db, info = B.detect_pair(a, b, spec)
            s.detections[keys[0]] = da.to_dict()
            s.detections[keys[1]] = db.to_dict()
            good = da.ok and db.ok and info.get("ordering_consistent", True)
            s.status = "valid" if good else "invalid"
            s.reason = "" if good else (da.reason or db.reason or "两视角不一致")
            if good:
                s.signature = B.pose_signature(da, spec)
                ok_stereo += 1
            cv2.imwrite(os.path.join(sess.overlay_dir, s.images[keys[0]]["file"]),
                        B.draw_overlay(a, da, spec, label_step=3))
            cv2.imwrite(os.path.join(sess.overlay_dir, s.images[keys[1]]["file"]),
                        B.draw_overlay(b, db, spec, label_step=3))
    sess.save()
    print("重检测完成：有效单相机样本 %d，有效帧对 %d（共 %d 个样本）"
          % (ok_single, ok_stereo, len(sess.samples)))
    print("叠加图已更新到 %s" % sess.overlay_dir)
    return 0


def cmd_intrinsic(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    r = svc.run_intrinsic(args.cameras, exclude=args.exclude)
    _print_intrinsic(r["intrinsics"])
    for cam, msg in (r.get("errors") or {}).items():
        print("\n相机 %s 未完成：%s" % (cam, msg), file=sys.stderr)
    print("\n中间结果：%s" % r["saved_to"])
    return 0


def cmd_stereo(args) -> int:
    """stereoCalibrate 交叉校验（不是主路线，主路线是 solvePnP）。"""
    svc = service()
    svc.reload_session(args.session)
    if args.intrinsic:
        if os.path.isfile(args.intrinsic):
            with open(args.intrinsic, encoding="utf-8") as fh:
                data = json.load(fh)
            svc._intrinsics = data.get("intrinsics", data)
        else:
            _die("内参文件不存在：%s" % args.intrinsic)
    else:
        svc.run_intrinsic(None, exclude=args.exclude)
    res = svc.run_stereo(exclude=args.exclude,
                         auto_flip=not args.no_auto_flip,
                         flip_b=args.flip_b)
    _print_stereo(res)
    print("\n中间结果：%s" % os.path.join(svc.session.root, "result_stereo.json"))
    return 0


def cmd_extrinsic(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    if args.intrinsic:
        if os.path.isfile(args.intrinsic):
            with open(args.intrinsic, encoding="utf-8") as fh:
                data = json.load(fh)
            svc._intrinsics = data.get("intrinsics", data)
        else:
            _die("内参文件不存在：%s" % args.intrinsic)
    else:
        svc.run_intrinsic(None, exclude=args.exclude)
    rig = svc.solve_extrinsic(flip180=args.flip180, exclude=args.exclude)
    _print_rig(rig)
    print("\n中间结果：%s" % os.path.join(svc.session.root, "result_extrinsic.json"))
    return 0


def cmd_validate_rig(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    svc.run_intrinsic(None)
    if os.path.isfile(os.path.join(svc.session.root, "result_extrinsic.json")):
        with open(os.path.join(svc.session.root, "result_extrinsic.json"),
                  encoding="utf-8") as fh:
            svc._extrinsic = json.load(fh)
    else:
        svc._extrinsic = svc.solve_extrinsic(flip180=args.flip180)
    val = svc.validate_rig(holdout=args.holdout)
    _print_rig_validation(val)
    return 0


def cmd_save(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    svc.run_intrinsic(None, exclude=args.exclude)
    svc.solve_extrinsic(flip180=args.flip180, exclude=args.exclude)
    if args.validate:
        svc.validate_rig(holdout=False)
    r = svc.save_result(args.out)
    print("\n已保存标定结果：%s" % r["path"])
    print("  会话副本：%s" % r["session_copy"])
    print("  方法 %s  基线 %.2fmm  重投影 %.4fpx"
          % (r["method"], (r["baseline_m"] or 0) * 1000, r["stereo_rms_px"] or -1))
    return 0


def cmd_validate(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    svc.run_intrinsic(None)
    r = svc.validate(holdout=args.holdout,
                     auto_flip=None if not args.no_auto_flip else False,
                     expected_mm=args.expected_mm,
                     corner_pair=tuple(args.corner_pair) if args.corner_pair else None)
    print("\n切分：训练 %d 对 / 验证 %d 对" % (r["split"]["train"], r["split"]["validation"]))

    def show(tag, d):
        if not d:
            return
        print("\n%s" % tag)
        print("  极线 median %.4f / p95 %.4f / max %.4f px"
              % (d["epipolar"]["median_px"], d["epipolar"]["p95_px"], d["epipolar"]["max_px"]))
        print("  三角化 median %.4f / p95 %.4f px"
              % (d["triangulation"]["median_px"], d["triangulation"]["p95_px"]))
        kd = d.get("known_distance") or {}
        if kd:
            print("  已知间距误差 mean %.4fmm（相对 %.3f%%）最大 %.4fmm"
                  % (kd["mean_error_mm"], kd["mean_relative_error_pct"],
                     kd["max_abs_error_mm"]))
        print("  落在两相机前方 %.1f%%" % (100 * d["checks"]["in_front_ratio"]))
        worst = sorted(d["per_pair"], key=lambda p: -(p["epipolar_median_px"] or 0))[:3]
        if worst:
            print("  极线误差最大的帧对：%s"
                  % "，".join("%s=%.2fpx" % (p["sample"], p["epipolar_median_px"]) for p in worst))

    show("训练集（自洽评估，偏乐观）", r["on_train"])
    show("验证集（独立参考）", r["on_validation"])
    for s in r["interpretation"]:
        print("\n· %s" % s)
    return 0


def cmd_save(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    svc.run_intrinsic(None, exclude=args.exclude)
    svc.run_stereo(exclude=args.exclude, auto_flip=not args.no_auto_flip)
    if args.validate:
        svc.validate(holdout=args.holdout)
    r = svc.save_result(args.out)
    print("\n已保存标定结果：%s" % r["path"])
    print("  会话副本：%s" % r["session_copy"])
    print("  基线 %.2fmm  双目 RMS %.4fpx"
          % (r["baseline_m"] * 1000, r["stereo_rms_px"]))
    return 0


def cmd_info(args) -> int:
    path = args.result or DEFAULT_RESULT_PATH
    if not os.path.isfile(path):
        _die("没有标定结果文件：%s" % path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    print("文件：%s" % path)
    print("标定时间：%s   schema：%s v%s"
          % (d.get("created"), d.get("schema"), d.get("version")))
    b = d.get("board", {})
    print("标定板：%sx%s 内角点（%s×%s 方格），方格 %s mm，板面高 %s mm"
          % (b.get("cols"), b.get("rows"),
             (b.get("squares") or ["?", "?"])[0], (b.get("squares") or ["?", "?"])[1],
             b.get("square_mm"), round(1000 * (b.get("surface_height_m") or 0), 1)))
    wf = d.get("world_frame") or {}
    if wf:
        print("世界系：%s；%s；%s" % (wf.get("origin"), wf.get("x_axis"), wf.get("z_axis")))
        print("        地面 Z = %.3f m" % wf.get("floor_z_m", 0.0))
    for cam, it in (d.get("intrinsics") or {}).items():
        print("相机 %s：RMS %.4fpx  fx=%.3f  fy=%.3f"
              % (cam, it["rms_px"], it["K"][0][0], it["K"][1][1]))
    e = d.get("extrinsics") or {}
    if e:
        print("外参方法：%s" % e.get("method"))
        for cam, c in (e.get("cameras") or {}).items():
            pos = c.get("camera_centre_world_m") or [0, 0, 0]
            print("  相机 %s：位置 [%.3f, %.3f, %.3f] m  离地 %.3f m  重投影 %.4f px"
                  % (cam, pos[0], pos[1], pos[2],
                     c.get("height_above_floor_m") or 0,
                     c.get("reprojection_rms_px") or 0))
        print("  基线 %.3f mm  相对旋转 %.3f°  flip180=%s"
              % ((e.get("baseline_m") or 0) * 1000,
                 e.get("rotation_angle_deg") or 0, e.get("flip180_global")))
    v = d.get("validation")
    if v and v.get("aggregate"):
        agg = v["aggregate"]
        print("独立验证：%d 个位置  间距误差 mean %.3fmm  max %.3fmm  重投影 %.3fpx"
              % (agg["placements"], agg["spacing_abs_error_mm"]["mean"],
                 agg["spacing_abs_error_mm"]["max"],
                 agg["reprojection_rms_px"]["mean"]))
    else:
        print("独立验证：未做（当前结果只有自洽评估）")
    for c in d.get("caveats", []):
        print("· %s" % c)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="双相机内外参标定命令行工具")
    p.add_argument("--root", default=None, help="标定会话根目录（默认 captures/calibration）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="列出已有标定会话")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("new", help="新建标定会话")
    sp.add_argument("--cols", type=int, default=B.DEFAULT_COLS, help="内角点列数（12 格 → 11）")
    sp.add_argument("--rows", type=int, default=B.DEFAULT_ROWS, help="内角点行数（9 格 → 8）")
    sp.add_argument("--square-mm", type=float, default=B.DEFAULT_SQUARE_MM, help="方格边长 mm")
    sp.add_argument("--surface-height-mm", type=float, default=0.0,
                    help="板面高于地面的高度 mm（用于地面基准换算）")
    sp.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    sp.add_argument("--id", default=None, help="会话名（默认时间戳）")
    sp.add_argument("--format", default="", help="采集格式记录（NV12/MJPG）")
    sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_new)

    sp = sub.add_parser("capture-single", help="采单相机标定图")
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--count", type=int, default=10)
    sp.add_argument("--interval", type=float, default=1.5, help="两次采集间隔秒")
    sp.add_argument("--phase", choices=["intrinsic", "extrinsic"], default="intrinsic",
                    help="intrinsic=移动标定板采内参；extrinsic=固定板采外参")
    sp.add_argument("--session", default=None)
    sp.add_argument("--force", action="store_true", help="允许重复姿态")
    sp.add_argument("--keep-going", action="store_true")
    sp.set_defaults(func=cmd_capture_single)

    sp = sub.add_parser("capture-stereo", help="采双相机同步帧对")
    sp.add_argument("--count", type=int, default=10)
    sp.add_argument("--interval", type=float, default=2.0)
    sp.add_argument("--session", default=None)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--keep-going", action="store_true")
    sp.set_defaults(func=cmd_capture_stereo)

    sp = sub.add_parser("detect", help="对已落盘图像重新检测（离线）")
    sp.add_argument("--session", required=True)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("intrinsic", help="内参标定（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--cameras", type=int, nargs="*", default=None)
    sp.add_argument("--exclude", nargs="*", default=(), help="要排除的样本 id")
    sp.set_defaults(func=cmd_intrinsic)

    sp = sub.add_parser("extrinsic", help="世界系外参标定（固定板 + solvePnP，离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--intrinsic", default=None, help="直接指定 result_intrinsic.json")
    sp.add_argument("--exclude", nargs="*", default=())
    sp.add_argument("--flip180", action="store_true",
                    help="全局绕板法向转 180°（两台一起，人工确认用）")
    sp.set_defaults(func=cmd_extrinsic)

    sp = sub.add_parser("validate-rig", help="独立验证：多位置/高度三角化精度（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--holdout", action="store_true", help="只取三分之一的验证帧对")
    sp.add_argument("--flip180", action="store_true")
    sp.set_defaults(func=cmd_validate_rig)

    sp = sub.add_parser("stereo", help="stereoCalibrate 交叉校验（非主路线，离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--intrinsic", default=None, help="直接指定 result_intrinsic.json")
    sp.add_argument("--exclude", nargs="*", default=())
    sp.add_argument("--no-auto-flip", action="store_true", help="关闭角点顺序自动择向")
    sp.add_argument("--flip-b", dest="flip_b", action="store_true", default=None,
                    help="强制翻转相机1角点顺序")
    sp.set_defaults(func=cmd_stereo)

    sp = sub.add_parser("validate", help="留出集独立验证（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--holdout", type=float, default=0.3, help="验证集比例，0 表示不切分")
    sp.add_argument("--expected-mm", type=float, default=None,
                    help="自定义已知物理距离（需配 --corner-pair）")
    sp.add_argument("--corner-pair", type=int, nargs=2, default=None,
                    metavar=("I", "J"), help="自定义距离对应的两个角点编号")
    sp.add_argument("--no-auto-flip", action="store_true")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("save", help="求解并保存 calibration.json（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--out", default=None, help="输出路径（默认 api/calibration/calibration.json）")
    sp.add_argument("--exclude", nargs="*", default=())
    sp.add_argument("--flip180", action="store_true")
    sp.add_argument("--validate", action="store_true", help="顺便跑一次独立验证并写入结果")
    sp.set_defaults(func=cmd_save)

    sp = sub.add_parser("info", help="查看已保存的标定结果")
    sp.add_argument("--result", default=None)
    sp.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "root", None) is None:
        pass
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
