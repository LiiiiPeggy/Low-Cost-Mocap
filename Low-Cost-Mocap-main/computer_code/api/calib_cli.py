"""标定命令行工具：采集 / 重检测 / 内参 / 外参 / 验证 / 保存，全流程可在命令行完成。

在 ``Low-Cost-Mocap-main/computer_code`` 目录下运行::

    python api/calib_cli.py list
    python api/calib_cli.py new --cols 11 --rows 8 --square-mm 30 --surface-height-mm 3
    python api/calib_cli.py capture-single --camera 0 --count 20 --interval 1.5
    python api/calib_cli.py capture-stereo --count 3 --interval 2.0
    python api/calib_cli.py detect --session captures/calibration/20261113_120000
    python api/calib_cli.py intrinsic --session <dir>
    python api/calib_cli.py stereo    --session <dir>
    python api/calib_cli.py validate-rig --session <dir> --holdout
    python api/calib_cli.py save      --session <dir> [--out <file>]
    python api/calib_cli.py info      --result <dir>/calibration.json

``intrinsic / stereo / validate-rig / save / detect`` **不需要相机在线**：它们只读
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
from calibration.service import (  # noqa: E402
    DEFAULT_SESSION_ROOT, project_relative, resolve_path, service,
)

# 相机线程只存在于跑着 preview_server 的那个进程里，所以采集默认走 HTTP。
DEFAULT_API = os.environ.get("MOCAP_CALIB_API", "http://127.0.0.1:3002")


def _die(msg: str, code: int = 1):
    print("错误：%s" % msg, file=sys.stderr)
    raise SystemExit(code)


def _session(root: str) -> Session:
    if not root:
        _die("需要 --session <会话目录>")
    path = resolve_path(root)
    if not os.path.isdir(path):
        _die("会话目录不存在：%s" % path)
    return Session.load(path)


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
        grid = d.get("coverage") or {}
        if grid:
            print("  区域覆盖率 = %.0f%%（6×6 网格，%d 个角点命中）"
                  % (100 * grid.get("covered_ratio", 0), grid.get("corner_hits", 0)))
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
    root = resolve_path(args.root) if args.root else DEFAULT_SESSION_ROOT
    if not os.path.isdir(root):
        print("没有标定目录：%s" % project_relative(root))
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
    fc = rig.get("flip_cam") or {}
    print("\n外参（固定棋盘格 + solvePnP）  flip_cam=%s  方向已人工确认=%s"
          % (fc, rig.get("orientation_confirmed")))
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
        if st.get("centre_std_m") is None:
            print("  相机 %s 帧间抖动：%s" % (k, st.get("note") or "样本不足"))
        else:
            print("  相机 %s 帧间抖动：中心标准差 %.4f mm  最大偏离 %.4f mm（用第 %s 帧）"
                  % (k, st["centre_std_m"] * 1000, st["centre_range_m"] * 1000,
                     st.get("used_frame_index")))
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
    payload = {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm,
               "surface_height_mm": args.surface_height_mm, "cameras": args.cameras,
               "session_id": args.id, "format": args.format, "note": args.note,
               "root": args.root}
    st = None
    if not args.local:
        try:
            st = _api_call("/api/calib/session", payload, base=args.api)
        except RuntimeError as exc:
            # 服务没起来就本地建目录，尺寸退化为 1280×720（与硬件一致）
            print("[提示] 服务不可用（%s），改为本地创建会话" % exc, file=sys.stderr)
    if st is None:
        st = service().open_session(
            cols=args.cols, rows=args.rows, square_mm=args.square_mm,
            cameras=args.cameras, session_id=args.id,
            pixel_format=args.format, note=args.note,
            surface_height_m=args.surface_height_mm / 1000.0, root=args.root)
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
        _die("相机 %s 未启动。--local 只对本进程打开的相机有效；"
             "正常用法是先在另一个窗口运行 preview_server，再用 --api 通过 HTTP 采集"
             % missing)
    not_ready = [i for i in indices
                 if m.worker(i) is not None and m.worker(i).state != "streaming"]
    if not_ready:
        _die("相机 %s 还没进入 streaming 状态" % not_ready)


def _api_call(path: str, payload=None, base: str | None = None, timeout: float = 10.0):
    """调用运行中的预览服务（相机线程只存在于那个进程里）。"""
    import urllib.error
    import urllib.request

    base = (base or DEFAULT_API).rstrip("/")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            msg = json.loads(body).get("error", body)
        except ValueError:
            msg = body
        raise RuntimeError("%s 返回 %d：%s" % (path, exc.code, msg))
    except urllib.error.URLError as exc:
        raise RuntimeError("连不上 %s：%s（先运行 preview_server，或用 --local）"
                           % (base, exc.reason))


def _load_session_over_http(args) -> None:
    root = resolve_path(args.session) if args.session else ""
    if not root or not os.path.isdir(root):
        _die("需要 --session <会话目录>（calib_cli.py list 可查看已有会话）")
    _api_call("/api/calib/session/load", {"root": root}, base=args.api)


def _capture_loop(args, *, stereo: bool) -> int:
    _load_session_over_http(args)
    ok = 0
    for k in range(args.count):
        try:
            if stereo:
                r = _api_call("/api/calib/capture/stereo", {"force": args.force},
                              base=args.api)
            else:
                r = _api_call("/api/calib/capture/single",
                              {"camera": args.camera, "phase": args.phase,
                               "force": args.force}, base=args.api)
        except RuntimeError as exc:
            print("第 %d 次采集失败：%s" % (k + 1, exc), file=sys.stderr)
            if not args.keep_going:
                break
            time.sleep(args.interval)
            continue
        suffix = ("  Δt=%s ms" % r.get("dt_ms")) if stereo else ""
        print("[%2d/%2d] %s%s" % (k + 1, args.count, r.get("message", ""), suffix))
        if r.get("warning"):
            print("  ! %s" % r["warning"])
        ok += 1 if r.get("ok") else 0
        if k + 1 < args.count:
            time.sleep(args.interval)
    print("\n本次有效 %d / %d %s。" % (ok, args.count, "对" if stereo else "张"))
    return 0


def cmd_capture_single(args) -> int:
    if not args.local:
        return _capture_loop(args, stereo=False)
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
        if r.get("warning"):
            print("  ! %s" % r["warning"])
        ok += 1 if r["ok"] else 0
        if k + 1 < args.count:
            time.sleep(args.interval)
    print("\n本次有效 %d / %d 张。移动标定板换姿态后再次运行本命令。" % (ok, args.count))
    return 0


def cmd_capture_stereo(args) -> int:
    if not args.local:
        return _capture_loop(args, stereo=True)
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
    if r.get("latest_saved"):
        print("\n最新内参已保存：%s" % r["latest_path"])
    else:
        print("\n计算成功但最新内参文件保存失败：%s" % r.get("latest_error"),
              file=sys.stderr)
    for msg in (r.get("save_errors") or []):
        print("  ! %s" % msg, file=sys.stderr)
    print("\n中间结果：%s" % (r.get("saved_to") or "未保存（见上方错误）"))
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


def _flip_cam_arg(args):
    """把 --flip-cam0/--flip-cam1 转成 {"0":bool,"1":bool}；都没给就是 None。"""
    if not (getattr(args, "flip_cam0", False) or getattr(args, "flip_cam1", False)):
        return None
    return {"0": bool(args.flip_cam0), "1": bool(args.flip_cam1)}


def _ensure_intrinsics(svc, args) -> None:
    """优先用会话里已保存的内参，没有再重算（避免每次求解都覆盖内参）。"""
    if getattr(args, "intrinsic", None):
        if not os.path.isfile(args.intrinsic):
            _die("内参文件不存在：%s" % args.intrinsic)
        with open(args.intrinsic, encoding="utf-8") as fh:
            data = json.load(fh)
        svc._intrinsics = data.get("intrinsics", data)
        return
    if not svc._intrinsics:
        svc.run_intrinsic(None, exclude=getattr(args, "exclude", ()))


def cmd_extrinsic(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    _ensure_intrinsics(svc, args)
    rig = svc.solve_extrinsic(flip180=args.flip180, flip_cam=_flip_cam_arg(args),
                              orientation_confirmed=args.orientation_confirmed,
                              exclude=args.exclude)
    _print_rig(rig)
    if rig.get("latest_saved"):
        print("\n最新外参已保存：%s" % rig["latest_path"])
    else:
        print("\n计算成功但最新外参文件保存失败：%s" % rig.get("latest_error"),
              file=sys.stderr)
    for msg in (rig.get("save_errors") or []):
        print("  ! %s" % msg, file=sys.stderr)
    print("\n中间结果：%s"
          % (rig.get("saved_to") or "未保存（见上方错误）"))
    return 0


def cmd_validate_rig(args) -> int:
    """独立验证：只读已保存的内参/外参，**不会**改写它们。"""
    svc = service()
    svc.reload_session(args.session)
    _ensure_intrinsics(svc, args)
    if svc._extrinsic is None:
        svc.solve_extrinsic(flip180=args.flip180, flip_cam=_flip_cam_arg(args),
                            orientation_confirmed=args.orientation_confirmed)
    val = svc.validate_rig(holdout=args.holdout)
    _print_rig_validation(val)
    return 0


def cmd_save(args) -> int:
    svc = service()
    svc.reload_session(args.session)
    _ensure_intrinsics(svc, args)
    svc.solve_extrinsic(flip180=args.flip180, flip_cam=_flip_cam_arg(args),
                        orientation_confirmed=args.orientation_confirmed,
                        exclude=args.exclude)
    if args.validate:
        svc.validate_rig(holdout=False)
    r = svc.save_result(args.out)
    print("\n已保存标定结果：%s" % r["path"])
    print("  会话副本：%s" % r["session_copy"])
    print("  方法 %s  基线 %.2fmm  最大重投影 %.4fpx  三角化自洽 %.2fmm  方向已确认=%s"
          % (r["method"], (r["baseline_m"] or 0) * 1000,
             r["max_reprojection_rms_px"] if r["max_reprojection_rms_px"] is not None else -1,
             r["consistency_max_mm"] if r["consistency_max_mm"] is not None else -1,
             r["orientation_confirmed"]))
    return 0


def cmd_info(args) -> int:
    path = resolve_path(args.result)
    if not os.path.isfile(path):
        _die("没有标定结果文件：%s" % path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    print("文件：%s" % project_relative(path))
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
        print("  基线 %.3f mm  相对旋转 %.3f°  flip_cam=%s  方向已确认=%s"
              % ((e.get("baseline_m") or 0) * 1000,
                 e.get("rotation_angle_deg") or 0, e.get("flip_cam"),
                 e.get("orientation_confirmed")))
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


def _add_api_args(sp) -> None:
    sp.add_argument("--api", default=DEFAULT_API, help="标定服务地址（默认 %s）" % DEFAULT_API)
    sp.add_argument("--local", action="store_true",
                    help="不用 HTTP，直接在本进程采集（要求相机由本进程打开，一般不用）")


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
    sp.add_argument("--surface-height-mm", type=float, default=3.0,
                    help="板面高于地面的高度 mm（默认 3，用于地面基准换算）")
    sp.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    sp.add_argument("--id", default=None, help="会话名（默认时间戳）")
    sp.add_argument("--format", default="", help="采集格式记录（NV12/MJPG）")
    sp.add_argument("--note", default="")
    _add_api_args(sp)
    sp.set_defaults(func=cmd_new)

    sp = sub.add_parser("capture-single", help="采单相机标定图（默认走 HTTP）")
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--count", type=int, default=10)
    sp.add_argument("--interval", type=float, default=1.5, help="两次采集间隔秒")
    sp.add_argument("--phase", choices=["intrinsic", "extrinsic"], default="intrinsic",
                    help="intrinsic=移动标定板采内参；extrinsic=固定板采外参")
    sp.add_argument("--session", default=None, help="会话目录（必填）")
    sp.add_argument("--force", action="store_true", help="允许重复姿态")
    sp.add_argument("--keep-going", action="store_true")
    _add_api_args(sp)
    sp.set_defaults(func=cmd_capture_single)

    sp = sub.add_parser("capture-stereo", help="采双相机同步帧对（默认走 HTTP）")
    sp.add_argument("--count", type=int, default=10)
    sp.add_argument("--interval", type=float, default=2.0)
    sp.add_argument("--session", default=None, help="会话目录（必填）")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--keep-going", action="store_true")
    _add_api_args(sp)
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
                    help="两台一起绕板法向转 180°（人工确认用）")
    sp.add_argument("--flip-cam0", action="store_true", help="只翻相机 0 的 180°")
    sp.add_argument("--flip-cam1", action="store_true", help="只翻相机 1 的 180°")
    sp.add_argument("--orientation-confirmed", action="store_true",
                    help="已用世界轴叠加人工确认过方向（写入结果）")
    sp.set_defaults(func=cmd_extrinsic)

    sp = sub.add_parser("validate-rig", help="独立验证：多位置/高度三角化精度（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--intrinsic", default=None)
    sp.add_argument("--holdout", action="store_true", help="只取三分之一的验证帧对")
    sp.add_argument("--flip180", action="store_true")
    sp.add_argument("--flip-cam0", action="store_true")
    sp.add_argument("--flip-cam1", action="store_true")
    sp.add_argument("--orientation-confirmed", action="store_true")
    sp.set_defaults(func=cmd_validate_rig)

    sp = sub.add_parser("stereo", help="stereoCalibrate 交叉校验（非主路线，离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--intrinsic", default=None, help="直接指定 result_intrinsic.json")
    sp.add_argument("--exclude", nargs="*", default=())
    sp.add_argument("--no-auto-flip", action="store_true", help="关闭角点顺序自动择向")
    sp.add_argument("--flip-b", dest="flip_b", action="store_true", default=None,
                    help="强制翻转相机1角点顺序")
    sp.set_defaults(func=cmd_stereo)

    sp = sub.add_parser("save", help="保存当前会话的标定结果（离线）")
    sp.add_argument("--session", required=True)
    sp.add_argument("--out", default=None,
                    help="可选导出路径；不传时只写会话目录 calibration.json")
    sp.add_argument("--intrinsic", default=None)
    sp.add_argument("--exclude", nargs="*", default=())
    sp.add_argument("--flip180", action="store_true")
    sp.add_argument("--flip-cam0", action="store_true")
    sp.add_argument("--flip-cam1", action="store_true")
    sp.add_argument("--orientation-confirmed", action="store_true")
    sp.add_argument("--validate", action="store_true", help="顺便跑一次独立验证并写入结果")
    sp.set_defaults(func=cmd_save)

    sp = sub.add_parser("info", help="查看已保存的标定结果")
    sp.add_argument("--result", required=True,
                    help="会话目录下的 calibration.json 或显式导出的结果文件")
    sp.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
