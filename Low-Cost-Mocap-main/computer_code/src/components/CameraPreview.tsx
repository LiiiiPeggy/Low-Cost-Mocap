import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Col, Row } from 'react-bootstrap';

/**
 * 双路 USB 红外相机实时预览（Windows / Media Foundation 后端）。
 *
 * 数据来源：api/camera_preview.py
 *   GET  /api/cameras                 状态（fps / 分辨率 / 帧号 / 时间戳）
 *   POST /api/cameras/start|stop      启停采集
 *   GET  /api/camera/<i>/stream       MJPEG 视频流
 *   POST /api/cameras/save            保存两路原始灰度 PNG
 *
 * 采集线程在后端常驻，本组件只是消费者；页面刷新/多标签页都不会重复打开相机。
 */

const ENV: Record<string, string | undefined> =
  ((import.meta as unknown as { env?: Record<string, string | undefined> }).env) ?? {};

// 默认同源（走 vite 代理）；如后端不在同源，可设 VITE_API_BASE=http://127.0.0.1:3002
const API_BASE = ENV.VITE_API_BASE ?? '';
const CAMERA_INDICES: number[] = (ENV.VITE_CAMERA_INDICES ?? '0,1')
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => !Number.isNaN(n));

interface FrameMeta {
  seq: number;
  width: number;
  height: number;
  mf_timestamp: number;
  mf_timestamp_seconds: number;
  host_timestamp: number;
  dtype: string;
  mean: number;
}

interface CameraStatus {
  index: number;
  name: string | null;
  state: string;
  error: string | null;
  format: string | null;
  fps: number;
  frames_total: number;
  frame: FrameMeta | null;
}

interface SavedFrame {
  index: number;
  path: string;
  seq: number;
  mf_timestamp: number;
  host_time_iso: string;
}

const ACTIVE_STATES = ['opening', 'streaming', 'reconnecting'];

function stateColor(state: string): string {
  if (state === 'streaming') return '#35c46a';
  if (state === 'error') return '#ef5b5b';
  if (ACTIVE_STATES.includes(state)) return '#e8b13a';
  return '#8b93a7';
}

function stateText(state: string): string {
  const map: Record<string, string> = {
    streaming: '采集中',
    opening: '打开中',
    reconnecting: '重连中',
    error: '异常',
    stopped: '已停止',
    idle: '未启动',
  };
  return map[state] ?? state;
}

export default function CameraPreview() {
  const [status, setStatus] = useState<CameraStatus[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SavedFrame[]>([]);
  const [busy, setBusy] = useState(false);
  // 每次启动/重连时递增，用来强制 <img> 重新建立 MJPEG 连接
  const [streamKey, setStreamKey] = useState(0);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/cameras`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatus(data.cameras ?? []);
      setRunning((data.cameras ?? []).some((c: CameraStatus) => ACTIVE_STATES.includes(c.state)));
      setError(null);
      setNow(Date.now() / 1000);
    } catch (e) {
      setError(`后端未连接：${(e as Error).message}（请先启动 api/preview_server.py）`);
    }
  }, []);

  useEffect(() => {
    refresh();
    timer.current = window.setInterval(refresh, 1000);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, [refresh]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/cameras/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indices: CAMERA_INDICES }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setStreamKey((k) => k + 1);
      setStatus(data.cameras ?? []);
      setRunning(true);
    } catch (e) {
      setError(`启动失败：${(e as Error).message}`);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/cameras/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setStatus(data.cameras ?? []);
      setRunning(false);
    } catch (e) {
      setError(`停止失败：${(e as Error).message}`);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const saveRaw = async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/cameras/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setSaved(data.saved ?? []);
      if (!data.count) setError('没有可保存的帧：相机还没开始采集');
    } catch (e) {
      setError(`保存失败：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const byIndex = useMemo(() => {
    const map: Record<number, CameraStatus> = {};
    status.forEach((c) => { map[c.index] = c; });
    return map;
  }, [status]);

  const liveCount = status.filter((c) => c.state === 'streaming').length;

  return (
    <div>
      <Row className="mb-2" style={{ alignItems: 'center' }}>
        <Col xs="auto">
          <h4 className="mb-0">IR Cameras (Windows)</h4>
        </Col>
        <Col>
          <Button size="sm" className="me-2" variant={running ? 'outline-danger' : 'outline-primary'}
            disabled={busy} onClick={running ? stop : start}>
            {running ? 'Stop' : 'Start'}
          </Button>
          <Button size="sm" variant="outline-secondary" className="me-3"
            disabled={busy || !status.length} onClick={saveRaw}>
            Save raw
          </Button>
          <span style={{ fontSize: 13, color: '#6c757d' }}>
            {status.length ? `在线 ${liveCount}/${status.length} 路` : '未启动'}
            {status[0]?.format ? ` · ${status[0].format}` : ''}
          </span>
        </Col>
      </Row>

      {error && (
        <div className="mb-2" style={{ fontSize: 13, color: '#c0392b' }}>{error}</div>
      )}

      <Row className="g-3">
        {CAMERA_INDICES.map((idx) => {
          const cam = byIndex[idx];
          const frame = cam?.frame ?? null;
          const lagMs = frame ? Math.max(0, (now - frame.host_timestamp) * 1000) : null;
          return (
            <Col md={6} key={idx}>
              <div style={{ border: '1px solid #dee2e6', borderRadius: 8, overflow: 'hidden' }}>
                <div className="d-flex align-items-center px-2 py-1"
                  style={{ gap: 8, borderBottom: '1px solid #dee2e6', background: '#f8f9fa' }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: stateColor(cam?.state ?? 'idle'), display: 'inline-block',
                  }} />
                  <b style={{ fontSize: 13 }}>Camera {idx}</b>
                  <span style={{ fontSize: 12, color: '#6c757d' }}>
                    {cam?.name ?? '—'} · {cam ? stateText(cam.state) : '未启动'}
                  </span>
                </div>

                <div style={{ background: '#000', aspectRatio: '16 / 9', position: 'relative' }}>
                  {running ? (
                    <img
                      alt={`camera ${idx}`}
                      src={`${API_BASE}/api/camera/${idx}/stream?k=${streamKey}`}
                      style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                    />
                  ) : (
                    <div style={{
                      position: 'absolute', inset: 0, display: 'flex',
                      alignItems: 'center', justifyContent: 'center',
                      color: '#8b93a7', fontSize: 13,
                    }}>
                      {cam?.state === 'error' ? cam.error ?? '异常' : '点击 Start 开始采集'}
                    </div>
                  )}
                </div>

                <div className="px-2 py-1"
                  style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', fontSize: 12 }}>
                  <div><span style={{ color: '#6c757d' }}>分辨率</span><br />
                    <b>{frame ? `${frame.width}×${frame.height}` : '—'}</b></div>
                  <div><span style={{ color: '#6c757d' }}>FPS</span><br />
                    <b>{cam ? cam.fps.toFixed(1) : '—'}</b></div>
                  <div><span style={{ color: '#6c757d' }}>帧号</span><br />
                    <b>{frame ? frame.seq : '—'}</b></div>
                  <div><span style={{ color: '#6c757d' }}>滞后</span><br />
                    <b>{lagMs === null ? '—' : `${lagMs.toFixed(0)} ms`}</b></div>
                </div>

                {frame && (
                  <div className="px-2 pb-1"
                    style={{ fontSize: 11, color: '#8b93a7', fontFamily: 'monospace' }}>
                    mf_ts={frame.mf_timestamp} ({frame.mf_timestamp_seconds.toFixed(3)}s)
                  </div>
                )}
                {cam?.error && (
                  <div className="px-2 pb-1" style={{ fontSize: 11, color: '#c0392b' }}>
                    {cam.error}
                  </div>
                )}
              </div>
            </Col>
          );
        })}
      </Row>

      {saved.length > 0 && (
        <div className="mt-2" style={{ fontSize: 12, fontFamily: 'monospace', color: '#495057' }}>
          {saved.map((s) => (
            <div key={s.path}>
              已保存 cam {s.index} → {s.path}（seq={s.seq}，mf_ts={s.mf_timestamp}）
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
