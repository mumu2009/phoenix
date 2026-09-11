import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';

export default function OpsPanel({ onError, isAdmin }) {
  const [monitor, setMonitor] = useState(null);
  const [modules, setModules] = useState([]);
  const [db, setDb] = useState(null);
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState('');

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const [m, mods, database] = await Promise.all([api.opsMonitor(), api.opsModules(), api.opsDatabase()]);
      setMonitor(m);
      setModules(mods?.modules || []);
      setDb(database);
      setInfo('');
    } catch (e) {
      onError?.(e.message);
    } finally {
      setBusy(false);
    }
  }, [onError]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function toggle(id, enabled) {
    setBusy(true);
    try {
      const out = await api.opsModuleSet(id, enabled);
      setModules(out?.modules || []);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function backup() {
    setBusy(true);
    try {
      const out = await api.opsDatabaseBackup();
      setInfo(`备份已写入 ${out?.backupPath || ''}`);
      const database = await api.opsDatabase();
      setDb(database);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="cfg-card">
      <div className="cfg-card-title">运行状态</div>
      <div className="cfg-label-hint" style={{ marginBottom: 12 }}>
        推理进程只看 pid 与端口，不会请求其 HTTP 健康接口，也不会提供停止按钮。说明见仓库{' '}
        <code>phoenix/doc/product_ops.md</code>。
      </div>
      <button className="btn btn-ghost" onClick={refresh} disabled={busy}>
        {busy ? '刷新中…' : '刷新'}
      </button>
      <div className="status-row" style={{ marginTop: 12 }}>
        <span className="muted">
          本进程 pid={monitor?.pid ?? '-'} 内存={Number(monitor?.memory?.rssMB || 0).toFixed(1)}MB
        </span>
      </div>
      <div className="status-row">
        <span className="muted">
          请求数={monitor?.requests?.total ?? 0} 错误={monitor?.requests?.errors ?? 0}
        </span>
      </div>
      <div className="section-title" style={{ marginTop: 16 }}>端口与存活</div>
      {(monitor?.endpoints || []).map((ep) => (
        <div key={ep.id} className="status-row">
          <span className={`dot ${ep.alive ? 'ok' : 'bad'}`} />
          <span className="muted">
            {ep.title} {ep.host}:{ep.port} {ep.pid ? `pid=${ep.pid}` : ''} 探测={ep.probe}
            {ep.alive ? ' 存活' : ' 未连通'}
          </span>
        </div>
      ))}

      <div className="section-title" style={{ marginTop: 16 }}>逻辑模块</div>
      {modules.map((m) => (
        <div key={m.id} className="cfg-row">
          <div className="cfg-label">
            <div className="cfg-label-title">{m.title}</div>
            <div className="cfg-label-hint">{m.readonly ? m.note || '只读' : m.id}</div>
          </div>
          <div className="cfg-control">
            {m.readonly || !isAdmin ? (
              <span className="muted">{m.enabled ? '开' : '关'}</span>
            ) : (
              <button className="btn btn-ghost" disabled={busy} onClick={() => toggle(m.id, !m.enabled)}>
                {m.enabled ? '关闭' : '开启'}
              </button>
            )}
          </div>
        </div>
      ))}

      <div className="section-title" style={{ marginTop: 16 }}>数据库</div>
      <div className="muted">引擎：{db?.engine || '-'}</div>
      <div className="muted">路径：{db?.path || '-'}</div>
      <div className="muted">LMDB：{db?.legacyDir || '-'}</div>
      <div className="muted">备份目录：{db?.backupDir || '-'}</div>
      <div className="muted">状态：{db?.healthy ? '可用' : '未就绪'}</div>
      {isAdmin ? (
        <button className="btn" style={{ marginTop: 10 }} disabled={busy} onClick={backup}>
          备份主库
        </button>
      ) : null}
      {info ? <div className="cfg-label-hint" style={{ marginTop: 8 }}>{info}</div> : null}
    </div>
  );
}
