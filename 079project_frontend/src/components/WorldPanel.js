import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

const asNum = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

function Field({ label, children }) {
  return (
    <label className="world-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function JsonPreview({ title, data }) {
  if (!data) return null;
  return (
    <div className="world-card world-card-pre">
      <div className="world-card-head">
        <div className="world-card-title">{title}</div>
      </div>
      <pre className="world-pre">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default function WorldPanel({ activeSessionId, ensureSession, onError }) {
  const [loading, setLoading] = useState(false);
  const [runtimeStatus, setRuntimeStatus] = useState(null);
  const [worldStatus, setWorldStatus] = useState(null);
  const [worldState, setWorldState] = useState(null);
  const [importBusy, setImportBusy] = useState(false);
  const [simulateBusy, setSimulateBusy] = useState(false);
  const [importResult, setImportResult] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [earthForm, setEarthForm] = useState({
    enabled: true,
    sourceUri: '',
    format: 'heightfield',
    regionLabel: 'china-relief-demo',
    lod: '6',
    metersPerCell: '750'
  });
  const [simForm, setSimForm] = useState({
    maxAgents: '6',
    maxSteps: '4',
    physicsEnabled: true,
    physicsBackend: 'bullet3',
    physicsSubsteps: '4',
    include3DMap: true,
    includeEmbodiedAgents: true,
    includeEcologyFromVideo: true,
    persist: true
  });

  const getSessionId = () => {
    const existing = activeSessionId || ensureSession?.()?.id;
    return existing || '';
  };

  const refreshWorldState = async (sessionId = activeSessionId) => {
    if (!sessionId) return;
    try {
      const state = await api.worldState(sessionId, 12);
      setWorldState(state || null);
    } catch (e) {
      onError?.(e.message);
    }
  };

  const refreshRuntime = async () => {
    setLoading(true);
    try {
      const [runtime, status] = await Promise.all([
        api.worldPhysicsStatus(),
        api.worldStatus().catch(() => null)
      ]);
      setRuntimeStatus(runtime || null);
      setWorldStatus(status || null);
      const defaults = runtime?.defaults || {};
      const earthDefaults = defaults.earthMap || {};
      setEarthForm((prev) => ({
        ...prev,
        enabled: earthDefaults.enabled ?? prev.enabled,
        sourceUri: earthDefaults.sourceUri || prev.sourceUri,
        format: earthDefaults.format || prev.format,
        regionLabel: earthDefaults.regionLabel || prev.regionLabel,
        lod: String(earthDefaults.lod ?? prev.lod),
        metersPerCell: String(earthDefaults.metersPerCell ?? prev.metersPerCell)
      }));
      setSimForm((prev) => ({
        ...prev,
        physicsEnabled: defaults.physicsEnabled ?? prev.physicsEnabled,
        physicsBackend: defaults.physicsBackend || prev.physicsBackend,
        physicsSubsteps: String(defaults.physicsSubsteps ?? prev.physicsSubsteps)
      }));
    } catch (e) {
      onError?.(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshRuntime();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeSessionId) {
      refreshWorldState(activeSessionId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  const importEarthMap = async () => {
    const sessionId = getSessionId();
    if (!sessionId) {
      onError?.('sessionId unavailable');
      return;
    }
    setImportBusy(true);
    try {
      const result = await api.worldEarthMapImport({
        sessionId,
        persist: true,
        earthMap: {
          enabled: Boolean(earthForm.enabled),
          sourceUri: earthForm.sourceUri,
          format: earthForm.format,
          regionLabel: earthForm.regionLabel,
          lod: asNum(earthForm.lod, 6),
          metersPerCell: asNum(earthForm.metersPerCell, 750)
        }
      });
      setImportResult(result || null);
      await refreshWorldState(sessionId);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setImportBusy(false);
    }
  };

  const runSimulation = async () => {
    const sessionId = getSessionId();
    if (!sessionId) {
      onError?.('sessionId unavailable');
      return;
    }
    setSimulateBusy(true);
    try {
      const result = await api.worldSimulate({
        sessionId,
        persist: Boolean(simForm.persist),
        maxAgents: asNum(simForm.maxAgents, 6),
        maxSteps: asNum(simForm.maxSteps, 4),
        physicsEnabled: Boolean(simForm.physicsEnabled),
        physicsBackend: simForm.physicsBackend,
        physicsSubsteps: asNum(simForm.physicsSubsteps, 4),
        include3DMap: Boolean(simForm.include3DMap),
        includeEmbodiedAgents: Boolean(simForm.includeEmbodiedAgents),
        includeEcologyFromVideo: Boolean(simForm.includeEcologyFromVideo),
        earthMap: {
          enabled: Boolean(earthForm.enabled),
          sourceUri: earthForm.sourceUri,
          format: earthForm.format,
          regionLabel: earthForm.regionLabel,
          lod: asNum(earthForm.lod, 6),
          metersPerCell: asNum(earthForm.metersPerCell, 750)
        }
      });
      setSimulation(result || null);
      await refreshWorldState(sessionId);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setSimulateBusy(false);
    }
  };

  const recentEvidence = Array.isArray(worldState?.recentEvidence) ? worldState.recentEvidence : [];
  const bodySummaries = Array.isArray(simulation?.physicsExecution?.bodySummaries) ? simulation.physicsExecution.bodySummaries : [];

  return (
    <div className="world-grid">
      <div className="world-card">
        <div className="world-card-head">
          <div>
            <div className="world-card-title">Native Physics Runtime</div>
            <div className="world-card-sub">Bullet embedded-source backend + bundled China heightfield</div>
          </div>
          <button className="btn btn-ghost" onClick={refreshRuntime} disabled={loading}>
            {loading ? '刷新中…' : '刷新 Runtime'}
          </button>
        </div>
        <div className="world-kv">
          <div><span>mode</span><strong>{runtimeStatus?.physicsRuntime?.runtimeMode || 'unknown'}</strong></div>
          <div><span>native</span><strong>{runtimeStatus?.physicsRuntime?.nativeCompiled ? 'compiled' : 'not compiled'}</strong></div>
          <div><span>format</span><strong>{runtimeStatus?.physicsRuntime?.preferredEarthFormat || 'heightfield'}</strong></div>
          <div><span>asset</span><strong>{runtimeStatus?.physicsRuntime?.bundledEarthHeightfieldUri || earthForm.sourceUri}</strong></div>
        </div>
        <div className="world-summary">{runtimeStatus?.physicsRuntime?.summary || '等待 physics runtime 信息。'}</div>
      </div>

      <div className="world-card">
        <div className="world-card-head">
          <div>
            <div className="world-card-title">Earth Terrain Import</div>
            <div className="world-card-sub">导入 heightfield 地形，并将其写入 world model 供后续训练消费</div>
          </div>
          <button className="btn" onClick={importEarthMap} disabled={importBusy}>
            {importBusy ? '导入中…' : '导入地形'}
          </button>
        </div>
        <div className="world-form-grid">
          <Field label="启用地形">
            <input type="checkbox" checked={earthForm.enabled} onChange={(e) => setEarthForm((prev) => ({ ...prev, enabled: e.target.checked }))} />
          </Field>
          <Field label="格式">
            <select value={earthForm.format} onChange={(e) => setEarthForm((prev) => ({ ...prev, format: e.target.value }))}>
              <option value="heightfield">heightfield</option>
              <option value="3dtiles">3dtiles</option>
            </select>
          </Field>
          <Field label="资源路径">
            <input value={earthForm.sourceUri} onChange={(e) => setEarthForm((prev) => ({ ...prev, sourceUri: e.target.value }))} />
          </Field>
          <Field label="区域标签">
            <input value={earthForm.regionLabel} onChange={(e) => setEarthForm((prev) => ({ ...prev, regionLabel: e.target.value }))} />
          </Field>
          <Field label="LOD">
            <input value={earthForm.lod} onChange={(e) => setEarthForm((prev) => ({ ...prev, lod: e.target.value }))} />
          </Field>
          <Field label="每格米数">
            <input value={earthForm.metersPerCell} onChange={(e) => setEarthForm((prev) => ({ ...prev, metersPerCell: e.target.value }))} />
          </Field>
        </div>
        <div className="world-summary">{importResult?.earthMap?.summary || '当前默认值会使用 bundled China relief heightfield。'}</div>
      </div>

      <div className="world-card">
        <div className="world-card-head">
          <div>
            <div className="world-card-title">Embodied Simulation</div>
            <div className="world-card-sub">运行 world rollout、Bullet 真实物理、并把 runtime 样本回灌到 trainSamples</div>
          </div>
          <button className="btn" onClick={runSimulation} disabled={simulateBusy}>
            {simulateBusy ? '仿真中…' : '运行仿真'}
          </button>
        </div>
        <div className="world-form-grid">
          <Field label="Agent 数">
            <input value={simForm.maxAgents} onChange={(e) => setSimForm((prev) => ({ ...prev, maxAgents: e.target.value }))} />
          </Field>
          <Field label="Step 数">
            <input value={simForm.maxSteps} onChange={(e) => setSimForm((prev) => ({ ...prev, maxSteps: e.target.value }))} />
          </Field>
          <Field label="Physics 子步">
            <input value={simForm.physicsSubsteps} onChange={(e) => setSimForm((prev) => ({ ...prev, physicsSubsteps: e.target.value }))} />
          </Field>
          <Field label="Backend">
            <input value={simForm.physicsBackend} onChange={(e) => setSimForm((prev) => ({ ...prev, physicsBackend: e.target.value }))} />
          </Field>
          <Field label="持久化 world state">
            <input type="checkbox" checked={simForm.persist} onChange={(e) => setSimForm((prev) => ({ ...prev, persist: e.target.checked }))} />
          </Field>
          <Field label="启用 Physics">
            <input type="checkbox" checked={simForm.physicsEnabled} onChange={(e) => setSimForm((prev) => ({ ...prev, physicsEnabled: e.target.checked }))} />
          </Field>
        </div>
        <div className="world-summary">{simulation?.physicsExecution?.summary || simulation?.physicsScene?.executionSummary || '运行后会显示 Bullet 执行摘要。'}</div>
        {bodySummaries.length ? (
          <div className="world-list">
            {bodySummaries.slice(0, 4).map((body) => (
              <div key={body.id} className="world-list-item">
                <strong>{body.id}</strong>
                <span>{body.role || body.bodyClass}</span>
                <span>位移 {Number(body.displacementMeters || 0).toFixed(2)}m</span>
                <span>峰值 {Number(body.peakSpeedMps || 0).toFixed(2)}m/s</span>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      <div className="world-card">
        <div className="world-card-head">
          <div>
            <div className="world-card-title">World State Feedback</div>
            <div className="world-card-sub">查看 persisted evidence，确认新增地形与物理轨迹已经回流到 world model</div>
          </div>
          <button className="btn btn-ghost" onClick={() => refreshWorldState(getSessionId())}>
            刷新 World State
          </button>
        </div>
        <div className="world-kv">
          <div><span>session</span><strong>{activeSessionId || 'auto-create on demand'}</strong></div>
          <div><span>evidence</span><strong>{recentEvidence.length}</strong></div>
          <div><span>trainSamples</span><strong>{Array.isArray(simulation?.trainSamples) ? simulation.trainSamples.length : 0}</strong></div>
          <div><span>world status</span><strong>{worldStatus?.ok === false ? 'not-ready' : 'ready'}</strong></div>
        </div>
        {recentEvidence.length ? (
          <div className="world-list">
            {recentEvidence.slice(0, 6).map((entry, index) => (
              <div key={`${entry.modality || 'e'}_${index}`} className="world-list-item">
                <strong>{entry.modality || 'unknown'}</strong>
                <span>{(entry.graphSummary || entry.text || '').slice(0, 120) || 'no summary'}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="world-summary">当前 session 还没有 world evidence，导入地形或运行仿真后会在这里出现。</div>
        )}
      </div>

      <JsonPreview title="Latest Earth Import" data={importResult} />
      <JsonPreview title="Latest Simulation" data={simulation} />
    </div>
  );
}