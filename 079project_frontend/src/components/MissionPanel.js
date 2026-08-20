import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';

const MISSION_STATES = ['Idle', 'Running', 'Completed', 'Failed'];

// Parse a numeric input to a finite number, or undefined for empty/invalid text
// so the payload omits fields the operator left blank (backend defaults apply).
const asNum = (v) => {
  const t = String(v ?? '').trim();
  if (!t) return undefined;
  const n = Number(t);
  return Number.isFinite(n) ? n : undefined;
};

const display = (v) => (v === undefined || v === null || v === '' ? '-' : String(v));

const stateLabel = (s) => {
  const n = Number(s);
  if (Number.isFinite(n) && n >= 0 && n < MISSION_STATES.length) return MISSION_STATES[n];
  return display(s);
};

const fmtDuration = (ms) => {
  if (ms == null || !Number.isFinite(Number(ms))) return '-';
  const totalSec = Math.floor(Number(ms) / 1000);
  if (totalSec < 0) return '-';
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

function FieldRow({ label, hint, children }) {
  return (
    <div className="cfg-row">
      <div className="cfg-label">
        <div className="cfg-label-title">{label}</div>
        {hint ? <div className="cfg-label-hint">{hint}</div> : null}
      </div>
      <div className="cfg-control">{children}</div>
    </div>
  );
}

function Kv({ items }) {
  return (
    <div className="world-kv">
      {items.map((it) => (
        <div key={it.label}>
          <span>{it.label}</span>
          <strong>{it.value}</strong>
        </div>
      ))}
    </div>
  );
}

export default function MissionPanel({ onError }) {
  // 1. Lifecycle assignment card
  const [goal, setGoal] = useState('');
  const [deadlineSec, setDeadlineSec] = useState('300');
  const [painGainPerSec, setPainGainPerSec] = useState('0.01');
  const [pressureMode, setPressureMode] = useState('logarithmic');
  const [pressureHorizonSec, setPressureHorizonSec] = useState('');
  const [maxPain, setMaxPain] = useState('1.0');
  const [mutationRate, setMutationRate] = useState('0.05');
  const [maxReplicas, setMaxReplicas] = useState('4');
  const [assignBusy, setAssignBusy] = useState(false);
  const [assignedMission, setAssignedMission] = useState(null);

  // 2. Live monitor + estop latch (polled together)
  const [missionStatus, setMissionStatus] = useState(null);
  const [autonomyStatus, setAutonomyStatus] = useState(null);
  const [estopStatus, setEstopStatus] = useState(null);
  const [backendDown, setBackendDown] = useState(false);

  // 3. Human verdict card
  const [reportBusy, setReportBusy] = useState(false);

  // 4. Interjection card
  const [interjectText, setInterjectText] = useState('');
  const [amendGoal, setAmendGoal] = useState('');
  const [interjectBusy, setInterjectBusy] = useState(false);
  const [interjectResult, setInterjectResult] = useState(null);

  // 5. Autonomy loop card
  const [loopStatus, setLoopStatus] = useState(null);
  const [loopBusy, setLoopBusy] = useState(false);
  const [loopIntervalSec, setLoopIntervalSec] = useState('');
  const [loopMaxSteps, setLoopMaxSteps] = useState('');
  const [loopPersistEvery, setLoopPersistEvery] = useState('');

  // 6. E-stop card
  const [estopReason, setEstopReason] = useState('');
  const [estopBusy, setEstopBusy] = useState(false);

  const refreshStatuses = useCallback(async () => {
    let anyOk = false;
    const [m, a, e] = await Promise.all([
      api.missionStatus().catch(() => null),
      api.autonomyStatus().catch(() => null),
      api.estopStatus().catch(() => null)
    ]);
    if (m) { setMissionStatus(m); anyOk = true; }
    if (a) { setAutonomyStatus(a); anyOk = true; }
    if (e) { setEstopStatus(e); anyOk = true; }
    setBackendDown(!anyOk);
  }, []);

  // Poll every 4s and clean the timer up on unmount.
  useEffect(() => {
    refreshStatuses();
    const timer = setInterval(refreshStatuses, 4000);
    return () => clearInterval(timer);
  }, [refreshStatuses]);

  // Loop status is a POST action; fetch it once on mount and on demand.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await api.autonomyLoop({ action: 'status' });
        if (!cancelled) setLoopStatus(r || null);
      } catch (e) {
        if (!cancelled) onError?.(e.message);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitAssign = async () => {
    const text = goal.trim();
    if (!text) {
      onError?.('请填写目标');
      return;
    }
    setAssignBusy(true);
    try {
      const payload = { goal: text };
      const d = asNum(deadlineSec); if (d !== undefined) payload.deadlineSec = d;
      const p = asNum(painGainPerSec); if (p !== undefined) payload.painGainPerSec = p;
      const mp = asNum(maxPain); if (mp !== undefined) payload.maxPain = mp;
      const mode = String(pressureMode || 'logarithmic').trim();
      payload.pressureMode = mode === 'linear' ? 'linear' : 'logarithmic';
      const ph = asNum(pressureHorizonSec); if (ph !== undefined) payload.pressureHorizonSec = ph;
      const mr = asNum(mutationRate); if (mr !== undefined) payload.mutationRate = mr;
      const mx = asNum(maxReplicas); if (mx !== undefined) payload.maxReplicas = mx;
      const r = await api.missionAssign(payload);
      setAssignedMission(r?.result?.mission || r?.result || null);
      await refreshStatuses();
    } catch (e) {
      onError?.(e.message);
    } finally {
      setAssignBusy(false);
    }
  };

  const submitReport = async (goalAchieved) => {
    setReportBusy(true);
    try {
      await api.missionReport(goalAchieved);
      await refreshStatuses();
    } catch (e) {
      onError?.(e.message);
    } finally {
      setReportBusy(false);
    }
  };

  const submitInterject = async () => {
    const text = interjectText.trim();
    if (!text) {
      onError?.('请填写插话内容');
      return;
    }
    setInterjectBusy(true);
    try {
      const payload = { text };
      const amended = amendGoal.trim();
      if (amended) payload.amendGoal = amended;
      const r = await api.autonomyInterject(payload);
      setInterjectResult(r || null);
      setInterjectText('');
      setAmendGoal('');
    } catch (e) {
      onError?.(e.message);
    } finally {
      setInterjectBusy(false);
    }
  };

  const runLoopAction = async (action, extra) => {
    setLoopBusy(true);
    try {
      const r = await api.autonomyLoop({ action, ...(extra || {}) });
      setLoopStatus(r || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setLoopBusy(false);
    }
  };

  const configureLoop = async () => {
    const extra = {};
    const iv = asNum(loopIntervalSec); if (iv !== undefined) extra.intervalSec = iv;
    const ms = asNum(loopMaxSteps); if (ms !== undefined) extra.maxStepsPerTick = ms;
    const pe = asNum(loopPersistEvery); if (pe !== undefined) extra.persistEveryTicks = pe;
    await runLoopAction('configure', extra);
  };

  const pressEstop = async () => {
    const confirmed = window.confirm('确定要执行系统急停吗？此操作会锁存并停止所有实例。');
    if (!confirmed) return;
    setEstopBusy(true);
    try {
      const reason = estopReason.trim();
      await api.estop(reason ? { reason } : {});
      await refreshStatuses();
    } catch (e) {
      onError?.(e.message);
    } finally {
      setEstopBusy(false);
    }
  };

  const stats = missionStatus?.result?.stats || {};
  const mission = stats.mission || {};
  const children = Array.isArray(stats.children) ? stats.children : [];
  const elapsedMs = mission.startMs != null ? Date.now() - Number(mission.startMs) : null;
  const completionTimeMs = Number(stats.completionTimeMs);
  const completionDisplay =
    completionTimeMs != null && Number.isFinite(completionTimeMs) && completionTimeMs >= 0 ? `${completionTimeMs}ms` : '-';
  const agi = autonomyStatus?.result?.agi || {};
  const agiGoals = Array.isArray(autonomyStatus?.result?.agiGoals) ? autonomyStatus.result.agiGoals : [];
  const agiGoalsRecent = agiGoals.slice(-5);

  return (
    <div className="cfg">
      <div className="cfg-head">
        <div>
          <div className="cfg-title">Mission</div>
          <div className="cfg-sub">生命周期任务模式：设立目标，模型自主求解，控制台监控 / 插话 / 判定</div>
        </div>
      </div>

      {backendDown ? (
        <div className="mission-banner">后端不可达：无法获取任务或自主状态，请确认后端服务已启动。</div>
      ) : null}

      <div className="cfg-grid">
        <section className="card">
          <div className="card-title">生命周期启动</div>
          <div className="muted">在模型生命周期开始时设立目标；提交后模型将自主求解直至完成或失败。</div>

          <FieldRow label="目标 (goal)" hint="必填，描述本次生命周期要完成的任务">
            <textarea
              aria-label="目标"
              className="textarea"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="例如：分析并总结最近一周的日志异常"
            />
          </FieldRow>

          <FieldRow label="高级参数" hint="数值字段留空则使用后端默认值">
            <div className="cfg-inline">
              <input aria-label="deadlineSec" className="input" value={deadlineSec} onChange={(e) => setDeadlineSec(e.target.value)} placeholder="deadlineSec 300" />
              <input aria-label="painGainPerSec" className="input" value={painGainPerSec} onChange={(e) => setPainGainPerSec(e.target.value)} placeholder="painGainPerSec 0.01 (linear)" />
              <select aria-label="pressureMode" className="input" value={pressureMode} onChange={(e) => setPressureMode(e.target.value)} title="压力增长模式">
                <option value="logarithmic">压力: 对数增长 (默认)</option>
                <option value="linear">压力: 线性增长</option>
              </select>
              <input aria-label="pressureHorizonSec" className="input" value={pressureHorizonSec} onChange={(e) => setPressureHorizonSec(e.target.value)} placeholder="pressureHorizonSec 3600 (对数)" />
              <input aria-label="maxPain" className="input" value={maxPain} onChange={(e) => setMaxPain(e.target.value)} placeholder="maxPain 1.0" />
              <input aria-label="mutationRate" className="input" value={mutationRate} onChange={(e) => setMutationRate(e.target.value)} placeholder="mutationRate 0.05" />
              <input aria-label="maxReplicas" className="input" value={maxReplicas} onChange={(e) => setMaxReplicas(e.target.value)} placeholder="maxReplicas 4" />
            </div>
          </FieldRow>

          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <button className="btn" onClick={submitAssign} disabled={assignBusy}>
              {assignBusy ? '设立中…' : '设立任务'}
            </button>
          </div>

          {assignedMission ? (
            <div className="world-kv" style={{ marginTop: 10 }}>
              <div><span>id</span><strong>{display(assignedMission.id)}</strong></div>
              <div><span>state</span><strong>{stateLabel(assignedMission.state)}</strong></div>
              <div><span>goal</span><strong>{display(assignedMission.goal)}</strong></div>
            </div>
          ) : null}
        </section>

        <section className="card mission-wide">
          <div className="card-title">实时监控</div>
          <div className="muted">每 4 秒轮询一次任务与自主智能体状态。</div>

          <Kv
            items={[
              { label: '任务状态', value: stateLabel(mission.state) },
              { label: '已运行时长', value: fmtDuration(elapsedMs) },
              { label: 'pressure', value: display(stats.pressure) },
              { label: 'generations', value: display(stats.generations) },
              { label: 'spawns', value: display(stats.spawns) },
              { label: 'completions', value: display(stats.completions) },
              { label: 'completionTimeMs', value: completionDisplay },
              { label: 'agi.enabled', value: display(agi.enabled) },
              { label: 'iteration', value: display(autonomyStatus?.result?.iteration) },
              { label: 'mission.enabled', value: display(missionStatus?.result?.enabled) }
            ]}
          />

          <div className="card-subtitle">deliverable（交付物，模型实际产出）</div>
          <pre className="deliverable-box">{display(mission.deliverable) === '-' ? '（尚无产出：自主循环启动后，模型每 tick 写一段交付物）' : mission.deliverable}</pre>

          <div className="card-subtitle">children</div>
          {children.length ? (
            <div className="world-list">
              {children.map((c) => (
                <div key={c.id || c.bornMs} className="world-list-item">
                  <strong>{display(c.id)}</strong>
                  <span>世代 {display(c.generation)} · {display(c.goal)}</span>
                  <span>born {c.bornMs ? new Date(c.bornMs).toLocaleString() : '-'}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="muted">暂无 children</div>
          )}

          <div className="card-subtitle">agiGoals（最近 5 条）</div>
          {agiGoalsRecent.length ? (
            <div className="world-list">
              {agiGoalsRecent.map((g, i) => (
                <div key={i} className="world-list-item">
                  <span>{typeof g === 'string' ? g : JSON.stringify(g)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="muted">暂无 agiGoals</div>
          )}
        </section>

        <section className="card">
          <div className="card-title">人工判定</div>
          <div className="muted">由人工确认任务完成或失败，写入报告并刷新状态。</div>
          <div className="cfg-inline">
            <button className="btn" onClick={() => submitReport(true)} disabled={reportBusy}>
              判定完成
            </button>
            <button className="btn btn-ghost" onClick={() => submitReport(false)} disabled={reportBusy}>
              判定失败
            </button>
          </div>
        </section>

        <section className="card">
          <div className="card-title">插话</div>
          <div className="muted">向运行中的模型注入一条人类插话；amendGoal 可选，用于重定向目标。</div>

          <FieldRow label="插话内容 (text)" hint="必填">
            <input
              aria-label="插话内容"
              className="input"
              value={interjectText}
              onChange={(e) => setInterjectText(e.target.value)}
              placeholder="给模型的一条提示"
            />
          </FieldRow>
          <FieldRow label="重定向目标 (amendGoal)" hint="可选，填写后运行中任务目标被重定向">
            <input
              aria-label="重定向目标"
              className="input"
              value={amendGoal}
              onChange={(e) => setAmendGoal(e.target.value)}
              placeholder="新的目标描述"
            />
          </FieldRow>

          <div className="cfg-inline">
            <button className="btn" onClick={submitInterject} disabled={interjectBusy}>
              {interjectBusy ? '发送中…' : '发送插话'}
            </button>
          </div>

          {interjectResult ? (
            <pre className="world-pre" style={{ marginTop: 10 }}>{JSON.stringify(interjectResult, null, 2)}</pre>
          ) : null}
        </section>

        <section className="card">
          <div className="card-title">自主循环</div>
          <div className="muted">控制自主心跳循环：查询状态、配置参数、启动、停止。</div>

          <FieldRow label="intervalSec" hint="心跳间隔秒数">
            <input aria-label="intervalSec" className="input" value={loopIntervalSec} onChange={(e) => setLoopIntervalSec(e.target.value)} placeholder="intervalSec" />
          </FieldRow>
          <FieldRow label="maxStepsPerTick" hint="每 tick 最大迭代步数">
            <input aria-label="maxStepsPerTick" className="input" value={loopMaxSteps} onChange={(e) => setLoopMaxSteps(e.target.value)} placeholder="maxStepsPerTick" />
          </FieldRow>
          <FieldRow label="persistEveryTicks" hint="每 N tick 持久化一次">
            <input aria-label="persistEveryTicks" className="input" value={loopPersistEvery} onChange={(e) => setLoopPersistEvery(e.target.value)} placeholder="persistEveryTicks" />
          </FieldRow>

          <div className="cfg-inline">
            <button className="btn btn-ghost" onClick={() => runLoopAction('status')} disabled={loopBusy}>查询状态</button>
            <button className="btn" onClick={configureLoop} disabled={loopBusy}>配置</button>
            <button className="btn" onClick={() => runLoopAction('start')} disabled={loopBusy}>启动</button>
            <button className="btn btn-ghost" onClick={() => runLoopAction('stop')} disabled={loopBusy}>停止</button>
          </div>

          {loopStatus ? (
            <pre className="world-pre" style={{ marginTop: 10 }}>{JSON.stringify(loopStatus, null, 2)}</pre>
          ) : (
            <div className="muted" style={{ marginTop: 10 }}>暂无循环状态</div>
          )}
        </section>

        <section className="card mission-estop">
          <div className="card-title">E-stop 急停</div>
          <div className="muted">系统级急停：锁存并停止所有实例，需二次确认。</div>

          <FieldRow label="急停原因 (reason)" hint="可选">
            <input aria-label="急停原因" className="input" value={estopReason} onChange={(e) => setEstopReason(e.target.value)} placeholder="例如：行为越界" />
          </FieldRow>

          <div className="cfg-inline">
            <button className="btn btn-danger" onClick={pressEstop} disabled={estopBusy}>
              {estopBusy ? '急停中…' : '急停'}
            </button>
            <span className="pill">{estopStatus?.latched ? '已锁存 (latched)' : '未锁存'}</span>
          </div>
        </section>
      </div>
    </div>
  );
}
