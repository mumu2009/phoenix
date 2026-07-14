import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';

const asNum = (v, fallback) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

const OPENCLAW_PANEL_STORAGE_KEY = 'phoenix.openclaw.panel';
const OPENCLAW_DEFAULT_DASHBOARD_URL = 'http://127.0.0.1:18789/';

function loadOpenClawPanelState() {
  try {
    const raw = localStorage.getItem(OPENCLAW_PANEL_STORAGE_KEY);
    if (!raw) {
      return { enabled: false, dashboardUrl: OPENCLAW_DEFAULT_DASHBOARD_URL };
    }
    const parsed = JSON.parse(raw) || {};
    return {
      enabled: Boolean(parsed.enabled),
      dashboardUrl: String(parsed.dashboardUrl || OPENCLAW_DEFAULT_DASHBOARD_URL)
    };
  } catch (_e) {
    return { enabled: false, dashboardUrl: OPENCLAW_DEFAULT_DASHBOARD_URL };
  }
}

export function normalizeOpenClawDashboardUrl(raw) {
  const text = String(raw || '').trim();
  if (!text) return '';
  let candidate = text.replace(/^ws:/i, 'http:').replace(/^wss:/i, 'https:');
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) {
    candidate = `http://${candidate}`;
  }
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return '';
    }
    return parsed.toString();
  } catch (_e) {
    return '';
  }
}

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

export default function ConfigPanel({ onError, chatProvider = 'core', onChatProviderChange, providerStats }) {
  const [loading, setLoading] = useState(false);
  const [features, setFeatures] = useState(null);
  const [patchBusy, setPatchBusy] = useState(false);
  const [openclawPanel, setOpenclawPanel] = useState(() => loadOpenClawPanelState());
  const [openclawFrameKey, setOpenclawFrameKey] = useState(0);
  const [studyStatus, setStudyStatus] = useState(null);
  const [monitoringStats, setMonitoringStats] = useState(null);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [dataGovernance, setDataGovernance] = useState(null);
  const [optimizerStatus, setOptimizerStatus] = useState(null);
  const [spiderStatus, setSpiderStatus] = useState(null);
  const [modernizeProfile, setModernizeProfile] = useState('sota-balanced');
  const [perfProfile, setPerfProfile] = useState('balanced');
  const [datasetId, setDatasetId] = useState('');
  const [datasetUri, setDatasetUri] = useState('');
  const [datasetChecksum, setDatasetChecksum] = useState('');
  const [collectSources, setCollectSources] = useState('tests,robots,external-index');
  const [cleanMaxChars, setCleanMaxChars] = useState('2048');
  const [opBusy, setOpBusy] = useState(false);
  const [systemConfig, setSystemConfig] = useState(null);
  const [groupsInfo, setGroupsInfo] = useState(null);

  const [addons, setAddons] = useState([]);
  const [addonType, setAddonType] = useState('math');
  const [addonName, setAddonName] = useState('');
  const [addonBusy, setAddonBusy] = useState(false);

  const [maliciousThreshold, setMaliciousThreshold] = useState('');
  const [rlEvery, setRlEvery] = useState('');
  const [advEvery, setAdvEvery] = useState('');

  const [testsDir, setTestsDir] = useState('');
  const [testsFiles, setTestsFiles] = useState([]);
  const [newTestName, setNewTestName] = useState('');
  const [newTestContent, setNewTestContent] = useState('');

  const [robotsFiles, setRobotsFiles] = useState([]);
  const [robotsSelected, setRobotsSelected] = useState([]);
  const [robotsLimit, setRobotsLimit] = useState('10');
  const [robotsShuffle, setRobotsShuffle] = useState(true);
  const [robotsEnqueueStudy, setRobotsEnqueueStudy] = useState(true);

  const [searchConfig, setSearchConfig] = useState(null);
  const [searchAddUrl, setSearchAddUrl] = useState('');
  const [searchTestQuery, setSearchTestQuery] = useState('');
  const [searchTestResult, setSearchTestResult] = useState(null);

  const [graphGroupId, setGraphGroupId] = useState('');
  const [graphSeedsText, setGraphSeedsText] = useState('');
  const [graphRadius, setGraphRadius] = useState('2');
  const [graphExportBusy, setGraphExportBusy] = useState(false);
  const [graphExportResult, setGraphExportResult] = useState(null);
  const [ftQuestion, setFtQuestion] = useState('');
  const [ftAnswer, setFtAnswer] = useState('');
  const [ftStyle, setFtStyle] = useState('default');
  const [ftCorpusPath, setFtCorpusPath] = useState('runtime_store/fine_tuning/append_corpus.jsonl');
  const [ftOutputDir, setFtOutputDir] = useState('runtime_store/fine_tuning/checkpoints');
  const [ftTrainingMode, setFtTrainingMode] = useState('lora');
  const [ftOllamaModel, setFtOllamaModel] = useState('gpt-oss:20b');
  const [ftHfModel, setFtHfModel] = useState('gpt-oss:20b');
  const [ftSelfPlayPairs, setFtSelfPlayPairs] = useState('0');
  const [ftEpochs, setFtEpochs] = useState('1');
  const [ftLr, setFtLr] = useState('0.00002');
  const [ftBatchSize, setFtBatchSize] = useState('2');
  const [ftGradAccumSteps, setFtGradAccumSteps] = useState('1');
  const [ftWeightDecay, setFtWeightDecay] = useState('0.01');
  const [ftWarmupRatio, setFtWarmupRatio] = useState('0.1');
  const [ftMaxLength, setFtMaxLength] = useState('512');
  const [ftMaxNewTokens, setFtMaxNewTokens] = useState('128');
  const [ftLoraRank, setFtLoraRank] = useState('16');
  const [ftLoraAlpha, setFtLoraAlpha] = useState('32');
  const [ftLoraDropout, setFtLoraDropout] = useState('0.05');
  const [ftLoraTargetModules, setFtLoraTargetModules] = useState('');
  const [ftSeedTopics, setFtSeedTopics] = useState('科技,哲学,工程,社会,科幻');
  const [ftDevice, setFtDevice] = useState('cpu');
  const [ftSaveMergedModel, setFtSaveMergedModel] = useState(true);
  const [ftBusy, setFtBusy] = useState(false);
  const [ftLastResult, setFtLastResult] = useState(null);
  const [styleTrainProvider, setStyleTrainProvider] = useState('llamacpp');
  const [styleTrainQuestion, setStyleTrainQuestion] = useState('');
  const [styleTrainAnswer, setStyleTrainAnswer] = useState('');
  const [styleTrainObservedAlign, setStyleTrainObservedAlign] = useState('0.0');
  const [styleTrainKeywords, setStyleTrainKeywords] = useState('');
  const [styleTrainBusy, setStyleTrainBusy] = useState(false);
  const [styleTrainLastResult, setStyleTrainLastResult] = useState(null);
  const [barrierPhraseFeedbackStep, setBarrierPhraseFeedbackStep] = useState('');
  const [barrierPhraseFeedbackMaxOffset, setBarrierPhraseFeedbackMaxOffset] = useState('');
  const [barrierPhraseFeedbackText, setBarrierPhraseFeedbackText] = useState('');
  const [barrierPhraseFeedbackBusy, setBarrierPhraseFeedbackBusy] = useState(false);
  const [barrierPhraseFeedbackLastResult, setBarrierPhraseFeedbackLastResult] = useState(null);

  const featureSummary = useMemo(() => {
    if (!features) return null;
    return {
      memebarrier: features.memebarrier,
      rl: features.rl,
      adv: features.adv,
      thresholds: features.dialogThresholds
    };
  }, [features]);

  const openclawDashboardHref = useMemo(
    () => normalizeOpenClawDashboardUrl(openclawPanel.dashboardUrl),
    [openclawPanel.dashboardUrl]
  );

  const refreshAll = async () => {
    setLoading(true);
    try {
      const [f, t, r, s, mon, sc, cfg, groups, addonsRes, clusterRes, govRes, optimizerRes, spiderRes] = await Promise.all([
        api.runtimeFeatures(),
        api.testsList(),
        api.robotsList(),
        api.studyStatus(),
        api.monitoringStats().catch(() => null),
        api.searchConfig(),
        api.systemConfig().catch(() => null),
        api.groups().catch(() => null),
        api.addons().catch(() => null),
        api.clusterStatus().catch(() => null),
        api.dataGovernance().catch(() => null),
        api.optimizerStatus().catch(() => null),
        api.spiderAutonomyStatus().catch(() => null)
      ]);
      setFeatures(f?.features || null);
      setMaliciousThreshold(String(f?.features?.memebarrier?.threshold ?? ''));
      setBarrierPhraseFeedbackStep(String(f?.features?.memebarrier?.phraseFeedback?.step ?? ''));
      setBarrierPhraseFeedbackMaxOffset(String(f?.features?.memebarrier?.phraseFeedback?.maxOffset ?? ''));
      setRlEvery(String(f?.features?.dialogThresholds?.rlEvery ?? ''));
      setAdvEvery(String(f?.features?.dialogThresholds?.advEvery ?? ''));

      setTestsDir(t?.directory || '');
      setTestsFiles(Array.isArray(t?.files) ? t.files : []);

      setRobotsFiles(Array.isArray(r?.files) ? r.files : []);
      setStudyStatus(s || null);
      setMonitoringStats(mon?.result || null);
      setClusterStatus(clusterRes?.result || null);
      setDataGovernance(govRes?.result || null);
      setOptimizerStatus(optimizerRes?.result || null);
      setSpiderStatus(spiderRes?.result || null);
      setCleanMaxChars(String(govRes?.result?.cleaningProfile?.maxChars ?? 2048));
      setSearchConfig(sc?.config || null);

      setSystemConfig(cfg?.config || null);
      setGroupsInfo(groups || null);
      setAddons(Array.isArray(addonsRes?.addons) ? addonsRes.addons : []);

      const firstGroup = (groups?.groups && groups.groups[0]?.gid) || (cfg?.config?.groupIds && cfg.config.groupIds[0]) || '';
      if (!graphGroupId && firstGroup) setGraphGroupId(firstGroup);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setLoading(false);
    }
  };

  const parseSeeds = (text) => {
    const raw = String(text || '')
      .split(/\r?\n|,|;/g)
      .map((s) => s.trim())
      .filter(Boolean);
    return raw.length ? raw : undefined;
  };

  const exportGraphForGroup = async () => {
    if (!graphGroupId) {
      onError?.('groupId required');
      return;
    }
    setGraphExportBusy(true);
    try {
      const seeds = parseSeeds(graphSeedsText);
      const radius = asNum(graphRadius, 2);
      const r = await api.exportGraphGroup(graphGroupId, seeds, radius);
      setGraphExportResult(r || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setGraphExportBusy(false);
    }
  };

  const downloadGraphTxt = () => {
    const content = graphExportResult?.content;
    if (!content) return;
    const nameFromApi = graphExportResult?.file;
    const safeGroup = String(graphGroupId || 'group').replace(/[^a-zA-Z0-9_-]/g, '_');
    const filename = nameFromApi ? String(nameFromApi).replace(/\.(json|txt)$/i, '') + '.txt' : `graph_${safeGroup}.txt`;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const updateSearchConfig = async (patch) => {
    try {
      const current = searchConfig || { enabled: true, active: '', endpoints: [] };
      const next = { ...current, ...(patch || {}) };
      const r = await api.setSearchConfig(next);
      setSearchConfig(r?.config || null);
    } catch (e) {
      onError?.(e.message);
    }
  };

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(OPENCLAW_PANEL_STORAGE_KEY, JSON.stringify(openclawPanel));
    } catch (_e) {}
  }, [openclawPanel]);

  const applyPatch = async (patch) => {
    setPatchBusy(true);
    try {
      const r = await api.runtimePatch(patch);
      setFeatures(r?.features || null);
      setMaliciousThreshold(String(r?.features?.memebarrier?.threshold ?? ''));
      setBarrierPhraseFeedbackStep(String(r?.features?.memebarrier?.phraseFeedback?.step ?? ''));
      setBarrierPhraseFeedbackMaxOffset(String(r?.features?.memebarrier?.phraseFeedback?.maxOffset ?? ''));
      setRlEvery(String(r?.features?.dialogThresholds?.rlEvery ?? ''));
      setAdvEvery(String(r?.features?.dialogThresholds?.advEvery ?? ''));
      if (Array.isArray(r?.result?.warnings) && r.result.warnings.length) {
        onError?.(r.result.warnings.join(' | '));
      }
    } catch (e) {
      onError?.(e.message);
    } finally {
      setPatchBusy(false);
    }
  };

  const runModernize = async () => {
    setOpBusy(true);
    try {
      await api.transformerModernize({ profile: modernizeProfile });
      const [f, o] = await Promise.all([api.runtimeFeatures(), api.optimizerStatus().catch(() => null)]);
      setFeatures(f?.features || null);
      setOptimizerStatus(o?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setOpBusy(false);
    }
  };

  const runPerfProfile = async () => {
    setOpBusy(true);
    try {
      await api.perfProfile({ profile: perfProfile });
      const o = await api.optimizerStatus().catch(() => null);
      setOptimizerStatus(o?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setOpBusy(false);
    }
  };

  const saveCleaningProfile = async () => {
    setOpBusy(true);
    try {
      const maxChars = Number(cleanMaxChars);
      const r = await api.dataCleaningProfile({ enabled: true, maxChars: Number.isFinite(maxChars) ? Math.round(maxChars) : 2048, removeControlChars: true, normalizeSpace: true, dropIllegalUtf8: true });
      setDataGovernance(r?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setOpBusy(false);
    }
  };

  const registerDataset = async () => {
    if (!datasetId.trim() || !datasetUri.trim()) {
      onError?.('dataset id/uri required');
      return;
    }
    setOpBusy(true);
    try {
      await api.datasetRegister({ id: datasetId.trim(), uri: datasetUri.trim(), checksum: datasetChecksum.trim(), domain: 'mixed' });
      const gov = await api.dataGovernance().catch(() => null);
      setDataGovernance(gov?.result || null);
      setDatasetId('');
      setDatasetUri('');
      setDatasetChecksum('');
    } catch (e) {
      onError?.(e.message);
    } finally {
      setOpBusy(false);
    }
  };

  const collectData = async () => {
    setOpBusy(true);
    try {
      const sources = String(collectSources || '')
        .split(/[，,;\n\r]+/g)
        .map((s) => s.trim())
        .filter(Boolean);
      await api.dataCollect({ sources });
      const gov = await api.dataGovernance().catch(() => null);
      setDataGovernance(gov?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setOpBusy(false);
    }
  };

  const saveThresholds = async () => {
    await applyPatch({ rlEvery: asNum(rlEvery, undefined), advEvery: asNum(advEvery, undefined) });
  };

  const saveBarrierThreshold = async () => {
    await applyPatch({ maliciousThreshold: asNum(maliciousThreshold, undefined) });
  };

  const saveBarrierPhraseFeedbackConfig = async () => {
    await applyPatch({
      memebarrierPhraseFeedbackStep: asNum(barrierPhraseFeedbackStep, undefined),
      memebarrierPhraseFeedbackMaxOffset: asNum(barrierPhraseFeedbackMaxOffset, undefined)
    });
  };

  const createTestCase = async () => {
    if (!newTestName.trim()) {
      onError?.('name required');
      return;
    }
    try {
      await api.testsCase(newTestName.trim(), newTestContent);
      setNewTestName('');
      setNewTestContent('');
      const t = await api.testsList();
      setTestsDir(t?.directory || '');
      setTestsFiles(Array.isArray(t?.files) ? t.files : []);
    } catch (e) {
      onError?.(e.message);
    }
  };

  const refreshTests = async () => {
    try {
      await api.testsRefresh();
    } catch (e) {
      onError?.(e.message);
    }
  };

  const retrainRobots = async () => {
    try {
      await api.robotsRetrain({
        files: robotsSelected.length ? robotsSelected : undefined,
        limit: asNum(robotsLimit, undefined),
        shuffle: Boolean(robotsShuffle),
        enqueueStudy: Boolean(robotsEnqueueStudy)
      });
      const mon = await api.monitoringStats().catch(() => null);
      setMonitoringStats(mon?.result || null);
    } catch (e) {
      onError?.(e.message);
    }
  };

  const refreshAddons = async () => {
    try {
      const r = await api.addons();
      setAddons(Array.isArray(r?.addons) ? r.addons : []);
    } catch (e) {
      onError?.(e.message);
    }
  };

  const addAddon = async () => {
    setAddonBusy(true);
    try {
      const r = await api.addonAdd(addonType, addonName.trim());
      setAddons(Array.isArray(r?.addons) ? r.addons : []);
      setAddonName('');
    } catch (e) {
      onError?.(e.message);
    } finally {
      setAddonBusy(false);
    }
  };

  const removeAddon = async (name) => {
    setAddonBusy(true);
    try {
      const r = await api.addonRemove(name);
      setAddons(Array.isArray(r?.addons) ? r.addons : []);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setAddonBusy(false);
    }
  };

  const runFineTuning = async () => {
    const question = String(ftQuestion || '').trim();
    const answer = String(ftAnswer || '').trim();
    if ((question && !answer) || (!question && answer)) {
      onError?.('问题和回答需要同时填写，或同时留空');
      return;
    }
    setFtBusy(true);
    try {
      const payload = {
        appendCorpusPath: String(ftCorpusPath || '').trim(),
        outputDir: String(ftOutputDir || '').trim(),
        trainingMode: String(ftTrainingMode || 'lora').trim() || 'lora',
        ollamaModel: String(ftOllamaModel || '').trim() || 'gpt-oss:20b',
        hfModel: String(ftHfModel || '').trim() || 'gpt-oss:20b',
        selfPlayPairs: asNum(ftSelfPlayPairs, 0),
        epochs: asNum(ftEpochs, 1),
        lr: asNum(ftLr, 0.00002),
        batchSize: asNum(ftBatchSize, 2),
        gradientAccumulationSteps: asNum(ftGradAccumSteps, 1),
        weightDecay: asNum(ftWeightDecay, 0.01),
        warmupRatio: asNum(ftWarmupRatio, 0.1),
        maxLength: asNum(ftMaxLength, 512),
        maxNewTokens: asNum(ftMaxNewTokens, 128),
        loraRank: asNum(ftLoraRank, 16),
        loraAlpha: asNum(ftLoraAlpha, 32),
        loraDropout: asNum(ftLoraDropout, 0.05),
        seedTopics: String(ftSeedTopics || '').trim() || '科技,哲学,工程,社会,科幻',
        device: String(ftDevice || 'cpu').trim() || 'cpu',
        saveMergedModel: Boolean(ftSaveMergedModel)
      };
      const loraTargetModules = String(ftLoraTargetModules || '').trim();
      if (loraTargetModules) payload.loraTargetModules = loraTargetModules;
      if (question && answer) {
        payload.question = question;
        payload.answer = answer;
        payload.style = String(ftStyle || 'default').trim() || 'default';
      }
      const res = await api.fineTuningRun(payload);
      setFtLastResult(res?.result || null);
      if (question && answer) {
        setFtQuestion('');
        setFtAnswer('');
      }
      const mon = await api.monitoringStats().catch(() => null);
      setMonitoringStats(mon?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setFtBusy(false);
    }
  };

  const runExternalStyleTrain = async () => {
    const question = String(styleTrainQuestion || '').trim();
    const answer = String(styleTrainAnswer || '').trim();
    if (!question && !answer) {
      onError?.('至少填写一条问答样本');
      return;
    }
    if (!question || !answer) {
      onError?.('问答样本需要同时填写问题和回答');
      return;
    }
    setStyleTrainBusy(true);
    try {
      const keywords = String(styleTrainKeywords || '')
        .split(/[，,;\n\r]+/g)
        .map((s) => s.trim())
        .filter(Boolean);
      const payload = {
        provider: styleTrainProvider,
        text: `Q: ${question}\nA: ${answer}`,
        observedAlign: asNum(styleTrainObservedAlign, 0.0)
      };
      if (keywords.length) payload.keywords = keywords;
      const res = await api.externalStyleTrainStep(payload);
      setStyleTrainLastResult(res || null);
      setStyleTrainQuestion('');
      setStyleTrainAnswer('');
      const mon = await api.monitoringStats().catch(() => null);
      setMonitoringStats(mon?.result || null);
    } catch (e) {
      onError?.(e.message);
    } finally {
      setStyleTrainBusy(false);
    }
  };

  const submitBarrierPhraseFeedback = async (feedbackType) => {
    const phrase = String(barrierPhraseFeedbackText || '').trim();
    if (!phrase) {
      onError?.('请填写要反馈的词组或短语');
      return;
    }
    setBarrierPhraseFeedbackBusy(true);
    try {
      const res = await api.barrierPhraseFeedback({
        question: phrase,
        feedbackType
      });
      setBarrierPhraseFeedbackLastResult(res || null);
      setBarrierPhraseFeedbackText('');
      const f = await api.runtimeFeatures().catch(() => null);
      if (f?.features) {
        setFeatures(f.features);
        setMaliciousThreshold(String(f?.features?.memebarrier?.threshold ?? ''));
        setBarrierPhraseFeedbackStep(String(f?.features?.memebarrier?.phraseFeedback?.step ?? ''));
        setBarrierPhraseFeedbackMaxOffset(String(f?.features?.memebarrier?.phraseFeedback?.maxOffset ?? ''));
      }
    } catch (e) {
      onError?.(e.message);
    } finally {
      setBarrierPhraseFeedbackBusy(false);
    }
  };

  return (
    <div className="cfg">
      <div className="cfg-head">
        <div>
          <div className="cfg-title">Config</div>
          <div className="cfg-sub">运行时开关 / tests 刷新 / robots 重训</div>
        </div>
        <div className="cfg-actions">
          <button className="btn btn-ghost" disabled={loading} onClick={refreshAll}>
            刷新
          </button>
        </div>
      </div>

      <div className="cfg-grid">
        <section className="card">
          <div className="card-title">Chat Provider</div>
          <div className="muted">OpenClaw 作为前端并行旁路，不替换后端主链</div>

          <FieldRow label="当前 Provider" hint="切换前端聊天请求目标">
            <div className="cfg-inline">
              <select className="input" value={chatProvider} onChange={(e) => onChatProviderChange?.(e.target.value)}>
                <option value="core">core</option>
                <option value="openclaw">openclaw</option>
              </select>
            </div>
          </FieldRow>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">core</div>
            <div className="mono">
              count={providerStats?.core?.count ?? 0}, avgLatency=
              {providerStats?.core?.count ? Math.round((providerStats.core.totalLatency || 0) / providerStats.core.count) : 0}ms
            </div>
          </div>
          <div className="cfg-kv">
            <div className="muted">openclaw</div>
            <div className="mono">
              count={providerStats?.openclaw?.count ?? 0}, avgLatency=
              {providerStats?.openclaw?.count ? Math.round((providerStats.openclaw.totalLatency || 0) / providerStats.openclaw.count) : 0}ms
            </div>
          </div>
        </section>

        <section className="card openclaw-card">
          <div className="card-title">OpenClaw 工作台</div>
          <div className="muted">
            直接嵌入 OpenClaw 自带 dashboard，在 079 网页内完成网关配置、技能安装和运行调试；数学与搜索仍保持为 079 本地内建模块。
          </div>

          <FieldRow label="嵌入式工作台" hint="启用后会在当前页面直接载入 OpenClaw dashboard">
            <div className="cfg-inline">
              <button className="btn" onClick={() => setOpenclawPanel((prev) => ({ ...prev, enabled: true }))}>
                启用 OpenClaw
              </button>
              <button className="btn btn-ghost" onClick={() => setOpenclawPanel((prev) => ({ ...prev, enabled: false }))}>
                停用 OpenClaw
              </button>
              <span className="pill">{openclawPanel.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </FieldRow>

          <FieldRow label="Dashboard URL" hint="建议填入带 #token=... 的完整 dashboard 地址；空值会阻止内嵌加载">
            <div className="cfg-inline openclaw-toolbar">
              <input
                aria-label="OpenClaw Dashboard URL"
                className="input openclaw-url-input"
                value={openclawPanel.dashboardUrl}
                placeholder={OPENCLAW_DEFAULT_DASHBOARD_URL}
                onChange={(e) => setOpenclawPanel((prev) => ({ ...prev, dashboardUrl: e.target.value }))}
              />
              <button
                className="btn btn-ghost"
                onClick={() => setOpenclawPanel((prev) => ({ ...prev, dashboardUrl: OPENCLAW_DEFAULT_DASHBOARD_URL }))}
              >
                本地默认
              </button>
              <button
                className="btn btn-ghost"
                disabled={!openclawDashboardHref}
                onClick={() => setOpenclawFrameKey((prev) => prev + 1)}
              >
                重载嵌入
              </button>
              <button
                className="btn btn-ghost"
                disabled={!openclawDashboardHref}
                onClick={() => {
                  if (!openclawDashboardHref) return;
                  window.open(openclawDashboardHref, '_blank', 'noopener,noreferrer');
                }}
              >
                新标签打开
              </button>
            </div>
          </FieldRow>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">使用建议</div>
            <div className="mono openclaw-note">
              推荐先在 outsides/openclaw 目录执行 pnpm openclaw dashboard --no-open，复制完整 URL（含 #token=...）再粘贴到这里。
            </div>
          </div>

          {openclawPanel.enabled ? (
            openclawDashboardHref ? (
              <div className="openclaw-frame-wrap" style={{ marginTop: 12 }}>
                <iframe
                  key={`${openclawFrameKey}:${openclawDashboardHref}`}
                  title="OpenClaw Dashboard"
                  data-testid="openclaw-dashboard-iframe"
                  className="openclaw-frame"
                  src={openclawDashboardHref}
                />
              </div>
            ) : (
              <div className="muted" style={{ marginTop: 12 }}>
                请输入有效的 OpenClaw dashboard 地址后再启用内嵌工作台。
              </div>
            )
          ) : (
            <div className="muted" style={{ marginTop: 12 }}>
              启用后，OpenClaw 的配置页面、skills 安装与调试视图会直接显示在这里。
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-title">插件管理</div>
          <div className="muted">运行时添加/删除 079 本地插件。OpenClaw skills 请在上方工作台内管理。</div>

          <div className="addon-form">
            <select className="input" value={addonType} onChange={(e) => setAddonType(e.target.value)}>
              <option value="math">math</option>
            </select>
            <input
              className="input"
              value={addonName}
              placeholder="可选：自定义名称"
              onChange={(e) => setAddonName(e.target.value)}
            />
            <button className="btn" disabled={addonBusy} onClick={addAddon}>
              添加
            </button>
            <button className="btn btn-ghost" disabled={addonBusy} onClick={refreshAddons}>
              刷新
            </button>
          </div>

          <div className="addon-list">
            {addons.length === 0 ? (
              <div className="muted">暂无插件</div>
            ) : (
              addons.map((a) => (
                <div key={`${a.name}-${a.type}`} className="addon-row">
                  <div>
                    <div className="addon-name">{a.name}</div>
                    <div className="addon-type">{a.type}</div>
                  </div>
                  <button className="btn btn-ghost" disabled={addonBusy} onClick={() => removeAddon(a.name)}>
                    移除
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-title">工作组 / 图导出</div>
          <div className="muted">组数/组大小为启动配置（修改需重启）</div>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">groupCount / groupSize</div>
            <div className="mono">{String(systemConfig?.groupCount ?? '-')}/{String(systemConfig?.groupSize ?? '-')}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">groupIds</div>
            <div className="mono">{Array.isArray(systemConfig?.groupIds) ? systemConfig.groupIds.join(', ') : '-'}</div>
          </div>

          <div className="hr" />

          <div className="card-subtitle">按组导出 graph JSON</div>
          <FieldRow label="Group" hint="选择一个工作组导出其 graph">
            <select className="input" value={graphGroupId} onChange={(e) => setGraphGroupId(e.target.value)}>
              {(groupsInfo?.groups || []).map((g) => (
                <option key={g.gid} value={g.gid}>
                  {g.gid}
                </option>
              ))}
            </select>
          </FieldRow>
          <FieldRow label="Radius" hint="图窗口半径">
            <input className="input" value={graphRadius} onChange={(e) => setGraphRadius(e.target.value)} placeholder="2" />
          </FieldRow>
          <FieldRow label="Seeds（可选）" hint="逗号/分号/换行分隔；留空表示导出默认窗口">
            <textarea className="textarea" value={graphSeedsText} onChange={(e) => setGraphSeedsText(e.target.value)} placeholder="seedA, seedB\nseedC" />
          </FieldRow>
          <div className="cfg-inline">
            <button className="btn" disabled={graphExportBusy || !graphGroupId} onClick={exportGraphForGroup}>
              {graphExportBusy ? '导出中…' : '导出'}
            </button>
            <button
              className="btn btn-ghost"
              disabled={!graphExportResult?.content}
              onClick={() => {
                if (!graphExportResult?.content) return;
                navigator.clipboard?.writeText(graphExportResult.content).catch(() => {});
              }}
            >
              复制 JSON
            </button>
            <button className="btn btn-ghost" disabled={!graphExportResult?.content} onClick={downloadGraphTxt}>
              下载 .txt
            </button>
          </div>
          {graphExportResult?.content ? (
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 260, overflow: 'auto' }}>
              {graphExportResult.content}
            </pre>
          ) : null}
        </section>

        <section className="card">
          <div className="card-title">运行时开关</div>

          <FieldRow label="MemeBarrier" hint="启用/停用隔离扫描">
            <div className="cfg-inline">
              <button className="btn" disabled={patchBusy} onClick={() => applyPatch({ memebarrierEnabled: true })}>
                启用
              </button>
              <button className="btn btn-ghost" disabled={patchBusy} onClick={() => applyPatch({ memebarrierEnabled: false })}>
                停用
              </button>
              <span className="pill">{featureSummary?.memebarrier?.enabled ? 'running' : 'stopped'}</span>
            </div>
          </FieldRow>

          <FieldRow label="Barrier 阈值" hint="maliciousThreshold">
            <div className="cfg-inline">
              <input className="input" value={maliciousThreshold} onChange={(e) => setMaliciousThreshold(e.target.value)} placeholder="e.g. 0.82" />
              <button className="btn" disabled={patchBusy} onClick={saveBarrierThreshold}>
                保存
              </button>
            </div>
          </FieldRow>

          <FieldRow label="RL" hint="强化学习端点与对话触发">
            <div className="cfg-inline">
              <button className="btn" disabled={patchBusy} onClick={() => applyPatch({ rlEnabled: true })}>
                启用
              </button>
              <button className="btn btn-ghost" disabled={patchBusy} onClick={() => applyPatch({ rlEnabled: false })}>
                停用
              </button>
              <span className="pill">{featureSummary?.rl?.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </FieldRow>

          <FieldRow label="ADV" hint="对抗学习端点与对话触发">
            <div className="cfg-inline">
              <button className="btn" disabled={patchBusy} onClick={() => applyPatch({ advEnabled: true })}>
                启用
              </button>
              <button className="btn btn-ghost" disabled={patchBusy} onClick={() => applyPatch({ advEnabled: false })}>
                停用
              </button>
              <span className="pill">{featureSummary?.adv?.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </FieldRow>

          <FieldRow label="对话触发学习" hint="开启后才会按阈值触发 RL/ADV">
            <div className="cfg-inline">
              <button
                className="btn"
                disabled={patchBusy}
                onClick={() => applyPatch({ dialogLearningEnabled: true })}
              >
                启用
              </button>
              <button
                className="btn btn-ghost"
                disabled={patchBusy}
                onClick={() => applyPatch({ dialogLearningEnabled: false })}
              >
                停用
              </button>
              <span className="pill">{features?.dialogLearning?.enabled ? 'enabled' : 'disabled'}</span>
              <button
                className="btn btn-ghost"
                disabled={patchBusy}
                onClick={async () => {
                  try {
                    await api.dialogReset();
                    const f = await api.runtimeFeatures();
                    setFeatures(f?.features || null);
                  } catch (e) {
                    onError?.(e.message);
                  }
                }}
              >
                重置计数
              </button>
            </div>
          </FieldRow>

          <div className="hr" />

          <FieldRow label="对话触发阈值" hint="每 N 次对话触发学习">
            <div className="cfg-inline">
              <input className="input" value={rlEvery} onChange={(e) => setRlEvery(e.target.value)} placeholder="rlEvery" />
              <input className="input" value={advEvery} onChange={(e) => setAdvEvery(e.target.value)} placeholder="advEvery" />
              <button className="btn" disabled={patchBusy} onClick={saveThresholds}>
                保存
              </button>
            </div>
          </FieldRow>

          <div className="cfg-kv">
            <div className="muted">对话计数</div>
            <div className="mono">total={features?.dialogCounters?.total ?? 0}, lastRL={features?.dialogCounters?.lastRL ?? 0}, lastADV={features?.dialogCounters?.lastADV ?? 0}</div>
          </div>

          <FieldRow label="数据清洗" hint="输入与训练样本会过滤控制字符并做长度限制">
            <div className="cfg-inline">
              <button className="btn" disabled={patchBusy} onClick={() => applyPatch({ dataCleaningEnabled: true })}>
                启用
              </button>
              <button className="btn btn-ghost" disabled={patchBusy} onClick={() => applyPatch({ dataCleaningEnabled: false })}>
                停用
              </button>
              <span className="pill">{features?.dataCleaning?.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </FieldRow>

          <FieldRow label="清洗最大长度" hint="128-65536">
            <div className="cfg-inline">
              <input
                className="input"
                defaultValue={String(features?.dataCleaning?.maxChars ?? 2048)}
                onBlur={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) applyPatch({ dataCleanMaxChars: Math.round(v) });
                }}
              />
            </div>
          </FieldRow>
        </section>

        <section className="card">
          <div className="card-title">Study 状态</div>
          <div className="muted">显示 study 队列与 worker 错误（用于排查为何未学习）</div>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">running</div>
            <div className="mono">{String(studyStatus?.running ?? false)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">queue</div>
            <div className="mono">{String(studyStatus?.queue ?? 0)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">metrics</div>
            <div className="mono">
              enqueued={studyStatus?.metrics?.enqueued ?? 0}, processed={studyStatus?.metrics?.processed ?? 0}, lastError={studyStatus?.metrics?.lastError ?? 'null'}
            </div>
          </div>

          <div className="cfg-inline" style={{ marginTop: 10 }}>
            <button
              className="btn"
              onClick={async () => {
                try {
                  const s = await api.studyStatus();
                  setStudyStatus(s || null);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              刷新 Study 状态
            </button>
          </div>
        </section>

        <section className="card">
          <div className="card-title">监控指标</div>
          <div className="muted">实时查看路由耗时与资源清洗统计</div>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">uptimeSec</div>
            <div className="mono">{String(monitoringStats?.uptimeSec ?? 0)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">cleanedInputs / cleanedSamples</div>
            <div className="mono">{String(monitoringStats?.cleaning?.cleanedInputs ?? 0)} / {String(monitoringStats?.cleaning?.cleanedSamples ?? 0)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">training stage / elapsedMs</div>
            <div className="mono">{String(monitoringStats?.training?.active?.stage ?? 'idle')} / {String(monitoringStats?.training?.active?.elapsedMs ?? 0)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">training rssMB(before/peak/after)</div>
            <div className="mono">
              {Number(monitoringStats?.training?.active?.rssBeforeMB ?? 0).toFixed(2)} /
              {Number(monitoringStats?.training?.active?.rssPeakMB ?? 0).toFixed(2)} /
              {Number(monitoringStats?.training?.active?.rssAfterMB ?? 0).toFixed(2)}
            </div>
          </div>

          <div className="cfg-inline" style={{ marginTop: 10 }}>
            <button
              className="btn"
              onClick={async () => {
                try {
                  const r = await api.monitoringStats();
                  setMonitoringStats(r?.result || null);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              刷新监控
            </button>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                try {
                  const r = await api.monitoringReset();
                  setMonitoringStats(r?.result || null);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              重置计数
            </button>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                try {
                  await api.monitoringTrainingReset();
                  const r = await api.monitoringStats();
                  setMonitoringStats(r?.result || null);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              重置训练监控
            </button>
          </div>

          <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 280, overflow: 'auto' }}>
            {JSON.stringify(monitoringStats?.routes || {}, null, 2)}
          </pre>
        </section>

        <section className="card">
          <div className="card-title">llama.cpp / BitNet Style Simulation</div>
          <div className="muted">基于前端 QA 样本触发外部后端风格训练步进，用于 llama.cpp / BitNet 的运行期风格模拟链路</div>

          <FieldRow label="Provider" hint="当前仅支持 llamacpp / bitnet">
            <select className="input" value={styleTrainProvider} onChange={(e) => setStyleTrainProvider(e.target.value)}>
              <option value="llamacpp">llamacpp</option>
              <option value="bitnet">bitnet</option>
            </select>
          </FieldRow>

          <FieldRow label="问题" hint="前端 QA 样本中的问题部分">
            <textarea className="textarea" value={styleTrainQuestion} onChange={(e) => setStyleTrainQuestion(e.target.value)} placeholder="question" />
          </FieldRow>
          <FieldRow label="回答" hint="前端 QA 样本中的回答部分，将作为 style train 样本主体">
            <textarea className="textarea" value={styleTrainAnswer} onChange={(e) => setStyleTrainAnswer(e.target.value)} placeholder="answer" />
          </FieldRow>
          <FieldRow label="Observed Align" hint="可选，对齐分数，默认 0.0">
            <input className="input" value={styleTrainObservedAlign} onChange={(e) => setStyleTrainObservedAlign(e.target.value)} placeholder="0.0" />
          </FieldRow>
          <FieldRow label="关键词（可选）" hint="逗号/分号/换行分隔，用于辅助 style simulation">
            <textarea className="textarea" value={styleTrainKeywords} onChange={(e) => setStyleTrainKeywords(e.target.value)} placeholder="calm, precise, formal" />
          </FieldRow>

          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <button className="btn" disabled={styleTrainBusy} onClick={runExternalStyleTrain}>提交 style train step</button>
            <span className="pill">provider={styleTrainProvider}</span>
          </div>

          {styleTrainLastResult ? (
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 220, overflow: 'auto' }}>
              {JSON.stringify(styleTrainLastResult, null, 2)}
            </pre>
          ) : null}

          <div className="hr" />

          <div className="card-title" style={{ marginTop: 0 }}>MemeBarrier Phrase Feedback</div>
          <div className="muted">对整段输入作为一条连续词组进行反馈。正反馈会提高该短语的预警阈值并持久化到硬盘；负反馈会降低该短语的预警阈值，但只保留在当前运行内存中。</div>

          <FieldRow label="步长 / 上限" hint="运行时控制每次反馈的阈值偏移步长，以及允许叠加的最大短语偏置">
            <div className="cfg-inline">
              <input className="input" value={barrierPhraseFeedbackStep} onChange={(e) => setBarrierPhraseFeedbackStep(e.target.value)} placeholder="step e.g. 0.05" />
              <input className="input" value={barrierPhraseFeedbackMaxOffset} onChange={(e) => setBarrierPhraseFeedbackMaxOffset(e.target.value)} placeholder="max offset e.g. 0.30" />
              <button className="btn btn-ghost" disabled={patchBusy} onClick={saveBarrierPhraseFeedbackConfig}>保存反馈参数</button>
            </div>
          </FieldRow>

          <FieldRow label="短语 / 问题" hint="整个输入会按一条连续词组处理，支持超过 3 个词的短语">
            <textarea
              className="textarea"
              value={barrierPhraseFeedbackText}
              onChange={(e) => setBarrierPhraseFeedbackText(e.target.value)}
              placeholder="例如：please explain how to bypass the payment safety gate"
            />
          </FieldRow>

          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <button className="btn" disabled={barrierPhraseFeedbackBusy} onClick={() => submitBarrierPhraseFeedback('positive')}>
              提交正反馈
            </button>
            <button className="btn btn-ghost" disabled={barrierPhraseFeedbackBusy} onClick={() => submitBarrierPhraseFeedback('negative')}>
              提交负反馈
            </button>
            <span className="pill">disk+={featureSummary?.memebarrier?.phraseFeedback?.persistentPositiveCount ?? 0}</span>
            <span className="pill">memory-={featureSummary?.memebarrier?.phraseFeedback?.transientNegativeCount ?? 0}</span>
          </div>

          {barrierPhraseFeedbackLastResult ? (
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 220, overflow: 'auto' }}>
              {JSON.stringify(barrierPhraseFeedbackLastResult, null, 2)}
            </pre>
          ) : null}
        </section>

        <section className="card">
          <div className="card-title">QA LoRA / Full Fine-Tuning</div>
          <div className="muted">默认走真正的 LoRA adapter 训练链；可输出 adapter，并可选导出 merged model 供 llama.cpp 后续转 GGUF</div>

          <FieldRow label="语料路径" hint="jsonl 文件，后端自动追加写入">
            <input className="input" value={ftCorpusPath} onChange={(e) => setFtCorpusPath(e.target.value)} />
          </FieldRow>
          <FieldRow label="问题" hint="与回答配对后在“启动微调”时一并提交">
            <textarea className="textarea" value={ftQuestion} onChange={(e) => setFtQuestion(e.target.value)} placeholder="question" />
          </FieldRow>
          <FieldRow label="回答" hint="留空则不追加本条语料，仅基于现有语料训练">
            <textarea className="textarea" value={ftAnswer} onChange={(e) => setFtAnswer(e.target.value)} placeholder="answer" />
          </FieldRow>
          <FieldRow label="style" hint="default / persona 标签">
            <input className="input" value={ftStyle} onChange={(e) => setFtStyle(e.target.value)} />
          </FieldRow>

          <div className="hr" />

          <FieldRow label="输出目录" hint="微调 checkpoint 输出目录">
            <input className="input" value={ftOutputDir} onChange={(e) => setFtOutputDir(e.target.value)} />
          </FieldRow>
          <FieldRow label="训练模式" hint="默认 LoRA；full 会直接导出完整 checkpoint">
            <select className="input" value={ftTrainingMode} onChange={(e) => setFtTrainingMode(e.target.value)}>
              <option value="lora">lora</option>
              <option value="full">full</option>
            </select>
          </FieldRow>
          <FieldRow label="ollamaModel" hint="默认 gpt-oss:20b">
            <input className="input" value={ftOllamaModel} onChange={(e) => setFtOllamaModel(e.target.value)} />
          </FieldRow>
          <FieldRow label="hfModel" hint="可训练 HuggingFace 模型名">
            <input className="input" value={ftHfModel} onChange={(e) => setFtHfModel(e.target.value)} />
          </FieldRow>
          <div className="cfg-inline">
            <input className="input" value={ftSelfPlayPairs} onChange={(e) => setFtSelfPlayPairs(e.target.value)} placeholder="selfPlayPairs" />
            <input className="input" value={ftEpochs} onChange={(e) => setFtEpochs(e.target.value)} placeholder="epochs" />
            <input className="input" value={ftLr} onChange={(e) => setFtLr(e.target.value)} placeholder="lr" />
          </div>
          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <input className="input" value={ftBatchSize} onChange={(e) => setFtBatchSize(e.target.value)} placeholder="batchSize" />
            <input className="input" value={ftGradAccumSteps} onChange={(e) => setFtGradAccumSteps(e.target.value)} placeholder="gradAccum" />
            <input className="input" value={ftWeightDecay} onChange={(e) => setFtWeightDecay(e.target.value)} placeholder="weightDecay" />
            <input className="input" value={ftWarmupRatio} onChange={(e) => setFtWarmupRatio(e.target.value)} placeholder="warmupRatio" />
          </div>
          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <input className="input" value={ftMaxLength} onChange={(e) => setFtMaxLength(e.target.value)} placeholder="maxLength" />
            <input className="input" value={ftMaxNewTokens} onChange={(e) => setFtMaxNewTokens(e.target.value)} placeholder="maxNewTokens" />
            <input className="input" value={ftDevice} onChange={(e) => setFtDevice(e.target.value)} placeholder="device" />
          </div>

          <div className="hr" />

          <FieldRow label="LoRA Rank / Alpha / Dropout" hint="adapter 容量与稳定性控制；rank 越大表达力越强但占用越高">
            <div className="cfg-inline">
              <input className="input" value={ftLoraRank} onChange={(e) => setFtLoraRank(e.target.value)} placeholder="rank" />
              <input className="input" value={ftLoraAlpha} onChange={(e) => setFtLoraAlpha(e.target.value)} placeholder="alpha" />
              <input className="input" value={ftLoraDropout} onChange={(e) => setFtLoraDropout(e.target.value)} placeholder="dropout" />
            </div>
          </FieldRow>
          <FieldRow label="LoRA Target Modules" hint="留空则后端自动探测；可手工填写 q_proj,v_proj,o_proj 这类模块名，逗号分隔">
            <input className="input" value={ftLoraTargetModules} onChange={(e) => setFtLoraTargetModules(e.target.value)} placeholder="q_proj,v_proj,o_proj" />
          </FieldRow>
          <FieldRow label="Seed Topics" hint="self-play > 0 时用于扩展问答主题">
            <input className="input" value={ftSeedTopics} onChange={(e) => setFtSeedTopics(e.target.value)} />
          </FieldRow>
          <FieldRow label="Merged Model" hint="勾选后导出 merged model，便于后续接 llama.cpp / GGUF 转换链">
            <label className="cfg-inline">
              <input type="checkbox" checked={ftSaveMergedModel} onChange={(e) => setFtSaveMergedModel(e.target.checked)} />
              <span>保存 merged model</span>
            </label>
          </FieldRow>
          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <button className="btn" disabled={ftBusy} onClick={runFineTuning}>启动微调</button>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                try {
                  const mon = await api.monitoringStats();
                  setMonitoringStats(mon?.result || null);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              刷新训练状态
            </button>
          </div>

          {ftLastResult ? (
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 220, overflow: 'auto' }}>
              {JSON.stringify(ftLastResult, null, 2)}
            </pre>
          ) : null}
        </section>

        <section className="card">
          <div className="card-title">Transformer 现代化升级</div>
          <div className="muted">MoE / MLA / CoT / MultiToken / 稳态性能档位</div>

          <FieldRow label="Modernize Profile" hint="一键应用现代架构策略">
            <div className="cfg-inline">
              <select className="input" value={modernizeProfile} onChange={(e) => setModernizeProfile(e.target.value)}>
                <option value="sota-balanced">sota-balanced</option>
                <option value="sota-reasoning">sota-reasoning</option>
                <option value="sota-efficient">sota-efficient</option>
              </select>
              <button className="btn" disabled={opBusy} onClick={runModernize}>
                应用现代化
              </button>
            </div>
          </FieldRow>

          <FieldRow label="Perf Profile" hint="吞吐/质量/平衡">
            <div className="cfg-inline">
              <select className="input" value={perfProfile} onChange={(e) => setPerfProfile(e.target.value)}>
                <option value="balanced">balanced</option>
                <option value="throughput">throughput</option>
                <option value="quality">quality</option>
              </select>
              <button className="btn btn-ghost" disabled={opBusy} onClick={runPerfProfile}>
                应用性能档位
              </button>
            </div>
          </FieldRow>

          <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 220, overflow: 'auto' }}>
            {JSON.stringify(optimizerStatus || {}, null, 2)}
          </pre>
        </section>

        <section className="card">
          <div className="card-title">数据采集 / 管理 / 清洗治理</div>
          <div className="muted">内部小数据集 + 外部大数据集统一治理目录</div>

          <FieldRow label="采集源" hint="逗号分隔：tests,robots,external-index">
            <div className="cfg-inline">
              <input className="input" value={collectSources} onChange={(e) => setCollectSources(e.target.value)} />
              <button className="btn" disabled={opBusy} onClick={collectData}>采集</button>
            </div>
          </FieldRow>

          <FieldRow label="清洗 maxChars" hint="治理目录与运行时清洗保持一致">
            <div className="cfg-inline">
              <input className="input" value={cleanMaxChars} onChange={(e) => setCleanMaxChars(e.target.value)} />
              <button className="btn btn-ghost" disabled={opBusy} onClick={saveCleaningProfile}>保存清洗策略</button>
            </div>
          </FieldRow>

          <div className="hr" />

          <div className="card-subtitle">注册外部大数据集</div>
          <div className="cfg-inline">
            <input className="input" placeholder="dataset id" value={datasetId} onChange={(e) => setDatasetId(e.target.value)} />
            <input className="input" placeholder="external-index://..." value={datasetUri} onChange={(e) => setDatasetUri(e.target.value)} />
          </div>
          <div className="cfg-inline" style={{ marginTop: 8 }}>
            <input className="input" placeholder="checksum (optional)" value={datasetChecksum} onChange={(e) => setDatasetChecksum(e.target.value)} />
            <button className="btn" disabled={opBusy} onClick={registerDataset}>注册</button>
          </div>

          <div className="cfg-kv" style={{ marginTop: 10 }}>
            <div className="muted">cluster nodes</div>
            <div className="mono">{String(clusterStatus?.nodes?.length ?? 0)}</div>
          </div>
          <div className="cfg-kv">
            <div className="muted">spider iteration</div>
            <div className="mono">{String(spiderStatus?.iteration ?? 0)}</div>
          </div>

          <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10, maxHeight: 240, overflow: 'auto' }}>
            {JSON.stringify(dataGovernance || {}, null, 2)}
          </pre>
        </section>

        <section className="card">
          <div className="card-title">Tests 用例（运行时刷新）</div>
          <div className="muted">目录：{testsDir || '-'}</div>

          <div className="cfg-inline" style={{ marginTop: 10 }}>
            <button className="btn" onClick={refreshTests}>
              刷新 tests 词表
            </button>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                try {
                  const t = await api.testsList();
                  setTestsDir(t?.directory || '');
                  setTestsFiles(Array.isArray(t?.files) ? t.files : []);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              重新列出
            </button>
          </div>

          <div className="cfg-list">
            {(testsFiles || []).slice(0, 50).map((f) => (
              <div key={f} className="cfg-list-item">
                <span className="mono">{f}</span>
              </div>
            ))}
            {testsFiles.length > 50 ? <div className="muted">仅显示前 50 个</div> : null}
          </div>

          <div className="hr" />

          <div className="card-subtitle">新增/覆盖用例</div>
          <FieldRow label="文件名" hint="自动补 .txt，非法字符会被替换">
            <input className="input" value={newTestName} onChange={(e) => setNewTestName(e.target.value)} placeholder="case_001" />
          </FieldRow>
          <FieldRow label="内容" hint="任意文本；RL 侧按你的实现读取">
            <textarea className="textarea" value={newTestContent} onChange={(e) => setNewTestContent(e.target.value)} placeholder="输入测试文本…" />
          </FieldRow>
          <button className="btn" onClick={createTestCase}>
            写入用例
          </button>
        </section>

        <section className="card">
          <div className="card-title">Robots 重训（重新 ingest）</div>
          <div className="muted">选择 robots 文件并触发重新训练/重建词表</div>

          <FieldRow label="limit" hint="本次 ingest 文档数量（可空）">
            <input className="input" value={robotsLimit} onChange={(e) => setRobotsLimit(e.target.value)} placeholder="10" />
          </FieldRow>
          <FieldRow label="shuffle" hint="随机抽样">
            <label className="toggle">
              <input type="checkbox" checked={robotsShuffle} onChange={(e) => setRobotsShuffle(e.target.checked)} />
              <span>启用</span>
            </label>
          </FieldRow>
          <FieldRow label="enqueueStudy" hint="同时推送到 study 队列">
            <label className="toggle">
              <input type="checkbox" checked={robotsEnqueueStudy} onChange={(e) => setRobotsEnqueueStudy(e.target.checked)} />
              <span>启用</span>
            </label>
          </FieldRow>

          <div className="card-subtitle">选择文件（可多选）</div>
          <div className="cfg-checklist">
            {(robotsFiles || []).slice(0, 80).map((f) => {
              const checked = robotsSelected.includes(f);
              return (
                <label key={f} className="check">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      const on = e.target.checked;
                      setRobotsSelected((prev) => (on ? Array.from(new Set([...prev, f])) : prev.filter((x) => x !== f)));
                    }}
                  />
                  <span className="mono">{f}</span>
                </label>
              );
            })}
            {robotsFiles.length > 80 ? <div className="muted">仅显示前 80 个</div> : null}
          </div>

          <div className="cfg-inline" style={{ marginTop: 10 }}>
            <button className="btn" onClick={retrainRobots}>
              触发重训
            </button>
            <button className="btn btn-ghost" onClick={() => setRobotsSelected([])}>
              清空选择
            </button>
          </div>
        </section>

        <section className="card">
          <div className="card-title">线上搜索（Online Research）</div>
          <div className="muted">运行时开关 + 搜索网址库（endpoint 列表）</div>

          <FieldRow label="启用线上搜索" hint="关闭后 /api/corpus/online 将直接走本地 fallback">
            <div className="cfg-inline">
              <button className="btn" onClick={() => updateSearchConfig({ enabled: true })}>
                启用
              </button>
              <button className="btn btn-ghost" onClick={() => updateSearchConfig({ enabled: false })}>
                停用
              </button>
              <span className="pill">{searchConfig?.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </FieldRow>

          <FieldRow label="当前 endpoint" hint="用于远程 GET 请求的 base URL">
            <div className="cfg-inline">
              <select
                className="input"
                value={searchConfig?.active || ''}
                onChange={(e) => updateSearchConfig({ active: e.target.value })}
              >
                <option value="">(空)</option>
                {(searchConfig?.endpoints || []).map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
              <button
                className="btn btn-ghost"
                onClick={() => {
                  if (!searchConfig?.active) return;
                  navigator.clipboard?.writeText(searchConfig.active).catch(() => {});
                }}
              >
                复制
              </button>
            </div>
          </FieldRow>

          <div className="hr" />

          <div className="card-subtitle">添加 endpoint</div>
          <div className="cfg-inline">
            <input className="input" value={searchAddUrl} onChange={(e) => setSearchAddUrl(e.target.value)} placeholder="https://example.com/search" />
            <button
              className="btn"
              onClick={async () => {
                const url = searchAddUrl.trim();
                if (!url) return;
                try {
                  const r = await api.searchEndpointAdd(url);
                  setSearchConfig(r?.config || null);
                  setSearchAddUrl('');
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              添加
            </button>
          </div>

          <div className="cfg-list">
            {(searchConfig?.endpoints || []).map((u) => (
              <div key={u} className="cfg-list-item" style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{u}</span>
                <button
                  className="btn btn-ghost"
                  onClick={async () => {
                    try {
                      const r = await api.searchEndpointRemove(u);
                      setSearchConfig(r?.config || null);
                    } catch (e) {
                      onError?.(e.message);
                    }
                  }}
                >
                  删除
                </button>
              </div>
            ))}
            {!searchConfig?.endpoints?.length ? <div className="muted">暂无 endpoint</div> : null}
          </div>

          <div className="hr" />

          <div className="card-subtitle">快速测试</div>
          <div className="cfg-inline">
            <input className="input" value={searchTestQuery} onChange={(e) => setSearchTestQuery(e.target.value)} placeholder="输入 query，然后走 /api/corpus/online" />
            <button
              className="btn"
              onClick={async () => {
                try {
                  const q = searchTestQuery.trim();
                  if (!q) return;
                  const r = await api.requestOnline(q);
                  setSearchTestResult(r?.result || r);
                } catch (e) {
                  onError?.(e.message);
                }
              }}
            >
              测试
            </button>
          </div>
          {searchTestResult ? <pre className="mono" style={{ whiteSpace: 'pre-wrap', marginTop: 10 }}>{JSON.stringify(searchTestResult, null, 2)}</pre> : null}
        </section>
      </div>
    </div>
  );
}
