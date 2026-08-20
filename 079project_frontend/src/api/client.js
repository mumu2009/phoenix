const API_BASE = process.env.REACT_APP_API_BASE || '';
const OPENCLAW_CHAT_PATH = process.env.REACT_APP_OPENCLAW_CHAT_PATH || '/openclaw/chat';
const DEFAULT_CHAT_PROVIDER = process.env.REACT_APP_CHAT_PROVIDER || 'core';
const REQUEST_TIMEOUT_MS = Number(process.env.REACT_APP_REQUEST_TIMEOUT_MS || 300000);
const TOKEN_KEY = 'phoenix_auth_token';

export function getAuthToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch (_e) {
    return '';
  }
}

export function setAuthToken(token) {
  try {
    if (!token) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, String(token));
  } catch (_e) {
    // ignore
  }
}

async function request(path, { method = 'GET', body, headers } = {}) {
  const token = getAuthToken();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(headers || {})
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal
    });
  } catch (e) {
    if (e?.name === 'AbortError') {
      const err = new Error(`request-timeout:${path}`);
      err.code = 'REQUEST_TIMEOUT';
      throw err;
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (_e) {
    json = { ok: false, error: 'invalid-json', raw: text };
  }
  if (!res.ok) {
    const msg = json && typeof json === 'object' ? (json.error || json.message || res.statusText) : res.statusText;
    const err = new Error(msg);
    err.status = res.status;
    err.payload = json;
    throw err;
  }
  return json;
}

function normalizeChatResponse(raw, provider) {
  if (provider !== 'openclaw') return raw;
  if (!raw || typeof raw !== 'object') {
    return { ok: false, result: { reply: '' }, provider: 'openclaw' };
  }
  if (raw.result && typeof raw.result === 'object') {
    return { ...raw, provider: 'openclaw' };
  }
  const reply = raw.reply ?? raw.text ?? raw.output ?? '';
  const latency = raw.latency ?? raw.costMs;
  return {
    ok: raw.ok !== false,
    provider: 'openclaw',
    result: {
      reply,
      latency,
      raw
    }
  };
}

async function chatWithProvider(text, sessionId, extra) {
  const provider = String(extra?.provider || DEFAULT_CHAT_PROVIDER || 'core').toLowerCase();
  const payload = { text, sessionId, ...(extra || {}) };
  delete payload.provider;
  if (provider === 'openclaw') {
    const out = await request(OPENCLAW_CHAT_PATH, { method: 'POST', body: payload });
    return normalizeChatResponse(out, 'openclaw');
  }
  return request('/api/chat', { method: 'POST', body: payload });
}

export const api = {
  authConfig: () => request('/auth/config'),
  authBootstrap: (username, email, password) => request('/auth/bootstrap', { method: 'POST', body: { username, email, password } }),
  authRegister: (username, email, password) => request('/auth/register', { method: 'POST', body: { username, email, password } }),
  authLogin: async (username, password) => {
    const out = await request('/auth/login', { method: 'POST', body: { username, password } });
    if (out?.token) setAuthToken(out.token);
    return out;
  },
  authVerifyRequest: (payload) => request('/auth/verify/request', { method: 'POST', body: payload || {} }),
  authVerify: (payload) => request('/auth/verify', { method: 'POST', body: payload || {} }),
  authForgot: (email) => request('/auth/forgot', { method: 'POST', body: { email } }),
  authReset: (email, token, password) => request('/auth/reset', { method: 'POST', body: { email, token, password } }),
  authMe: () => request('/auth/me'),
  authLogout: async () => {
    const out = await request('/auth/logout', { method: 'POST', body: {} });
    setAuthToken('');
    return out;
  },
  chat: (text, sessionId, extra) => chatWithProvider(text, sessionId, extra),
  arrayChat: (text, sessionId, options) => request('/api/array/chat', { method: 'POST', body: { text, sessionId, options } }),
  visionAnalyze: (imageBase64) => request('/vision/analyze', { method: 'POST', body: { imageBase64 } }),
  speechAnalyze: (audioBase64) => request('/speech/analyze', { method: 'POST', body: { audioBase64 } }),
  speechIngest: (audioBase64, sessionId, mode) => request('/speech/ingest', { method: 'POST', body: { audioBase64, sessionId, mode } }),
  speechSynthesize: (text, options) => request('/speech/synthesize', { method: 'POST', body: { text, ...(options || {}) } }),
  platformStatus: () => request('/api/platform/status'),
  platformRefresh: () => request('/api/platform/refresh', { method: 'POST', body: {} }),
  platformBudget: () => request('/api/platform/budget'),
  platformConfigPatch: (patch) => request('/api/platform/config', { method: 'PATCH', body: patch || {} }),
  platformPlan: (payload) => request('/api/platform/plan', { method: 'POST', body: payload || {} }),
  platformPlanMobility: (payload) => request('/api/platform/plan/mobility', { method: 'POST', body: payload || {} }),
  platformDispatchCompute: (payload) => request('/api/platform/dispatch/compute', { method: 'POST', body: payload || {} }),
  platformDispatchPeripheral: (payload) => request('/api/platform/dispatch/peripheral', { method: 'POST', body: payload || {} }),
  platformDispatchMobility: (payload) => request('/api/platform/dispatch/mobility', { method: 'POST', body: payload || {} }),
  platformSelfTest: (payload) => request('/api/platform/self-test', { method: 'POST', body: payload || {} }),
  runtimeFeatures: () => request('/api/runtime/features'),
  runtimePatch: (patch) => request('/api/runtime/features', { method: 'PATCH', body: patch || {} }),
  addons: () => request('/api/addons'),
  addonAdd: (type, name) => request('/api/addons/add', { method: 'POST', body: { type, name } }),
  addonRemove: (name) => request('/api/addons/remove', { method: 'POST', body: { name } }),
  studyStatus: () => request('/api/study/status'),
  dialogReset: () => request('/api/learn/dialog/reset', { method: 'POST', body: {} }),
  searchConfig: () => request('/api/search/config'),
  setSearchConfig: (config) => request('/api/search/config', { method: 'PUT', body: config || {} }),
  searchEndpointAdd: (url) => request('/api/search/endpoints/add', { method: 'POST', body: { url } }),
  searchEndpointRemove: (url) => request('/api/search/endpoints/remove', { method: 'POST', body: { url } }),
  getParams: () => request('/api/model/params'),
  setParams: (params) => request('/api/model/params', { method: 'POST', body: params }),
  resetParams: () => request('/api/model/params/reset', { method: 'POST', body: {} }),
  snapshots: () => request('/api/snapshots'),
  snapshotCreate: (name) => request('/api/snapshots/create', { method: 'POST', body: { name } }),
  snapshotRestore: (id) => request(`/api/snapshots/restore/${encodeURIComponent(id)}`, { method: 'POST', body: {} }),
  snapshotDelete: (id) => request(`/api/snapshots/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  systemStatus: () => request('/api/system/status'),
  systemConfig: () => request('/api/system/config'),
  worldStatus: () => request('/world/status'),
  worldState: (sessionId, limit = 8) => request(`/world/state?sessionId=${encodeURIComponent(sessionId)}&limit=${encodeURIComponent(limit)}`),
  worldPhysicsStatus: () => request('/world/physics/status'),
  worldEarthMapImport: (payload) => request('/world/earth-map/import', { method: 'POST', body: payload || {} }),
  worldSimulate: (payload) => request('/world/simulate', { method: 'POST', body: payload || {} }),
  groups: () => request('/api/groups'),
  groupMetrics: (gid) => request(`/api/groups/${encodeURIComponent(gid)}/metrics`),
  requestOnline: (query, options) => request('/api/corpus/online', { method: 'POST', body: { query, options: options || {} } }),
  shards: () => request('/api/shards'),
  robotsList: () => request('/robots/list'),
  robotsIngest: (options) => request('/robots/ingest', { method: 'POST', body: options || {} }),
  robotsRetrain: (options) => request('/api/robots/retrain', { method: 'POST', body: options || {} }),
  testsList: () => request('/api/tests/list'),
  testsCase: (name, content) => request('/api/tests/case', { method: 'POST', body: { name, content } }),
  testsRefresh: () => request('/api/tests/refresh', { method: 'POST', body: {} }),
  exportGraph: (seeds, radius) => request('/api/export/graph', { method: 'POST', body: { seeds, radius } }),
  exportGraphGroup: (groupId, seeds, radius) => request('/api/export/graph/group', { method: 'POST', body: { groupId, seeds, radius } }),
  rlLearn: (cycles) => request('/api/learn/reinforce', { method: 'POST', body: { cycles } }),
  rlLatest: () => request('/api/learn/reinforce/latest'),
  advLearn: (samples) => request('/api/learn/adversarial', { method: 'POST', body: { samples } }),
  advLatest: () => request('/api/learn/adversarial/latest'),
  learnThresholds: (rlEvery, advEvery) => request('/api/learn/thresholds', { method: 'POST', body: { rlEvery, advEvery } }),
  barrierStart: (maliciousThreshold) => request('/api/memebarrier/start', { method: 'POST', body: { maliciousThreshold } }),
  barrierStop: () => request('/api/memebarrier/stop', { method: 'POST', body: {} }),
  barrierStats: () => request('/api/memebarrier/stats'),
  barrierPhraseFeedback: (payload) => request('/api/memebarrier/phrase_feedback', { method: 'POST', body: payload || {} }),
  externalStyleTrainStep: (payload) => request('/api/external_style/train_step', { method: 'POST', body: payload || {} }),
  monitoringStats: () => request('/api/monitoring/stats'),
  monitoringReset: () => request('/api/monitoring/reset', { method: 'POST', body: {} }),
  monitoringTraining: () => request('/api/monitoring/training'),
  monitoringTrainingReset: () => request('/api/monitoring/training/reset', { method: 'POST', body: {} }),
  clusterStatus: () => request('/api/cluster/status'),
  clusterRoute: (payload) => request('/api/cluster/route', { method: 'POST', body: payload || {} }),
  clusterNodes: (payload) => request('/api/cluster/nodes', { method: 'POST', body: payload || {} }),
  clusterFeedback: (payload) => request('/api/cluster/feedback', { method: 'POST', body: payload || {} }),
  dataGovernance: () => request('/api/data/governance'),
  datasetCatalog: () => request('/api/dataset/catalog'),
  datasetRegister: (payload) => request('/api/dataset/register', { method: 'POST', body: payload || {} }),
  datasetActivate: (id) => request('/api/dataset/activate', { method: 'POST', body: { id } }),
  dataCollect: (payload) => request('/api/data/collect', { method: 'POST', body: payload || {} }),
  dataCleaningProfile: (payload) => request('/api/data/cleaning/profile', { method: 'POST', body: payload || {} }),
  optimizerStatus: () => request('/api/optimizer/autonomy/status'),
  optimizerIterate: (payload) => request('/api/optimizer/autonomy/iterate', { method: 'POST', body: payload || {} }),
  perfProfile: (payload) => request('/api/perf/profile', { method: 'POST', body: payload || {} }),
  gnnUpgrade: (payload) => request('/api/gnn/upgrade', { method: 'POST', body: payload || {} }),
  transformerUpgrade: (payload) => request('/api/transformer/upgrade', { method: 'POST', body: payload || {} }),
  transformerModernize: (payload) => request('/api/transformer/modernize', { method: 'POST', body: payload || {} }),
  fineTuningCorpusAdd: (payload) => request('/api/fine_tuning/corpus/add', { method: 'POST', body: payload || {} }),
  fineTuningRun: (payload) => request('/api/fine_tuning/run', { method: 'POST', body: payload || {} }),
  spiderAutonomyStatus: () => request('/api/spider/autonomy/status'),
  spiderAutonomyAdapt: (payload) => request('/api/spider/autonomy/adapt', { method: 'POST', body: payload || {} }),
  missionStatus: () => request('/api/mission/status'),
  missionAssign: (payload) => request('/api/mission/assign', { method: 'POST', body: payload || {} }),
  missionReport: (goalAchieved) => request('/api/mission/report', { method: 'POST', body: { goalAchieved } }),
  missionReplicate: (payload) => request('/api/mission/replicate', { method: 'POST', body: payload || {} }),
  autonomyInterject: (payload) => request('/api/cognition/autonomy/interject', { method: 'POST', body: payload || {} }),
  autonomyLoop: (payload) => request('/api/cognition/autonomy/loop', { method: 'POST', body: payload || {} }),
  autonomyStatus: () => request('/api/cognition/autonomy/status'),
  estopStatus: () => request('/api/safety/estop/status'),
  estop: (payload) => request('/api/safety/estop', { method: 'POST', body: payload || {} })
};
