describe('api.chat provider routing', () => {
  beforeEach(() => {
    jest.resetModules();
    localStorage.clear();
    global.fetch = jest.fn();
    delete process.env.REACT_APP_CHAT_PROVIDER;
    delete process.env.REACT_APP_OPENCLAW_CHAT_PATH;
  });

  test('uses core provider by default', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      text: async () => JSON.stringify({ ok: true, result: { reply: 'ok-core' } })
    });

    const { api } = require('./client');
    const out = await api.chat('hello', 's1');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/chat');
    expect(out?.result?.reply).toBe('ok-core');
  });

  test('routes openclaw provider and normalizes response', async () => {
    process.env.REACT_APP_CHAT_PROVIDER = 'openclaw';
    process.env.REACT_APP_OPENCLAW_CHAT_PATH = '/openclaw/chat';

    fetch.mockResolvedValueOnce({
      ok: true,
      text: async () => JSON.stringify({ reply: 'ok-openclaw', costMs: 12 })
    });

    const { api } = require('./client');
    const out = await api.chat('hello', 's2');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/openclaw/chat');
    expect(out?.provider).toBe('openclaw');
    expect(out?.result?.reply).toBe('ok-openclaw');
    expect(out?.result?.latency).toBe(12);
  });

  test('routes world runtime endpoints through the shared request helper', async () => {
    fetch
      .mockResolvedValueOnce({ ok: true, text: async () => JSON.stringify({ ok: true, physicsRuntime: { runtimeMode: 'native-embedded-source' } }) })
      .mockResolvedValueOnce({ ok: true, text: async () => JSON.stringify({ ok: true, physicsExecution: { status: 'executed' } }) });

    const { api } = require('./client');
    const runtime = await api.worldPhysicsStatus();
    const simulation = await api.worldSimulate({ sessionId: 's-world' });

    expect(fetch).toHaveBeenNthCalledWith(1, '/world/physics/status', expect.any(Object));
    expect(fetch).toHaveBeenNthCalledWith(2, '/world/simulate', expect.objectContaining({ method: 'POST' }));
    expect(runtime?.physicsRuntime?.runtimeMode).toBe('native-embedded-source');
    expect(simulation?.physicsExecution?.status).toBe('executed');
  });
});

describe('mission & autonomy api helpers', () => {
  beforeEach(() => {
    jest.resetModules();
    localStorage.clear();
    global.fetch = jest.fn();
  });

  const json = (data) => ({ ok: true, text: async () => JSON.stringify(data) });

  test('missionStatus GETs /api/mission/status', async () => {
    fetch.mockResolvedValueOnce(json({ ok: true, result: { enabled: true, stats: { generations: 3 } } }));

    const { api } = require('./client');
    const out = await api.missionStatus();

    expect(fetch).toHaveBeenCalledWith('/api/mission/status', expect.objectContaining({ method: 'GET' }));
    expect(out?.result?.stats?.generations).toBe(3);
  });

  test('missionAssign POSTs the lifecycle payload', async () => {
    fetch.mockResolvedValueOnce(json({ ok: true, result: { mission: { id: 'm1' } } }));

    const { api } = require('./client');
    await api.missionAssign({ goal: 'g', deadlineSec: 300 });

    const [url, opts] = fetch.mock.calls[0];
    expect(url).toBe('/api/mission/assign');
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body)).toEqual({ goal: 'g', deadlineSec: 300 });
  });

  test('missionReport sends goalAchieved flag', async () => {
    fetch.mockResolvedValueOnce(json({ ok: true, result: {} }));

    const { api } = require('./client');
    await api.missionReport(true);

    expect(fetch.mock.calls[0][0]).toBe('/api/mission/report');
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ goalAchieved: true });
  });

  test('autonomy interject/loop/status and estop route to the contract endpoints', async () => {
    fetch
      .mockResolvedValueOnce(json({ ok: true }))
      .mockResolvedValueOnce(json({ ok: true }))
      .mockResolvedValueOnce(json({ ok: true }))
      .mockResolvedValueOnce(json({ ok: true }));

    const { api } = require('./client');
    await api.autonomyInterject({ text: 'hi', amendGoal: 'new' });
    await api.autonomyLoop({ action: 'start' });
    await api.autonomyStatus();
    await api.estop({ reason: 'r' });

    expect(fetch).toHaveBeenNthCalledWith(1, '/api/cognition/autonomy/interject', expect.objectContaining({ method: 'POST' }));
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ text: 'hi', amendGoal: 'new' });
    expect(fetch).toHaveBeenNthCalledWith(2, '/api/cognition/autonomy/loop', expect.objectContaining({ method: 'POST' }));
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ action: 'start' });
    expect(fetch).toHaveBeenNthCalledWith(3, '/api/cognition/autonomy/status', expect.objectContaining({ method: 'GET' }));
    expect(fetch).toHaveBeenNthCalledWith(4, '/api/safety/estop', expect.objectContaining({ method: 'POST' }));
    expect(JSON.parse(fetch.mock.calls[3][1].body)).toEqual({ reason: 'r' });
  });

  test('non-ok response throws with the backend error text', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => JSON.stringify({ error: 'mission disabled' })
    });

    const { api } = require('./client');
    await expect(api.missionAssign({ goal: 'g' })).rejects.toThrow('mission disabled');
  });
});
