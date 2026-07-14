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
