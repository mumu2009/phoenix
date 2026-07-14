import React, { useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import AuthGate from './components/AuthGate';
import { api } from './api/client';
import ConfigPanel from './components/ConfigPanel';
import WorldPanel from './components/WorldPanel';

const loadJson = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch (_e) {
    return fallback;
  }
};

const saveJson = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (_e) {}
};

function App() {
  const [account, setAccount] = useState(() => loadJson('phoenix.account', { id: 'local', name: 'Local User' }));
  const [sessions, setSessions] = useState(() => loadJson('phoenix.sessions', []));
  const [activeSessionId, setActiveSessionId] = useState(() => loadJson('phoenix.activeSessionId', null));
  const [activePage, setActivePage] = useState(() => loadJson('phoenix.activePage', 'chat'));
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [chatProvider, setChatProvider] = useState(() => loadJson('phoenix.chatProvider', 'core'));
  const [providerStats, setProviderStats] = useState(() =>
    loadJson('phoenix.providerStats', {
      core: { count: 0, totalLatency: 0 },
      openclaw: { count: 0, totalLatency: 0 }
    })
  );
  const [imageInfo, setImageInfo] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [imageBusy, setImageBusy] = useState(false);
  const [voiceReady, setVoiceReady] = useState(false);
  const [listening, setListening] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [recording, setRecording] = useState(false);
  const [speechInfo, setSpeechInfo] = useState(null);
  const [speechBusy, setSpeechBusy] = useState(false);
  const [perf, setPerf] = useState({ cpu: 0, rssMB: 0, uptime: 0 });
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const activeSession = useMemo(() => {
    const found = sessions.find((s) => s.id === activeSessionId);
    return found || null;
  }, [sessions, activeSessionId]);

  useEffect(() => {
    saveJson('phoenix.account', account);
  }, [account]);
  useEffect(() => {
    saveJson('phoenix.sessions', sessions);
  }, [sessions]);
  useEffect(() => {
    saveJson('phoenix.activeSessionId', activeSessionId);
  }, [activeSessionId]);
  useEffect(() => {
    saveJson('phoenix.activePage', activePage);
  }, [activePage]);
  useEffect(() => {
    saveJson('phoenix.chatProvider', chatProvider);
  }, [chatProvider]);
  useEffect(() => {
    saveJson('phoenix.providerStats', providerStats);
  }, [providerStats]);

  const trackProviderStat = (provider, latency) => {
    const key = String(provider || 'core').toLowerCase() === 'openclaw' ? 'openclaw' : 'core';
    const n = Number(latency);
    setProviderStats((prev) => {
      const current = prev?.[key] || { count: 0, totalLatency: 0 };
      return {
        ...(prev || {}),
        [key]: {
          count: current.count + 1,
          totalLatency: current.totalLatency + (Number.isFinite(n) ? n : 0)
        }
      };
    });
  };

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const refresh = async () => {
      try {
        const r = await api.systemStatus();
        if (cancelled) return;
        setStatus(r);
        const rss = Number(r?.memory?.rss || 0);
        const cpu = Number(r?.processCpuPercent || 0);
        const uptime = Number(r?.uptime || 0);
        setPerf({
          cpu: Number.isFinite(cpu) ? cpu : 0,
          rssMB: Number.isFinite(rss) ? rss / 1024 / 1024 : 0,
          uptime: Number.isFinite(uptime) ? uptime : 0
        });
      } catch (e) {
        if (!cancelled) setStatus({ ok: false, error: e.message });
      }
    };
    refresh();
    timer = setInterval(refresh, 5000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceReady(false);
      return;
    }
    const rec = new SpeechRecognition();
    rec.lang = 'zh-CN';
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        transcript += event.results[i][0].transcript;
      }
      setMessage(transcript.trim());
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recognitionRef.current = rec;
    setVoiceReady(true);
  }, []);

  const ensureSession = () => {
    if (activeSession) return activeSession;
    const id = `s_${Date.now().toString(36)}_${Math.random().toString(16).slice(2)}`;
    const session = { id, title: 'New chat', createdAt: Date.now(), messages: [] };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(id);
    return session;
  };

  const appendMessage = (sessionId, msg) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              messages: [...s.messages, msg],
              title: s.title === 'New chat' ? (msg.role === 'user' ? msg.text.slice(0, 24) || 'Chat' : s.title) : s.title
            }
          : s
      )
    );
  };

  const onSend = async () => {
    const text = message.trim();
    if (!text || busy) return;
    setError(null);
    setBusy(true);
    const session = ensureSession();
    const sid = session.id;
    const imagePayload = imageInfo?.ok
      ? {
          image: {
            embedding: imageInfo.embedding || [],
            graphContext: imageInfo.graphContext || '',
            details: imageInfo.details || [],
            detections: imageInfo.detections || []
          }
        }
      : null;
    appendMessage(sid, {
      id: `m_${Date.now()}`,
      role: 'user',
      text,
      ts: Date.now(),
      meta: imagePayload
        ? {
            image: {
              preview: imagePreview,
              graphContext: imageInfo?.graphContext || ''
            }
          }
        : null
    });
    setMessage('');
    try {
      const r = await api.chat(text, sid, { ...(imagePayload || {}), provider: chatProvider });
      const disconnected = r?.connected === false || r?.error === 'disconnected' || r?.result?.error === 'disconnected';
      if (disconnected) {
        setStatus({ ok: false, error: 'disconnected' });
        appendMessage(sid, { id: `m_${Date.now()}_err`, role: 'system', text: 'Error: disconnected', ts: Date.now() });
        return;
      }
      setStatus({ ok: true });
      const isWeakReply = (textOut) => {
        const t = String(textOut || '').trim();
        if (!t) return true;
        const low = t.toLowerCase();
        if (low === 'unk' || low === '<unk>') return true;
        const parts = low.split(/\s+/g).filter(Boolean);
        if (!parts.length) return true;
        const unkCount = parts.filter((x) => x === 'unk' || x === '<unk>').length;
        return unkCount / parts.length >= 0.35;
      };
      let reply = r?.result?.reply ?? '';
      if (isWeakReply(reply) && chatProvider !== 'openclaw') {
        try {
          const fallback = await api.chat(text, sid, { provider: 'openclaw' });
          const fallbackReply = fallback?.result?.reply ?? '';
          if (!isWeakReply(fallbackReply) && fallbackReply) {
            reply = fallbackReply;
          }
        } catch (_e) {
        }
      }
      if (!reply) {
        return;
      }
      appendMessage(sid, {
        id: `m_${Date.now()}_ai`,
        role: 'assistant',
        text: reply,
        ts: Date.now(),
        meta: {
          provider: r?.provider || chatProvider,
          latency: r?.result?.latency,
          seeds: r?.result?.seeds,
          memes: r?.result?.memes,
          addon: r?.result?.addon,
          imageContext: r?.result?.imageContext,
          imageEmbeddingCount: r?.result?.imageEmbeddingCount
        }
      });
      trackProviderStat(r?.provider || chatProvider, r?.result?.latency);
      if (ttsEnabled && reply) {
        try {
          window.speechSynthesis.cancel();
          const utter = new SpeechSynthesisUtterance(reply);
          utter.lang = 'zh-CN';
          window.speechSynthesis.speak(utter);
        } catch (_e) {}
      }
    } catch (e) {
      setError(e.message);
      const msg = String(e?.message || '').toLowerCase();
      if (msg.includes('disconnected') || msg.includes('ai-proxy-failed') || msg.includes('failed to fetch')) {
        setStatus({ ok: false, error: 'disconnected' });
      }
      appendMessage(sid, { id: `m_${Date.now()}_err`, role: 'system', text: `Error: ${e.message}`, ts: Date.now() });
    } finally {
      setBusy(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  };

  const onToggleVoice = () => {
    if (!voiceReady || !recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
      setListening(false);
    } else {
      try {
        setListening(true);
        recognitionRef.current.start();
      } catch (_e) {
        setListening(false);
      }
    }
  };

  const onRecordSpeech = async () => {
    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }
    setSpeechBusy(true);
    setSpeechInfo(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        setSpeechBusy(true);
        try {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const wav = await convertToWavBase64(blob);
          const sessionId = activeSessionId || ensureSession().id;
          const res = await api.speechIngest(wav, sessionId, 'auto');
          const speech = res?.speech || {};
          setSpeechInfo(speech);
          if (speech?.text) {
            setMessage((prev) => (prev ? `${prev}\n${speech.text}` : speech.text));
          }
        } catch (err) {
          setError(err.message || 'speech analyze failed');
        } finally {
          setSpeechBusy(false);
          stream.getTracks().forEach((t) => t.stop());
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (err) {
      setError(err.message || 'record failed');
      setSpeechBusy(false);
    }
  };

  const convertToWavBase64 = async (blob) => {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const wav = encodeWav(audioBuffer);
    const base64 = btoa(String.fromCharCode(...new Uint8Array(wav)));
    return `data:audio/wav;base64,${base64}`;
  };

  const encodeWav = (audioBuffer) => {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length * numChannels;
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => {
      for (let i = 0; i < str.length; i += 1) view.setUint8(off + i, str.charCodeAt(i));
    };
    let offset = 0;
    writeStr(offset, 'RIFF'); offset += 4;
    view.setUint32(offset, 36 + length * 2, true); offset += 4;
    writeStr(offset, 'WAVE'); offset += 4;
    writeStr(offset, 'fmt '); offset += 4;
    view.setUint32(offset, 16, true); offset += 4;
    view.setUint16(offset, 1, true); offset += 2;
    view.setUint16(offset, numChannels, true); offset += 2;
    view.setUint32(offset, sampleRate, true); offset += 4;
    view.setUint32(offset, sampleRate * numChannels * 2, true); offset += 4;
    view.setUint16(offset, numChannels * 2, true); offset += 2;
    view.setUint16(offset, 16, true); offset += 2;
    writeStr(offset, 'data'); offset += 4;
    view.setUint32(offset, length * 2, true); offset += 4;
    const channelData = [];
    for (let c = 0; c < numChannels; c += 1) channelData.push(audioBuffer.getChannelData(c));
    for (let i = 0; i < audioBuffer.length; i += 1) {
      for (let c = 0; c < numChannels; c += 1) {
        const sample = Math.max(-1, Math.min(1, channelData[c][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
      }
    }
    return buffer;
  };

  const onPickImage = () => {
    fileRef.current?.click();
  };

  const onImageSelected = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageBusy(true);
    setError(null);
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('read image failed'));
        reader.readAsDataURL(file);
      });
      setImagePreview(String(dataUrl));
      const result = await api.visionAnalyze(String(dataUrl));
      setImageInfo(result);
    } catch (err) {
      setError(err.message || 'image analyze failed');
      setImageInfo(null);
      setImagePreview('');
    } finally {
      setImageBusy(false);
      e.target.value = '';
    }
  };

  const clearImage = () => {
    setImageInfo(null);
    setImagePreview('');
  };

  const newSession = () => {
    const id = `s_${Date.now().toString(36)}_${Math.random().toString(16).slice(2)}`;
    const session = { id, title: 'New chat', createdAt: Date.now(), messages: [] };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(id);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const deleteSession = (id) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      setActiveSessionId(null);
    }
  };

  return (
    <AuthGate>
      <div className="phoenix-root">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-title">079 Phoenix</div>
          <div className="brand-sub">AI workspace</div>
        </div>

        <div className="account">
          <div className="account-row">
            <div className="avatar">{(account?.name || 'U').slice(0, 1).toUpperCase()}</div>
            <div className="account-meta">
              <div className="account-name">{account?.name || 'User'}</div>
              <div className="account-id">{account?.id || 'local'}</div>
            </div>
          </div>
          <div className="account-actions">
            <button className="btn btn-ghost" onClick={() => setAccount((a) => ({ ...a, name: a.name === 'Local User' ? 'Operator' : 'Local User' }))}>
              切换昵称
            </button>
            <button className="btn" onClick={newSession}>
              新会话
            </button>
          </div>
        </div>

        <div className="sessions">
          <div className="section-title">会话</div>
          <div className="session-list">
            {sessions.length === 0 ? (
              <div className="muted">暂无会话</div>
            ) : (
              sessions.map((s) => (
                <div key={s.id} className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}>
                  <button className="session-main" onClick={() => setActiveSessionId(s.id)}>
                    <div className="session-title">{s.title || 'Chat'}</div>
                    <div className="session-sub">{new Date(s.createdAt).toLocaleString()}</div>
                  </button>
                  <button className="session-del" onClick={() => deleteSession(s.id)} title="删除">
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="nav">
          <div className="section-title">导航</div>
          <div className="nav-list">
            <button className={`nav-item ${activePage === 'chat' ? 'active' : ''}`} onClick={() => setActivePage('chat')}>
              Chat
            </button>
            <button className={`nav-item ${activePage === 'config' ? 'active' : ''}`} onClick={() => setActivePage('config')}>
              Config
            </button>
            <button className={`nav-item ${activePage === 'world' ? 'active' : ''}`} onClick={() => setActivePage('world')}>
              World
            </button>
          </div>
        </div>

        <div className="status">
          <div className="section-title">后端</div>
          <div className="status-row">
            <span className={`dot ${status?.ok ? 'ok' : 'bad'}`} />
            <span className="muted">{status?.ok ? 'connected' : 'disconnected'}</span>
          </div>
          <div className="status-row"><span className="muted">CPU: {perf.cpu.toFixed(2)}%</span></div>
          <div className="status-row"><span className="muted">MEM: {perf.rssMB.toFixed(2)}MB</span></div>
          <div className="status-row"><span className="muted">UP: {perf.uptime.toFixed(1)}s</span></div>
        </div>
      </aside>

      <main className="main">
        {activePage === 'chat' ? (
          <>
            <header className="topbar">
              <div className="topbar-title">{activeSession ? activeSession.title : '请选择或新建会话'}</div>
              <div className="topbar-actions">
                <select className="provider-select" value={chatProvider} onChange={(e) => setChatProvider(e.target.value)}>
                  <option value="core">core</option>
                  <option value="openclaw">openclaw</option>
                </select>
                <span className="provider-summary">
                  core={providerStats?.core?.count || 0} / openclaw={providerStats?.openclaw?.count || 0}
                </span>
                <button
                  className="btn btn-ghost"
                  onClick={() => api.snapshotCreate('ui').then(() => api.systemStatus().then(setStatus)).catch((e) => setError(e.message))}
                >
                  保存快照
                </button>
                <button className="btn btn-ghost" onClick={() => api.barrierStats().then(() => {}).catch((e) => setError(e.message))}>
                  Barrier
                </button>
              </div>
            </header>

            <section className="chat">
              <div className="chat-scroll">
                {(activeSession?.messages || []).map((m) => (
                  <div key={m.id} className={`msg ${m.role}`}>
                    <div className="msg-role">{m.role}</div>
                    <div className="msg-bubble">
                      <div className="msg-text">{m.text}</div>
                      {m.meta?.image?.preview ? (
                        <div className="msg-image">
                          <img src={m.meta.image.preview} alt="attached" />
                          <div className="msg-image-caption">{m.meta.image.graphContext || '图像已附加'}</div>
                        </div>
                      ) : null}
                      {m.meta ? (
                        <div className="msg-meta">
                          {m.meta.latency != null ? <span>latency: {m.meta.latency}ms</span> : null}
                          {m.meta.provider ? <span> provider: {m.meta.provider}</span> : null}
                          {Array.isArray(m.meta.seeds) ? <span> seeds: {m.meta.seeds.length}</span> : null}
                          {Array.isArray(m.meta.memes) ? <span> memes: {m.meta.memes.length}</span> : null}
                          {m.meta.addon ? (
                            <span> addon: {m.meta.addon.name || m.meta.addon.addon || m.meta.addon.type || 'custom'}</span>
                          ) : null}
                          {m.meta.imageContext ? <span> imageCtx: {m.meta.imageContext.slice(0, 60)}</span> : null}
                          {m.meta.imageEmbeddingCount ? <span> imageEmb: {m.meta.imageEmbeddingCount}</span> : null}
                        </div>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>

              <div className="composer">
                <div className="composer-panel">
                  <div className="composer-tools">
                    <button className={`btn btn-ghost ${listening ? 'active' : ''}`} disabled={!voiceReady} onClick={onToggleVoice}>
                      {listening ? '停止语音' : '语音输入'}
                    </button>
                    <button className={`btn btn-ghost ${recording ? 'active' : ''}`} onClick={onRecordSpeech} disabled={speechBusy}>
                      {recording ? '结束录音' : speechBusy ? '语音解析中…' : '语音识别'}
                    </button>
                    <button className="btn btn-ghost" onClick={() => setTtsEnabled((v) => !v)}>
                      {ttsEnabled ? '关闭朗读' : '开启朗读'}
                    </button>
                    <button className="btn btn-ghost" onClick={onPickImage} disabled={imageBusy}>
                      {imageBusy ? '分析中…' : '添加图像'}
                    </button>
                    <input ref={fileRef} type="file" accept="image/*" onChange={onImageSelected} style={{ display: 'none' }} />
                  </div>
                  {speechInfo?.ok ? (
                    <div className="speech-attach">
                      <div className="speech-title">语音识别</div>
                      <div className="speech-text">{speechInfo.text || '未识别文字'}</div>
                      <div className="speech-sub">{speechInfo.stage05}</div>
                    </div>
                  ) : null}
                  {imagePreview ? (
                    <div className="image-attach">
                      <img src={imagePreview} alt="preview" className="image-preview" />
                      <div className="image-meta">
                        <div className="image-title">图像已附加</div>
                        <div className="image-sub">
                          {imageInfo?.graphContext ? imageInfo.graphContext : imageInfo?.ok ? '已解析视觉特征' : '未完成解析'}
                        </div>
                        <button className="btn btn-ghost" onClick={clearImage}>移除</button>
                      </div>
                    </div>
                  ) : null}
                  <div className="composer-row">
                    <input
                      ref={inputRef}
                      className="composer-input"
                      value={message}
                      placeholder={busy ? '生成中…' : '输入消息，Enter 发送'}
                      onChange={(e) => setMessage(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          onSend();
                        }
                      }}
                    />
                    <button className="btn" disabled={busy || !message.trim()} onClick={onSend}>
                      发送
                    </button>
                  </div>
                </div>
              </div>

              {error ? <div className="error">{error}</div> : null}
            </section>
          </>
        ) : activePage === 'world' ? (
          <>
            <header className="topbar">
              <div className="topbar-title">World</div>
              <div className="topbar-actions">
                <button className="btn btn-ghost" onClick={() => api.systemStatus().then(setStatus).catch((e) => setError(e.message))}>
                  刷新后端状态
                </button>
              </div>
            </header>
            <section className="cfg-wrap">
              <WorldPanel
                activeSessionId={activeSessionId}
                ensureSession={ensureSession}
                onError={(msg) => setError(msg)}
              />
              {error ? <div className="error">{error}</div> : null}
            </section>
          </>
        ) : (
          <>
            <header className="topbar">
              <div className="topbar-title">Config</div>
              <div className="topbar-actions">
                <button className="btn btn-ghost" onClick={() => api.systemStatus().then(setStatus).catch((e) => setError(e.message))}>
                  刷新后端状态
                </button>
              </div>
            </header>
            <section className="cfg-wrap">
              <ConfigPanel
                onError={(msg) => setError(msg)}
                chatProvider={chatProvider}
                onChatProviderChange={setChatProvider}
                providerStats={providerStats}
              />
              {error ? <div className="error">{error}</div> : null}
            </section>
          </>
        )}
      </main>
    </div>
    </AuthGate>
  );
}

export default App;
