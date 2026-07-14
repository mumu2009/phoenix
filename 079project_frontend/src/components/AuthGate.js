import React, { useEffect, useState } from 'react';
import { api, getAuthToken } from '../api/client';

export default function AuthGate({ children }) {
  const [state, setState] = useState({ loading: true, user: null, error: null, needsBootstrap: false, allowRegister: true, requireEmailVerify: true });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const token = getAuthToken();
        if (!token) {
          const cfg = await api.authConfig();
          if (!alive) return;
          setState({
            loading: false,
            user: null,
            error: null,
            needsBootstrap: !!cfg?.allowBootstrap,
            allowRegister: !!cfg?.allowRegister,
            requireEmailVerify: cfg?.requireEmailVerify !== false
          });
          return;
        }
        const out = await api.authMe();
        if (!alive) return;
        setState({ loading: false, user: out.user, error: null, needsBootstrap: false, allowRegister: true, requireEmailVerify: true });
      } catch (e) {
        if (!alive) return;
        try {
          const cfg = await api.authConfig();
          setState({
            loading: false,
            user: null,
            error: e,
            needsBootstrap: !!cfg?.allowBootstrap,
            allowRegister: !!cfg?.allowRegister,
            requireEmailVerify: cfg?.requireEmailVerify !== false
          });
        } catch (_cfgErr) {
          setState({ loading: false, user: null, error: e, needsBootstrap: false, allowRegister: false, requireEmailVerify: true });
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (state.loading) {
    return (
      <div style={{ padding: 24, color: '#e5e7eb' }}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>正在验证身份…</div>
      </div>
    );
  }

  if (state.user) {
    return children;
  }

  return (
    <LoginPanel
      needsBootstrap={state.needsBootstrap}
      allowRegister={state.allowRegister}
      requireEmailVerify={state.requireEmailVerify}
      onAuthed={(user) =>
        setState({
          loading: false,
          user,
          error: null,
          needsBootstrap: false,
          allowRegister: state.allowRegister,
          requireEmailVerify: state.requireEmailVerify
        })
      }
    />
  );
}

function LoginPanel({ onAuthed, needsBootstrap, allowRegister, requireEmailVerify }) {
  const [mode, setMode] = useState(needsBootstrap ? 'bootstrap' : 'login'); // login | register | bootstrap | verify | forgot | reset
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [verifyToken, setVerifyToken] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [status, setStatus] = useState({ busy: false, error: null, info: null });

  async function handleLogin() {
    setStatus({ busy: true, error: null, info: null });
    try {
      const out = await api.authLogin(username, password);
      const me = await api.authMe();
      onAuthed(me.user);
      setStatus({ busy: false, error: null, info: `欢迎 ${out.user?.username || username}` });
    } catch (e) {
      if (e?.message && String(e.message).includes('email not verified')) {
        setMode('verify');
        setStatus({ busy: false, error: '邮箱未验证，请完成验证。', info: null });
      } else {
        setStatus({ busy: false, error: e?.message || '登录失败', info: null });
      }
    }
  }

  async function handleBootstrap() {
    setStatus({ busy: true, error: null, info: null });
    try {
      const out = await api.authBootstrap(username, email, password);
      if (out?.verifyRequired) {
        setMode('verify');
        setStatus({ busy: false, error: null, info: '已发送验证信息，请完成邮箱验证。' });
        if (out?.verifyToken) setVerifyToken(out.verifyToken);
      } else {
        await handleLogin();
      }
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '初始化失败', info: null });
    }
  }

  async function handleRegister() {
    if (password !== confirm) {
      setStatus({ busy: false, error: '两次密码不一致', info: null });
      return;
    }
    setStatus({ busy: true, error: null, info: null });
    try {
      const out = await api.authRegister(username, email, password);
      if (out?.verifyRequired) {
        setMode('verify');
        setStatus({ busy: false, error: null, info: '已发送验证信息，请完成邮箱验证。' });
        if (out?.verifyToken) setVerifyToken(out.verifyToken);
      } else {
        await api.authLogin(username, password);
        const me = await api.authMe();
        onAuthed(me.user);
        setStatus({ busy: false, error: null, info: `欢迎 ${username}` });
      }
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '注册失败', info: null });
    }
  }

  async function handleVerifyRequest() {
    setStatus({ busy: true, error: null, info: null });
    try {
      const payload = email ? { email } : { username };
      const out = await api.authVerifyRequest(payload);
      if (out?.verifyToken) setVerifyToken(out.verifyToken);
      setStatus({ busy: false, error: null, info: '验证码已发送（请检查本地 outbox 或邮箱）。' });
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '发送失败', info: null });
    }
  }

  async function handleVerify() {
    setStatus({ busy: true, error: null, info: null });
    try {
      const payload = email ? { email, token: verifyToken } : { username, token: verifyToken };
      await api.authVerify(payload);
      setStatus({ busy: false, error: null, info: '验证成功，请登录。' });
      setMode('login');
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '验证失败', info: null });
    }
  }

  async function handleForgot() {
    setStatus({ busy: true, error: null, info: null });
    try {
      const out = await api.authForgot(email);
      if (out?.resetToken) setResetToken(out.resetToken);
      setMode('reset');
      setStatus({ busy: false, error: null, info: '重置码已发送（请检查本地 outbox 或邮箱）。' });
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '发送失败', info: null });
    }
  }

  async function handleReset() {
    if (password !== confirm) {
      setStatus({ busy: false, error: '两次密码不一致', info: null });
      return;
    }
    setStatus({ busy: true, error: null, info: null });
    try {
      await api.authReset(email, resetToken, password);
      setStatus({ busy: false, error: null, info: '密码已重置，请登录。' });
      setMode('login');
    } catch (e) {
      setStatus({ busy: false, error: e?.message || '重置失败', info: null });
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0b0f19', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: 420, background: '#121a2a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 18 }}>
        <div style={{ color: '#e5e7eb', fontSize: 18, fontWeight: 800, marginBottom: 8 }}>身份验证</div>
        <div style={{ color: '#9aa4b2', fontSize: 12, marginBottom: 12 }}>
          {mode === 'login'
            ? '请输入用户名与密码登录。'
            : mode === 'register'
            ? '创建一个新账号并登录。'
            : mode === 'verify'
            ? '请输入邮箱验证码完成验证。'
            : mode === 'forgot'
            ? '输入邮箱获取重置验证码。'
            : mode === 'reset'
            ? '输入重置验证码与新密码。'
            : '首次启动：创建管理员账号（只允许一次）。'}
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <label style={{ color: '#cbd5e1', fontSize: 12 }}>
            用户名
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={inputStyle}
              placeholder="admin"
              autoComplete="username"
            />
          </label>
          {mode !== 'login' || requireEmailVerify ? (
            <label style={{ color: '#cbd5e1', fontSize: 12 }}>
              邮箱
              <input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={inputStyle}
                placeholder="you@example.com"
                autoComplete="email"
              />
            </label>
          ) : null}
          <label style={{ color: '#cbd5e1', fontSize: 12 }}>
            密码
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={inputStyle}
              placeholder="至少 6 位"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
          </label>
          {mode === 'register' || mode === 'reset' ? (
            <label style={{ color: '#cbd5e1', fontSize: 12 }}>
              确认密码
              <input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                style={inputStyle}
                placeholder="再次输入密码"
                autoComplete="new-password"
              />
            </label>
          ) : null}
          {mode === 'verify' ? (
            <label style={{ color: '#cbd5e1', fontSize: 12 }}>
              验证码
              <input
                value={verifyToken}
                onChange={(e) => setVerifyToken(e.target.value)}
                style={inputStyle}
                placeholder="邮箱验证码"
              />
            </label>
          ) : null}
          {mode === 'reset' ? (
            <label style={{ color: '#cbd5e1', fontSize: 12 }}>
              重置码
              <input
                value={resetToken}
                onChange={(e) => setResetToken(e.target.value)}
                style={inputStyle}
                placeholder="重置验证码"
              />
            </label>
          ) : null}

          {status.error ? <div style={{ color: '#fca5a5', fontSize: 12 }}>{String(status.error)}</div> : null}
          {status.info ? <div style={{ color: '#86efac', fontSize: 12 }}>{String(status.info)}</div> : null}

          <div style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
            <button
              onClick={
                mode === 'login'
                  ? handleLogin
                  : mode === 'register'
                  ? handleRegister
                  : mode === 'verify'
                  ? handleVerify
                  : mode === 'forgot'
                  ? handleForgot
                  : mode === 'reset'
                  ? handleReset
                  : handleBootstrap
              }
              disabled={status.busy}
              style={primaryBtn}
            >
              {status.busy
                ? '处理中…'
                : mode === 'login'
                ? '登录'
                : mode === 'register'
                ? '注册并登录'
                : mode === 'verify'
                ? '完成验证'
                : mode === 'forgot'
                ? '发送重置码'
                : mode === 'reset'
                ? '重置密码'
                : '创建管理员并登录'}
            </button>

            <button
              onClick={() =>
                setMode((prev) => {
                  if (prev === 'login') return allowRegister ? 'register' : needsBootstrap ? 'bootstrap' : 'login';
                  if (prev === 'register') return needsBootstrap ? 'bootstrap' : 'login';
                  if (prev === 'verify') return 'login';
                  if (prev === 'forgot') return 'login';
                  if (prev === 'reset') return 'login';
                  return 'login';
                })
              }
              disabled={status.busy}
              style={linkBtn}
            >
              {mode === 'login'
                ? allowRegister
                  ? '去注册'
                  : needsBootstrap
                  ? '首次启动？去初始化'
                  : '登录'
                : mode === 'register'
                ? needsBootstrap
                  ? '去初始化'
                  : '返回登录'
                : mode === 'verify'
                ? '返回登录'
                : mode === 'forgot'
                ? '返回登录'
                : mode === 'reset'
                ? '返回登录'
                : '返回登录'}
            </button>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'space-between' }}>
            {mode === 'login' ? (
              <button onClick={() => setMode('forgot')} disabled={status.busy} style={linkBtn}>
                忘记密码
              </button>
            ) : null}
            {mode === 'verify' ? (
              <button onClick={handleVerifyRequest} disabled={status.busy} style={linkBtn}>
                重新发送验证码
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  width: '100%',
  marginTop: 6,
  padding: '10px 12px',
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.12)',
  background: '#0b1220',
  color: '#e5e7eb',
  outline: 'none'
};

const primaryBtn = {
  padding: '10px 12px',
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.10)',
  background: '#2563eb',
  color: 'white',
  fontWeight: 700,
  cursor: 'pointer'
};

const linkBtn = {
  padding: '10px 8px',
  borderRadius: 10,
  border: 'none',
  background: 'transparent',
  color: '#93c5fd',
  cursor: 'pointer'
};
