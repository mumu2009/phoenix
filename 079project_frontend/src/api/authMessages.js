const ZH = {
  unauthorized: '用户名或密码不正确，或登录已过期。',
  'invalid credentials': '用户名或密码不正确。',
  'email not verified': '邮箱尚未验证，请先完成验证后再登录。',
  'already bootstrapped': '管理员账号已创建，请直接登录。',
  'register disabled': '当前未开放自行注册，请联系管理员。',
  'missing username or password': '请填写用户名和密码。',
  'missing username, password or email': '请填写用户名、邮箱和密码。',
  'invalid username': '用户名长度需为 3–32 个字符。',
  'invalid email': '邮箱格式不正确。',
  'password too short': '密码至少需要 6 位。',
  'username exists': '该用户名已被使用。',
  'email exists': '该邮箱已被使用。',
  'user not found': '找不到对应账号。',
  'email not found': '找不到对应账号。',
  'invalid token': '验证码无效或已过期。',
  'reset failed': '验证码无效或已过期。',
  'admin only': '此操作仅管理员可用。',
  'module forbidden': '该模块不允许在此面板启停。',
  'unknown module': '未知逻辑模块。',
  'db missing': '数据库文件尚不存在。',
  'backup failed': '备份失败，请检查目录权限。'
};

export function humanizeAuthError(error, fallback) {
  const key = String(error || '').trim().toLowerCase();
  if (ZH[key]) return ZH[key];
  if (fallback) return String(fallback);
  return error ? String(error) : '操作失败';
}
