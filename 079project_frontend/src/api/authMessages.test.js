import { humanizeAuthError } from './authMessages';

test('maps machine auth codes to Chinese', () => {
  expect(humanizeAuthError('Invalid credentials')).toMatch(/密码/);
  expect(humanizeAuthError('email not verified')).toMatch(/验证/);
  expect(humanizeAuthError('admin only')).toMatch(/管理员/);
  expect(humanizeAuthError('unknown-code', '回退文案')).toBe('回退文案');
});
