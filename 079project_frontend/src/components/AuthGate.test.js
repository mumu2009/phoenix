import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AuthGate from './AuthGate';

jest.mock('../api/client', () => ({
  getAuthToken: jest.fn(() => ''),
  api: {
    authConfig: jest.fn(),
    authMe: jest.fn(),
    authLogin: jest.fn(),
    authRegister: jest.fn(),
    authBootstrap: jest.fn()
  }
}));

const { api, getAuthToken } = require('../api/client');

describe('AuthGate login flow', () => {
  beforeEach(() => {
    getAuthToken.mockReturnValue('');
    api.authConfig.mockResolvedValue({ allowBootstrap: false, allowRegister: true, requireEmailVerify: true });
  });

  test('shows login form and surfaces Chinese errors', async () => {
    render(
      <AuthGate>
        <div>inside</div>
      </AuthGate>
    );
    await waitFor(() => expect(screen.getByText('登录')).toBeInTheDocument());
    await userEvent.type(screen.getByPlaceholderText('admin'), 'alice');
    await userEvent.type(screen.getByPlaceholderText('至少 6 位'), 'secret1');
    api.authLogin.mockRejectedValueOnce(new Error('用户名或密码不正确。'));
    await userEvent.click(screen.getByText('登录'));
    await waitFor(() => expect(screen.getByText('用户名或密码不正确。')).toBeInTheDocument());
    expect(screen.queryByText('inside')).not.toBeInTheDocument();
  });

  test('can switch to register', async () => {
    render(
      <AuthGate>
        <div>inside</div>
      </AuthGate>
    );
    await waitFor(() => expect(screen.getByText('去注册')).toBeInTheDocument());
    await userEvent.click(screen.getByText('去注册'));
    expect(screen.getByText('注册并登录')).toBeInTheDocument();
  });
});
