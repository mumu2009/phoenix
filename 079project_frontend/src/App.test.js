import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

jest.mock('./api/client', () => ({
  api: {
    systemStatus: jest.fn().mockResolvedValue({ ok: true, uptime: 1, processCpuPercent: 0, memory: { rss: 1024 } }),
    snapshotCreate: jest.fn().mockResolvedValue({ ok: true }),
    barrierStats: jest.fn().mockResolvedValue({ ok: true })
  }
}));

jest.mock('./components/AuthGate', () => ({
  __esModule: true,
  default: ({ children }) => children
}));

jest.mock('./components/WorldPanel', () => ({
  __esModule: true,
  default: () => <div>world-panel-mock</div>
}));

test('renders shell', async () => {
  render(<App />);
  expect(screen.getByText(/079 phoenix/i)).toBeInTheDocument();
  expect(screen.getByText('新会话')).toBeInTheDocument();
  expect(screen.getByDisplayValue('core')).toBeInTheDocument();
  expect(screen.getByText('World')).toBeInTheDocument();
  await waitFor(() => expect(require('./api/client').api.systemStatus).toHaveBeenCalled());
});
