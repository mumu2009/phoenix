import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import OpsPanel from './OpsPanel';

jest.mock('../api/client', () => ({
  api: {
    opsMonitor: jest.fn(),
    opsModules: jest.fn(),
    opsDatabase: jest.fn(),
    opsModuleSet: jest.fn(),
    opsDatabaseBackup: jest.fn()
  }
}));

const { api } = require('../api/client');

describe('OpsPanel', () => {
  beforeEach(() => {
    api.opsMonitor.mockResolvedValue({
      ok: true,
      pid: 11,
      memory: { rssMB: 32 },
      requests: { total: 4, errors: 1 },
      endpoints: [{ id: 'gateway', title: '网关', host: '127.0.0.1', port: 5080, probe: 'tcp-port', alive: true }]
    });
    api.opsModules.mockResolvedValue({
      modules: [
        { id: 'search', title: '联网搜索', enabled: true, readonly: false },
        { id: 'inference-llama', title: '推理进程', enabled: true, readonly: true, note: '只读' }
      ]
    });
    api.opsDatabase.mockResolvedValue({
      engine: 'sqlite+lmdb',
      path: 'runtime_store/ai_store.sqlite',
      legacyDir: 'lmdb',
      backupDir: 'runtime_store/db_backups',
      healthy: true
    });
  });

  test('renders monitor and lets admin toggle a safe module', async () => {
    api.opsModuleSet.mockResolvedValue({
      modules: [{ id: 'search', title: '联网搜索', enabled: false, readonly: false }]
    });
    render(<OpsPanel isAdmin onError={jest.fn()} />);
    await waitFor(() => expect(screen.getByText(/网关 127.0.0.1:5080/)).toBeInTheDocument());
    expect(screen.getAllByText(/推理进程/).length).toBeGreaterThan(0);
    expect(screen.queryByText('停止推理')).not.toBeInTheDocument();
    await userEvent.click(screen.getByText('关闭'));
    await waitFor(() => expect(api.opsModuleSet).toHaveBeenCalledWith('search', false));
  });
});
