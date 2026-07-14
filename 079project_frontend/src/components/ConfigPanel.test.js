import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConfigPanel from './ConfigPanel';

jest.mock('../api/client', () => ({
  api: {
    runtimeFeatures: jest.fn(() => Promise.resolve({ features: {} })),
    testsList: jest.fn(() => Promise.resolve({ files: [], directory: '' })),
    robotsList: jest.fn(() => Promise.resolve({ files: [] })),
    studyStatus: jest.fn(() => Promise.resolve({})),
    monitoringStats: jest.fn(() => Promise.resolve({ result: null })),
    searchConfig: jest.fn(() => Promise.resolve({ config: {} })),
    systemConfig: jest.fn(() => Promise.resolve({ config: {} })),
    groups: jest.fn(() => Promise.resolve({ groups: [] })),
    addons: jest.fn(() => Promise.resolve({ addons: [] })),
    clusterStatus: jest.fn(() => Promise.resolve({ result: null })),
    dataGovernance: jest.fn(() => Promise.resolve({ result: null })),
    optimizerStatus: jest.fn(() => Promise.resolve({ result: null })),
    spiderAutonomyStatus: jest.fn(() => Promise.resolve({ result: null }))
  }
}));

describe('ConfigPanel OpenClaw workspace', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  test('persists and restores the embedded dashboard settings', async () => {
    const props = {
      onError: jest.fn(),
      chatProvider: 'core',
      onChatProviderChange: jest.fn(),
      providerStats: {
        core: { count: 0, totalLatency: 0 },
        openclaw: { count: 0, totalLatency: 0 }
      }
    };

    const { unmount } = render(<ConfigPanel {...props} />);

    const input = screen.getByLabelText('OpenClaw Dashboard URL');
  await userEvent.clear(input);
  await userEvent.type(input, '127.0.0.1:19999/#token=abc');
  await userEvent.click(screen.getByRole('button', { name: '启用 OpenClaw' }));

    expect(screen.getByTestId('openclaw-dashboard-iframe')).toHaveAttribute(
      'src',
      'http://127.0.0.1:19999/#token=abc'
    );
    expect(JSON.parse(window.localStorage.getItem('phoenix.openclaw.panel'))).toEqual({
      enabled: true,
      dashboardUrl: '127.0.0.1:19999/#token=abc'
    });

    unmount();
    render(<ConfigPanel {...props} />);

    expect(screen.getByLabelText('OpenClaw Dashboard URL')).toHaveValue('127.0.0.1:19999/#token=abc');
    expect(screen.getByTestId('openclaw-dashboard-iframe')).toHaveAttribute(
      'src',
      'http://127.0.0.1:19999/#token=abc'
    );
  });
});