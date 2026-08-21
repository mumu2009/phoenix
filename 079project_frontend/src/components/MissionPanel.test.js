import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import MissionPanel from './MissionPanel';

jest.mock('../api/client', () => ({
  api: {
    missionStatus: jest.fn(),
    missionAssign: jest.fn(),
    missionReport: jest.fn(),
    missionReplicate: jest.fn(),
    autonomyInterject: jest.fn(),
    autonomyLoop: jest.fn(),
    autonomyStatus: jest.fn(),
    estopStatus: jest.fn(),
    estop: jest.fn()
  }
}));

const { api } = require('../api/client');

const okMissionStatus = () => ({
  ok: true,
  result: {
    enabled: true,
    stats: {
      generations: 3,
      spawns: 2,
      completions: 1,
      completionTimeMs: 1234,
      pressure: 0.5,
      mission: { id: 'm1', goal: '分析文档', deadlineSec: 300, state: 1, startMs: 1000, endMs: 0 },
      children: [{ id: 'c1', generation: 2, goal: '分析文档', bornMs: 2000 }]
    },
    genome: {}
  }
});

beforeEach(() => {
  jest.clearAllMocks();
  api.missionStatus.mockResolvedValue(okMissionStatus());
  api.autonomyStatus.mockResolvedValue({
    ok: true,
    result: { enabled: true, iteration: 7, agi: { enabled: false }, agiGoals: ['g1', 'g2', 'g3', 'g4', 'g5', 'g6'] }
  });
  api.estopStatus.mockResolvedValue({ latched: false });
  api.autonomyLoop.mockResolvedValue({ ok: true, result: { enabled: false, iteration: 0 } });
  api.missionAssign.mockResolvedValue({ ok: true, result: { mission: { id: 'm2', goal: '新目标', state: 1 }, generations: 0 } });
  api.missionReport.mockResolvedValue({ ok: true, result: { mission: { id: 'm1', state: 2 } } });
  api.autonomyInterject.mockResolvedValue({ ok: true, accepted: true });
  api.estop.mockResolvedValue({ ok: true, latched: true });
});

test('assign submits the lifecycle payload with advanced params', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  fireEvent.change(screen.getByLabelText('目标'), { target: { value: '写一份周报' } });
  fireEvent.change(screen.getByLabelText('deadlineSec'), { target: { value: '600' } });
  fireEvent.change(screen.getByLabelText('maxReplicas'), { target: { value: '8' } });
  fireEvent.click(screen.getByRole('button', { name: '设立任务' }));

  await waitFor(() => expect(api.missionAssign).toHaveBeenCalled());
  const payload = api.missionAssign.mock.calls[0][0];
  expect(payload.goal).toBe('写一份周报');
  expect(payload.deadlineSec).toBe(600);
  expect(payload.painGainPerSec).toBe(0.01);
  expect(payload.maxPain).toBe(1.0);
  expect(payload.mutationRate).toBe(0.05);
  expect(payload.maxReplicas).toBe(8);
  expect(payload.ctxSize).toBe(4096);
  expect(payload.contextPack).toBe('full_and_summary');
  expect(payload.includeGnnSummary).toBe(false);

  expect(await screen.findByText('m2')).toBeInTheDocument();
});

test('context pack controls are submitted on assign', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  fireEvent.change(screen.getByLabelText('目标'), { target: { value: '摘要模式任务' } });
  fireEvent.change(screen.getByLabelText('ctxSize'), { target: { value: '16384' } });
  fireEvent.change(screen.getByLabelText('contextPack'), { target: { value: 'summary' } });
  fireEvent.click(screen.getByLabelText('includeGnnSummary'));
  fireEvent.click(screen.getByRole('button', { name: '设立任务' }));

  await waitFor(() => expect(api.missionAssign).toHaveBeenCalled());
  const payload = api.missionAssign.mock.calls[0][0];
  expect(payload.ctxSize).toBe(16384);
  expect(payload.contextPack).toBe('summary');
  expect(payload.includeGnnSummary).toBe(true);
});

test('polls mission/autonomy/estop and loop status on mount', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  await waitFor(() => expect(api.missionStatus).toHaveBeenCalled());
  expect(api.autonomyStatus).toHaveBeenCalled();
  expect(api.estopStatus).toHaveBeenCalled();
  expect(api.autonomyLoop).toHaveBeenCalledWith({ action: 'status' });

  expect(await screen.findByText('Running')).toBeInTheDocument();
  expect(screen.getByText('7')).toBeInTheDocument();
  expect(screen.getByText('c1')).toBeInTheDocument();
});

test('sets up the polling interval and clears it on unmount', () => {
  const setSpy = jest.spyOn(global, 'setInterval');
  const clearSpy = jest.spyOn(global, 'clearInterval');

  const { unmount } = render(<MissionPanel onError={jest.fn()} />);
  expect(setSpy).toHaveBeenCalled();

  unmount();
  expect(clearSpy).toHaveBeenCalled();

  setSpy.mockRestore();
  clearSpy.mockRestore();
});

test('judge buttons call missionReport with the goalAchieved flag', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  fireEvent.click(screen.getByRole('button', { name: '判定完成' }));
  await waitFor(() => expect(api.missionReport).toHaveBeenCalledWith(true));

  const failBtn = screen.getByRole('button', { name: '判定失败' });
  await waitFor(() => expect(failBtn).toBeEnabled());
  fireEvent.click(failBtn);
  await waitFor(() => expect(api.missionReport).toHaveBeenCalledWith(false));
});

test('interject sends text and optional amendGoal', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  fireEvent.change(screen.getByLabelText('插话内容'), { target: { value: '注意截止时间' } });
  fireEvent.change(screen.getByLabelText('重定向目标'), { target: { value: '改为分析性能' } });
  fireEvent.click(screen.getByRole('button', { name: '发送插话' }));

  await waitFor(() => expect(api.autonomyInterject).toHaveBeenCalledWith({ text: '注意截止时间', amendGoal: '改为分析性能' }));
});

test('autonomy loop configure/start/stop follow the action contract', async () => {
  render(<MissionPanel onError={jest.fn()} />);

  await waitFor(() => expect(api.autonomyLoop).toHaveBeenCalledWith({ action: 'status' }));

  fireEvent.change(screen.getByLabelText('intervalSec'), { target: { value: '5' } });
  fireEvent.change(screen.getByLabelText('maxStepsPerTick'), { target: { value: '10' } });
  fireEvent.change(screen.getByLabelText('persistEveryTicks'), { target: { value: '3' } });
  fireEvent.click(screen.getByRole('button', { name: '配置' }));
  await waitFor(() => expect(api.autonomyLoop).toHaveBeenCalledWith({ action: 'configure', intervalSec: 5, maxStepsPerTick: 10, persistEveryTicks: 3 }));

  fireEvent.click(screen.getByRole('button', { name: '启动' }));
  await waitFor(() => expect(api.autonomyLoop).toHaveBeenCalledWith({ action: 'start' }));

  fireEvent.click(screen.getByRole('button', { name: '停止' }));
  await waitFor(() => expect(api.autonomyLoop).toHaveBeenCalledWith({ action: 'stop' }));
});

test('estop requires confirmation and sends the reason', async () => {
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(false);

  render(<MissionPanel onError={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('急停原因'), { target: { value: '行为越界' } });
  fireEvent.click(screen.getByRole('button', { name: '急停' }));

  await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
  expect(api.estop).not.toHaveBeenCalled();

  confirmSpy.mockReturnValue(true);
  fireEvent.click(screen.getByRole('button', { name: '急停' }));
  await waitFor(() => expect(api.estop).toHaveBeenCalledWith({ reason: '行为越界' }));

  confirmSpy.mockRestore();
});

test('shows 后端不可达 when every status fetch fails', async () => {
  api.missionStatus.mockRejectedValue(new Error('network'));
  api.autonomyStatus.mockRejectedValue(new Error('network'));
  api.estopStatus.mockRejectedValue(new Error('network'));

  render(<MissionPanel onError={jest.fn()} />);

  expect(await screen.findByText(/后端不可达/)).toBeInTheDocument();
});
