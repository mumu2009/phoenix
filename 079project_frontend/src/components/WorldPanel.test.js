import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import WorldPanel from './WorldPanel';

jest.mock('../api/client', () => ({
  api: {
    worldPhysicsStatus: jest.fn(),
    worldStatus: jest.fn(),
    worldState: jest.fn(),
    worldEarthMapImport: jest.fn(),
    worldSimulate: jest.fn()
  }
}));

const { api } = require('../api/client');

beforeEach(() => {
  jest.clearAllMocks();
  api.worldPhysicsStatus.mockResolvedValue({
    ok: true,
    physicsRuntime: {
      runtimeMode: 'native-embedded-source',
      nativeCompiled: true,
      preferredEarthFormat: 'heightfield',
      bundledEarthHeightfieldUri: 'static/earth_maps/china_relief_heightfield.json',
      summary: 'bullet3 repo detected; runtime mode is native-embedded-source'
    },
    defaults: {
      physicsEnabled: true,
      physicsBackend: 'bullet3',
      physicsSubsteps: 4,
      earthMap: {
        enabled: true,
        sourceUri: 'static/earth_maps/china_relief_heightfield.json',
        format: 'heightfield',
        regionLabel: 'china-relief-demo',
        lod: 6,
        metersPerCell: 750
      }
    }
  });
  api.worldStatus.mockResolvedValue({ ok: true });
  api.worldState.mockResolvedValue({ recentEvidence: [{ modality: 'simulation_physics_runtime', graphSummary: 'native bullet trace persisted' }] });
  api.worldEarthMapImport.mockResolvedValue({
    ok: true,
    earthMap: { summary: 'Bundled China relief heightfield imported.', regionLabel: 'china-relief-demo' }
  });
  api.worldSimulate.mockResolvedValue({
    ok: true,
    trainSamples: [{ source: 'sim_physics_runtime' }],
    physicsExecution: {
      status: 'executed',
      summary: 'native Bullet execution produced 2 body traces',
      bodySummaries: [
        { id: 'agent-planner-1', role: 'planner', displacementMeters: 3.2, peakSpeedMps: 1.4 },
        { id: 'agent-explorer-2', role: 'explorer', displacementMeters: 4.8, peakSpeedMps: 1.9 }
      ]
    }
  });
});

test('loads runtime defaults and exposes import and simulation actions', async () => {
  render(<WorldPanel activeSessionId="s1" ensureSession={() => ({ id: 's1' })} onError={jest.fn()} />);

  expect(await screen.findByText('native-embedded-source')).toBeInTheDocument();
  expect(screen.getByDisplayValue('static/earth_maps/china_relief_heightfield.json')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: '导入地形' }));
  await waitFor(() => expect(api.worldEarthMapImport).toHaveBeenCalled());
  expect(await screen.findByText('Bundled China relief heightfield imported.')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: '运行仿真' }));
  await waitFor(() => expect(api.worldSimulate).toHaveBeenCalled());
  expect((await screen.findAllByText(/native Bullet execution produced 2 body traces/i)).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/agent-explorer-2/i).length).toBeGreaterThan(0);
});