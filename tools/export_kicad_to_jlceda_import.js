const fs = require('fs');
const path = require('path');

const MM_TO_MIL = 39.37;
const DSN_UNIT_TO_MIL = MM_TO_MIL / 1000;

const repoRoot = process.cwd();
const dsnPath = path.join(repoRoot, 'catastrophe', 'ProPrj_catastrophe_2026-04-29', 'ProPrj_catastrophe_2026-04-29.dsn');
const pcbPath = path.join(repoRoot, 'catastrophe', 'ProPrj_catastrophe_2026-04-29', 'ProPrj_catastrophe_2026-04-29.kicad_pcb');
const outputDir = path.join(repoRoot, 'tools', 'generated');
const outputPath = path.join(outputDir, 'kicad_to_jlceda_import.json');

function round(value) {
  return Math.round(value * 10000) / 10000;
}

function mmToMil(value) {
  return round(Number(value) * MM_TO_MIL);
}

function dsnToMil(value) {
  return round(Number(value) * DSN_UNIT_TO_MIL);
}

function layerNameToId(layerName) {
  const normalized = String(layerName).trim();
  const map = {
    'F.Cu': 1,
    'Top Layer': 1,
    'B.Cu': 2,
    'Bottom Layer': 2,
    'In1.Cu': 15,
    Inner1: 15,
    'In2.Cu': 16,
    Inner2: 16,
    'In3.Cu': 17,
    Inner3: 17,
    'In4.Cu': 18,
    Inner4: 18,
    'In5.Cu': 19,
    Inner5: 19,
    'In6.Cu': 20,
    Inner6: 20,
  };
  return map[normalized] ?? null;
}

function countParenDelta(line) {
  let delta = 0;
  for (const char of line) {
    if (char === '(') delta += 1;
    if (char === ')') delta -= 1;
  }
  return delta;
}

function collectBlocks(text, blockName) {
  const lines = text.split(/\r?\n/);
  const blocks = [];
  let collecting = false;
  let depth = 0;
  let current = [];
  const marker = `(${blockName}`;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!collecting && trimmed.startsWith(marker)) {
      collecting = true;
      depth = 0;
      current = [];
    }
    if (!collecting) {
      continue;
    }
    current.push(line);
    depth += countParenDelta(line);
    if (collecting && depth === 0) {
      blocks.push(current.join('\n'));
      collecting = false;
      current = [];
    }
  }

  return blocks;
}

function parseDsnPlacements(dsnText) {
  const placements = {};
  const placementRegex = /\(place\s+([^\s]+)\s+([-\d.]+)\s+([-\d.]+)\s+(front|back)\s+([-\d.]+)\)/g;
  let match = placementRegex.exec(dsnText);

  while (match) {
    const ref = match[1];
    placements[ref] = {
      x: dsnToMil(match[2]),
      y: dsnToMil(match[3]),
      layerId: match[4] === 'back' ? 2 : 1,
      rotation: round(Number(match[5])),
    };
    match = placementRegex.exec(dsnText);
  }

  return placements;
}

function parseDsnBoardOutline(dsnText) {
  const boundaryMatch = dsnText.match(/\(boundary\s*\(path pcb 0\s+([\s\S]*?)\)\s*\)/);
  if (!boundaryMatch) {
    return null;
  }

  const numbers = boundaryMatch[1]
    .trim()
    .split(/\s+/)
    .map(Number)
    .filter((value) => !Number.isNaN(value));

  const points = [];
  for (let index = 0; index < numbers.length; index += 2) {
    points.push({
      x: dsnToMil(numbers[index]),
      y: dsnToMil(numbers[index + 1]),
    });
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);

  return {
    left,
    right,
    top,
    bottom,
    polygonSource: [[
      left,
      top,
      'L',
      right,
      top,
      'L',
      right,
      bottom,
      'L',
      left,
      bottom,
      'L',
      left,
      top,
    ]],
  };
}

function parseSegmentBlock(block) {
  const start = block.match(/\(start\s+([-\d.]+)\s+([-\d.]+)\)/);
  const end = block.match(/\(end\s+([-\d.]+)\s+([-\d.]+)\)/);
  const width = block.match(/\(width\s+([-\d.]+)\)/);
  const layer = block.match(/\(layer\s+"([^"]+)"\)/);
  const net = block.match(/\(net\s+"([^"]+)"\)/);
  if (!start || !end || !width || !layer || !net) {
    return null;
  }

  const layerId = layerNameToId(layer[1]);
  if (layerId === null) {
    return null;
  }

  return {
    net: net[1],
    layerId,
    startX: mmToMil(start[1]),
    startY: mmToMil(-Number(start[2])),
    endX: mmToMil(end[1]),
    endY: mmToMil(-Number(end[2])),
    lineWidth: mmToMil(width[1]),
  };
}

function parseViaBlock(block) {
  const at = block.match(/\(at\s+([-\d.]+)\s+([-\d.]+)\)/);
  const size = block.match(/\(size\s+([-\d.]+)\)/);
  const drill = block.match(/\(drill\s+([-\d.]+)\)/);
  const net = block.match(/\(net\s+"([^"]+)"\)/);
  if (!at || !size || !drill || !net) {
    return null;
  }

  return {
    net: net[1],
    x: mmToMil(at[1]),
    y: mmToMil(-Number(at[2])),
    diameter: mmToMil(size[1]),
    holeDiameter: mmToMil(drill[1]),
    viaType: 0,
  };
}

function main() {
  const dsnText = fs.readFileSync(dsnPath, 'utf8');
  const pcbText = fs.readFileSync(pcbPath, 'utf8');

  const placements = parseDsnPlacements(dsnText);
  const boardOutline = parseDsnBoardOutline(dsnText);
  const tracks = collectBlocks(pcbText, 'segment').map(parseSegmentBlock).filter(Boolean);
  const vias = collectBlocks(pcbText, 'via').map(parseViaBlock).filter(Boolean);

  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        meta: {
          sourceDsn: path.relative(repoRoot, dsnPath).replace(/\\/g, '/'),
          sourcePcb: path.relative(repoRoot, pcbPath).replace(/\\/g, '/'),
          placements: Object.keys(placements).length,
          tracks: tracks.length,
          vias: vias.length,
        },
        boardOutline,
        placements,
        tracks,
        vias,
      },
      null,
      2,
    ),
    'utf8',
  );

  console.log(JSON.stringify({
    outputPath,
    placements: Object.keys(placements).length,
    tracks: tracks.length,
    vias: vias.length,
    boardOutline,
  }, null, 2));
}

main();