import argparse
import json
import os
from pathlib import Path
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=r'D:\_phoenix\_079\v6.0Alixander\phoenix\runtime_store\datasets\musan_16k')
    args = parser.parse_args()
    d = Path(args.data_dir)
    out = d / 'manifest.json'
    files = sorted([f for f in d.iterdir() if f.suffix == '.wav'])
    total_samples = 0
    entries = []
    for i, f in enumerate(files, 1):
        try:
            info = sf.info(str(f))
            total_samples += info.frames
            entries.append({'id': i, 'file': f.name, 'samplerate': info.samplerate, 'frames': info.frames, 'channels': info.channels})
        except Exception as e:
            print('skip', f, e)
    manifest = {'count': len(entries), 'total_samples_16k': total_samples, 'total_seconds': total_samples / 16000, 'entries': entries}
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {out}: {len(entries)} files, {total_samples/16000/3600:.2f} hours')

if __name__ == '__main__':
    main()
