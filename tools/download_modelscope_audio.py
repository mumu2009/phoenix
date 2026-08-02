import argparse
import os
import sys

try:
    from modelscope.msdatasets import MsDataset
except Exception as e:
    print("modelscope not available:", e, file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="manyeyes/MUSAN", help="ModelScope dataset id")
    parser.add_argument("--split", default="train", help="split to download")
    parser.add_argument("--max-samples", type=int, default=1000, help="max samples to save")
    parser.add_argument("--target-sr", type=int, default=16000, help="resample to this rate")
    parser.add_argument("--out-dir", required=True, help="output directory for 16-bit PCM WAV files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.dataset}/{args.split} ...")
    ds = MsDataset.load(args.dataset, split=args.split)
    print("dataset:", ds)

    try:
        import soundfile as sf
        import librosa
        import numpy as np
    except ImportError as e:
        print("missing audio dependency:", e, file=sys.stderr)
        sys.exit(1)

    saved = 0
    for i, item in enumerate(ds):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        try:
            if "audio" in item:
                audio = item["audio"]
                arr = audio["array"]
                sr = audio["sampling_rate"]
            elif "path" in item:
                path = item["path"]
                arr, sr = librosa.load(path, sr=None, mono=True)
            else:
                continue
            arr = librosa.resample(arr, orig_sr=sr, target_sr=args.target_sr)
            # normalize to [-1,1] and convert to 16-bit PCM
            arr = arr / (max(abs(arr.max()), abs(arr.min())) + 1e-8)
            pcm = (arr * 32767).astype(np.int16)
            out_path = os.path.join(args.out_dir, f"sample_{i:06d}.wav")
            sf.write(out_path, pcm, args.target_sr, subtype="PCM_16")
            saved += 1
            if saved % 100 == 0:
                print(f"saved {saved} files ...")
        except Exception as e:
            print(f"skip sample {i}: {e}", file=sys.stderr)
    print(f"Done. Saved {saved} files to {args.out_dir}")

if __name__ == "__main__":
    main()
