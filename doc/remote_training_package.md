# Remote Video/Audio Training Package

This package trains the Phoenix video and audio autoencoders on the
remote Windows 10 Pro GPU box and produces ONNX / BPU-compatible `.bin` files
for the RDK X5.

## Files to copy to the remote Windows box

From `D:\_phoenix\_079\v6.0Alixander\phoenix`:

```
tools/train_audio.py
tools/train_video.py
tools/export_multimodal.py
tools/export_speech.py
tools/compile_bpu.sh
tools/run_hb_mapper.py
tools/hb_mapper_patch.py
tools/train_pilot.py
doc/remote_training_package.md
```

Also copy `/home/kali/decoder_trained.pt` from Kali to the remote box and put it
in the same directory you will use for vision training (e.g. `D:\_phoenix\_079\v6.0Alixander\phoenix\decoder_trained.pt`).

## Prerequisites (remote Windows)

Open an elevated PowerShell / CMD and install:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install soundfile numpy pillow
```

> If the remote uses a different CUDA version, replace `cu118` with `cu121` or
> `cu124`.  The training scripts also run on CPU if CUDA is unavailable.

Optional but recommended for 6 GB cards:

```powershell
pip install torch-tensorrt  # only if you plan to experiment with torch.compile
```

## Datasets

- **Speech**: MUSAN 16 kHz mono WAV files.  On the remote box point `--data-dir`
  at the folder containing `*.wav`.  Example: `D:\datasets\musan_16k`.
- **Vision**: a directory of images.  Accepts ImageFolder class subdirectories
  or a flat folder of `*.jpg / *.png`.  Images are resized to 224x224 and
  normalized with ImageNet mean/std.

## Quick sanity check (tiny, CPU/GPU)

```powershell
# audio ~1 M params, 3 epochs, quick smoke test
python tools\train_pilot.py `
    --data-dir D:\datasets\musan_16k `
    --out-dir D:\checkpoints\audio_pilot `
    --epochs 3 --batch-size 8 --device auto
```

This creates `D:\checkpoints\audio_pilot\best.pt` and `model_*.onnx`.

## Example: 5 M "small" audio run

```powershell
python tools\train_audio.py `
    --data-dir D:\datasets\musan_16k `
    --out-dir D:\checkpoints\audio_small `
    --scale small `
    --epochs 50 `
    --batch-size 32 `
    --lr 1e-3 `
    --device cuda
```

After training, export and compile on Kali (or copy back to Kali):

```powershell
python tools\export_multimodal.py `
    --modality speech `
    --checkpoint D:\checkpoints\audio_small\best.pt `
    --out-dir D:\checkpoints\audio_small\onnx
```

Then on Kali, inside the OpenExplorer Docker:

```bash
bash /workspace/tools/compile_bpu.sh \
    --modality speech \
    --onnx-dir /workspace/checkpoints/audio_small/onnx \
    --out-dir /workspace/checkpoints/audio_small/bin
```

## Example: 5 M "small" video run

```powershell
python tools\train_video.py `
    --data-dir D:\datasets\images `
    --out-dir D:\checkpoints\video_small `
    --resnet resnet18 `
    --concept 128 `
    --decoder-pt none `
    --decoder-width 1.25 `
    --epochs 50 `
    --batch-size 16 `
    --lr 1e-3 `
    --device cuda
```

> `--decoder-width 1.25` gives a decoder of about 4.7 M trainable parameters
> (ResNet base is frozen by default).  Use `--decoder-width 1.0` if you want to
> warm-start from `decoder_trained.pt`.

Export and compile:

```powershell
python tools\export_multimodal.py `
    --modality image `
    --checkpoint D:\checkpoints\video_small\best.pt `
    --out-dir D:\checkpoints\video_small\onnx
```

This writes:
  - `model_encoder.onnx` (ResNet base, 3x224x224 -> 512-D embedding)
  - `model_encoder_head.onnx` (tiny 512 -> concept linear head)
  - `encoder_head.json` (same head weights, read by C++ at runtime)
  - `model_decoder.onnx` (concept -> 3x224x224)

```bash
bash /workspace/tools/compile_bpu.sh \
    --modality image \
    --onnx-dir /workspace/checkpoints/video_small/onnx \
    --out-dir /workspace/checkpoints/video_small/bin
```

## Larger scales

| scale  | audio params | video decoder params | notes |
|--------|---------------|-----------------------|-------|
| tiny   | ~1.1 M        | ~0.8 M (w=0.5)        | Kali CPU smoke test |
| small  | ~5.0 M        | ~4.7 M (w=1.25)       | default, 6 GB OK |
| medium | ~22 M         | ~16 M (w=3.0)         | use `--amp` |
| large  | ~48 M         | ~32 M (w=4.0)         | use `--amp --grad-accum 2` |
| xlarge | ~96 M         | ~55 M (w=5.0,d=3)     | may not compile on X5; for R&D |

Video decoder width examples:

```powershell
# medium
--decoder-width 3.0 --decoder-depth 0

# large
--decoder-width 4.0 --decoder-depth 1

# xlarge
--decoder-width 5.0 --decoder-depth 3
```

## VRAM / batch-size guide

The remote GPU is a GTX 16-series with ~6 GB.  Use these rules of thumb:

- **tiny / small**: batch 32-64, no AMP needed.
- **medium**: batch 16-24, add `--amp`.
- **large**: batch 8-12, `--amp --grad-accum 2`.
- **xlarge**: batch 4-8, `--amp --grad-accum 4`.  If you still get OOM:
  - reduce `--batch-size`
  - increase `--grad-accum`
  - disable `--compile`
  - reduce `--decoder-width`
  - use `resnet18` instead of `resnet50`

## Important flags

| flag | what it does |
|------|--------------|
| `--scale tiny/small/medium/large/xlarge` | speech model size preset |
| `--encoder-channels`, `--decoder-channels` | explicit comma lists for speech |
| `--concept` | concept dim (default 128, keep this for X5) |
| `--resnet resnet18/34/50` | vision encoder base |
| `--unfreeze-encoder` | train the ResNet base (needs more VRAM) |
| `--decoder-width` | video decoder channel multiplier (must be `1.0` to load `decoder_trained.pt`) |
| `--decoder-depth` | extra 3x3 residual blocks after the video decoder upsampling stack |
| `--pretrained 0` | use randomly initialized ResNet instead of downloading ImageNet weights |
| `--amp` | use PyTorch automatic mixed precision (fp16) |
| `--grad-accum N` | accumulate gradients over N mini-batches before stepping |
| `--compile` | try `torch.compile` (PyTorch 2.x only) |

## Training on the remote box for long runs

Because the remote is Windows, use `Start-Process` with a log file rather than
`nohup`:

```powershell
$log = "D:\checkpoints\speech_medium.log"
Start-Process -FilePath python -ArgumentList @(
    "tools\train_audio.py",
    "--data-dir", "D:\datasets\musan_16k",
    "--out-dir", "D:\checkpoints\audio_medium",
    "--scale", "medium",
    "--epochs", "100",
    "--batch-size", "16",
    "--amp",
    "--grad-accum", "2"
) -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError $log
```

On Kali/X5, continue to use `nohup`:

```bash
nohup python tools/train_pilot.py \
    --data-dir /home/kali/phoenix/datasets/musan_16k \
    --out-dir /tmp/speech_pilot > /tmp/speech_pilot.log 2>&1 &
echo $!
```

## Calibration notes

`export_multimodal.py` writes synthetic random calibration bins by
default.  For the best BPU accuracy on real data, you can later replace those
with real samples:

- **audio encoder**: one `1x1x1x16000` float32 `.bin` per sample
- **audio decoder**: one `1xCONCEPTx1x1` float32 `.bin` per sample
- **video encoder**: one `1x3x224x224` float32 `.bin` per sample, NCHW,
  ImageNet-normalized
- **video decoder**: one `1xCONCEPTx1x1` float32 `.bin` per sample

## Output layout for the X5

After compiling, copy the artifacts to the X5 model directory, e.g.:

```
runtime_store/models/ijepa/audio-16k/
  model_encoder.bin
  model_decoder.bin
  model.manifest.json

runtime_store/models/ijepa/video-encoder/
  model_encoder.bin       # ResNet base -> 512-D embedding
  model_decoder.bin       # concept -> 3x224x224
  encoder_head.json       # 512 -> concept linear head (read by C++)
  model_encoder_head.onnx # optional CPU/onnxruntime version of the head
  model.manifest.json
```

Then set `JEPA_SPEECH_HORIZON_MODEL` or `JEPA_IMAGE_HORIZON_MODEL` accordingly.
