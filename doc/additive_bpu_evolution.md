# Additive Residual BPU Evolution Framework

This document describes the additive residual BPU evolution framework in the
Phoenix repo.  It grows a single ONNX model from a zero output by adding one
small residual block per round.  A host (typically the Kali compile box)
generates candidate ONNX files and compiles them with the Horizon BPU toolchain;
the RDK X5 evaluates each candidate on a fresh batch from a local data pool and
returns the MSE.  The best candidate becomes the next model.  The loop is
gradient-free: Kali only creates and compiles ONNX graphs; no PyTorch training
is performed on the compile host.

## Architecture

`tools/additive_jpea.py` defines `AdditiveResidualModel`, a generic container
that owns:

* an optional **frozen base** (e.g. ResNet18 for the vision encoder)
* a list of **residual blocks**, all sharing the same input shape and producing
  the same output shape

The forward pass is:

```text
output = base(input) + sum(block(input) for block in blocks)
```

If `base` is `None` and no blocks exist, the output is a constant zero tensor.

### Residual block builders

| Model | Input (NCHW, batch=1) | Output (NCHW, batch=1) | ~params |
|-------|----------------------|------------------------|---------|
| `SpeechEncoderBlock` | `1x1x1x16000` | `1x128x1x1` | ~124k |
| `SpeechDecoderBlock` | `1x128x1x1` | `1x1x1x15872` | ~113-210k (configurable `first`) |
| `VisionEncoderBlock` | `1x3x224x224` | `1x128x1x1` | ~242k |
| `VisionDecoderBlock` | `1x128x1x1` | `1x3x224x224` | ~127-244k (configurable `first`) |

The vision encoder can optionally load a frozen pretrained ResNet18 base from
`torchvision` or from a `.pt` checkpoint.  When a base is present, its output is
added to the residual sum, so the residual blocks learn to refine or augment the
base embedding while the base itself remains unchanged.

### ONNX compatibility

All blocks use only BPU-friendly operators:

* `Conv2d`, `ConvTranspose2d`
* `Linear`
* `BatchNorm2d`
* `ReLU`
* `AdaptiveAvgPool2d`
* `view` reshapes with fixed batch dimension

Batch size is fixed at 1.  No dynamic axes are exported, and no training-only
ops are used.

## File inventory

| File | Purpose |
|------|---------|
| `tools/additive_jpea.py` | `AdditiveResidualModel`, block builders, export helpers |
| `tools/export_additive_jpea.py` | CLI to export one `.pt` checkpoint to ONNX + calibration + manifest |
| `tools/bpu_evolve_additive.py` | `(1+lambda)` evolution controller |
| `tools/compile_bpu_jepa_v2.sh` | Updated to compile a **single** generic ONNX or a speech/image pair |
| `tools/x5_bpu_evaluate.py` | Updated to evaluate a single `.bin` or a directory of `.bin` files against `.npy` or `.bin` input/target batches |
| `doc/additive_bpu_evolution.md` | This document |

## Workflow

1. **Prepare a data pool** (`inputs.npy` / `targets.npy`) for the model you want
to evolve.  A pool can be created manually, with an existing teacher model, or
with the built-in synthetic generator for smoke tests.

2. **Initialize the model**.  With `--start-from-zero` the first round starts
from an empty model (zero output).  For the vision encoder you can load a frozen
ResNet18 base with `--base-path`.

3. **Run the controller**.  For each round it:
   * loads the current `best.pt`
   * draws a fresh `--batch-size` sample from the pool
   * generates `--lambda` candidates, each = parent + one new residual block
   * exports each candidate ONNX and writes calibration bins
   * compiles candidates in parallel using
     `tools/compile_bpu_jepa_v2.sh --model-name ... --onnx ...`
   * copies the compiled `.bin` files and the batch to the X5
   * runs `tools/x5_bpu_evaluate.py` on the X5
   * selects the lowest-MSE candidate and promotes it to `best.pt/best.onnx`
   * updates `work_dir/<model_name>/evolve_state.json`

4. **Stop** after `--max-rounds` or when the loss stops improving.

## Commands

### 1. Create an additive model checkpoint manually

```bash
python tools/additive_jpea.py \
    --model-name speech_encoder \
    --n-blocks 0 \
    --out-dir /tmp/additive_speech_encoder_init
```

For the vision encoder with a ResNet18 base:

```bash
python tools/additive_jpea.py \
    --model-name vision_encoder \
    --base-path /path/to/resnet18_base.pt \
    --n-blocks 0 \
    --out-dir /tmp/additive_vision_encoder_init
```

### 2. Export a checkpoint to ONNX

```bash
python tools/export_additive_jpea.py \
    --model-name speech_encoder \
    --checkpoint /tmp/additive_speech_encoder_init/model.pt \
    --out-dir /tmp/additive_speech_encoder_onnx
```

### 3. Compile one ONNX with the generic compile script

```bash
bash tools/compile_bpu_jepa_v2.sh \
    --model-name speech_encoder \
    --onnx /tmp/additive_speech_encoder_onnx/model.onnx \
    --calib-dir /tmp/additive_speech_encoder_onnx/calibration \
    --input-name waveform \
    --input-shape 1x1x1x16000 \
    --out-dir /tmp/additive_speech_encoder_bin \
    --per-channel True \
    --calib-type kl
```

For a decoder use `--per-channel False --calib-type max`.

### 4. Generate a synthetic smoke-test pool

```bash
python tools/bpu_evolve_additive.py \
    --model-name speech_decoder \
    --prepare-synthetic-pool 1000 \
    --data-pool /tmp/pools/speech_decoder
```

### 5. Prepare a real pool from a dataset and teacher

```bash
# Speech decoder
python tools/bpu_evolve_additive.py \
    --model-name speech_decoder \
    --prepare-pool \
    --data-pool /tmp/pools/speech_decoder \
    --dataset /path/to/musan_16k \
    --teacher-pt /path/to/speech_encoder_teacher.pt \
    --pool-size 10000

# Vision encoder with ResNet18 base
python tools/bpu_evolve_additive.py \
    --model-name vision_encoder \
    --prepare-pool \
    --data-pool /tmp/pools/vision_encoder \
    --dataset /path/to/images \
    --teacher-pt /path/to/vision_encoder_teacher.pt \
    --pool-size 5000
```

### 6. Run evolution

```bash
# Speech decoder
python tools/bpu_evolve_additive.py \
    --model-name speech_decoder \
    --data-pool /tmp/pools/speech_decoder \
    --work-dir /tmp/additive_work \
    --start-from-zero \
    --max-rounds 10 \
    --lambda 4 \
    --batch-size 1000 \
    --parallel 2 \
    --x5-host 192.168.0.107 \
    --x5-user sunrise \
    --x5-pass sunrise

# Vision encoder (starts from ResNet18 base)
python tools/bpu_evolve_additive.py \
    --model-name vision_encoder \
    --data-pool /tmp/pools/vision_encoder \
    --work-dir /tmp/additive_work \
    --base-path /path/to/resnet18_base.pt \
    --max-rounds 10 \
    --lambda 4 \
    --batch-size 256 \
    --parallel 2 \
    --x5-host 192.168.0.107 \
    --x5-user sunrise \
    --x5-pass sunrise
```

### 7. Evaluate a single compiled `.bin` on the X5

```bash
python3 x5_bpu_evaluate.py \
    --bin /home/sunrise/phoenix/evolve/round_0001/candidate_0000.bin \
    --inputs /home/sunrise/phoenix/evolve/round_0001_inputs.bin \
    --targets /home/sunrise/phoenix/evolve/round_0001_targets.bin \
    --input-shape 1x1x1x16000 \
    --target-shape 1x1x1x15872 \
    --format bin \
    --out /home/sunrise/phoenix/evolve/losses.json
```

## Input / output shapes

| Model | Input name | Input shape | Output name | Output shape |
|-------|------------|-------------|-------------|--------------|
| `speech_encoder` | `waveform` | `1x1x1x16000` | `concept` | `1x<concept>x1x1` |
| `speech_decoder` | `concept` | `1x<concept>x1x1` | `reconstruction` | `1x1x1x15872` |
| `vision_encoder` | `pixel_values` | `1x3x224x224` | `concept` | `1x<concept>x1x1` |
| `vision_decoder` | `concept` | `1x<concept>x1x1` | `reconstruction` | `1x3x224x224` |

The default `<concept>` is 128 and can be changed with `--concept`.  The data
pool must match the chosen concept; `DataPool.load()` raises a clear error if
not.

## Data pool format

A data pool is a directory with two files:

* `inputs.npy` -- float32 array of shape `[N, *input_shape]`
* `targets.npy` -- float32 array of shape `[N, *output_shape]`

`input_shape` and `output_shape` include the leading batch dimension of 1,
matching the shapes above.  Each round samples `--batch-size` entries and writes
a raw float32 `.bin` batch for the X5.

## Calibration

`export_additive_jpea.py` and `AdditiveResidualModel.export_to_onnx` write a
`calibration/` directory containing raw float32 `.bin` files.  The generic
compile command uses this directory directly.  For encoders, `per_channel=True`
and `calib_type=kl` work well.  For decoders, `per_channel=False` and
`calib_type=max` are the defaults.

## Notes and constraints

* No PyTorch gradient training runs on the compile host.  New blocks are
  randomly initialized and selected by the BPU.
* The ResNet18 vision base stays frozen; its weights are baked into the ONNX.
* All candidate compilations can run in parallel up to `--parallel`.
* `--eval-local` and `--no-bpu` run the candidate ONNX files with local
  `onnxruntime` for quick debugging; real BPU selection requires an X5.
* Compiled artifacts and candidate history live under
  `work_dir/<model_name>/candidates/round_####/`.  Set `--keep-candidates` to
  retain them all, otherwise only the promoted best is kept.
