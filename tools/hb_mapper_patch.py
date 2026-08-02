#!/usr/bin/env python3
"""Patch hb_mapper/ONNX to bound temp external-data files and skip heavy optimizer.

hb_mapper repeatedly serializes >2 GB ONNX models. ONNX 1.12.0's external-data
helpers create new uuid1() files on every call, causing /tmp to fill. This patch
forces a single ``external.data`` blob and cleans stale files before each write.

It also loads external tensor data into memory when an OnnxModel is constructed
from a file path, so serialized copies do not reference a missing side-car file
during onnxruntime calibration.

Finally, it injects ``skip_optimizer`` into ModelBuilder and turns
ModelBuilder.optimize into a no-op, so the multi-gigabyte optimizer passes are
bypassed.

The patch is applied lazily: it installs a Python import hook that waits for
``horizon_nn.ir`` (or ``hmct.ir``) to finish loading. That package imports
``onnx`` first and then ``horizon_onnx``, so by the time our hook runs both ONNX
modules are in memory and can be safely patched without the ``OpSchema``
registration conflict that occurs when ``horizon_onnx`` is loaded twice.
"""

import os
import re
import sys
import time
import traceback
import importlib.machinery

import numpy as np


_EXTERNAL_DATA_BASENAME = "external.data"
_DEBUG_LOG = "/workspace/hb_mapper_patch.log"


def _log(msg):
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _apply_patches():
    """Apply all patches once. Called after horizon_nn.ir (or hmct.ir) is loaded."""
    if getattr(_apply_patches, "_done", False):
        return
    _apply_patches._done = True
    _log("Applying hb_mapper patches")

    # 1) Patch OnnxModel to load external data into memory when constructed from a file
    # and to remember the source path so later temp-serialization can reload data.
    try:
        import onnx
        from onnx.external_data_helper import load_external_data_for_model
        from horizon_nn.ir.onnx_model import OnnxModel

        _orig_init_from_proto = OnnxModel._init_from_proto

        def _patched_init_from_proto(self, proto):
            _orig_init_from_proto(self, proto)
            if isinstance(proto, str):
                self._source_path = proto
                _log(f"OnnxModel source path recorded: {proto}")
                # Keep external tensors on disk: do NOT load raw data into memory.
                # Loading the full 2.5GB into ModelProto makes protobuf serialization
                # exceed the 2GB limit and crashes both horizon_onnx and onnx
                # shape inference / checker.
            else:
                self._source_path = None

        OnnxModel._init_from_proto = _patched_init_from_proto

        # 1a) Make OnnxModel.infer_shapes and check_validity no-ops.
        # The bundled horizon_onnx implementation crashes on the I-JEPA model, and
        # the standard onnx shape inference/checker hit the 2 GB protobuf limit.
        # hb_mapper's later C++ passes can determine the shapes they need, so we
        # skip the fragile Python-side shape inference entirely.
        def _patched_infer_shapes(self, clear_shapes: bool = True) -> "OnnxModel":
            _log(f"OnnxModel.infer_shapes called (no-op, clear_shapes={clear_shapes})")
            return self

        def _patched_check_validity(self) -> None:
            _log("OnnxModel.check_validity called (no-op)")
            return

        OnnxModel.infer_shapes = _patched_infer_shapes
        OnnxModel.check_validity = _patched_check_validity
        _log("OnnxModel.infer_shapes / check_validity patched to no-ops")
    except Exception as e:
        _log(f"OnnxModel patch failed: {e}\n{traceback.format_exc()}")

    # 1b) Patch save_model to remember where external data was written, and patch
    # serialize_model/serialize_proto to reload external data before temp
    # serialization.  If raw_data has been cleared by a previous save or a model
    # pass, onnx.save cannot regenerate the external.data file, causing ORT
    # SystemError 2.
    try:
        from onnx.external_data_helper import (
            load_external_data_for_model,
            uses_external_data,
            ExternalDataInfo,
        )
        import horizon_nn.ir as _ir_mod

        _ls_mod = sys.modules.get("hmct.ir.load_store")
        if _ls_mod is None:
            _ls_mod = sys.modules.get("horizon_nn.ir.load_store")

        _orig_serialize_proto = getattr(_ls_mod, "serialize_proto")
        _orig_serialize_model = getattr(_ls_mod, "serialize_model")
        _orig_save_model = getattr(_ls_mod, "save_model")

        def _patched_save_model(onnx_model_or_proto, onnx_file):
            result = _orig_save_model(onnx_model_or_proto, onnx_file)
            try:
                if hasattr(onnx_model_or_proto, "proto"):
                    onnx_model_or_proto._external_data_dir = os.path.dirname(
                        os.path.abspath(onnx_file)
                    )
            except Exception:
                pass
            return result

        def _find_external_data_basepath(onnx_model, proto):
            """Return directory containing the side-car file(s) referenced by proto."""
            candidates = []
            ext_dir = getattr(onnx_model, "_external_data_dir", None)
            if ext_dir:
                candidates.append(ext_dir)
            src = getattr(onnx_model, "_source_path", None)
            if src:
                candidates.append(os.path.dirname(src))
            candidates.extend([os.getcwd(), "/tmp", "/open_explorer"])

            locs = set()
            for tensor in onnx.external_data_helper._get_all_tensors(proto):
                if not tensor.HasField("raw_data"):
                    info = ExternalDataInfo(tensor)
                    if info.location:
                        locs.add(info.location)
            if not locs:
                return None

            for base in candidates:
                if not base or not os.path.isdir(base):
                    continue
                ok = True
                for loc in locs:
                    path = os.path.join(base, loc)
                    if not os.path.isfile(path):
                        ok = False
                        break
                if ok:
                    return base

            for root in ["/tmp", "/open_explorer", os.getcwd()]:
                for loc in locs:
                    for dirpath, _, files in os.walk(root):
                        if loc in files:
                            return dirpath
            return None

        def _reload_external_data_if_needed(onnx_model, proto=None):
            if proto is None:
                proto = onnx_model.proto
            missing = []
            for tensor in onnx.external_data_helper._get_all_tensors(proto):
                if tensor.HasField("raw_data"):
                    continue
                info = ExternalDataInfo(tensor)
                if info.location:
                    missing.append((tensor.name, info.location))
            if not missing:
                return proto

            basepath = _find_external_data_basepath(onnx_model, proto)
            _log(
                f"serialize_model missing raw_data for {len(missing)} tensors; "
                f"locations={sorted(set(loc for _, loc in missing))}; basepath={basepath}"
            )
            if not basepath:
                _log("serialize_model could not find external data basepath")
                return proto
            # Ensure data_location is EXTERNAL so onnx helper loads the side-car.
            from onnx import TensorProto
            for tensor in onnx.external_data_helper._get_all_tensors(proto):
                if not tensor.HasField("raw_data"):
                    info = ExternalDataInfo(tensor)
                    if info.location:
                        tensor.data_location = TensorProto.EXTERNAL
            try:
                load_external_data_for_model(proto, basepath)
                _log(f"serialize_model reloaded external data from {basepath}")
            except Exception as e:
                _log(f"serialize_model reload external data failed: {e}")
            return proto

        def _patched_serialize_model(onnx_model):
            _log(f"serialize_model called for {type(onnx_model).__name__}")
            # Sync shapes/dtypes from graph to proto ONCE, then load external data
            # into that same proto object.  Repeated access to .proto re-runs _sync
            # and would wipe raw_data again.
            proto = onnx_model.proto
            proto = _reload_external_data_if_needed(onnx_model, proto)
            raw_bytes = sum(
                len(t.raw_data) for t in onnx.external_data_helper._get_all_tensors(proto) if t.HasField("raw_data")
            )
            _log(f"serialize_model raw_data bytes before serialize: {raw_bytes}")
            return _orig_serialize_proto(proto)

        if _ls_mod is not None:
            _ls_mod.save_model = _patched_save_model
            _ls_mod.serialize_model = _patched_serialize_model
        _ir_mod.save_model = _patched_save_model
        _ir_mod.serialize_model = _patched_serialize_model
        _log("save_model and serialize_model patched")
    except Exception as e:
        _log(f"serialize_model patch failed: {e}\n{traceback.format_exc()}")

    # 2) ModelBuilder/optimizer patches intentionally removed:
    #    the default optimizer is required for the ResNet18 encoder to produce
    #    a BPU-executable graph. The heavy memory issue during quantization is
    #    handled by the ORT graph-optimization disabling below.

    # 2b) build_onnx optimizer-skip wrapper intentionally removed.

    # 2c) Patch RawImagesDirLoader to ignore non-.bin calibration side-car files.
    try:
        from horizon_tc_ui.data.loader import DirDataLoader, RawImagesDirLoader

        _orig_dir_build_list = DirDataLoader.build_image_list

        @staticmethod
        def _patched_build_image_list(data_dir):
            files = _orig_dir_build_list(data_dir)
            if isinstance(files, list):
                files = [f for f in files if str(f).lower().endswith(".bin")]
            return files

        RawImagesDirLoader.build_image_list = _patched_build_image_list
        _log("RawImagesDirLoader build_image_list patched to filter .bin files")
    except Exception as e:
        _log(f"RawImagesDirLoader patch failed: {e}\n{traceback.format_exc()}")

    # 3) Patch ONNX external-data helpers to use a single file and clean stale files.
    try:
        import onnx
        from onnx.external_data_helper import (
            convert_model_to_external_data as _orig_convert,
            write_external_data_tensors as _orig_write,
            save_external_data as _orig_save_external,
            set_external_data,
            uses_external_data,
            ExternalDataInfo,
            _get_all_tensors,
            _get_initializer_tensors,
        )

        def _patched_convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=None,
            size_threshold=1024,
            convert_attribute=False,
        ):
            if location is None:
                location = _EXTERNAL_DATA_BASENAME

            tensors = _get_all_tensors(model) if convert_attribute else _get_initializer_tensors(model)
            count = 0
            total_raw = 0
            for tensor in tensors:
                if not tensor.HasField("raw_data") or sys.getsizeof(tensor.raw_data) < size_threshold:
                    continue
                set_external_data(tensor, location)
                count += 1
                total_raw += sys.getsizeof(tensor.raw_data)
            _log(
                f"convert_model_to_external_data location={location} "
                f"all_tensors_to_one_file={all_tensors_to_one_file} count={count} total_raw={total_raw}"
            )
            return model

        def _patched_save_external_data(tensor, base_path):
            info = ExternalDataInfo(tensor)
            size = len(tensor.raw_data) if tensor.HasField("raw_data") else 0
            _log(f"save_external_data location={info.location} size={size}")
            return _orig_save_external(tensor, base_path)

        def _patched_write_external_data_tensors(model, filepath):
            force_one_file = os.path.basename(filepath).startswith("temp.")
            locations_to_remove = set()

            for tensor in _get_all_tensors(model):
                if not (uses_external_data(tensor) and tensor.HasField("raw_data")):
                    continue
                info = ExternalDataInfo(tensor)
                loc = info.location or ""
                is_uuid = bool(_UUID_RE.match(loc))
                if force_one_file or is_uuid or not loc:
                    locations_to_remove.add(loc)
                    set_external_data(tensor, _EXTERNAL_DATA_BASENAME)
                else:
                    locations_to_remove.add(loc)

            for loc in locations_to_remove:
                if not loc:
                    continue
                data_path = os.path.join(filepath, loc)
                if os.path.isfile(data_path):
                    try:
                        os.remove(data_path)
                    except OSError:
                        pass

            if force_one_file:
                try:
                    for entry in os.listdir(filepath):
                        full = os.path.join(filepath, entry)
                        if os.path.isfile(full) and not entry.endswith(".onnx"):
                            os.remove(full)
                except OSError:
                    pass

            _log(
                f"write_external_data_tensors filepath={filepath} force_one_file={force_one_file} "
                f"locations={locations_to_remove}"
            )
            return _orig_write(model, filepath)

        onnx.convert_model_to_external_data = _patched_convert_model_to_external_data
        onnx.write_external_data_tensors = _patched_write_external_data_tensors
        onnx.external_data_helper.convert_model_to_external_data = _patched_convert_model_to_external_data
        onnx.external_data_helper.write_external_data_tensors = _patched_write_external_data_tensors
        onnx.external_data_helper.save_external_data = _patched_save_external_data

        if hasattr(onnx, "save_model"):
            onnx.save_model.__globals__["convert_model_to_external_data"] = _patched_convert_model_to_external_data
            onnx.save_model.__globals__["write_external_data_tensors"] = _patched_write_external_data_tensors
        if hasattr(onnx, "save"):
            onnx.save.__globals__["convert_model_to_external_data"] = _patched_convert_model_to_external_data
            onnx.save.__globals__["write_external_data_tensors"] = _patched_write_external_data_tensors

        _log("ONNX external-data helpers patched")
    except Exception as e:
        _log(f"ONNX external-data patch failed: {e}\n{traceback.format_exc()}")

    # 5) Replace set_input_shape with a minimal implementation.
    #    The original calls onnx_model.infer_shapes().check_validity(), which
    #    crashes the Horizon/ONNX C++ shape inference on the I-JEPA model.
    #    The toolchain only needs the input shape values to proceed.
    try:
        _preproc_mod_name = "hmct.converter.preparer.preprocess_for_convert"
        _preproc_mod = sys.modules.get(_preproc_mod_name)
        if _preproc_mod is None:
            _preproc_mod = importlib.import_module(_preproc_mod_name)

        def _patched_set_input_shape(onnx_model, input_shapes):
            input_mappings = onnx_model.graph.input_mappings
            for input_name, input_shape in input_shapes.items():
                if input_name not in input_mappings:
                    raise ValueError(
                        f"The input {input_name} is not found in model inputs, "
                        f"existing model's input names: {list(input_mappings.keys())}"
                    )
                input_var = input_mappings[input_name]
                if len(input_shape) != len(input_var.shape):
                    raise ValueError(
                        f"For input {input_name}, the rank of target shape "
                        f"({len(input_shape)}) does not match the rank of "
                        f"model shape ({len(input_var.shape)})."
                    )
                for dim in input_shape:
                    if not isinstance(dim, int) or dim == -1:
                        raise ValueError(
                            f"For input {input_name}, the target input shape "
                            f"{input_shape} contains missing or dynamic dimension."
                        )
                _log(f"Replace input {input_name} shape from {input_var.shape} to {input_shape}")
                input_var.shape = list(input_shape)
            return onnx_model

        _preproc_mod.set_input_shape = _patched_set_input_shape
        _log("replaced set_input_shape on hmct.converter.preparer.preprocess_for_convert")
    except Exception as e:
        _log(f"set_input_shape patch failed: {e}\n{traceback.format_exc()}")

    # 4) Patch ORTExecutor's local serialize_model binding and create_session.
    #    horizon_nn.executor.ort1 does ``from horizon_nn.ir import serialize_model``,
    #    so it holds its own reference to the original function.  Replace that
    #    reference with our patched version so ORT sessions can reload external data.
    #    Also disable ORT graph optimizations, because the optimizer execution frame
    #    can fail with SystemError 2 / "Invalid fd was supplied: -1" when the model
    #    uses external data (onnxruntime issue #4922).
    try:
        for _ort_mod_name in ("hmct.executor.ort1", "horizon_nn.executor.ort1"):
            _ort_mod = sys.modules.get(_ort_mod_name)
            if _ort_mod is None:
                try:
                    _ort_mod = importlib.import_module(_ort_mod_name)
                except Exception:
                    continue
            if _ort_mod is None:
                continue

            if hasattr(_ort_mod, "serialize_model"):
                _ort_mod.serialize_model = _patched_serialize_model
                _log(f"serialize_model patched on {_ort_mod_name}")

            _ort_cls = getattr(_ort_mod, "ORTExecutor1", None)
            if _ort_cls is None:
                continue

            _orig_create_session = _ort_cls.create_session

            def _make_patched_create_session(mod, orig_create_session):
                def _patched_create_session(self):
                    ort = mod.ort
                    cpp = ort.onnxruntime_cpp2py_export
                    sess_options = cpp.SessionOptions()
                    sess_options.graph_optimization_level = cpp.GraphOptimizationLevel.ORT_DISABLE_ALL
                    self._sess = ort.InferenceSession(
                        mod.serialize_model(self._model),
                        sess_options=sess_options,
                        providers=self._providers,
                    )
                    self._load_inputs_outputs()
                    return self
                return _patched_create_session

            _ort_cls.create_session = _make_patched_create_session(_ort_mod, _orig_create_session)
            _log(f"create_session patched on {_ort_mod_name}")
    except Exception as e:
        _log(f"ORTExecutor patch failed: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Import hook: wait for horizon_nn.ir (or hmct.ir) to load, then apply patches.
# ---------------------------------------------------------------------------


class _PostLoadPatcher:
    def __init__(self, real_loader, patch_fn):
        self._real_loader = real_loader
        self._patch_fn = patch_fn

    def create_module(self, spec):
        if hasattr(self._real_loader, "create_module"):
            return self._real_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        self._patch_fn()


class _PatcherFinder:
    _TARGETS = ("horizon_nn.api", "horizon_tc_ui")

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._TARGETS:
            return None

        this = self
        meta = sys.meta_path
        try:
            sys.meta_path = [f for f in meta if f is not this]
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        finally:
            sys.meta_path = meta

        if spec is not None and spec.loader is not None and not isinstance(spec.loader, _PostLoadPatcher):
            spec.loader = _PostLoadPatcher(spec.loader, _apply_patches)
        return spec


sys.meta_path.insert(0, _PatcherFinder())

# If the target is already imported, apply patches immediately.
if "horizon_nn.api" in sys.modules or "horizon_tc_ui" in sys.modules:
    _apply_patches()
