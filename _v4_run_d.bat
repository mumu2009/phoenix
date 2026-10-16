@echo off
cd /d d:\_phoenix\_079\v6.0Alixander\phoenix
python tools\remote_slice_pipeline_v4.py --corpus wikipedia2023 --start 0 --slices 6 > runtime_store\_v4_pipe_d.log 2>&1
