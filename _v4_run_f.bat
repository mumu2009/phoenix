@echo off
cd /d d:\_phoenix\_079\v6.0Alixander\phoenix
python tools\remote_slice_pipeline_v4.py --corpus cnndailymail --start 3 --slices 3 > runtime_store\_v4_pipe_f.log 2>&1
