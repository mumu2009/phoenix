@echo off
cd /d d:\_phoenix\_079\v6.0Alixander\phoenix
python tools\remote_slice_pipeline_v4.py --corpus arxiv > runtime_store\_v4_pipe_arxiv.log 2>&1
python tools\remote_slice_pipeline_v4.py --corpus wikipedia2023 >> runtime_store\_v4_pipe_arxiv.log 2>&1
