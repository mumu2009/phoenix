@echo off
setlocal
pushd "%~dp0.."
python tools\run_single_variable_experiments.py --execute --module learning --reverse --skip learning.advAttackRounds,learning.advBenchLimit --state-dir experiments\single_variable_learning_offline_reverse
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
