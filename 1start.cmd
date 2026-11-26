@echo off
rem Start IR camera preview / calibration / LED service
rem URL:   http://127.0.0.1:3002/preview
rem Usage: 1start.cmd [--port 3003] [--indices 0 1] [--verbose]
cd /d "%~dp0Low-Cost-Mocap-main\computer_code"
set PYTHONIOENCODING=utf-8
"C:\Users\Peggy\AppData\Local\anaconda3\envs\lowcost_mocap\python.exe" api\preview_server.py %*
