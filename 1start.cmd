@echo off
rem Start IR camera preview / calibration / LED service
rem URL:   http://127.0.0.1:3002/preview
rem LAN:   1start.cmd --lan  -> phone on same WiFi uses printed LAN URL
rem Usage: 1start.cmd [--lan] [--port 3003] [--indices 0 1] [--verbose]
rem Firewall (run once as admin, match current network profile):
rem   netsh advfirewall firewall add rule name="Low-Cost-Mocap 3002" dir=in action=allow protocol=TCP localport=3002 profile=domain remoteip=localsubnet
cd /d "%~dp0Low-Cost-Mocap-main\computer_code"
set PYTHONIOENCODING=utf-8
"C:\Users\Peggy\AppData\Local\anaconda3\envs\lowcost_mocap\python.exe" api\preview_server.py %*
