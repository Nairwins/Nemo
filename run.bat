@echo off

echo Activating virtual environment...
call "[YOUR VIRTUALENV PATH]"\bin\activate

echo Starting Nemo...
start "Nemo" cmd /k python main.py

timeout /t 2 > nul

echo Starting Main...
start "Main" cmd /k python nemo.py

pause
