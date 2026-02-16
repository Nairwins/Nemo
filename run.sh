#!/bin/
source "[YOUR VIRTUALENV PATH]"/bin/activate

echo "Starting Nemo..."
python nemo.py &

sleep 2

echo "Starting Main..."
python main.py &

wait
