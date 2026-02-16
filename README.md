# Project Nemo

Nemo is a future plugin for Omniverse that integrates a Vision-Language Model (VLM) with the NVIDIA Isaac Sim robotics simulator, creating an interactive digital companion named Jarvis. You can control the simulated robot using natural language commands (text or speech) to perform tasks like navigation, object tracking.

## Core Features

-   **Natural Language Control**: Interact with the robot using text and voice commands.

-   **VLM Integration**: Powered by an NVIDIA Nemotron VLM for understanding and responding to complex queries involving both text and images.

-   **Real-time Vision**: Utilizes a Vision model for real-time object detection, enabling target following capabilities.

-   **Robotics Simulation**: Built on top of NVIDIA Isaac Sim for a realistic simulation of the robot and its environment.

-   **Interactive GUI**: A user-friendly chat interface built with PyQt6 allows for seamless interaction with Jarvis.

-   **Drag & Drop**: drag any external file insde the chat buble either videos or documenets, Jarvis will be able to analyze the files and answer you.

## How It Works

The project is composed of several key components that work together:

-   **`nemo.py`**: The NVIDIA Isaac Sim application. It launches the simulation, loads the robot and environment, and streams the robot's first-person camera view to the main application using shared memory.

-   **`main.py`**: The central orchestrator. It launches the GUI, manages the connection to the shared memory stream from Isaac Sim, and processes user input.

-   **`config.txt`**: Defines the persona, capabilities, and command syntax for the "Jarvis" AI. This is used as the system prompt for the VLM.

-   **`config.yaml`**: Contains configuration for the NVIDIA API endpoint, model parameters, and your API key.

## Installation and Setup

1.  **Prerequisites**:
    -   An NVIDIA GPU with **CUDA** is required.
    -   You must have **NVIDIA Isaac Sim** installed. This project is built to work with it.
    -   You need an NVIDIA API key to use the Nemotron model. you can aquire one for free here : https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/nairwins/nemo.git
    cd nemo
    ```

3.  **Set up a Python virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install dependencies**:
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```

5.  **Configuration**:
    -   Open `config.yaml` and replace `[YOUR API KEY]` with your NVIDIA API key.
    
    -   Open the `run.sh` (for Linux/macOS) or `run.bat` (for Windows) script and replace `[YOUR VIRTUALENV PATH]` with the absolute path to your virtual environment's directory. For example, `/path/to/nemo/venv`.

## Usage

To run the application, execute the appropriate script for your operating system. This will launch both the Isaac Sim environment and the main control interface.

**On Linux/macOS:**
```bash
./run.sh
```

**On Windows:**
```bat
run.bat
```

Once running, two main windows will appear:
1.  **Isaac Sim**: The 3D simulation environment.
2.  **Jarvis Interface**: The chat window where you can interact with the robot.

You can type messages or use the microphone button for speech-to-text commands.