from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid
import isaacsim.core.utils.numpy.rotations as rot_utils
from multiprocessing import shared_memory
import numpy as np
import cv2

# Open stage
usd_path = "/home/nairs/Desktop/Projects/Nvidia/Nemo.usd"
omni.usd.get_context().open_stage(usd_path)
my_world = World(stage_units_in_meters=1.0)

# Write metadata
W, H, C = 1024, 768, 3
META_FILE = "scripts/Apdata.txt"
shm = shared_memory.SharedMemory(create=True, size=H * W * C)
buffer = np.ndarray((H, W, C), dtype=np.uint8, buffer=shm.buf)

with open(META_FILE, "w") as f:
    f.write(f"{shm.name}\n{H}\n{W}\n{C}\nuint8\n")



# Setup
camera = Camera(
    prim_path="/World/carter_v1/chassis_link/camera_mount/carter_camera_first_person",
    frequency=20,
    resolution=(W, H),
)


camera.initialize()
my_world.reset()


camera.add_motion_vectors_to_frame()
i = 0

print("==================================================================================================")
while simulation_app.is_running():
    my_world.step(render=True)
    camera.get_current_frame()
    print(i)


    if i == 100:
        i -= 1
        rgba = camera.get_rgba()
        print(rgba.shape)
        
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        buffer[:] = bgr

    
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

    i += 1


shm.unlink()
cv2.destroyAllWindows()
simulation_app.close()
