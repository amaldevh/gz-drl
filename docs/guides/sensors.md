# Read sensors and record a simulation

<span id="cameras-lidar-contacts-and-recording"></span>

`DRLServer` exposes camera and LiDAR frames, callbacks, contact data, and camera
recording. Use sensor access when observations or diagnostics require more
than vehicle state. The world must define the relevant sensors, and rendered
sensors must be enabled when constructing the server.

## Discover and read frames

After initialization, query the sensor names from the loaded world:

```python
import gzdrl

server = gzdrl.DRLServer(
    "sensor-demo",
    str(gzdrl.get_sdf_path("world_simple.sdf")),
    ["quadrotor"],
    True,
)
server.run_N(10)
print(server.camera_sensor_names())
print(server.lidar_sensor_names())
```

The methods return the names defined by the SDF; an empty list means no
corresponding sensor was discovered. Pass a discovered name to
`get_sensor_image(name)` or `get_sensor_gpu_lidar(name)` to read its current
frame.

## Receive callbacks

`bind_img_cb(callback, name)` invokes a Python callback with an image array.
`bind_lidar_cb(callback, name)` passes a LiDAR array, or `None` when data is
unavailable. For example, after selecting a camera name:

```python
camera_names = server.camera_sensor_names()
if camera_names:
    def report_image(image):
        print("Image shape", image.shape)

    server.bind_img_cb(report_image, camera_names[0])
    server.run_N(100)
```

Python callbacks acquire the GIL. Keep them short, and copy LiDAR data within
the callback if it must outlive the callback's view of the sensor buffer.
Callbacks may run alongside application code; keep simulation stepping in one
thread.

## Contacts and recording

Use `request_contact_data(model_name)` and `get_contacts(...)` for contact
information. Camera recording uses `start_camera_recording(...)`,
`update_camera_recording_pose(...)`, and `stop_camera_recording(...)` to create,
move, and stop a recording camera. See {doc}`/reference/python-api` for argument
names and the installed method docstrings for overload types and defaults.

## Timing and reset limitations

Gazebo's rendering and sensor pipelines have their own update schedules.
Measure frame availability and latency for your world; the paper's
reproducibility results cover state observations, not rendered sensor output.

```{warning}
Use `reset_pos()` for pose changes when sensors are enabled. A full
`respawn_model()` can invalidate rendering resources and cause a crash.
Multiple depth cameras across environments in one process are also a known
limitation of the current sensor path.
```
