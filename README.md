# Fat-Bots in 2D 

> A gym like environment for fat-robots in a 2-D world.

### Sample World with 4 bots

https://github.com/Nelson-iitp/fatbot/blob/main/fig/vid_ppo_world_x4_1.mp4

<video controls>
  <source src="fig/vid_ppo_world_x4_1.mp4" type="video/mp4">
</video>

### Sample World with 5 bots
<video controls>
  <source src="fig/vid_ppo_world_x5_1.mp4" type="video/mp4">
</video>

### Sample World with 6 bots
<video controls>
  <source src="fig/vid_ppo_world_x6_1.mp4" type="video/mp4">
</video>

# World

![1](fig/01_world.png)

# Movement Model - Velocity and Speed

![2](fig/02_velocity.png)


# Sensor Data

> Sensor can be a device like range-finder, distance-sensor, lidar, radar, sonar, depth-camera

* Sensor in fat-bot simulates 2 types of sensors:
    * X-ray : can detect center and boundary of all bots within its scan-distance (occlusion clearly detectable)
    * D-ray : (distance-ray) a simple range-finder device

![3](fig/03_arcs.png)

# Robot's View

![4](fig/04_robot_view.png)
![5](fig/05_sensor_data.png)

# Visibility Model

![6](fig/06_occlusion.png)

# Sensor Data for Occluded bots

![7](fig/07_occlusion_detect.png)

### Red-Bot: all neighbours fully visible

![8](fig/08_not_occluded.png)

### Blue-Bot: Occlusion Detected

![9](fig/09_occluded.png)

### Green-Bot: Occlusion Detected

![10](fig/10_occluded.png)

## All Fully Visible

![11](fig/11_visible.png)
![12](fig/12_all_visible.png)

# Prebuilt Worlds

![show4](fig/show4.png)
![show6](fig/show6.png)
![world](fig/world.png)
![world2](fig/world2.png)
![world3](fig/world3.png)
![world4](fig/world4.png)
