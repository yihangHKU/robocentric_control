# robocentric perception and estimation
ros package used in Robocentric model-based visual servoing for quadrotor flights (2023 T-MECH accepted) 
## related video: [Video](https://www.youtube.com/watch?v=iAODWE3eTCo).

## visual targets detection
Change the value ```blob_target_num ``` in ```vision_detect.cpp``` to switch the mode.
1. Set ```blob_taget_num = 1``` when only one visual target is utilized for nagvigation. 
2. Set ```blob_taget_num = 2``` when two visual targets are utilized for nagvigation. 
```
rosrun robocentric_contorl vision_detect
```

## robocentric state estimation
It uses [IKFOM](https://github.com/hku-mars/IKFoM) as the state estimator.
```
rosrun robocentric_contorl state_estimate
```
