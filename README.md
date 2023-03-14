# robocentric perception and estimation
ros package used in Robocentric model-based visual servoing for quadrotor flights (2023 T-MECH accepted) 
## related video: [Video](https://www.youtube.com/watch?v=iAODWE3eTCo).

## visual targets detection
```
rosrun robocentric_contorl vision_detect
```

## robocentric state estimation
It uses [IKFOM](https://github.com/hku-mars/IKFoM) as the state estimator.
```
rosrun robocentric_contorl state_estimate
```
