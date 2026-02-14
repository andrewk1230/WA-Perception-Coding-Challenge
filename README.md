smooth_trajectory : It removes the high-frequency "shaking" while preserving the actual direction and velocity of the movement.

process_data: Handles data extraction and coordinate math. It iterates through the .npz files, which starts with converting 0 and inf values to NaN. Uses CSV bounding boxes to crop a patch of the 3D data and find distances. This also calculates both world frame and ego frame which centers the traffic light and the car, respectively. 

save_world_animation: In this view, the camera follows the action from above and the traffic light is at origin (0,0). You can also see the track for ego path and cart path.

save_ego_view_gif: The car(camera perspective) is set as the cennter and never moves. The traffic light and gold cart in this case moves towards & away from the car. 