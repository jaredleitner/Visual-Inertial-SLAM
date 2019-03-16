import numpy as np
from utils import *


if __name__ == '__main__':
    filename = "./data/0020.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    
    tau = t[0,1] - t[0,0]

    # (a) IMU Localization via EKF Prediction
    
    # Construct M
    M = contstruct_M(K, b)
    
    poses = np.zeros((4,4,linear_velocity.shape[1]))
    inv_poses = np.zeros((4,4,linear_velocity.shape[1]))
    
    current_pose = np.eye(4)
    for i in range(linear_velocity.shape[1]):
        next_pose = predict_EKF(current_pose, linear_velocity[:,i], rotational_velocity[:,i], tau)
        poses[:,:,i] = next_pose
        inv_poses[:,:,i] = np.linalg.inv(next_pose)
        current_pose = next_pose
        
    visualize_trajectory_2d(inv_poses, path_name='Trajectory',plot_title='Trajectory for Test Dataset', show_ori=True)

    # (b) Landmark Mapping via EKF Update
    
    # Determine where each landmark is initially located    
    initial_landmarks = np.ones((4, features.shape[1]))
    
    for i in range(features.shape[2]):
        for j in range(features.shape[1]):
            current_pixel = features[:,j,i]
            if current_pixel[0] == -1:
                continue
            else:
                camera_frame = image_to_camera_frame(current_pixel, M, b)
                world_frame = camera_to_world_frame(camera_frame, inv_poses[:,:,i], cam_T_imu)
                initial_landmarks[:3,j] = world_frame[:3].reshape(3,)

    visualize_trajectory_2d(inv_poses,path_name='Trajectory', plot_title='Initial Landmarks for Test Dataset',
                              show_ori=True, show_landmarks=True, landmarks=initial_landmarks)
    
    # Determine final locations of each landmark
    
    final_landmarks = np.zeros((4, features.shape[1]))

    for i in range(features.shape[1]):

        for j in range(features.shape[2]):
            if features[0,i,j] == -1:
                continue
            else:
                break

        current_mean = image_to_camera_frame(features[:,i,j], M, b)
        current_mean = camera_to_world_frame(current_mean, inv_poses[:,:,j], cam_T_imu)

        current_cov = np.eye(3)

        for k in range(j, features.shape[2]):
            if features[0,i,k] == -1:            # When the landmark is not observed
                continue
            else:
                current_mean, current_cov = update_EKF(current_mean.reshape(4,), current_cov, poses[:,:,j], features[:,i,j], cam_T_imu, M)

        final_landmarks[:,i] = current_mean

    visualize_trajectory_2d(inv_poses,path_name='Trajectory', plot_title='Final Landmarks for Test Dataset',
                              show_ori=True, show_landmarks=True, landmarks=final_landmarks)

    
    
    
    
    
    
    
    
    
    

	