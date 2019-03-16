import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
    Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
    '''
    with np.load(file_name) as data:
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
    return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu


def visualize_trajectory_2d(pose,path_name="Unknown",plot_title=None,show_ori=False, show_landmarks=False, landmarks=None):
    '''
    function to visualize the trajectory in 2D
    Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    
    if show_landmarks:
        ax.scatter(landmarks[0,:],landmarks[1,:],s=10, label='landmarks')

    
    if show_ori:
        select_ori_index = list(range(0,n_pose,int(n_pose/50)))
        yaw_list = []
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    if plot_title != None:
        plt.title(plot_title)
        plt.savefig('Plots/{}'.format(plot_title))
    plt.show(block=True)
    return fig, ax



def contstruct_M(K, b):
    """
    Constructs the M matrix for the stereo camera model which transforms from the camera frame to 
    the optical frame given the intrinsic calibration parameters K and b    
    """
    return np.array([[ K[0,0], 0, K[0,2], 0 ],
             [ 0, K[1,1], K[1,2], 0 ],
             [ K[0,0], 0, K[0,2], -b*K[0,0] ],
             [ 0, K[1,1], K[1,2], 0 ]])


def predict_EKF(current_pose, lin_velocity, ang_velocity, tau):
    """
    Prediction step for EKF
    
    Arguments:
    current_pose: current inverse IMU pose
    lin_velocity: current input linear velocity
    ang_velocity: current input angular velocity
    """
    
    u_hat = np.zeros((4,4))
    u_hat[:3,:3] = -tau * np.array([[0, -ang_velocity[2], ang_velocity[1]],
                            [ang_velocity[2], 0, -ang_velocity[0]],
                            [-ang_velocity[1], ang_velocity[0], 0]])
    
    u_hat[:3,-1] = -tau * lin_velocity
    
    ang_velocity_norm = np.linalg.norm(ang_velocity)
    
    # Rodriguez Formula
    exp_u_hat = np.eye(4) + u_hat + ( (1 - np.cos(ang_velocity_norm)) / ang_velocity_norm**2 ) * np.dot(u_hat,u_hat) 
    exp_u_hat += ( (ang_velocity_norm - np.sin(ang_velocity_norm)) / ang_velocity_norm**3 ) * np.dot(np.dot(u_hat,u_hat),u_hat)
    
    next_pose = np.dot(exp_u_hat, current_pose)
    return next_pose
    

def image_to_camera_frame(pixel, M, b):
    """
    Tranforms pixels from stereo camera pixels to coordinates in the camera frame
    """
    
    z = M[0,0]*b/(pixel[0]-pixel[2])
    x = (pixel[0] - M[0,2])*z/M[0,0]
    y = (pixel[1] - M[1,2])*z/M[1,1]
    
    return np.array([[x], [y], [z], [1]])


def camera_to_world_frame(camera_coord, current_inv_pose, cam_T_imu):
    """
    Transforms coordinate in camera frame to world frame
    
    current_inv_pose: current pose of imu with respect to the world frame (inverse of what we are keeping track of, iTw)
    """
    
    imu_T_cam = np.linalg.inv(cam_T_imu)
    world_T_imu = current_inv_pose
    
    return np.dot( np.dot(world_T_imu, imu_T_cam), camera_coord)


def jacobian_of_pi(x, current_pose, D, cam_T_imu, M):
    """
    Evaluates the jacobian of pi at x
    
    x: current optical frame coordinate
    """
    
    A = np.array([[1/x[2], 0, -x[0]/x[2]**2, 0],
                     [0, 1/x[2], -x[1]/x[2]**2, 0],
                     [0, 0, 0, 0],
                     [0, 0, -1/x[2]**2, 1/x[2]]])
    
    return np.dot( M, np.dot(A, np.dot(cam_T_imu, np.dot(current_pose, D))))


def update_EKF(current_mean, current_cov, current_pose, actual_pixel, cam_T_imu, M):
    """
    Update step for EKF
    
    current_mean: current landmark position in world frame
    current_cov: current covariance of landmark position
    current_pose: current transform from world to imu frame
    actual_pixel: current measured pixel values 
    """
    camera_coord = np.dot( np.dot(cam_T_imu, current_pose), current_mean)
    estimated_pixel = np.dot(M, camera_coord/camera_coord[2])
    
    D = np.zeros((4,3))
    D[:3,:] = np.eye(3)
    
    H = jacobian_of_pi(camera_coord, current_pose, D, cam_T_imu, M)
    
    matrix_to_be_inverted = np.dot( np.dot(H, current_cov), H.T)
    # If matrix values are too small, return current mean and cov
    for i in range(matrix_to_be_inverted.shape[0]):
        for j in range(matrix_to_be_inverted.shape[1]):
            if matrix_to_be_inverted[i,j] <= 10**-320:
                return current_mean, current_cov
            
    K = np.linalg.pinv( matrix_to_be_inverted )
    K = np.dot( np.dot( current_cov, H.T), K)
    
    new_mean = current_mean + np.dot( np.dot(D, K), actual_pixel - estimated_pixel)
    
    new_cov = np.eye(3) - np.dot(K,H)
    new_cov = np.dot(new_cov, current_cov)
    
    return new_mean, new_cov    
    
    
