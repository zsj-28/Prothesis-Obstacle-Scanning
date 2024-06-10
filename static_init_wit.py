import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import serial

from helpers import q_to_R, S, hat, norm, R_to_Euler, Euler_to_R, R_pad_to_T
import WitSensor as ws


def ori_comp_filter_noMag(q_prev, gyr, acc, g_n, dt, beta):
    """
    Estimate orientation quaternion q at time t between navigation (n) 
    and IMU frame (b) using high-pass (for w) and low-pass 
    (for acc, mag) complementary filter with one-step gradient descent 
    for the orientation estimate coming from acc and mag information.

    Input: q_prev: orientation quaternion estimate of previous 
                   time step, format: q = [q0, qx, qy, qz]
           gyr: current time gyro measurement, format: [wx, wy, wz]
           acc: current time accelerometer measurement: [ax, ay, az]
           mag: current time magentometer measurement: [mx, my, mz]

           g_n: gravity vector in navigation frame: [gx, gy, gz]
           m_n: normalized earth magnetic field vector in navigation 
               frame: [mx, my, mz], norm(m_n) = 1

           dt: time interval between current sensor readings
               and previous quaternion estimate, dt = t - t_prev 
           beta: step size of normalized gradient descent 
                (filter tuning parameter; reasonable choice is 
                beta = sqrt(3) * sigma, with sigma being the standard 
                deviation of the gyroscope noise; based on standard 
                deviation of integration drift for unit quaternion: 
                sqrt(3) * sigma * dt)

    Output: q: orientation quaternion estimate at time t:
               [q0, qx, qy, qz], norm(q) = 1
    """

    # normalize acceleration and magnetometer vectors
    if norm(acc) != 0:
        acc = acc / norm(acc)
    else:
        acc = np.array([0, 0, 0])
    # mag = mag / norm(mag)

    # compute gradient of cost function V (eq 12b)
    if norm(g_n) != 0:
        g_n = g_n / norm(g_n)
    else:
        g_n = np.array([0, 0, 0])
        
    R_bn = (q_to_R(q_prev)).T
    g_b = R_bn @ g_n
    # print(g_n)
    # print(R_bn)
    # m_b = R_bn @ m_n
    # grad_V = -hat(g_b) @ (acc + g_b) + hat(m_b) @ (mag - m_b)
    grad_V = -hat(g_b) @ (acc + g_b)
    # print(grad_V)

    # estimate angular velocity in body frame (eq 15b)
    if norm(grad_V) != 0:
        w_est = gyr - beta * grad_V / norm(grad_V)
        q = q_prev + dt / 2 * S(q_prev) @ w_est
    else:
        q = q_prev

    return q / norm(q)

def quaternion_mean(quaternions):
    """
    Compute the mean of an array of quaternions.
    Args:
    quaternions (np.array): An array of shape (n, 4) where n is the number of quaternions.

    Returns:
    np.array: The mean quaternion.
    """
    M = quaternions.shape[0]
    A = np.zeros((4, 4))

    for q in quaternions:
        A += np.outer(q, q)
    A /= M

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    mean_quat = eigenvectors[:, np.argmax(eigenvalues)]

    return mean_quat

def init_wit():
    print("\033[H\033[2J")  # clear interpreter console
    baud = int(115200)
    ser = serial.Serial('COM4', baud, timeout=0.5)
    print(ser.is_open)
    
    # Preallocate memory for results
    n_frames = 1000
    q_t = np.zeros((n_frames, 4))  # Quaternion history

    # Initialize attitude
    q_t[0] = np.array([1, 0, 0, 0])

    # INS initialization loop
    tx = 1
    INIT_TIME = 3.0  # [s] INS initialization time
    INCL = 66.77 * np.pi / 180 
    g_n = np.array([0, 0, -9.81])
    euler_t = np.zeros((n_frames, 3))
    time_t = []

    start_time = time.time()
    print("Start at:", datetime.datetime.now().strftime('%H:%M:%S.%f'))
    
    while (time.time() - start_time) <= INIT_TIME and tx < n_frames:
        datahex = ser.read(33)
        a, g = ws.DueData(datahex)
        acc = np.array(a) * 9.81
        gyr = np.array(g)
        dt = 1.0 / 200.0  # 200Hz
        q = ori_comp_filter_noMag(q_t[tx - 1], gyr, acc, g_n, dt, 3.0)
        R = q_to_R(q)
        roll, pitch, yaw = R_to_Euler(R)
        euler_t[tx] = [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        elapsed_time = datetime.datetime.now().strftime('%H:%M:%S.%f')
        time_t.append(elapsed_time)
        print(elapsed_time, acc, gyr, euler_t[tx])
        q_t[tx] = q
        tx += 1
        time.sleep(0.005)

    # Convert time_t to a NumPy array
    time_t = np.array(time_t)

    # Compute the mean quaternion from the later half of the frames
    later_half_quaternions = q_t[tx // 2:tx]
    mean_q = quaternion_mean(later_half_quaternions)

    # Convert mean quaternion to Euler angles
    R = q_to_R(mean_q)
    roll, pitch, yaw = R_to_Euler(R)
    print(f'Mean quaternion: {mean_q}')
    print(f'Roll: {np.degrees(roll)}, Pitch: {np.degrees(pitch)}, Yaw: {np.degrees(yaw)}')
    R = Euler_to_R(0, pitch, 0) #assume the camera is mounted at 0 degrees yaw angle
    T = R_pad_to_T(R)
    print(f'transformation matrix: {T}')
    # Plot Euler angles
    fig_number = 20
    fig = plt.figure(fig_number)
    fig.clear()
    fig.set_size_inches(6, 6)
    plt.plot(time_t, euler_t[1:tx, 0], label="Roll")
    plt.plot(time_t, euler_t[1:tx, 1], label="Pitch")
    plt.plot(time_t, euler_t[1:tx, 2], label="Yaw")
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel("Time (H:M:S.F)")
    plt.ylim(-180, 180)  # Set y-axis limits from -180 to 180
    plt.ylabel("Angle (Degrees)")
    plt.show()
    return T

if __name__ == "__main__":
    init_wit()
