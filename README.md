# Prothesis-Obstacle-Scanning
The code works with the RealSense D435 Camera and WitMotion HWT906 IMU.
It is used for scannign the narrow terrain in front of the lower limb prothesis leg. The RealSense Camera's depth image point cloud will be corrected to the global navigation frame by the IMU and the orientaion complementary filter.

Note: the current code only works for the initialization stage of the prothesis. Further works on converting the point cloud to global frame when walking is needed. A ESKF will be a possible solution.
