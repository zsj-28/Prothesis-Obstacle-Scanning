import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import static_init_wit as wit
import csv
import pythoncom
from open3d import *

global point_cloud_data
class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

# Variable to store point cloud data
point_cloud_data = None

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True
    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False
    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True
    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False
    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True
    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False
    if event == cv2.EVENT_MOUSEMOVE:
        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]
        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2
        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)
        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz
    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz
    state.prev_mouse = (x, y)

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)

def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)

def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height
    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p
        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)
        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)

def pointcloud(out, verts, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))
    if state.scale:
        proj *= 0.5**state.decimate
    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm
    out[i[m], j[m]] = color

# Function to get the current point cloud data
def get_point_cloud_data():
    return point_cloud_data

# Function to crop point cloud to the specified ROI
def crop_point_cloud(verts, depth_threshold=3.0, width=0.4):
    mask = (verts[:, 2] < depth_threshold) & (np.abs(verts[:, 0]) < width / 2)
    return verts[mask]

out = np.empty((h, w, 3), dtype=np.uint8)
start_loop_time = time.time()
while time.time() - start_loop_time < 2:
    if not state.paused:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        depth_image = np.asanyarray(depth_frame.get_data())

        points = pc.calculate(depth_frame)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)

        # Crop point cloud to the specified ROI
        verts = crop_point_cloud(verts)

        # Measure time to update the point cloud data
        store_time = time.time()
        point_cloud_data = verts  # Store point cloud data
        
    now = time.time()
    out.fill(0)
    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)
    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, (255, 255, 255))  # Render in white
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, (255, 255, 255))  # Render in white
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)
    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)
    dt = time.time() - now
    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
    cv2.imshow(state.WIN_NAME, out)

    key = cv2.waitKey(1)
    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Save the point cloud data to a CSV file
def save_point_cloud_to_csv(point_cloud_data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header (optional)
        writer.writerow(['X', 'Y', 'Z'])
        # Write the point cloud data
        for point in point_cloud_data:
            writer.writerow(point)

def transformation(point_cloud, transformation_matrix):
    """
    Apply a transformation matrix to a point cloud.
    
    Parameters:
    point_cloud (numpy.ndarray): Nx3 array of 3D points.
    transformation_matrix (numpy.ndarray): 4x4 transformation matrix.
    
    Returns:
    numpy.ndarray: Transformed point cloud.
    """
    transformed_points = []
    
    for point in point_cloud:
        # Convert the point to homogeneous coordinates
        # print ("Point: ", point)
        homogeneous_point = np.append(point, 1)  # Convert to 1x4
        
        # Apply the transformation matrix
        transformed_homogeneous_point = np.dot(transformation_matrix, homogeneous_point)  # Result is 1x4
        
        # Convert back to 3D coordinates
        transformed_point = transformed_homogeneous_point[:3] 
        # print ("Transformed Point: ", transformed_point)
        # Add the transformed point to the list
        transformed_points.append(transformed_point)
    
    return np.array(transformed_points)


cv2.destroyAllWindows()
pythoncom.CoInitialize()
try:
    T = wit.init_wit()
    print("T matrix: ", T)
    transformation_camera_to_robot = np.array([
    [0, 0, 1, 0],  
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
    ])
    # T = np.dot(transformation_camera_to_robot, T)
    print("Transformed T matrix: ", T)
    point_cloud_data = get_point_cloud_data()
    save_point_cloud_to_csv(point_cloud_data, 'point_cloud_data_1.csv')
    # Apply transformation and other operations
    transformed_point_cloud_data = transformation(point_cloud_data, transformation_camera_to_robot)
    transformed_point_cloud_data = transformation(transformed_point_cloud_data, T)  
    print("Transformation Matrix Applied")
    # print(point_cloud_data)
    # Save the transformed point cloud data to a CSV file
    save_point_cloud_to_csv(transformed_point_cloud_data, 'transformed_point_cloud_data_1.csv')
    print("Point Cloud Data Saved to CSV")
        
finally:
    pythoncom.CoUninitialize