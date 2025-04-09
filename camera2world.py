import cv2
import numpy as np
from wpimath.geometry import *
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import convertor,time
import torch  # Add this import for tensor operations

class CoralOrientationSolver:
    def __init__(self, camera_matrix, distortion, height, pitch, coral_radius, coral_length, offset_angle=0, enable_visualization=True):
        # 初始化相机和物理参数
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.height = height
        self.coral_radius = coral_radius
        self.coral_length = coral_length
        self.pitch = pitch
        self.offset_angle = offset_angle
        self.enable_visualization = enable_visualization  # Add visualization toggle

        # 预计算珊瑚几何参数
        self.coral_width = coral_radius * 2
        self.coral_diagonal_length, self.coral_diagonal_angle = self._get_diagonal_info(
            self.coral_width, self.coral_length)

        # 初始化可视化
        if self.enable_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self._init_visualization()

    def _init_visualization(self):
        """初始化可视化组件"""
        self.ax.set_title('Orientation Solutions')
        self.ax.set_xlim(-0.3, 0.3)
        self.ax.set_ylim(-0.3, 0.3)
        self.ax.grid(True)
        self.ax.set_aspect('equal')

        # 创建可视化线条
        self.line_ccw = self.ax.add_line(
            Line2D([], [], color='blue', lw=2, label='Counter Clockwise'))
        self.line_cw = self.ax.add_line(
            Line2D([], [], color='red', lw=2, label='Clockwise'))
        self.ax.legend()

    @staticmethod
    def _get_diagonal_info(width, height):
        """计算对角线长度和夹角（保持原始算法不变）"""
        diagonal = math.hypot(width, height)
        angle_rad = math.atan(width / height) * 2  # 注意原始算法中的*2
        return diagonal, angle_rad

    @staticmethod
    def _get_angle(m1, m2):
        """计算两条直线的夹角"""
        denominator = 1 + m1 * m2
        if denominator == 0:
            return math.pi / 2
        tan_theta = abs((m1 - m2) / denominator)
        return math.atan(tan_theta)

    def _clamp(self, value, min_val, max_val):
        """数值范围限制"""
        if value < min_val or value > max_val:
            print("Domain error: value clamped")
        return max(min_val, min(value, max_val))

    def _update_visualization(self, angle_ccw_deg, angle_cw_deg):
        """更新可视化显示"""
        if not self.enable_visualization:  # Skip if visualization is disabled
            return
        # 转换为极坐标
        theta_ccw = np.deg2rad(angle_ccw_deg)
        theta_cw = np.deg2rad(angle_cw_deg)
        length = 0.4

        # 更新逆时针解
        x_ccw = length * np.cos(theta_ccw)
        y_ccw = length * np.sin(theta_ccw)
        self.line_ccw.set_data([-x_ccw/2, x_ccw/2], [-y_ccw/2, y_ccw/2])

        # 更新顺时针解
        x_cw = length * np.cos(theta_cw)
        y_cw = length * np.sin(theta_cw)
        self.line_cw.set_data([-x_cw/2, x_cw/2], [-y_cw/2, y_cw/2])

        # 重绘图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def solve(self, u_center, v_center, box_length):
        """主求解函数"""
        # 计算中心点坐标（使用调整后的高度）
        center_height = self.height - self.coral_radius
        cam_center = Camera2World(self.camera_matrix, self.distortion, 
                                center_height, self.pitch)
        center_point = cam_center.getObjectToCameraPose(u_center, v_center)

        # 计算边界点坐标（使用原始高度）
        cam_default = Camera2World(self.camera_matrix, self.distortion,
                                 self.height, self.pitch)
        left_point = cam_default.getObjectToCameraPose(u_center - box_length/2, v_center)
        right_point = cam_default.getObjectToCameraPose(u_center + box_length/2, v_center)

        # 计算关键斜率
        left_slope = left_point.Y() / left_point.X()
        right_slope = right_point.Y() / right_point.X()
        center_slope = center_point.Y() / center_point.X()

        # 计算基准角度
        base_angle = self._get_angle(center_slope, 0)

        def calculate_angle(main_slope, edge_slope, is_second_solution):
            """角度计算核心逻辑"""
            bound_angle = self._get_angle(main_slope, edge_slope)
            distance = math.hypot(center_point.X(), center_point.Y())

            # 正弦定理计算
            sin_ratio = math.sin(bound_angle) / (self.coral_diagonal_length/2)
            sin_value = self._clamp(distance * sin_ratio, -1.0, 1.0)
            upper_angle = math.asin(sin_value)

            # 处理第二解
            if is_second_solution:
                upper_angle = math.pi - upper_angle + self.coral_diagonal_angle/2
            else:
                upper_angle -= self.coral_diagonal_angle/2

            return upper_angle + bound_angle + base_angle

        # 计算两个解
        angle_ccw = calculate_angle(center_slope, right_slope, False)
        angle_cw = calculate_angle(center_slope, left_slope, True)

        # 更新可视化 (convert radians to degrees for visualization)
        self._update_visualization(np.rad2deg(angle_ccw) + 90, np.rad2deg(angle_cw) + 90)

        # 打印输出仍然是度数
        print(f"逆时针解：{np.rad2deg(angle_ccw):.2f}°，顺时针解：{np.rad2deg(angle_cw):.2f}°，中心点坐标：({center_point.X():.2f}, {center_point.Y():.2f})")

        # 返回角度（弧度）和中心点坐标
        return angle_ccw, angle_cw, center_point.X(), center_point.Y()

# 保留原始Camera2World类
class Camera2World:
    def __init__(self, camera_matrix, distortion, height, pitch):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.height = height
        self.pitch = pitch

    def getAngle(self, x, y):
        image_points = np.array([[x, y]], dtype=np.float32)
        normalized_points = cv2.undistortPoints(image_points, self.camera_matrix, self.distortion)
        angle_x = -np.arctan(normalized_points[0][0][0])
        angle_y = np.arctan(normalized_points[0][0][1])
        return angle_x, angle_y

    def normalizeAngle(self, camera_rot, obj_rot):
        return obj_rot.rotateBy(convertor.inverseRotation(camera_rot))

    def getObjectToCameraPose(self, u, v):
        angle_x, angle_y = self.getAngle(u, v)
        ty = self.normalizeAngle(Rotation3d(0, self.pitch, 0), 
                               Rotation3d(0, angle_y, 0)).Y()
        x = self.height / np.tan(ty)
        y = x * np.tan(angle_x)
        return Translation2d(x, y)

class coralPositionEstimator:
    def __init__(self, cameraMatrix, distortionCoeffs, cameraPose: Pose3d) -> None:
        self.cameraMatrix = cameraMatrix
        self.distortionCoeffs = distortionCoeffs
        self.cameraPose = cameraPose
        self.solver = CoralOrientationSolver(
            cameraMatrix, distortionCoeffs, cameraPose.translation().z, cameraPose.rotation().y, coral_radius=0.055, coral_length=0.3, enable_visualization=False
        )
        self.target_id = 1  # Replace with the specific ID to filter

    def normalizeAngle2D(self, cameraYaw: Rotation2d, objectAngleToCamera: Rotation2d):
        """Convert object-to-camera angle to object-to-robot angle in 2D."""
        return objectAngleToCamera.rotateBy(cameraYaw)

    def transformCoordinates2D(self, cameraPose: Pose2d, objectTranslation: Translation2d):
        """Transform object coordinates from camera frame to robot frame in 2D."""
        transform = Transform2d(cameraPose.translation(), cameraPose.rotation())
        return transform.transformBy(Transform2d(objectTranslation, Rotation2d(0))).translation()

    def __call__(self, ids, boxes):
        results = []
        for obj_id, box in zip(ids, boxes):
            if obj_id != self.target_id:
                continue  # Skip if ID does not match the target

            u_center = box[0] + box[2] / 2  # x + width / 2
            v_center = box[1] + box[3] / 2  # y + height / 2
            box_length = max(box[2], box[3])  # Use the larger dimension as box length

            # Solve orientation
            ccw, cw, x, y = self.solver.solve(u_center, v_center, box_length)

            # Convert angles to robot's reference frame (2D rotation)
            camera_yaw = self.cameraPose.rotation().toRotation2d()
            ccw_robot = self.normalizeAngle2D(camera_yaw, Rotation2d(ccw)).radians()
            cw_robot = self.normalizeAngle2D(camera_yaw, Rotation2d(cw)).radians()

            # Convert coordinates to robot's reference frame (2D transformation)
            camera_pose_2d = self.cameraPose.toPose2d()
            object_translation = Translation2d(x, y)
            robot_translation = self.transformCoordinates2D(camera_pose_2d, object_translation)

            results.append([robot_translation.X(), robot_translation.Y(), ccw_robot, cw_robot])
            print(f"ID: {obj_id}, Robot Coordinates: ({robot_translation.X():.2f}, {robot_translation.Y():.2f}), "
                  f"CCW Angle: {np.rad2deg(ccw_robot):.2f}°, CW Angle: {np.rad2deg(cw_robot):.2f}°")
        while len(results) < 10:
            results.append([-9999, -9999, -9999, -9999])
        
        # Convert results to a tensor
        result_tensor = torch.tensor(results, dtype=torch.float32)

        return result_tensor

# 示例用法
if __name__ == "__main__":
    # 初始化参数
    camera_matrix = np.array([[905.32671946, 0, 679.6204086],
                             [0, 906.14946047, 331.96782248],
                             [0, 0, 1]])
    distortion = np.array([0.02907126, -0.03349167, 0.00055539, -0.00029301, -0.02025189])
    camera_pose = None  # Replace with actual camera pose if needed

    # 创建估计器实例
    estimator = coralPositionEstimator(camera_matrix, distortion, camera_pose)

    # 输入检测数据
    ids = [42, 7, 42]  # Example IDs
    boxes = [
        [306, 531, 219, 219],  # Example box for ID 42
        [100, 200, 50, 50],    # Example box for ID 7
        [135, 534, 176, 176]   # Example box for ID 42
    ]

    # 调用估计器
    results = estimator(ids, boxes)

    # 保持窗口打开
    input("按Enter键退出...")