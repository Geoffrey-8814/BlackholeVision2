import cv2
import numpy as np
from wpimath.geometry import *
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import convertor,time

class CoralOrientationSolver:
    def __init__(self, camera_matrix, distortion, height, coral_radius, coral_length, pitch, offset_angle=0):
        # 初始化相机和物理参数
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.height = height
        self.coral_radius = coral_radius
        self.coral_length = coral_length
        self.pitch = pitch
        self.offset_angle = offset_angle

        # 预计算珊瑚几何参数
        self.coral_width = coral_radius * 2
        self.coral_diagonal_length, self.coral_diagonal_angle = self._get_diagonal_info(
            self.coral_width, self.coral_length)

        # 初始化可视化
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

        # 转换为角度并应用偏移
        final_ccw = np.rad2deg(angle_ccw) + self.offset_angle
        final_cw = np.rad2deg(angle_cw) + self.offset_angle

        # 更新可视化
        self._update_visualization(final_ccw+90, final_cw+90)

        return final_ccw, final_cw

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

# 示例用法
if __name__ == "__main__":
    # 初始化参数
    camera_matrix = np.array([[905.32671946, 0, 679.6204086],
                             [0, 906.14946047, 331.96782248],
                             [0, 0, 1]])
    distortion = np.array([0.02907126, -0.03349167, 0.00055539, -0.00029301, -0.02025189])
    height = 0.383
    coral_radius = 0.055
    coral_length = 0.3
    pitch = np.deg2rad(0)

    # 创建求解器实例
    solver = CoralOrientationSolver(camera_matrix, distortion, height,
                                  coral_radius, coral_length, pitch)

    # 输入检测数据
    u_center, v_center, box_length = 306.3561, 531.6138, 219.1818

    # 求解并可视化
    ccw, cw = solver.solve(u_center, v_center, box_length)
    print(f"逆时针解：{ccw:.2f}°，顺时针解：{cw:.2f}°")
    
    
    time.sleep(5)
    u_center, v_center, box_length = 135.0338, 534.0199, 175.6627

    # 求解并可视化
    ccw, cw = solver.solve(u_center, v_center, box_length)



    # 保持窗口打开
    input("按Enter键退出...")