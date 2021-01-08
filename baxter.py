import time
import math
import random
import numpy as np
import pybullet as pb
import pybullet_data as pb_data
import pybullet_utils.bullet_client as bullet_client
import cv2 as cv
from getpositions import *


class BaxterEnvironment:

    def __init__(self):
        self.physics_client = bullet_client.BulletClient(pb.GUI)
        self.ik_back = bullet_client.BulletClient(pb.DIRECT)
        self.object_list = {}
        self.physics_client.setAdditionalSearchPath(pb_data.getDataPath())
        self.physics_client.loadURDF("plane.urdf")
        self.physics_client.setGravity(0, 0, -9.81)

        self.ik_back.setAdditionalSearchPath(pb_data.getDataPath())
        self.ik_back.loadURDF("plane.urdf")
        self.ik_back.setGravity(0, 0, -9.81)

        self.end_time = 0
        self.start_time = 0
        self.time_consumed = 0.0

        self.viewMatrix = self.physics_client.computeViewMatrix(
            cameraEyePosition=[0.5, 0.35, 0.98],
            cameraTargetPosition=[0.5, 0.35, .6],
            cameraUpVector=[0, 1, 0])

        self.projectionMatrix = self.physics_client.computeProjectionMatrixFOV(
            fov=45.0,  # degrees
            aspect=1.0,
            nearVal=0.1,
            farVal=1.5)

    def get_image(self):
        width, height, rgbImg, depthImg, segImg = self.physics_client.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)

        return rgbImg

    def prepare_table(self):
        self.physics_client.setAdditionalSearchPath(pb_data.getDataPath())
        self.physics_client.setAdditionalSearchPath("/Objects/")

        self.physics_client.loadURDF("table/table.urdf", (0.5, 0.35, 0))
        self.physics_client.setAdditionalSearchPath("/Objects/")
        pb.addUserDebugLine([0.37, 0.17, 0.65], [0.37, 0.47, 0.65], lineColorRGB=[1, 0, 1], lineWidth=1)
        pb.addUserDebugLine([0.63, 0.17, 0.65], [0.63, 0.47, 0.65], lineColorRGB=[1, 0, 1], lineWidth=1)
        pb.addUserDebugLine([0.37, 0.17, 0.65], [0.63, 0.17, 0.65], lineColorRGB=[1, 0, 1], lineWidth=1)
        pb.addUserDebugLine([0.37, 0.47, 0.65], [0.63, 0.47, 0.65], lineColorRGB=[1, 0, 1], lineWidth=1)
        pb.addUserDebugText("Spawn Area", [0.5, 0.32, 0.62], textColorRGB=[1, 0, 1])

        self.physics_client.setGravity(0, 0, -9.81)
        self.physics_client.loadURDF("basket_cube_green.urdf", (0.14, 0.35, 0.0), globalScaling=2, useFixedBase=1)
        self.physics_client.loadURDF("basket_cube_red.urdf", (0.86, 0.35, 0.0), globalScaling=2, useFixedBase=1)

    def set_baxter(self, baxter_gui, baxter_back):
        self.baxter_gui = baxter_gui
        self.baxter_back = baxter_back

    def transform(self, x, y):
        world_x = (0.1 * x + 30.8) / 84
        world_y = (40.5 - 0.1 * y) / 84

        return world_x, world_y

    def detect_object(self):
        #  detect object by using baxter_back

        rgbImg = self.get_image()
        hsv = cv.cvtColor(rgbImg, cv.COLOR_BGR2HSV)

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        green_mask = cv.inRange(hsv, lower_green, upper_green)

        lower_red = np.array([110, 50, 50])
        upper_red = np.array([130, 255, 255])
        red_mask = cv.inRange(hsv, lower_red, upper_red)
        object_color = "R"
        countGreen = cv.countNonZero(green_mask)
        if countGreen > 0:
            object_color = "G"
            res = cv.bitwise_and(rgbImg, rgbImg, mask=green_mask)
            l, r, t, b = getpositions(res)
            x = (l + r) / 2.
            y = (t + b) / 2.

            return self.transform(x, y), object_color

        countRed = cv.countNonZero(red_mask)
        if countRed > 0:
            object_color = "R"
            res = cv.bitwise_and(rgbImg, rgbImg, mask=red_mask)
            l, r, t, b = getpositions(res)
            x = (l + r) / 2.
            y = (t + b) / 2.
            return self.transform(x, y), object_color

        # not necessary but in any case
        res = cv.bitwise_and(rgbImg, rgbImg, mask=red_mask)
        l, r, t, b = getpositions(res)
        x = (l + r) / 2.
        y = (t + b) / 2.
        return self.transform(x, y), object_color

    def spawn_object(self):
        x = random.uniform(0.45, 0.6)
        y = random.uniform(0.25, 0.4)
        self.physics_client.setAdditionalSearchPath("/Objects/")
        color = "R"
        if random.randint(0, 1) == 0:
            color = "R"
            # print("Original\t R", x,y)
            self.cube = self.physics_client.loadURDF("cube_red.urdf", (x, y, 0.645), globalScaling=0.42)
        else:
            color = "G"
            # print("Original\t G", x,y)
            self.cube = self.physics_client.loadURDF("cube_green.urdf", (x, y, 0.645), globalScaling=0.42)

        self.run_simulation(5)

        return (x, y), color

    def set_home_position(self):
        self.baxter_gui.set_home_left()
        self.baxter_back.set_home_left()
        self.run_simulation(10)

    def grasp_object(self, object_pos):

        target_pos1 = [object_pos[0], object_pos[1], 0.9]
        joint_pos = self.baxter_back.accurateCalculateInverseKinematics(target_pos1, [0, 1, 0, 0],
                                                                        self.baxter_back.left_end_effector_id,
                                                                        self.baxter_gui.get_left_arm_state())
        self.baxter_gui.set_for_ik(joint_pos)
        self.run_simulation(30)

        self.baxter_gui.open_gripper()
        self.run_simulation(15)

        target_pos2 = [object_pos[0], object_pos[1], 0.64]
        joint_pos = self.baxter_back.accurateCalculateInverseKinematics(target_pos2, [0, 1, 0, 0],
                                                                        self.baxter_back.left_end_effector_id,
                                                                        self.baxter_gui.get_left_arm_state())
        self.baxter_gui.set_for_ik(joint_pos)
        self.run_simulation(30)

        self.baxter_gui.close_gripper()
        self.run_simulation(20)
        return target_pos2

    def move_into_basket(self, gripper_pos, object_color):

        if object_color == 'G':
            basket_pos = [0.14, 0.35]
        else:
            basket_pos = [0.8, 0.35]

        # ********** LIFTING THE GRASPED OBJECT ********** #

        for i in range(2):
            gripper_pos[2] = (gripper_pos[2] + 0.13)
            joint_pos = self.baxter_back.accurateCalculateInverseKinematics(gripper_pos, [0, 1, 0, 0],
                                                                            self.baxter_back.left_end_effector_id,
                                                                            self.baxter_gui.get_left_arm_state())
            self.baxter_gui.set_for_ik(joint_pos)
            self.run_simulation(30)

        # ************************************************ #

        # ******************** MOVING ON X AXIS ******************** #

        quantization = int(math.ceil(abs(basket_pos[1] - gripper_pos[1]) / 0.1))
        mov_on_y = (basket_pos[1] - gripper_pos[1]) / quantization

        for i in range(quantization):
            gripper_pos[1] = (gripper_pos[1] + mov_on_y)
            joint_pos = self.baxter_back.accurateCalculateInverseKinematics(gripper_pos, [0, 1, 0, 0],
                                                                            self.baxter_back.left_end_effector_id,
                                                                            self.baxter_gui.get_left_arm_state())
            self.baxter_gui.set_for_ik(joint_pos)
            self.run_simulation(30)

        # ********************************************************** #

        # ******************** MOVING ON Y AXIS ******************** #

        quantization = int(math.ceil(abs(basket_pos[0] - gripper_pos[0]) / 0.12))
        mov_on_x = (basket_pos[0] - gripper_pos[0]) / quantization

        for i in range(quantization):
            gripper_pos[0] = (gripper_pos[0] + mov_on_x)
            joint_pos = self.baxter_back.accurateCalculateInverseKinematics(gripper_pos, [0, 1, 0, 0],
                                                                            self.baxter_back.left_end_effector_id,
                                                                            self.baxter_gui.get_left_arm_state())
            self.baxter_gui.set_for_ik(joint_pos)
            self.run_simulation(30)

        # ********************************************************** #

        self.baxter_gui.open_gripper()
        self.end_time = time.clock()
        self.run_simulation(30)
        temp = self.end_time - self.start_time
        # print("time_consumed", temp)

    def move_into_basket2(self, gripper_pos, object_color):

        if object_color == 'G':
            basket_pos = [0.14, 0.35]
        else:
            basket_pos = [0.8, 0.35]

        # ********** LIFTING THE GRASPED OBJECT ********** #

        for i in range(2):
            gripper_pos[2] = (gripper_pos[2] + 0.13)
            joint_pos = self.baxter_back.accurateCalculateInverseKinematics(gripper_pos, [0, 1, 0, 0],
                                                                            self.baxter_back.left_end_effector_id,
                                                                            self.baxter_gui.get_left_arm_state())
            self.baxter_gui.set_for_ik(joint_pos)
            self.run_simulation(10)

        # ************************************************ #

        # ******************** MOVING ON X AND Y AXIS SIMULTANEOUSLY ******************** #

        discretization = int(math.ceil(abs(basket_pos[0] - gripper_pos[0]) / 0.12))
        mov_on_x = (basket_pos[0] - gripper_pos[0]) / discretization
        mov_on_y = (basket_pos[1] - gripper_pos[1]) / discretization

        for i in range(discretization):
            gripper_pos[0] = gripper_pos[0] + mov_on_x
            gripper_pos[1] = gripper_pos[1] + mov_on_y
            joint_pos = self.baxter_back.accurateCalculateInverseKinematics(gripper_pos, [0, 1, 0, 0],
                                                                            self.baxter_back.left_end_effector_id,
                                                                            self.baxter_gui.get_left_arm_state())

            self.baxter_gui.set_for_ik(joint_pos)
            self.run_simulation(20)

        # ******************************************************************************* #

        self.baxter_gui.open_gripper()
        # self.end_time = time.clock()
        self.run_simulation(20)
        # self.time_consumed += self.end_time - self.start_time
        # print(self.end_time - self.start_time)

    def put_into_basket(self, object_pos, object_color):

        self.start_time = time.clock()
        self.set_home_position()
        gripper_pos = self.grasp_object(object_pos)
        self.move_into_basket(gripper_pos, object_color)
        self.end_time = time.clock()
        self.time_consumed += self.end_time - self.start_time

    def set_gravity(self, acc_x, acc_y, acc_z):
        self.physics_client.setGravity(acc_x, acc_y, acc_z)
        self.ik_back.setGravity(acc_x, acc_y, acc_z)

    def run_simulation(self, nof_iteration):
        for i in range(nof_iteration):
            self.physics_client.stepSimulation()
        self.get_image()
        # time.sleep(1/100)

    def __del__(self):
        self.physics_client.disconnect(self.physics_client)
        self.ik_back.disconnect(self.physics_client)


class Baxter:

    def __init__(self, simulation):

        # 0.93 height in position makes pedestal of baxter to stand on the ground

        self.simulation = simulation
        self.position = (0, 1, 0.93)
        self.orientation = simulation.getQuaternionFromEuler((0, 0, -1.57))
        simulation.setGravity(0, 0, -9.81)
        simulation.setAdditionalSearchPath("/data/baxter_description/urdf")
        self.baxter_id = simulation.loadURDF("toms_baxter.urdf", self.position, self.orientation, useFixedBase=True)
        self.control_mode = simulation.POSITION_CONTROL

        self.baxter_joints = {}
        for i in range(simulation.getNumJoints(self.baxter_id)):
            self.baxter_joints[i] = simulation.getJointInfo(self.baxter_id, i)

        self.left_name_index = dict()
        self.left_name_index['left_s0'] = 34
        self.left_name_index['left_s1'] = 35
        self.left_name_index['left_e0'] = 36
        self.left_name_index['left_e1'] = 37
        self.left_name_index['left_w0'] = 38
        self.left_name_index['left_w1'] = 40
        self.left_name_index['left_w2'] = 41

        self.left_end_effector_id = 48
        self.number_of_joints = simulation.getNumJoints(self.baxter_id)

    def set_motor(self, joint_index, target_pos: float):
        self.simulation.setJointMotorControl2(self.baxter_id, joint_index, self.control_mode, target_pos)

    def compute_distance_error(self, current_pos, target_pos):

        """
        It is used in inverse kinematic calculation.
        It computes linear distance error between current pose and target pose of allegro hand.

        Parameters
        ----------
        current_pos: current position of an entity
        target_pos: target position of that entity  """

        x_axis = current_pos[0] - target_pos[0]
        y_axis = current_pos[1] - target_pos[1]
        z_axis = current_pos[2] - target_pos[2]
        return (x_axis ** 2 + y_axis ** 2 + z_axis ** 2) ** 0.5

    def set_home_left(self):
        self.set_motor(self.left_name_index['left_s0'], -0.1099309443985112)
        self.set_motor(self.left_name_index['left_s1'], -0.8599954373080283)
        self.set_motor(self.left_name_index['left_e0'], -0.15009375645443795)
        self.set_motor(self.left_name_index['left_e1'], 1.5799348915756681)
        self.set_motor(self.left_name_index['left_w0'], 0.13004237309466116)
        self.set_motor(self.left_name_index['left_w1'], 0.850078130010562)
        self.set_motor(self.left_name_index['left_w2'], 0.46992456562016155)

    def open_gripper(self):
        self.simulation.setJointMotorControl2(bodyIndex=self.baxter_id, jointIndex=49, controlMode=pb.POSITION_CONTROL,
                                              targetPosition=1)
        self.simulation.setJointMotorControl2(bodyIndex=self.baxter_id, jointIndex=51, controlMode=pb.POSITION_CONTROL,
                                              targetPosition=-1)

    def close_gripper(self):
        self.simulation.setJointMotorControl2(bodyIndex=self.baxter_id, jointIndex=49, controlMode=pb.POSITION_CONTROL,
                                              force=100, targetPosition=-1)
        self.simulation.setJointMotorControl2(bodyIndex=self.baxter_id, jointIndex=51, controlMode=pb.POSITION_CONTROL,
                                              force=100, targetPosition=1)

    def accurateCalculateInverseKinematics(self, targetPos, orientation, endEffectorId, initial):

        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_s0'], initial[0])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_s1'], initial[1])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_e0'], initial[2])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_e1'], initial[3])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_w0'], initial[4])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_w1'], initial[5])
        self.simulation.resetJointState(self.baxter_id, self.left_name_index['left_w2'], initial[6])

        closeEnough = False
        iter = 0
        dist2 = 1e30
        while not closeEnough and iter < 1000:
            joint_poses = self.simulation.calculateInverseKinematics(self.baxter_id, endEffectorId,
                                                                     targetPosition=targetPos,
                                                                     targetOrientation=orientation)
            j = 0
            for i in range(self.number_of_joints):
                if self.simulation.getJointInfo(self.baxter_id, i)[3] > -1:
                    if 34 < i < 42:
                        self.simulation.resetJointState(self.baxter_id, i, joint_poses[j])
                    j += 1
            ls = self.simulation.getLinkState(self.baxter_id, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < 0.00001)
            iter = iter + 1

        # print("Num iter: " + str(iter) + " error: " + str(dist2))
        return joint_poses

    def set_for_ik(self, joint_poses):
        j = 0
        for i in range(self.number_of_joints):
            # print(i, self.simulation.getJointInfo(self.baxter_id, i))
            # print("")
            if self.simulation.getJointInfo(self.baxter_id, i)[3] > -1:
                if 34 < i < 42:
                    self.simulation.setJointMotorControl2(bodyIndex=self.baxter_id, jointIndex=i,
                                                          controlMode=pb.POSITION_CONTROL,
                                                          targetPosition=joint_poses[j],
                                                          targetVelocity=0)
                j += 1

    def get_left_arm_state(self):

        to_return = [
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_s0'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_s1'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_e0'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_e1'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_w0'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_w1'])[0],
            self.simulation.getJointState(self.baxter_id, self.left_name_index['left_w2'])[0]
        ]

        return to_return
