#!/usr/bin/env python

"""
Abridged controller code for project 1
Author: Chris Lai, Chris Correa, Valmik Prabhu
"""

import numpy as np
import utils_kinematics as kin

class Controller:

    def __init__(self, sim=None):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        sim : SimpleArmSim object. Contains all information about simulator state (current joint angles and velocities, robot dynamics, etc.)
        """

        if sim:
            self.sim = sim

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly.  

        Parameters
        ----------
        target_position : (2,) numpy array 
            desired position or joint angles
        target_velocity : (2,) numpy array
            desired end-effector velocity or joint velocities
        target_acceleration : (2,) numpy array 
            desired end-effector acceleration or joint accelerations

        Returns
        ----------
        desired control input (joint velocities or torques) : (2,) numpy array
        """
        pass

# ---------------------------------------------------------------------------- #
#                        JointSpace Velocity Controller                        #
# ---------------------------------------------------------------------------- #
class JointVelocityController(Controller):

    def __init__(self, sim=None):
        super().__init__(sim)

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Sets the joint velocity equal to that computed through IK

        Parameters
        ----------
        target_position : (2,) numpy array 
            desired positions or joint angles
        target_velocity : (2,) numpy array
            desired end-effector velocity or joint velocities
        target_acceleration : (2,) numpy array 
            desired end-effector acceleration or joint accelerations

        ----------
        desired control input (joint velocities or torques) : (2,) numpy array
        """
        return target_velocity


# ---------------------------------------------------------------------------- #
#                        WorkSpace Velocity Controller                        #
# ---------------------------------------------------------------------------- #
class WorkspaceVelocityController(Controller):

    def __init__(self, sim=None):
        super().__init__(sim)

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Computes joint velocities to execute given end effector body velocity

        Parameters
        ----------
        target_position : (2,) numpy array 
            desired positions or joint angles
        target_velocity : (2,) numpy array
            desired end-effector velocity or joint velocities
        target_acceleration : (2,) numpy array 
            desired end-effector acceleration or joint accelerations

        Returns
        ----------
        desired control input (joint velocities or torques) : (2,) numpy array
        """
        # Calculate Jacobian
        xi_1 = np.array([0, 0, 0, 0, 0, 1]) # 6x1 Axis of rotation is z
        xi_2 = np.array([self.sim.l1, 0, 0, 0, 0, 1]) # 6x1 Axis of rotation is x

        translation = kin.spherical_to_cartesian((self.sim.q[0], 0, self.sim.l1))
        omega = np.array([0, 0, 1])
        adj_g = kin.adjoint(omega= omega, theta = self.sim.q[0], translation = translation)

        J_theta = np.array([xi_1, np.matmul(adj_g, xi_2)]).T # 6x2
        print("J_theta: ", J_theta)
        J_theta = self.sim.J_body_func(self.sim.q, self.sim.q_dot) # 6x2
        print("J_theta (correct): ", J_theta)

        # Calculate pseudo-inverse of Jacobian
        J_theta_pinv = np.linalg.pinv(J_theta) # 2x6
        print("J_theta_pinv: ", J_theta_pinv)

        # Multiply Jacobian by target velocity
        # target_velocity = np.array([target_velocity[0], target_velocity[1], 0, 0, 0, 0]).T
        theta_dot = np.matmul(J_theta_pinv, target_velocity)
        print("target_vel: ", target_velocity)
        print("theta_dot: ", theta_dot)

        return theta_dot

        # # Make sure you're using the latest time
        # while (self._curIndex < self._maxIndex and self._path.joint_trajectory.points[self._curIndex+1].time_from_start.to_sec() < t+0.001):
        #     self._curIndex = self._curIndex+1


        # # -------- Inquirng for the current position and velocity of the robot ------- #
        # current_position = np.array([self._limb.joint_angles()[joint_name] for joint_name in self._path.joint_trajectory.joint_names])
        # current_velocity = np.array([self._limb.joint_velocities()[joint_name] for joint_name in self._path.joint_trajectory.joint_names])


        # # # -------- Computing the target position and velocity ------- #
        # # if self._curIndex < self._maxIndex:

        # #     time_low = self._path.joint_trajectory.points[self._curIndex].time_from_start.to_sec()
        # #     time_high = self._path.joint_trajectory.points[self._curIndex+1].time_from_start.to_sec()

        # #     target_position_low = np.array(self._path.joint_trajectory.points[self._curIndex].positions)
        # #     target_velocity_low = np.array(self._path.joint_trajectory.points[self._curIndex].velocities)

        # #     target_position_high = np.array(self._path.joint_trajectory.points[self._curIndex+1].positions)
        # #     target_velocity_high = np.array(self._path.joint_trajectory.points[self._curIndex+1].velocities)

        # #     target_position = target_position_low + (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
        # #     target_velocity = target_velocity_low + (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)

        # # else:
        # #     target_position = np.array(self._path.joint_trajectory.points[self._curIndex].positions)
        # #     target_velocity = np.array(self._path.joint_trajectory.points[self._curIndex].velocities)


        # # Feed Forward Term
        # u_ff = target_velocity

        # # Error Term
        # error = target_position - current_position

        # # Integral Term
        # self._IntError = self._Kw * self._IntError + error
        
        # # Derivative Term
        # dt = t - self._LastTime
        # # We implement a moving average filter to smooth the derivative
        # curr_derivative = (error - self._LastError) / dt
        # self._ring_buff.append(curr_derivative)
        # ed = np.mean(self._ring_buff)

        # # Save terms for the next run
        # self._LastError = error
        # self._LastTime = t

        # ###################### YOUR CODE HERE #########################

        # # Note, you should load the Kp, Ki, Kd, and Kw constants with
        # # self._Kp
        # # and so on. This is better practice than hard-coding

        # # Feedforward 3.1
        # # u = u_ff 
        # print(u_ff)
        # print(error)
        # print(self._Kp)
        # u = u_ff + np.multiply(self._Kp,error) + np.multiply(self._Kd,ed) + np.multiply(self._Ki,self._IntError)

        # ###################### YOUR CODE END ##########################

        # return u


class JointTorqueController(Controller):

    def __init__(self, sim=None):
        super().__init__(sim)

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Recall that in order to implement a torque based controller you will need access to the 
        dynamics matrices M, C, G such that

        M ddq + C dq + G = u

        Look in section 4.5 of MLS for theory behind the computed torque control law.

        Parameters
        ----------
        target_position : (2,) numpy array 
            desired positions or joint angles
        target_velocity : (2,) numpy array
            desired end-effector velocity or joint velocities
        target_acceleration : (2,) numpy array 
            desired end-effector acceleration or joint accelerations

        Returns
        ----------
        desired control input (joint velocities or torques) : (2,) numpy array
        """
        pass
