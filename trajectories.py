#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import utils_kinematics as kin

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #
TOTAL_TIME_THRESHOLD = 0.1

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
        	desired duration of the trajectory in seconds 
        """
        if (total_time < TOTAL_TIME_THRESHOLD):
            raise ValueError("Total time must be greater than %f" % TOTAL_TIME_THRESHOLD)

        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        
        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

# ---------------------------------------------------------------------------- #
#                            LINEAR TRAJECTORY CLASS                           #
# ---------------------------------------------------------------------------- #
# - Create a LinearTrajectory that starts at zero velocity and ends at zero velocity

class LinearTrajectory(Trajectory):

    def __init__(self, total_time, init_pos, end_pos):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """

        # Are we going to implement a speed factor?
        # How are we going to define the speed factor, is it the speed of the end effector on the path, or the speed of the joints? like just a multiplier
            # We can Probably use Velocity control and make sure that the end-effector velocity is according to the speed factor
        # How will we handle the case where the trajectory is not possible?
    
        # - Trajectory 
        Trajectory.__init__(self, total_time)


        # --------------- Keeping track of the previous state variables -------------- #
        self.init_pos = init_pos
        self.end_pos  = end_pos
        self.curr_pos = init_pos
        self.prev_vel = 0
        self.prev_time = 0

        # ---------------- Quadratic Speed Model Velocity Coefficients --------------- #
        self.distance = np.linalg.norm(self.end_pos - self.init_pos)
        self.b = self.distance / (-self.total_time**3/3 + self.total_time**3/2)

        # Speed Check: If the maximum velocity going to be achieved is going to be dangerous


    def quadratic_speed_model(self, time):
        """
        Returns the ||speed|| of the end-effector at time t, calculated using 
        a quadratic speed model based on the total time of the trajectory and 
        the total distance need to be travelled.
        Implementation steps:
        1. Model the velocity as a quadratic function of time, from 2 points,
          (0, 0) and (total_time, 0), with the equation being scaled by `b`
        2. Take the integral of the Velocity Function to get the position function
        3. Using the total distance needed to be travelled by the end-effector,
            find the coefficient `b`.
        4. Return the velocity at time t.

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        float ||speed||
        """
        # ---- Find the distance between the current position and the end position --- #
        # distance = np.linalg.norm(self.end_pos - self.init_pos)
        # print("Distance: ", distance)

        # -------------------------- Find the coefficient b -------------------------- #
        # b = distance / (-self.total_time**3/3 + self.total_time**3/2)

        return self.b * (-(time - self.total_time/2)**2 + (self.total_time/2)**2)


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """

        # ----------------------- Vector equation for the line: ---------------------- #
        norm_vec = (self.end_pos - self.init_pos) / np.linalg.norm(self.end_pos - self.init_pos)
        vec_xyz = self.curr_pos + norm_vec * self.prev_vel * (time - self.prev_time)
        self.prev_vel = self.quadratic_speed_model(time)
        self.curr_pos = vec_xyz
        self.prev_time = time
        print("( time = " , time, ", vel = ", self.quadratic_speed_model(time) , ", curr_pos = " , self.curr_pos , ")")
        return [vec_xyz[0], vec_xyz[1], vec_xyz[2], 0, 1, 0, 0] # For now, assume the end effector is always pointing down

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector. 
        Implementation Steps:
        1. Find the vector in the direction of the line, normalized
        2. Multiply by the speed factor based on a speed map, calculated to be an exponent

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        # --------- Find the vector in the direction of t he line, normalized --------- #
        vec_dir = (self.end_pos - self.init_pos) / np.linalg.norm(self.end_pos - self.init_pos)

        # ----------------------- Multiply by the speed factor ----------------------- #
        return np.append(vec_dir * self.quadratic_speed_model(time), [0, 0, 0]) # for now, angular velocity is 0






# ---------------------------------------------------------------------------- #
#                           CIRCULAR TRAJECTORY CLASS                          #
# ---------------------------------------------------------------------------- #
# - Create a LinearTrajectory that starts at zero velocity and ends at zero velocity
class CircularTrajectory(Trajectory):

    # def __init__(self, center_position, radius, total_time):
    def __init__(self, axis_of_rotation, centerpoint, radius, total_time):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------

        ????? You're going to have to fill these in how you see fit
        """
        pass
        
        Trajectory.__init__(self, total_time)

        self.axis = axis_of_rotation
        self.center = centerpoint
        self.radius = radius

        # ------------------------------ Previous States ----------------------------- #
        self.prev_time = 0
        self.prev_ang_vel = 0
        self.curr_angle = 0
        integral = (-self.total_time**3/3 + self.total_time**3/2)
        self.b = 2*np.pi / integral

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        # ------ Get how much in theta was travelled from the previous timestep ------ #
        ang_travelled = self.prev_ang_vel * (time - self.prev_time) 

        # ------------------- Get the current angle of the end effector ---------------- #
        self.curr_angle = self.curr_angle + ang_travelled
        print("( curr_angle: ", self.curr_angle, ", time = " , time, ", ang_travelled = ", ang_travelled, ", ang_vel = ", self.quadratic_speed_model(time) , ", angle = " , self.curr_angle , ")")
        self.prev_ang_vel = self.quadratic_speed_model(time)

        # ------------------- Get the current position of the end effector -------------- #
        curr_pos = np.array([self.center[0] + self.radius*np.cos(self.curr_angle), self.center[1] + self.radius*np.sin(self.curr_angle), 0, 1, 0, 0]) # add initial offset to where the robot is starting the circle
        self.prev_time = time
        return curr_pos

    def quadratic_speed_model(self, time):
        """
        Returns the ||ang_vel|| of the end-effector at time t wrt the axis of rotation. Calculated using 
        a quadratic speed model based on the total time of the trajectory and 
        the total distance need to be travelled.
        Implementation steps:
        1. Model the velocity as a quadratic function of time, from 2 points,
          (0, 0) and (total_time, 0), with the equation being scaled by `b`
        2. Take the integral of the Velocity Function to get the position function
        3. Using the total distance needed to be travelled by the end-effector,
            find the coefficient `b`.
        4. Return the velocity at time t.

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        float ||speed||
        """
        # ---- Find the distance between the current position and the end position --- #
        distance =  2 * np.pi # Angle need to be travelled in radians
        # print("distance = ", distance)

        # -------------------------- Find the coefficient b -------------------------- #
        # integral = (-self.total_time**3/3 + self.total_time**3/2)
        # b = distance / integral
        # print("b = ", b)
        # print("denominator = ", (-self.total_time**3/3 + self.total_time**3/2))

        temp = self.b * (-(time - self.total_time/2)**2 + (self.total_time/2)**2)
        # print("temp_vel = ", temp)
        return temp #b * (-(time - self.total_time/2)**2 + (self.total_time/2)**2)

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        # Calculate the magnitude of the angular velocity
        ang_vel = self.quadratic_speed_model(time) # in radians per second
        # print("ang_vel = ", ang_vel)
        # v_magnitude = ang_vel * self.radius
        # print("v_vector = ", v_vector)
        # x_dot = v_magnitude * np.cos(self.curr_angle)
        # y_dot = v_magnitude * np.sin(self.curr_angle)
        # print("x_dot = ", x_dot)
        # print("y_dot = ", y_dot)
        # v = np.array([x_dot, y_dot, 0, x_dot/self.radius, y_dot/self.radius, 0])
        # print("v = ", v)

        # r = kin.spherical_to_cartesian(self.curr_angle, 0, self.radius)
        # r = np.array([x, y, z])
        # v = np.array([v_x, v_y, 0])

        # v_t = np.cross(v, np.cross(r, v)) / np.linalg.norm(r)**2
        current_angle = self.b*(-time**3/3 + self.total_time*time**2/2)
        r_vec = self.radius * ang_vel * np.array([ - np.sin(current_angle), np.cos(current_angle), 0])
        ret_val = np.array([r_vec[0],r_vec[1],r_vec[2], 0,0,ang_vel])
        print("ret_val, ", ret_val, ", angle (radians): ", current_angle)
        return ret_val







# ---------------------------------------------------------------------------- #
#                             POLYGONAL TRAJECTORY                             #
# ---------------------------------------------------------------------------- #
class PolygonalTrajectory(Trajectory):
    def __init__(self, points, total_time):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit

        """
        Trajectory.__init__(self, total_time)

        # -------------- Get a list of distances and the total distance -------------- #
        self.total_distance = 0
        self.distances = np.array([])
        self.points = np.array(points)
        print("points = ", self.points)
        print("points = ", self.points[0].reshape(1, -1))
        self.points = np.concatenate((self.points, self.points[0].reshape(1, -1))) # add the first point to the end of the list to make it a closed loop
        
        print("points = ", self.points)

        for i in range(len(self.points)-1): # for n points, there are n-1 intervals
            distance = np.linalg.norm(self.points[i] - self.points[i+1])
            self.total_distance += distance
            self.distances = np.append(self.distances, distance)
        print("distances = ", self.distances)
        print("total_distance = ", self.total_distance)

        # ------------------- Get the total time for the each trajectory ------------------ #
        self.times = np.array([])
        for i in range(len(self.distances)):
            self.times = np.append(self.times, self.distances[i]/self.total_distance * self.total_time)
        print("times = ", self.times)
        print("total_time = ", self.total_time)

        # -------- For Each Time in self.times, create a new Linear Trajectory ------- #
        self.trajectories = np.array([])
        for i in range(len(self.times)):
            self.trajectories = np.append(self.trajectories, LinearTrajectory(self.times[i], self.points[i], self.points[i+1]))
        


    def time_to_index(self, time):
        """
        Returns the index of the trajectory that the time belongs to.

        Parameters
        ----------
        time : float

        Returns
        -------
        int index  :index of the trajectory that the time belongs to
        float time : time in the trajectory that the time belongs to
        """
        for i in range(len(self.times)):
            if time < self.times[i]:
                return i, time
            else:
                time -= self.times[i]
            print(len(self.times)-1, time)
        
        return len(self.times)-1, time

    def quadratic_speed_model(self, time):
        """
        Returns the speed of the end effector at time t
        1) Find the right trajectory index, and the time in that trajectory
        2) Use the quadratic speed model function of that Linear trajectory to find the speed at that time

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        float ||speed||
        """
        trajectory_idx, time = self.time_to_index(time)
        trajectory = self.trajectories[trajectory_idx]
        return trajectory.quadratic_speed_model(time)


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        # ----------------------- Find the right trajectory ----------------------- #
        print("time = ", time)
        print(self.time_to_index(time))
        trajectory_idx, time = self.time_to_index(time)
        trajectory = self.trajectories[trajectory_idx]
        t_pos = trajectory.target_pose(time)
        print("t_pos = ", t_pos)
        return t_pos

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector. 
        Implementation Steps:
        1. Find the vector in the direction of the line, normalized
        2. Multiply by the speed factor based on a speed map, calculated to be an exponent

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        # ----------------------- Find the right trajectory ----------------------- #
        trajectory_idx, time = self.time_to_index(time)
        trajectory = self.trajectories[trajectory_idx]
        t_vel = trajectory.target_velocity(time)
        print("t_vel = ", t_vel)
        
        return t_vel




# ---------------------------------------------------------------------------- #
#                             MAIN HELPER FUNCTIONS                            #
# ---------------------------------------------------------------------------- #

def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory(total_time=3, init_pos=np.array([5,0,0]), end_pos= np.array([5,30,0]))
    elif args.task == 'circle':
        trajectory = CircularTrajectory([0, 0, 1], [0, 10, 0], 1, 10)
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 5)
    return trajectory


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)
