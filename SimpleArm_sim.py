from time import sleep
import SimpleArm_dynamics
import numpy as np
import matplotlib.pyplot as plt
import pyglet
from pyglet import shapes
from trajectories import LinearTrajectory, CircularTrajectory, PolygonalTrajectory
from controllers import JointVelocityController, JointTorqueController, WorkspaceVelocityController
import colorsys
import argparse

"""
2DOF manipulator simulation. Used for testing controllers and paths
C106B Project 1A
"""

# ---------------------------------------------------------------------------- #
#                                Debugging tools                               #
# ---------------------------------------------------------------------------- #
# let's use the pprint module for readability
from pprint import pprint

# import inspect module
import inspect
# ---------------------------------------------------------------------------- #
#                                Debugging tools                               #
# ---------------------------------------------------------------------------- #


trajectory = None
controller = None

class SimpleArmSim(pyglet.window.Window):

    def __init__(self, width, height):
        """
        Initialize the simulation window, constants for the manipulator, and other simulation parameters.
        Also sets up the visualization of the manipulator and its trajectory.

        CONSTANTS:
        dt      : The time step for the simulation.
        time    : The current time of the simulation.
        batch   : A batch object from the pyglet graphics library, which is used to group graphics primitives for drawing.
        pixel_origin: The origin of the coordinate system in pixels.
        link1, link2: The two links of the manipulator, represented as rectangles.
        q, q_dot    : The current joint angles and velocities of the manipulator.
        ik_sols, joint_velocity_sols, joint_acceleration_sols: Lists for storing the solutions for the inverse kinematics, joint velocities and joint accelerations, respectively.
        circles     : A list for storing circles that visualize the trajectory of the manipulator.

        Parameters
        ----------
        width: width of display window in pixels
        height: height of display window in pixels
        trajectory: desired trajectory for manipulator to follow. Will remove orientation and z-axis information
        controller: controller used to execute desired trajectory
        """

        # ---------------------------------------------------------------------------- #
        #                   DEFINE CONSTANTS for for the manipulator                   #
        # ---------------------------------------------------------------------------- #
        self.l1 = 8
        self.l2 = 6
        m1 = 2
        m2 = 1
        I1 = (1/12) * m1 * self.l1**2
        I2 = (1/12) * m2 * self.l2**2
        g = 9.8
        constants = [self.l1, self.l2, m1, m2, I1, I2, g]

        # Other simulation constants
        self.dt = 1/60 
        super().__init__(width, height, "2DOF Manipulator Simulation")
        self.time = 0
        self.batch = pyglet.graphics.Batch()
        self.pixel_origin = [width/2, 50]
        

        # Constants for visualization
        link_w = 0.25
        board_w = link_w/2
        self.pm = 25 # pixels per meter
        self.deg2rad = 180/np.pi

        # Setup These are condensed functions that take in the current joint angles and velocities, 
        # and return the mass matrix, Coriolis matrix, gravity matrix, and the body Jacobian, respectively.
        self.M_func = lambda q, q_dot: SimpleArm_dynamics.M_func(*(constants + [q, q_dot]))
        self.C_func = lambda q, q_dot: SimpleArm_dynamics.C_func(*(constants + [q, q_dot]))
        self.G_func = lambda q, q_dot: SimpleArm_dynamics.G_func(*(constants + [q, q_dot]))
        self.J_body_func = lambda q, q_dot: SimpleArm_dynamics.J_body_func(*(constants + [q, q_dot]))


        # ---------------------------------------------------------------------------- #
        #                          ROBOT ARM Manipulator Links                         #
        # ---------------------------------------------------------------------------- #
        self.link1 = shapes.Rectangle(
            x=self.pixel_origin[0],
            y=self.pixel_origin[1],
            width=link_w*self.pm,
            height=self.l1*self.pm,
            color=(87, 74, 226),
            batch=self.batch)
        self.link1.anchor_x = (link_w/2)*self.pm
        self.link1.anchor_y = 0

        self.link2 = shapes.Rectangle(
            x=self.pixel_origin[0],
            y=self.pixel_origin[1] + self.l1*self.pm,
            width=link_w*self.pm,
            height=self.l2*self.pm,
            color=(226, 173, 242),
            batch=self.batch)
        self.link2.anchor_x = (link_w/2)*self.pm
        self.link2.anchor_y = 0

        # Manipulator states
        self.q = np.array([
            0,
            0
        ], dtype=np.float32)
        self.q_dot = np.array([
            0,
            0
        ], dtype=np.float32)


        # ---------------------------------------------------------------------------- #
        #                           Visualizer for trajectory                          #
        # ---------------------------------------------------------------------------- #
        # - We start by solving the initial state using IK. 
        # - Then, for every point dt, we solve IK and store the solution.
        # - We also compute the joint velocities and accelerations (naively, by dividing dt).
        # - Finally, we draw circles at the end of each link to visualize the trajectory.
        # ---------------------------------------------------------------------------- #
        if trajectory:

            # ------------------ Check for trajectory reachability first ----------------- #
            if not self.trajectory_reachable(trajectory):
                raise Exception("Trajectory is not reachable!")
            
            # Set initial configuration to trajectory start
            sol = self.ik(trajectory.target_pose(0)[:2]) # We only need xy, screw z
            self.q = sol

            # Lists to store IK over entire trajectory
            self.ik_sols = [sol]
            self.joint_velocity_sols = []
            self.joint_acceleration_sols = []

            # Add circles to visualize trajectory
            self.circles = []
            count = 0
            for t in np.arange(0, trajectory.total_time, self.dt):
                pos = trajectory.target_pose(t)[:2] # We only need xy, screw z
                self.ik_sols.append(self.ik(pos, self.ik_sols[-1]))

                # Compute joint velocities (naively)
                # Joint velocities are computed by taking the difference between the current and previous joint angles, divided by dt
                self.joint_velocity_sols.append((self.ik_sols[-1] - self.ik_sols[-2])/self.dt)

                # Compute joint accelerations (naively)
                if len(self.joint_velocity_sols) > 1:
                    self.joint_acceleration_sols.append((self.joint_velocity_sols[-1] - self.joint_velocity_sols[-2])/self.dt)
                
                # For every 20th point, draw a circle to visualize the trajectory
                if count % 20 == 0:
                    rgb = colorsys.hsv_to_rgb(t/trajectory.total_time, 1, 1)
                    rgb_norm = [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)]
                    circle = shapes.Circle(
                        x=pos[0]*self.pm + self.pixel_origin[0],
                        y=pos[1]*self.pm + self.pixel_origin[1],
                        radius=3,
                        color=rgb_norm,
                        batch=self.batch)
                    self.circles.append(circle)
                count += 1
    
    def ang_diff(self, a1, a2):
        """ Helper function for IK """

        diff = np.mod(np.abs(a1 - a2), 2 * np.pi)
        if diff < np.pi:
            return diff
        else:
            return 2 * np.pi - diff

    def ik(self, pos, seed=np.zeros(2)):
        """
        Return IK solution for desired xy position using MLS 3.1

        Parameters                 
        ----------
        pos - xy position of end effector
        seed - prevous IK solution

        Returns
        -------
        sol - joint angles corresponding to xy position of end effector
        """

        r = np.linalg.norm(pos)
        cos_alpha = (self.l1**2 + self.l2**2 - r**2)/(2*self.l1*self.l2)
        if cos_alpha > 1 or cos_alpha < -1:
            raise Exception("IK Solver Failed")
        alpha = np.arccos(cos_alpha)
        theta2_sol1 = np.pi + alpha
        theta2_sol2 = np.pi - alpha

        cos_beta = (r**2 + self.l1**2 - self.l2**2)/(2*self.l1*r)
        beta = np.arccos(cos_beta)
        if cos_beta > 1 or cos_beta < -1:
            raise Exception("IK Solver Failed")
        theta1_sol1 = np.arctan2(-pos[0], pos[1]) + beta
        theta1_sol2 = np.arctan2(-pos[0], pos[1]) - beta
        diff1 = self.ang_diff(seed[0], theta1_sol1) + self.ang_diff(seed[1], theta2_sol1)
        diff2 = self.ang_diff(seed[0], theta1_sol2) + self.ang_diff(seed[1], theta2_sol2)

        # Return solution closest to previous solution
        if diff1 < diff2:
            return np.array([theta1_sol1, theta2_sol1])
        else:
            return np.array([theta1_sol2, theta2_sol2])

    def xy(self):
        """Return positions of junction and end effector"""

        first_tip = np.array([
            -self.l1*np.sin(self.q[0]),
            self.l1*np.cos(self.q[0])
        ])
        second_tip = np.array([
            -self.l2*np.sin(self.q[0] + self.q[1]),
            self.l2*np.cos(self.q[0] + self.q[1])
        ])
        return first_tip, first_tip + second_tip

    # ---------------------------------------------------------------------------- #
    #      on_draw() method updates the window with current manipulator state      #
    # ---------------------------------------------------------------------------- #
    def on_draw(self):
        """Clear the screen and draw shapes"""

        self.clear()
        self.batch.draw()
    
    def update_kinematic(self):
        """For controllers that require a kinematic (instantaneous velocity change) model"""

        if isinstance(controller, JointVelocityController):
            # Get current desired joint angle velocity
            index = int(self.time // self.dt)
            safe_index = lambda i: max(min(i, len(self.joint_velocity_sols) - 1), 0)
            target_velocity = self.joint_velocity_sols[safe_index(index)]
            joint_velocity = controller.step_control(None, target_velocity, None)
            self.q_dot = joint_velocity
        
        elif isinstance(controller, WorkspaceVelocityController):
            # Get current desired end effector velocity
            target_velocity = trajectory.target_velocity(self.time)[:2] # We only need xy
            joint_velocity = controller.step_control(None, target_velocity, None)
            self.q_dot = joint_velocity

        # Integrate state
        self.q = self.q + self.q_dot * self.dt
        self.update_frame()

    def update_dynamic(self):
        """ For controllers that require a dynamic model """

        M = self.M_func(self.q, self.q_dot) 
        C = self.C_func(self.q, self.q_dot) 
        G = self.G_func(self.q, self.q_dot) 

        joint_torque = np.zeros((4, 1))
        if isinstance(controller, JointTorqueController):
            # Get current desired joint torque
            index = int(self.time // self.dt)
            safe_index = lambda i: max(min(i, len(self.joint_acceleration_sols) - 1), 0)
            target_acceleration = self.joint_acceleration_sols[safe_index(index)]
            joint_torque = controller.step_control(None, None, target_acceleration)

        # Simulate dynamics
        q_ddot = np.matmul(np.linalg.inv(M), -(np.matmul(C, self.q_dot.reshape(-1, 1)) + G) + joint_torque).reshape(-1)
        self.q_dot = self.q_dot + q_ddot * self.dt
        self.q = self.q + self.q_dot * self.dt
        self.update_frame()
    
    def update_frame(self):
        """Animate the shapes"""

        self.time += self.dt

        self.link1.rotation = -self.q[0] * self.deg2rad
        self.link2.rotation = -(self.q[0] + self.q[1]) * self.deg2rad

        junction, end = self.xy()
        self.link2.x = junction[0]*self.pm + self.pixel_origin[0]
        self.link2.y = junction[1]*self.pm + self.pixel_origin[1]

    def trajectory_reachable(self, trajectory):
        '''
        Check if the end effector is within the workspace of the manipulator.
        If not, print a warning and exit the program.

        Parameters
        ----------
        sim (SimpleArmSim): The simulation object
        
        Returns:
        ----------    
        bool: True if the end effector is within the workspace, False otherwise
        '''
        if trajectory.__class__.__name__ == 'LinearTrajectory':
            # Check if Initial Position and Final Position is within the workspace
            initial_position = trajectory.init_pos
            final_position = trajectory.end_pos

            # Strip the z coordinate
            initial_position = initial_position[:2]
            final_position = final_position[:2]

            # # Boundary conditions
            # if( initial_position[0]  > self.width or 
            #     initial_position[1] > self.height or
            #     final_position[0] > self.width or
            #     final_position[1] > self.height or
            #     initial_position[0] < 0 or
            #     initial_position[1] < 0 or
            #     final_position[0] < 0 or
            #     final_position[1] < 0):
            #     print("Initial Position is outside the workspace")
            #     return False
            
            # Reachability conditions
            inner_radius = self.l1 - self.l2
            outer_radius = self.l1 + self.l2

            # If the distance between the initial and final position with the origin is less than the inner radius, return False
            origin = np.array([0, 0])
            if (np.linalg.norm(initial_position - origin)<inner_radius):
                print("Initial Position is within the inner radius")
                return False
            if (np.linalg.norm(final_position   - origin)<inner_radius):
                print("Final Position is within the inner radius")
                return False
            if (np.linalg.norm(initial_position - origin)>outer_radius):
                print("Initial Position = ", initial_position)
                print("Initial Position is outside the outer radius")
                return False
            if (np.linalg.norm(final_position   - origin)>outer_radius):
                print("Final Position is outside the outer radius")
                return False

            # Checking if the trajectory crosses the inner radius
            # If the trajectory crosses the inner radius, return False
            # idea: draw the line, if line passes the inner circle it is wrong
            return True


        elif trajectory.__class__.__name__ == 'CircularTrajectory':
            
            pass
        elif trajectory.__class__.__name__ == 'PolygonalTrajectory':
            trajectory = PolygonalTrajectory([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 5)
        return trajectory

# ---------------------------------------------------------------------------- #
#                            SimpleArmSim Class Ends                           #
# ---------------------------------------------------------------------------- #




# ---------------------------------------------------------------------------- #
#     Helper function to define the trajectory type from command line args     #
# ---------------------------------------------------------------------------- #
def define_trajectories(args, width, height):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory(total_time=3, init_pos=np.array([-2, 4, 0]), end_pos= np.array([2, 4, 0]))
    elif args.task == 'circle':
        trajectory = CircularTrajectory([0, 1, 1], [0, 0, 0], 1, 2)
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory(np.array([[-2, 4, 0], [2, 4, 0], [2, 8, 0], [-2, 8, 0]]), 5)
    return trajectory

# ---------------------------------------------------------------------------- #
#                             MAIN_HELPER_FUNCTIONS                            #
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='jointspace', 
        help='Options: jointspace, workspace, or torque.  Default: jointspace'
    )

    # ------------------ Get the arguments from the command line ----------------- #
    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    #                                  PARAMETERS                                  #
    # ---------------------------------------------------------------------------- #
    # --------- Width and height in pixels of simulator. Adjust as needed -------- #
    width = 500
    height = 500

    # For Line Trajectory
    goal_point = np.array([0.5, 0.5, 0])
    total_time = 3

    # For Polygonal Trajectory
    goal_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    # ------------------------- Initialize the trajectory ------------------------ #
    trajectory = define_trajectories(args, width, height)

    # ------------------------- Initialize the Simulator ------------------------- #
    sim = SimpleArmSim(width, height)        

    # -------------------------- Define the controllers -------------------------- #
    if args.controller_name == 'jointspace':
        controller = JointVelocityController(sim)
    if args.controller_name == 'workspace':
        controller = WorkspaceVelocityController(sim)
    elif args.controller_name == 'torque':
        controller = JointTorqueController(sim)

    if trajectory and controller:
        if args.controller_name == 'jointspace':
            update_func = sim.update_kinematic
        if args.controller_name == 'workspace':
            update_func = sim.update_kinematic
        elif args.controller_name == 'torque':
            update_func = sim.update_dynamic

        while(sim.time < trajectory.total_time):
            # Integrate the dynamics
            update_func()

            # Render scene
            sim.dispatch_events()
            sim.dispatch_event('on_draw')
            sim.flip()

            # Wait until finshed with dt if needed
            delta_time = pyglet.clock.tick()
            time_left = max(sim.dt - delta_time, 0)
            sleep(time_left)