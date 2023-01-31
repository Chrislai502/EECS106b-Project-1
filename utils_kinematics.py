#!/usr/bin/env python
"""
Kinematic function skeleton code for Lab 3 prelab.
Course: EE 106A, Fall 2021
Originally written by: Aaron Bestick, 9/10/14
Adapted for Fall 2020 by: Amay Saxena, 9/10/20
This Python file is a code skeleton for Lab 3 prelab. You should fill in
the body of the five empty methods below so that they implement the kinematic
functions described in the assignment.
When you think you have the methods implemented correctly, you can test your
code by running "python kin_func_skeleton.py at the command line.
"""

import numpy as np
import math

np.set_printoptions(precision=4,suppress=True)

#-----------------------------2D Examples---------------------------------------
#--(you don't need to modify anything here but you should take a look at them)--

def rotation_2d(theta):
    """
    Computes a 2D rotation matrix given the angle of rotation.
    Args:
    theta: the angle of rotation
    Returns:
    rot - (2,2) ndarray: the resulting rotation matrix
    """

    rot = np.zeros((2,2))
    rot[0,0] = np.cos(theta)
    rot[1,1] = np.cos(theta)
    rot[0,1] = -np.sin(theta)
    rot[1,0] = np.sin(theta)

    return rot

def hat_2d(xi):
    """
    Converts a 2D twist to its corresponding 3x3 matrix representation
    Args:
    xi - (3,) ndarray: the 2D twist
    Returns:
    xi_hat - (3,3) ndarray: the resulting 3x3 matrix
    """
    if not xi.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    xi_hat = np.zeros((3,3))
    xi_hat[0,1] = -xi[2]
    xi_hat[1,0] =  xi[2]
    xi_hat[0:2,2] = xi[0:2]

    return xi_hat

def homog_2d(xi, theta):
    """
    Computes a 3x3 homogeneous transformation matrix given a 2D twist and a
    joint displacement
    Args:
    xi - (3,) ndarray: the 2D twist
    theta: the joint displacement
    Returns:
    g - (3,3) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (3,):
        raise TypeError('xi must be a 3-vector')

    g = np.zeros((3,3))
    wtheta = xi[2]*theta
    R = rotation_2d(wtheta)
    p = np.dot(np.dot( \
        [[1 - np.cos(wtheta), np.sin(wtheta)],
        [-np.sin(wtheta), 1 - np.cos(wtheta)]], \
        [[0,-1],[1,0]]), \
        [[xi[0]/xi[2]],[xi[1]/xi[2]]])

    g[0:2,0:2] = R
    g[0:2,2:3] = p[0:2]
    g[2,2] = 1

    return g

#-----------------------------3D Functions--------------------------------------
#-------------(These are the functions you need to complete)--------------------

def twist(q, w):

    # assert w.shape == (3,)
    # assert q.shape == (3,)
    v = np.cross(-w, q)
    return np.hstack((v, w)).tolist()

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
    Also known as ^ hat
    Args:
    omega - (3,) ndarray: the rotation vector
    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """

    # YOUR CODE HERE
    s = np.zeros((3,3))
    s[0][1] = -omega[2]
    s[0][2] = omega[1]
    s[1][0] = omega[2]
    s[1][2] = -omega[0]
    s[2][0] = -omega[1]
    s[2][1] = omega[0]
    return s


def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation
    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix
    """

    # YOUR CODE HERE
    I = np.identity(3)
    s = skew_3d(omega)
    n = np.linalg.norm(omega)

    r = I + (s / n)*np.sin(n*theta) + (np.linalg.matrix_power(s,2)/n**2) *(1-np.cos(n*theta))
    return r



def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation
    Args:
    xi - (6,) ndarray: the 3D twist
    Returns:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix
    """

    # YOUR CODE HERE
    xi_hat = np.zeros((4,4))
    s = skew_3d(xi[3:6])
    for i in range(len(s)):
        for j in range(len(s[i])):
            xi_hat[i][j] = s[i][j]

    for k in range(3):
        xi_hat[k][3] = xi[k]

    return xi_hat


def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a
    joint displacement.
    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement
    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """

    # YOUR CODE HERE
    r = np.zeros((4,4))
    v = xi[0:3]
    w = xi[3:6]
    vt = v * theta
    wh = skew_3d(w)

    if (w[0] == 0 and w[1] == 0 and w[2] == 0):
        r[0,0] = 1
        r[1,1] = 1
        r[2,2] = 1
        r[0,3] = vt[0]
        r[1,3] = vt[1]
        r[2,3] = vt[2]
    else:
        rm = rotation_3d(w, theta)
        I_sub_rm = np.identity(3) - rm
        wh_times_v = np.matmul(wh, v)
        w_squared = np.outer(w, np.transpose(w))
        w2_times_vt = np.matmul(w_squared, vt)
        temp = np.matmul(I_sub_rm, wh_times_v) + w2_times_vt
        ivn = 1 / np.linalg.norm(w)**2
        temp = temp * ivn
        for i in range(len(rm)):
            for j in range(len(rm[i])):
                r[i][j] = rm[i][j]

        for k in range(len(temp)):
            r[k, 3] = temp[k]

    r[3][3] = 1

    return r



def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given
    the twists and displacements for each joint.
    Args:
    xi - (6, N) ndarray: the twists for each joint
    theta - (N,) ndarray: the displacement of each joint
    Returns:
    g - (4,4) ndarray: the resulting homogeneous transformation matrix
    """

    # YOUR CODE HERE
    r = np.identity(4)
    for i in range(len(theta)):
        r = np.matmul(r, homog_3d(xi[:,i], theta[i]))

    return r

def adjoint(omega, theta, translation):
    """
    Computes the adjoint representation of a homogeneous transformation matrix
    Args:
    omega - (3,) ndarray: the rotation axis
    theta               : the rotation displacement
    translation - (3,) ndarray: the displacement between the frame
    Returns:
    Adj - (6,6) ndarray: the resulting homogeneous transformation matrix
    """

    # YOUR CODE HERE
    R = rotation_3d(omega, theta)
    Right_top_corner = np.matmul(skew_3d(translation), R)
    Right_bottom_corner = R

    Adj = np.zeros((6,6))
    Adj[0:3,0:3] = R
    Adj[0:3,3:6] = Right_top_corner
    Adj[3:6,3:6] = Right_bottom_corner

    return Adj

# --------------- Converts Spherical Coordinates to XYZ Coordinates -------------- #
def spherical_to_cartesian(spr):
    phi, theta, rho = spr
    x = rho * math.sin(theta) * math.cos(phi)
    y = rho * math.sin(theta) * math.sin(phi)
    z = rho * math.cos(theta)
    return np.array([x, y, z])



# --------------- Converts Spherical Coordinates list to XYZ Coordinates -------------- #
def spherical2xyz(spr_arr):
    def spherical_to_cartesian(spr):
        phi, theta, rho = spr
        x = rho * math.sin(theta) * math.cos(phi)
        y = rho * math.sin(theta) * math.sin(phi)
        z = rho * math.cos(theta)
        return [x, y, z]
    # Apply Cartesian Conversion on each point
    return list(map(lambda x: spherical_to_cartesian(x), spr_arr))


# ------ Converts pointclouds(in camera frame) to spherical Coordinates ------ # (Spherical coordinates following Wikipedia definition here: https://en.wikipedia.org/wiki/Spherical_coordinate_system)
# Output points are in degrees
def xyz2spherical(ptc_arr):
    def cartesian_to_spherical(xyz):
        x, y, z = xyz # in camera frame
        rho   = math.sqrt(x**2 + y**2 + z**2)
        r     = math.sqrt(x**2 + y**2)
        
        # Determine theta
        if z>0:
            theta = math.atan(r/z)
        elif z<0:
                theta = math.pi + math.atan(r/z)
        elif z==0 and x!=0 and y!=0:
                theta = math.pi/2
        elif x==0 and y==0 and z==0:
                theta = None
        else:
                theta = math.acos(z/rho)
        
        # Determine Phi
        if x>0:
            phi = math.atan(y/x)
        elif x<0 and y>=0:
            phi = math.atan(y/x)
        elif x<0 and y<0:
            phi = math.atan(y/x) - math.pi
        elif x==0 and y>0:
            phi = math.pi
        elif x<0 and y<0:
            phi = math.pi
        elif x == 0 and y == 0:
            phi = None
        else:
            phi = (0 if y==0 else 1 if x > 0 else -1)*math.acos(x/r)

        return [phi, theta, rho]

    # Apply Spherical Conversion on each point
    return list(map(lambda x: cartesian_to_spherical(x), ptc_arr))

