import numpy as np
import CoordFrame

def g02(t, v2, R2):
    '''
    Problem (b)
        Returns the 4x4 rigid body pose g_02 at time t given that the
        satellite is orbitting at radius R2 and linear velocity v2.

        Args:
            t: time at which the configuration is computed.
            v2: linear speed of the satellite, in meters per second.
            R2: radius of the orbit, in meters.
        Returns:
            4x4 rigid pose of frame {2} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.sin
            numpy.cos
            numpy.array
            numpy.sqrt
            numpy.matmul
            numpy.eye

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    teta = (v2/R2)*t 
    # might be wrong
    g = np.array([[np.sin(teta), 0, -np.cos(teta), -R2*np.sin(teta)], 
                  [-np.cos(teta), 0, -np.sin(teta), R2*np.cos(teta)], 
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]) # unsure
    return g

def g01(t, v1, R1):
    '''
    Problem (c)
        Returns the 4x4 rigid body pose g_01 at time t given that the
        satellite is orbitting at radius R1 and linear velocity v1.

        Args:
            t: time at which the configuration is computed.
            v1: linear speed of the satellite, in meters per second.
            R1: radius of the orbit, in meters.
        Returns:
            4x4 rigid pose of frame {1} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.sin
            numpy.cos
            numpy.array
            numpy.sqrt
            numpy.matmul
            numpy.eye

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    
    teta = (v1/R1)*t
    
    t_2 = np.deg2rad(30) # deg or radians?
    g = np.array([[np.sin(teta), 0, -np.cos(teta), -R1*np.sin(teta)], 
                  [-np.cos(t_2)*np.cos(teta), -np.sin(t_2), -np.sin(teta)*np.cos(t_2), R1*np.cos(teta)*np.cos(t_2)], 
                  [-np.sin(t_2)*np.cos(teta), np.cos(t_2), -np.sin(t_2)*np.sin(teta), R1*np.cos(teta)*np.sin(t_2)],  
                  [0, 0, 0, 1]]) # unsure   
    return g

def g21(t, v1, R1, v2, R2):
    '''
    Problem (d)
        Returns the 4x4 rigid body pose g_21 at time t given that the
        first satellite is orbitting at radius R1 and linear velocity v1
        and the second satellite is orbitting at radius R2 and linear 
        velocity v2.

        Args:
            t: time at which the configuration is computed.
            v1: linear speed of satellite 1, in meters per second.
            R1: radius of the orbit of satellite 1, in meters.
            v2: linear speed of satellite 2, in meters per second.
            R2: radius of the orbit of satellite 2, in meters.
        Returns:
            4x4 rigid pose of frame {2} as seen from frame {0} as a numpy array.

        Functions you might find useful:
            numpy.matmul
            numpy.linalg.inv

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    g_20 = np.linalg.inv( g02(t,v2,R2) )
    return np.matmul(g_20, g01(t,v1,R1)) 

def xi02(v2, R2):
    '''
    Problem (e)
        Returns the 6x1 twist describing the motion of satellite 2
        given that it is rotating at radius R2 with linear velocity v2.

        Args:
            v2: linear speed of satellite 2, in meters per second.
            R2: radius of the orbit of satellite 2, in meters.
        Returns:
            6x1 twist describing the motion of frame {2} relative to frame {0}

        Functions you might find useful:
            numpy.array

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    xi = np.array([
        0,
        0,
        0,
        0,
        0,
        1
    ])
    return xi

def xi01(v1, R1):
    '''
    Problem (f)
        Returns the 6x1 twist describing the motion of satellite 1
        given that it is rotating at radius R1 with linear velocity v1.

        Args:
            v1: linear speed of satellite 1, in meters per second.
            R1: radius of the orbit of satellite 1, in meters.
        Returns:
            6x1 twist describing the motion of frame {1} relative to frame {0}

        Functions you might find useful:
            numpy.array

        Note: Feel free to use one or more of the other functions you have implemented
        in this file.
    '''
    # degrees or radians??
    t = np.deg2rad(30)
    
    xi = np.array([
        0,
        0,
        0,
        0,
        -np.sin(t),
        np.cos(t)
    ])
    return xi