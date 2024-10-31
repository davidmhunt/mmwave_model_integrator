import numpy as np

def cartesian_to_spherical(points_cart:np.ndarray):
    """Convert an array of points stored as (x,y,z) to (range,azimuth, elevation).
    Note that azimuth = 0 degrees for points on the positive x-axis

    Args:
        points_cart (np.ndarray): Nx3 matrix of points in cartesian (x,y,z)

    Returns:
        (np.ndarray): Nx3 matrix of points in spherical (range, azimuth, elevation) in radians
    """
    ranges = np.sqrt(points_cart[:, 0]**2 + points_cart[:, 1]**2 + points_cart[:, 2]**2)
    azimuths = np.arctan2(points_cart[:, 1], points_cart[:, 0])
    elevations = np.arccos(points_cart[:, 2] / ranges)

    return  np.column_stack((ranges,azimuths,elevations))
    
def spherical_to_cartesian(points_spherical:np.ndarray):
    """Convert an array of points stored as (range, azimuth, elevation) to (x,y,z)

    Args:
        points_spherical (np.ndarray): Nx3 matrix of points in spherical (range,azimuth, elevation)

    Returns:
        (np.ndarray): Nx3 matrix of  points in cartesian (x,y,z)
    """

    x = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.cos(points_spherical[:,1])
    y = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.sin(points_spherical[:,1])
    z = points_spherical[:,0] * np.cos(points_spherical[:,2])


    return np.column_stack((x,y,z))

def polar_to_cartesian(points_polar:np.ndarray)->np.ndarray:

    """Convert an array of points stored as (range, azimuth) to (x,y)

    Args:
        points_polar (np.ndarray): Nx2 matrix of points in spherical (range,azimuth)

    Returns:
        (np.ndarray): Nx2 matrix of  points in cartesian (x,y)
    """

    x = points_polar[:,0] *  np.cos(points_polar[:,1])
    y = points_polar[:,0] *  np.sin(points_polar[:,1])


    return np.column_stack((x,y))

def cartesian_to_polar(points_cart:np.ndarray)->np.ndarray:

    """Convert an array of points stored as (x, y) to (range,azimuth)

    Args:
        points_cart (np.ndarray): Nx2 matrix of points in cartesian (x,y)

    Returns:
        (np.ndarray): Nx2 matrix of  points in polar (range,azimuth)
    """

    ranges = np.sqrt(points_cart[:, 0]**2 + points_cart[:, 1]**2)
    azimuths = np.arctan2(points_cart[:, 1], points_cart[:, 0])


    return np.column_stack((ranges,azimuths))