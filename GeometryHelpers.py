import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import shapely.geometry


R_earth = 6371000.
Lake_geneva_elevation = 372.

# https://math.stackexchange.com/questions/1076177/3d-coordinates-of-circle-center-given-three-point-on-the-circle
def find_circle_center(A,B,C):
    u1 = B - A
    w1 = np.cross((C - A),u1)
    u = u1/np.linalg.norm(u1)
    w = w1/np.linalg.norm(w1)
    v = np.cross(w,u)
    bx = np.dot(B-A,u)
    cx = np.dot(C-A,u)
    cy = np.dot(C-A,v)
    h = ((cx - bx/2)**2 + cy**2 - (bx/2)**2)/(2*cy)
    return A + (bx/2)*u + h*v

class Circle:

    def __init__(self,x1,x2,x3): 
        # x1, x2, and x3 are three points on circle
        # define a 3D circle as the intersection of a sphere and a plane
        # plane eq: ax + by + cz + d = 0
        # sphere eq: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
        self.v1 = x1 - x2
        self.v2 = x1 - x3
        self.n = np.cross(self.v1,self.v2) # n = (a,b,c)
        self.d = - np.dot(self.n,x1)
        self.center = find_circle_center(x1,x2,x3)
        self.radius = np.linalg.norm(x1 - self.center)

    def find_elevation(self,lat,long):
        # find the elevation of the lat,long point on the circle
        # use the plane equation, double-check with sphere equation
        denom = self.n[0] * np.cos(math.radians(lat)) * np.cos(math.radians(long)) \
                + self.n[1] * np.cos(math.radians(lat)) * np.sin(math.radians(long)) \
                + self.n[2] * np.sin(math.radians(lat))
        elev = -self.d / denom - R_earth
        new_point = lat_long_to_xyz(lat,long,elev)
        new_r = np.linalg.norm(new_point - self.center)
        return elev,new_point
    
    def tangent_line(self,x,crossing=None):
        # find tangent to circle at point x
        # assume x is actually on the circle
        # use derivative of sphere and plane equation to calculate direction
        delta = x - self.center
        dzdx = ((self.n[0]/self.n[1]) * delta[1] - delta[0]) / (delta[2] - self.n[2]/self.n[1] * delta[1])
        dydx = -self.n[2]/self.n[1] * dzdx - self.n[0]/self.n[1]
        dir = np.array([1,dydx,dzdx])
        unit_dir = -dir/np.linalg.norm(dir)
        if crossing is None: return unit_dir
        if crossing[0] == 0: # vertical case
            rotation_axis = x - self.center
        elif crossing[0] == 1: # horizontal case:
            rotation_axis = np.cross(dir,x-self.center)
        else:
            return unit_dir, None, None # no crossing angle defined
        rotation_axis /= np.linalg.norm(rotation_axis)
        rot1 = Rotation.from_mrp(np.tan(crossing[1]/8.)*rotation_axis) # uses modified rodrigues parameters
        rot2 = Rotation.from_mrp(np.tan(-crossing[1]/8.)*rotation_axis) # uses modified rodrigues parameters
        return unit_dir, np.matmul(rot1.as_matrix(),unit_dir), np.matmul(rot2.as_matrix(),unit_dir)
        

def plot_tangent_line(circle,x,crossing,limit=10000,Lake_Crossings=None,label=None):
    
    fig, axs = plt.subplots(3,sharex=True)
    fig.set_size_inches(8,10)
    axs[2].set_xlabel('Distance from interaction Point [m]')
    dir0,dir1,dir2 = circle.tangent_line(x,crossing)
    if(type(limit)==list):
        trange = np.linspace(limit[0],limit[1],2*1000)
    else: 
        trange = np.linspace(-limit,limit,2*1000)
    dir = dir0 #only consider unshifted beam for now
    points = x.reshape(-1,1) + np.outer(dir,trange)
    FASER_envelope = np.abs(0.125/480*trange)
    earth_points = np.array([xyz_to_lat_long(*p) for p in points.transpose()])
    axs[0].plot(trange,earth_points[:,0],color='black')
    axs[1].plot(trange,earth_points[:,1],color='black')
    elev_range = earth_points[:,2] - Lake_geneva_elevation
    axs[2].plot(trange,elev_range,color='black',label=label)
    axs[2].fill_between(trange, elev_range - FASER_envelope, elev_range + FASER_envelope, color='black', alpha = 0.2, label = 'FASER envelope')
    

    # Plot lake crossings:
    pairs  = [[Lake_Crossings[i],Lake_Crossings[i+1]] for i in range(0,len(Lake_Crossings),2)]
    for i,pair in enumerate(pairs):
        diffs = (np.linalg.norm(np.array(pair[0])[None,:] - earth_points[:,0:2],axis=1),
                 np.linalg.norm(np.array(pair[1])[None,:] - earth_points[:,0:2],axis=1))
        
        t_intersects = (trange[np.argmin(diffs[0])],
                        trange[np.argmin(diffs[1])])
        axs[0].axvspan(t_intersects[0],t_intersects[1],color='blue',alpha=0.2)
        axs[1].axvspan(t_intersects[0],t_intersects[1],color='blue',alpha=0.2)
        X = np.linspace(t_intersects[0],t_intersects[1],100)
        axs[2].fill_between(X,-1e5 * np.ones_like(X), np.zeros_like(X),color='blue',alpha=0.2,label='Lake Geneva' if i==0 else None)
    
    fig.subplots_adjust(hspace=0)
    axs[2].legend()
    axs[2].set_ylim(np.min(earth_points[:,2] - Lake_geneva_elevation),
                    np.max(earth_points[:,2] - Lake_geneva_elevation))
    axs[0].set_ylabel('Latitude [deg]')
    axs[1].set_ylabel('Longitude [deg]')
    axs[2].set_ylabel('Elevation w.r.t. Lake [m]')

def plot_tangent_elevation(circle,x,crossing,limit=10000,Lake_Crossings=None,IPlabel=None,color="black",detector_position=10000):

    fig = plt.figure()
    fig.set_size_inches(9,6)
    dir0,dir1,dir2 = circle.tangent_line(x,crossing)
    if(type(limit)==list):
        trange = np.linspace(limit[0],limit[1],2*1000)
    else: 
        trange = np.linspace(-limit,limit,2*1000)
    dir = dir0 #only consider unshifted beam for now
    points = x.reshape(-1,1) + np.outer(dir,trange)
    FASER_envelope = np.abs(0.125/480*trange)
    earth_points = np.array([xyz_to_lat_long(*p) for p in points.transpose()])
    elev_range = earth_points[:,2] - Lake_geneva_elevation
    
    pipe_detector_radius = 5
    pipe_detector_length = 500
    pipe_detector_mask = np.logical_and(trange>detector_position-pipe_detector_length/2,trange<detector_position+pipe_detector_length/2)
    plt.fill_between(trange[pipe_detector_mask], elev_range[pipe_detector_mask] - pipe_detector_radius, elev_range[pipe_detector_mask] + pipe_detector_radius, color="black", alpha = 0.5, label = 'Pipe Detector')
    plt.plot(trange[pipe_detector_mask],elev_range[pipe_detector_mask] - pipe_detector_radius,color="black",lw=3)
    plt.plot(trange[pipe_detector_mask],elev_range[pipe_detector_mask] + pipe_detector_radius,color="black",lw=3)

    panel_detector_radius = 10
    plt.plot(18210.36*np.ones(2),[-panel_detector_radius,panel_detector_radius],color="black",label="Panel Detector")
    

    # Plot lake crossings:
    pairs  = [[Lake_Crossings[i],Lake_Crossings[i+1]] for i in range(0,len(Lake_Crossings),2)]
    prev_edge = trange[0]
    for i,pair in enumerate(pairs):
        diffs = (np.linalg.norm(np.array(pair[0])[None,:] - earth_points[:,0:2],axis=1),
                 np.linalg.norm(np.array(pair[1])[None,:] - earth_points[:,0:2],axis=1))
        
        t_intersects = (trange[np.argmin(diffs[0])],
                        trange[np.argmin(diffs[1])])
        X = np.linspace(t_intersects[0],t_intersects[1],2)
        plt.fill_between(X,-1e5 * np.ones_like(X), np.zeros_like(X),color='blue',alpha=0.4,label='Lake Geneva' if i==0 else None)
        X = np.linspace(prev_edge,t_intersects[0],2)
        plt.fill_between(X,-1e5 * np.ones_like(X), np.zeros_like(X),color=(0,1,0,0.2),label='Land' if i==0 else None)
        prev_edge = t_intersects[1]
    X = np.linspace(trange[0],trange[-1],2)
    plt.fill_between(X,np.zeros_like(X),np.max(elev_range),color="lightskyblue",alpha=0.2)

    plt.scatter([0],[xyz_to_lat_long(*x)[2] - Lake_geneva_elevation],marker='*',s=500,color=color,edgecolors="black",label="%s Interaction Point"%IPlabel,zorder=10)
    plt.plot(trange,elev_range,color=color,label="Beam from %s"%IPlabel)
    plt.fill_between(trange, elev_range - FASER_envelope, elev_range + FASER_envelope, color=color, alpha = 0.3, label = 'FASER envelope')
    
    plt.legend(loc="upper left",ncol=2)
    plt.xlim(trange[0],trange[-1])
    plt.ylim(np.min(earth_points[:,2] - Lake_geneva_elevation)-10,
                    np.max(earth_points[:,2] - Lake_geneva_elevation))
    plt.xlabel('Distance from interaction Point [m]')
    plt.ylabel('Elevation w.r.t. Lake [m]')


def plot_crossing_angle_impact(circle,x,crossing_angle,limit=10000,Lake_Crossings=None,label=None):
    fig = plt.figure(figsize=(6,4))
    dir = circle.tangent_line(x)
    if(type(limit)==list):
        trange = np.linspace(limit[0],limit[1],2*1000)
    else: 
        trange = np.linspace(-limit,limit,2*1000)
    points = x.reshape(-1,1) + np.outer(dir,trange)
    earth_points = np.array([xyz_to_lat_long(*p) for p in points.transpose()])
    displacement = np.tan(crossing_angle) * trange
    plt.plot(trange,displacement,color='black',label=label)

    # Plot lake crossings:
    pairs  = [[Lake_Crossings[i],Lake_Crossings[i+1]] for i in range(0,len(Lake_Crossings),2)]
    for i,pair in enumerate(pairs):
        diffs = (np.linalg.norm(np.array(pair[0])[None,:] - earth_points[:,0:2],axis=1),
                 np.linalg.norm(np.array(pair[1])[None,:] - earth_points[:,0:2],axis=1))
        
        t_intersects = (trange[np.argmin(diffs[0])],
                        trange[np.argmin(diffs[1])])
        plt.axvspan(t_intersects[0],t_intersects[1],color='blue',alpha=0.2,label='Lake Geneva' if i==0 else None)

    plt.legend()
    plt.xlabel('Distance from interaction Point [m]')
    plt.ylabel('Displacement from Crossing Angle [m]')




def lat_long_to_xyz(lat,long,elevation,
                    lat_0 = 46.26,
                    long_0 = 6.05):
    R = R_earth + elevation
    #r = Rotation.from_euler('zy',[long_0,lat_0],degrees=True)
    point = np.array([
        R * math.cos(math.radians(lat)) * math.cos(math.radians(long)),
        R * math.cos(math.radians(lat)) * math.sin(math.radians(long)),
        R * math.sin(math.radians(lat))]
    )
    return point#np.matmul(r.as_matrix(),point)

def xyz_to_lat_long(x,y,z):
    R = np.linalg.norm(np.array([x,y,z]))
    lat = math.degrees(np.arcsin(z/R))
    long = math.degrees(np.arctan2(y,x))
    return lat,long,R-R_earth

def equirectangular(lat,long,elevation,phi_0):
    R = R_earth + elevation
    return (
        R * long * math.cos(math.radians(phi_0)),
        R * lat,
        elevation,
    )

          

def plot_tangent_line_lat_long(circle,x,crossing,limit=10000,N=1000,**kwargs):
    
    dir0,dir1,dir2 = circle.tangent_line(x,crossing)
    lines = []
    for dir in [dir0,dir1,dir2]:
        if dir is None: continue
        trange = np.linspace(-limit,limit,2*N+1)
        points = x.reshape(-1,1) + np.outer(dir,trange)
        earth_points = np.array([xyz_to_lat_long(*p) for p in points.transpose()])
        lines.append([earth_points[N],earth_points])
    return lines


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
    
def rotation_to_beam_direction(beam_dir):

    # Follow https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(beam_dir, [0,0,1])
    s = np.linalg.norm(v)
    c = np.dot(beam_dir, [0,0,1])
    
    # use skey symmetric cross product matrix
    vx = skew(v)

    # Construct rotation matrix
    R = np.identity(3) + vx + np.matmul(vx,vx)*(1-c)/(s**2)
    
    # We want to rotate the beam away from [0,0,1] to the true direction
    return np.linalg.inv(R)

def calculate_single_lake_intersection(x,dir,lake_coordinates,limit=50000):
    point_lat_long = xyz_to_lat_long(*x)
    lake_polygon = shapely.geometry.Polygon([[p[0],p[1]] for p in lake_coordinates])
    R = np.linalg.norm(x)
    dR_dt = 1./R * np.dot(x,dir)
    dlat_dt = 1./(R*np.sqrt(R**2 - x[2]**2)) * (R*dir[2] - x[2]*dR_dt)
    dlong_dt = 1./(x[0]**2 + x[1]**2) * (-x[1]*dir[0] + x[0]*dir[1])
    line1 = shapely.geometry.LineString([[point_lat_long[0], point_lat_long[1]],
                                        [point_lat_long[0] + dlat_dt*limit, point_lat_long[1] + dlong_dt*limit]])
    line2 = shapely.geometry.LineString([[point_lat_long[0], point_lat_long[1]],
                                         [point_lat_long[0] - dlat_dt*limit, point_lat_long[1] - dlong_dt*limit]])
    return line1.intersection(lake_polygon),line2.intersection(lake_polygon)
    
    
def calculate_intersections_with_lake(circle,x,crossing,lake_coordinates,particle_unit_dirs=None,limit=50000):
    dir,dir1,dir2 = circle.tangent_line(x,crossing)
    if particle_unit_dirs is not None:
        # rotate particle dir wrt beam axis to global frame
        # first get rotation matrix
        R = rotation_to_beam_direction(dir)
        # then do rotations
        int1_list,int2_list = [],[]
        for particle_unit_dir in particle_unit_dirs:
            dir = np.dot(R,particle_unit_dir)
            int1,int2 = calculate_single_lake_intersection(x,dir,lake_coordinates,limit=limit)
            int1_list.append(int1)
            int2_list.append(int2)
        return int1_list,int2_list
    else:
        return calculate_single_lake_intersection(x,dir,lake_coordinates,limit=limit)
    
def calculate_intersections_with_surface(beam_pos,
                                         beam_dir,
                                         particle_positions,
                                         particle_unit_dirs,
                                         limit=500000,
                                         particle_position_beam_coordinates=True):
    R = rotation_to_beam_direction(beam_dir)
    trange = np.linspace(0,limit,50000)
    surface_intersections = []
    surface_intersections_lat_long = []
    for particle_position,particle_unit_dir in zip(particle_positions,
                                                   particle_unit_dirs):
        particle_dir = np.dot(R,particle_unit_dir)
        if particle_position_beam_coordinates:
            # we must translate particle positions to global coordinate frame
            norm = np.linalg.norm(particle_position)
            if norm>0: particle_unit_position = particle_position/np.linalg.norm(particle_position)
            else: particle_unit_position = particle_position
            particle_origin_unit_dir = np.dot(R,particle_unit_position)
            particle_position = beam_pos + norm*particle_origin_unit_dir
        xrange = particle_position.reshape(-1,1) + np.outer(particle_dir,trange)
        Rrange = np.linalg.norm(xrange,axis=0)
        crossing_idx = np.argmin(np.abs(Rrange - (R_earth+Lake_geneva_elevation)))
        surface_intersections.append(xrange.T[crossing_idx])
        surface_intersections_lat_long.append(xyz_to_lat_long(*xrange.T[crossing_idx]))
    return np.array(surface_intersections),np.array(surface_intersections_lat_long)
        