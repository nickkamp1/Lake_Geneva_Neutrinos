import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import shapely.geometry


R_earth = 6371000.
Lake_geneva_elevation = 372.

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
    
    def tangent_line(self,x,crossing):
        # find tangent to circle at point x
        # assume x is actually on the circle
        # use derivative of sphere and plane equation to calculate direction
        delta = x - self.center
        dzdx = ((self.n[0]/self.n[1]) * delta[1] - delta[0]) / (delta[2] - self.n[2]/self.n[1] * delta[1])
        dydx = -self.n[2]/self.n[1] * dzdx - self.n[0]/self.n[1]
        dir = np.array([1,dydx,dzdx])
        unit_dir = -dir/np.linalg.norm(dir)
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
    dir = dir0 #only consider non-crossed beam for now
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
    
    
def calculate_intersections_with_lake(circle,x,crossing,lake_coordinates,limit=50000):
    point_lat_long = xyz_to_lat_long(*x)
    lake_polygon = shapely.geometry.Polygon([[p[0],p[1]] for p in lake_coordinates])
    dir,dir1,dir2 = circle.tangent_line(x,crossing)
    R = np.linalg.norm(x)
    dR_dt = 1./R * np.dot(x,dir)
    dlat_dt = 1./(R*np.sqrt(R**2 - x[2]**2)) * (R*dir[2] - x[2]*dR_dt)
    dlong_dt = 1./(x[0]**2 + x[1]**2) * (-x[1]*dir[0] + x[0]*dir[1])
    line1 = shapely.geometry.LineString([[point_lat_long[0], point_lat_long[1]],
                                        [point_lat_long[0] + dlat_dt*limit, point_lat_long[1] + dlong_dt*limit]])
    line2 = shapely.geometry.LineString([[point_lat_long[0], point_lat_long[1]],
                                         [point_lat_long[0] - dlat_dt*limit, point_lat_long[1] - dlong_dt*limit]])
    return line1.intersection(lake_polygon),line2.intersection(lake_polygon)
