import numpy as np
import math

def ptDist(pt1, pt2):
    d = pt1 - pt2
    return math.sqrt(np.dot(d.transpose(),d))
    
def ptImpact(m, b, r_candle, cdl_ctr, pt0):
    A = (1.0 + m**2)
    B = 2 * (m * (b - cdl_ctr[1]) - cdl_ctr[0])
    C = ((b - cdl_ctr[1])**2 + cdl_ctr[0]**2 - r_candle**2)
    poi_x = np.roots(np.array([A,B,C]).reshape(3))
    poi_y = m * poi_x + b
    pt_a = np.array([poi_x[0],poi_y[0]]).reshape(2,1)
    pt_b = np.array([poi_x[1],poi_y[1]]).reshape(2,1)
    dist1 = ptDist(pt0,pt_a)
    dist2 = ptDist(pt0,pt_b)    
    if (dist1 < dist2):
        return pt_a
    else:
        return pt_b

def ptLine(pt1, pt2):
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]);  # check for 1/0
    b = pt1[1] - m * pt1[0];
    return m, b

def bounce(StateVect, LastPosn):
    
    # LastPosn = np.array([x_prev, y_prev]).reshape(2,1)
    # Estimated positions
    x_k1  = StateVect[0]    # next predicted x
    y_k1  = StateVect[1]    # next predicted y
    
    # Estimated velocities
    # If using velocity magnitude and heading,
    # Vx = V * cos(hdg)
    # Vy = V * sin(hdg)
    vel_x = StateVect[2]    # next predicted Vx
    vel_y = StateVect[3]    # next predicted Vy
    
    x_left = 81
    x_right = 563
    y_top = 325
    y_bot = 36
    r_candle = 38
    cdl_ctr = np.array([330,178]).reshape(2,1)
    
    ## Check for edges and corners
    rho_p = 0.5                 # Posn restitution coeff, bug-box collision
    rho_v = 0.9                 # Vel restitution coeff, bug-box collision
    
    if (x_k1 < x_left):
        x_k1 = x_left + rho_p * (x_left - x_k1)
        vel_x *= -rho_v
    elif (x_k1 > x_right):
        x_k1 = x_right + rho_p * (x_right - x_k1)
        vel_x *= -rho_v


    if (y_k1 < y_bot):
        y_k1 = y_bot + rho_p * (y_bot - y_k1)
        vel_y *= -rho_v
    elif (y_k1 > y_top):
        y_k1 = y_top + rho_p * (y_top - y_k1)
        vel_y *= -rho_v

    ## Check for candle
    rho_p = 0.5                 # Posn restitution coeff, bug-candle collision
    rho_v = 0.9                 # Vel restitution coeff, bug-candle collision

    pt2 = np.array([x_k1,y_k1]).reshape(2,1)
    if (ptDist(pt2, cdl_ctr) < r_candle):
        m, b = ptLine(LastPosn, pt2)
        poi = ptImpact(m, b, r_candle, cdl_ctr, LastPosn)
        d_t = ptDist(poi, cdl_ctr) - ptDist(pt2, cdl_ctr) # always > 0
        d_c = ptDist(poi, pt2)
        d_x = math.sqrt(d_c**2 - d_t**2)
        rad_new_pt = math.sqrt((r_candle + rho_p * d_t)**2 + (rho_p * d_x)**2)
        
        phi = math.atan2(poi[1]-cdl_ctr[1],poi[0]-cdl_ctr[0])
        if (phi < 0.0):
            phi += 2.0 * math.pi
        
        phi2 = math.atan2(pt2[1]-cdl_ctr[1],pt2[0]-cdl_ctr[0])
        if (phi2 < 0.0):
            phi2 += 2.0 * math.pi
        
        theta = math.atan2(rho_p * d_x, r_candle + rho_p * d_t)
        if (theta < 0.0):
            theta += 2.0 * math.pi
        
        if (phi > phi2):
            theta = -theta
        
        x_k1 = cdl_ctr[0] + rad_new_pt * math.cos(phi + theta)
        y_k1 = cdl_ctr[1] + rad_new_pt * math.sin(phi + theta)
        angl = 2.0 * theta - math.pi
        vel_x =  StateVect[2] * math.cos(angl) + StateVect[3] * math.sin(angl)
        vel_y = -StateVect[2] * math.sin(angl) + StateVect[3] * math.cos(angl)
        vel_x *= -rho_v
        vel_y *= rho_v
        
return np.array([x_k1, y_k1, vel_x, vel_y]).reshape(4,1)

