import sys, getopt
import csv
import numpy as np
import scipy as sc
import kinematics

def readSeg(fname):
    f = open(fname, 'r')
    csv_f = csv.reader(f)
    seg = []
    for row in csv_f:
        seg.append([float(row[0]),float(row[1])])
    return seg

def saveSeg(fname, data):
    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    return

def sigma_points(x, P):
    n = 4.0
    q = sc.linalg.sqrtm(n*P)
    q_wig = np.concatenate((q[0].reshape(4,1),q[1].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,q[2].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,q[3].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,-q[0].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,-q[1].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,-q[2].reshape(4,1)),1)
    q_wig = np.concatenate((q_wig,-q[3].reshape(4,1)),1)
    q_hat = x + q_wig
    return q_hat, q_wig

def h2_fcn_eval(x):
    c = len(x[0])
    h = x[:2,0].reshape(2,1)
    
    for k in range(1,c):
        h = np.concatenate((h,x[:2,k].reshape(2,1)),1)
    return h
    
def f2_fcn_eval(x,u):
    c = len(x[0])
    dT = 1.0 / 30.0
    F = np.array([[1, 0, dT, 0],
                  [0, 1, 0, dT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
                  
    G = np.array([[dT*dT/2, 0],
                  [0, dT*dT/2],
                  [dT, 0],
                  [0, dT]])
                  
    f = np.dot(F,x[:,0].reshape(4,1)) + np.dot(G,u)
    
    for k in range(1,c):
        fval = np.dot(F,x[:,k].reshape(4,1)) + np.dot(G,u)
        f = np.concatenate((f,fval),1)    
    return f
    
def main(argv):
    # Any mention of pages or equations in this Unscented Kalman Filter
    # refer to the book "Optimal State Estimation" by Dan Simon, 2006,
    # John Wiley & Sons, ISBN-10 0-471-70858-5
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'unscKF.py <inputfile>'
        sys.exit(2)
    
    fname = args[0]
    outputfile = fname[:-4] + '_ukf'+ fname[-4:]
    data = readSeg(fname)

    n = len(data)
    n = len(data)
    dT = 1.0 / 30.0
    
    # Variances, estimated from data post-processing
    varX = 46.2704      # Variance in x
    varY = 49.9576      # Variance in y
    varAx = 85954.0     # Variance in x-accel
    varAy = 89005.0     # Variance in y-accel

    R = np.array([[varX, 0],
                  [0, varY]])

    Q = np.array([[dT**4/4, 0, 0, 0],
                  [0, dT**4/4, 0, 0],
                  [0, 0, dT**2, 0],
                  [0, 0, 0, dT**2]])

    Q *= np.array([[varAx, 0, 0, 0],
                   [0, varAy, 0, 0],
                   [0, 0, varAx, 0],
                   [0, 0, 0, varAy]])
                 
    x_pos = np.array([data[0][0],data[0][1],0,0]).reshape(4,1)
    P_pos = Q * 2
    R *= 0.5
    
    alpha = 1.005
    
    x_est = [float(x_pos[0])]
    y_est = [float(x_pos[1])]
    
    for k in range(1,n):        
        LastPosn = np.array([float(x_pos[0]), float(x_pos[1])]).reshape(2,1)
        # Fading memory filter, pg 210
        if (k <= 150):
            Qk = alpha**(-2*k-2) * Q
            Rk = alpha**(-2*k) * R
        
        u = np.array([0.0,0.0]).reshape(2,1)                # Could add noise here
        x_hat, x_wig = sigma_points(x_pos, P_pos)           # Eqn 14.58
        x_neg_sigma = f2_fcn_eval(x_hat,u);                 # Eqn 14.59
        x_neg = np.sum(x_neg_sigma,axis=1).reshape(4,1) / 8   # Eqn 14.60
        
        P_neg = np.zeros((4,4))
        for j in range(8):
            z = x_neg_sigma[:,j].reshape(4,1) - x_neg
            P_neg = P_neg + z*z.transpose()
        P_neg = P_neg / 8 + Qk                              # Eqn 14.61
        
        x_hat, x_wig = sigma_points(x_neg,P_neg)            # Eqn 14.62
        y_sigma = h2_fcn_eval(x_hat)                        # Eqn 14.63
        y_hat = np.sum(y_sigma,axis=1).reshape(2,1) / 8     # Eqn 14.64
        
        P_y = np.zeros((2,2))
        P_xy = np.zeros((4,2))
        for j in range(8):
            z = y_sigma[:,j].reshape(2,1) - y_hat
            P_y = P_y + z*z.transpose()
            P_xy = P_xy + np.dot(x_wig[:,j].reshape(4,1),z.transpose())
        P_y = P_y / 8 + Rk;                                 # Eqn 14.65
        P_xy = P_xy / 8;                                    # Eqn 14.66
        K = np.dot(P_xy, np.linalg.inv(P_y))
        meas = np.array([data[k][0],data[k][1]]).reshape(2,1)
        resid = meas - y_hat
        x_pos = x_neg + np.dot(K, resid)
        x_pos = kinematics.bounce(x_pos, LastPosn)
        P_pos = P_neg - np.dot(K, np.dot(P_y, K.transpose())) # Eqn 14.67
        
        x_est.append(float(x_pos[0]))
        y_est.append(float(x_pos[1]))

    data_filt = [[x_est[i],y_est[i]] for i in range(n)]
    saveSeg(outputfile, data_filt)
    print '\nSaved ' + outputfile + '\n'
    return

if __name__ == "__main__":
    main(sys.argv[1:])

