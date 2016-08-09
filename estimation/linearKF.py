import sys, getopt
import csv
import numpy as np
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

def main(argv):
    # Any mention of pages or equations in this Kalman Filter refer
    # to the book "Optimal State Estimation" by Dan Simon, 2006,
    # John Wiley & Sons, ISBN-10 0-471-70858-5
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'linearKF.py <inputfile>'
        sys.exit(2)
    
    fname = args[0]
    outputfile = fname[:-4] + '_lkf'+ fname[-4:]
    data = readSeg(fname)

    n = len(data)
    dT = 1.0 / 30.0
    F = np.array([[1, 0, dT, 0],
                  [0, 1, 0, dT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
                  
    G = np.array([[dT*dT/2, 0],
                  [0, dT*dT/2],
                  [dT, 0],
                  [0, dT]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    I4 = np.eye(4)
    Ft = F.transpose()
    Ht = H.transpose()
    
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
                   [0, 0, 0, varAy]])   # Eqn 5.17
                 
    x_pos = np.array([data[0][0],data[0][1],0,0]).reshape(4,1)
    P_pos = Q * 10                      # Eqn 5.18
    
    alpha = 1.005
    
    x_est = [float(x_pos[0])]
    y_est = [float(x_pos[1])]
    
    for k in range(1,n):
        
        LastPosn = np.array([float(x_pos[0]), float(x_pos[1])]).reshape(2,1)
        # Fading memory filter, pg 210
        if (k <= 150):
            Qk = alpha**(-2*k-2) * Q
            Rk = alpha**(-2*k) * R
        
        # KF equations, pg 128
        P_neg = np.dot(np.dot(F,P_pos),Ft) + Qk
        inv = np.linalg.inv(np.dot(np.dot(H,P_neg),Ht) + Rk)
        K = np.dot(np.dot(P_neg,Ht),inv)
        u = np.array([0.0,0.0]).reshape(2,1)
        x_neg = np.dot(F,x_pos) + np.dot(G,u)
        z = np.array([data[k][0],data[k][1]]).reshape(2,1)
        resid = z - np.dot(H,x_neg)
        x_pos = x_neg + np.dot(K,resid)
        x_pos = kinematics.bounce(x_pos, LastPosn)
        P_pos = np.dot((I4 - np.dot(K,H)), P_neg)       # Eqn 5.19
        x_est.append(float(x_pos[0]))
        y_est.append(float(x_pos[1]))

    data_filt = [[x_est[i],y_est[i]] for i in range(n)]
    saveSeg(outputfile, data_filt)
    print '\nSaved ' + outputfile + '\n'
    return

if __name__ == "__main__":
    main(sys.argv[1:])
    
