# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

# Load and Read Data
data = pd.read_csv("pc1.csv",header=None)
x = np.array(data[0])
y = np.array(data[1])
z = np.array(data[2])
l = len(data)

mean_x = sum(x)/l
mean_y = sum(y)/l
mean_z = sum(z)/l

x_var = sum((x-mean_x)**2)/l
y_var = sum((y-mean_y)**2)/l
z_var = sum((z-mean_z)**2)/l
xy = sum((x-mean_x)*(y-mean_y))/l
yz = sum((y-mean_y)*(z-mean_z))/l
zx = sum((z-mean_z)*(x-mean_x))/l

C = np.array([[x_var,xy,zx],
              [xy,y_var,yz],
              [zx,yz,z_var]])
print(C)

eigenvalues, eigenvectors = np.linalg.eig(C)
eigenvalue_idx = np.argmin(eigenvalues)
normal = eigenvectors[:,eigenvalue_idx]
magnitude = np.linalg.norm(normal,ord=2)

print("normal: ",normal)
print("magnitude: ",magnitude)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,color='r')
# Length of the normal is scaled for visualization
ax.quiver(0,0,0,normal[0],normal[1],normal[2],color='g',length=10)
ax.set_title("normal for pc1.csv")
plt.show()


## Least Square Fitting
# pc1.csv

# For the equation of the plane aX + bY + c = Z
ones = np.ones(l)

A = np.column_stack([x,y,ones])
B = z

A_inv = np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)
coeff = np.dot(A_inv,B)
print("coefficents: ", coeff)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = coeff[0] * X[r,c] + coeff[1] * Y[r,c] + coeff[2]

ax.plot_wireframe(X,Y,Z, color='g', alpha=0.4)
ax.set_title("least sq fit for pc1")
plt.show()

# pc2.csv
data2 = pd.read_csv("pc2.csv",header=None)
x2 = np.array(data2[0])
y2 = np.array(data2[1])
z2 = np.array(data2[2])
l2 = len(data2)

mean_x2 = sum(x2)/l2
mean_y2 = sum(y2)/l2
mean_z2 = sum(z2)/l2

x_var2 = sum((x2-mean_x2)**2)/l2
y_var2 = sum((y2-mean_y2)**2)/l2
z_var2 = sum((z2-mean_z2)**2)/l2
xy2 = sum((x2-mean_x2)*(y2-mean_y2))/l2
yz2 = sum((y2-mean_y2)*(z2-mean_z2))/l2
zx2 = sum((z2-mean_z2)*(x2-mean_x2))/l2

C2 = np.array([[x_var2,xy2,zx2],
              [xy2,y_var2,yz2],
              [zx2,yz2,z_var2]])
print(C2)

# For the equation of the plane aX + bY + c = Z
ones2 = np.ones(l2)

A2 = np.column_stack([x2,y2,ones2])
B2 = z2

A_inv2 = np.dot(np.linalg.inv(np.dot(A2.T,A2)),A2.T)
coeff2 = np.dot(A_inv2,B2)
print("coeffiecents: ", coeff2)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(x2,y2,z2,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

X2,Y2 = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z2 = np.zeros(X2.shape)
for r in range(X2.shape[0]):
    for c in range(X2.shape[1]):
        Z2[r,c] = coeff2[0] * X2[r,c] + coeff2[1] * Y2[r,c] + coeff2[2]

ax.plot_wireframe(X2,Y2,Z2, color='g', alpha=0.4)
ax.set_title("least sq fit for pc2")
plt.show()

## Total Least Square Fitting
# pc1.csv

# aX + bY +cZ = d
# A = [x-xmean y-mean z-zmean]
# get normal -> a,b,c
# d = a*xmean + b*ymean + c*zmean

ones = np.ones(l)

# At = np.column_stack([x_var,y_var,z_var])
a,b,c = normal
norm = np.linalg.norm(normal,ord=2)

print("normal: ",normal)
print("magnitude: ",magnitude)

d = (a*mean_x)+(b*mean_y)+(c*mean_z)

# Zt = a*x + b*y + c*z + d

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

Xt,Yt = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))

Zt = np.zeros(Xt.shape)

for r in range(Xt.shape[0]):
    for col in range(Xt.shape[1]):
        Zt[r,col] = (d/c) - (a/c) * Xt[r,col] - (b/c) * Yt[r,col] 

ax.plot_wireframe(Xt,Yt,Zt, color='g', alpha=0.4)
ax.set_title("total least sq fit for pc1")
plt.show()
print(a,b,c,d)

#pc2.csv
eigenvalues2, eigenvectors2 = np.linalg.eig(C2)
eigenvalue_idx2 = np.argmin(eigenvalues2)
normal2 = eigenvectors2[:,eigenvalue_idx2]
magnitude2 = np.linalg.norm(normal2,ord=2)

print("normal: ",normal2)
print("magnitude: ",magnitude2)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(x2,y2,z2,color='r')
# Again length of the normal is scaled for visualization
ax.quiver(0,0,0,normal2[0],normal2[1],normal2[2],color='g',length=10)
plt.show()

# aX + bY +cZ = d

a2,b2,c2 = normal2
norm2 = np.linalg.norm(normal2,ord=2)

d2 = (a2*mean_x2)+(b2*mean_y2)+(c2*mean_z2)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x2,y2,z2,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

Xt2,Yt2 = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))

Zt2 = np.zeros(Xt2.shape)

for r in range(Xt.shape[0]):
    for col in range(Xt.shape[1]):
        Zt2[r,col] = (d2/c2) - (a2/c2) * Xt2[r,col] - (b2/c2) * Yt2[r,col] 

ax.plot_wireframe(Xt2,Yt2,Zt2, color='g', alpha=0.4)
ax.set_title("total least sq fit for pc2")
plt.show()
print(a2,b2,c2,d2)

## RANSAC
# pc1.csv

def ransac(data,i,thresh):
    inlier= []
    for _ in range(i):
        sample_idx = random.sample(range(len(data)), 3)
        points = np.array(data)[sample_idx]
        vector1 = points[1] - points[0]
        vector2 = points[2] - points[0]
        normal = np.cross(vector1, vector2)
        a,b,c = normal
        d = -np.sum(normal * points[1])

        dist = (a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        
        curr_inlier = np.where(np.abs(dist) <= thresh)[0]

        if len(curr_inlier) > len(inlier):
            coeffs = [a,b,c,d]
            inlier = curr_inlier

    return coeffs

coeff = ransac(data,1000,2)

fig= plt.figure()
ax= fig.add_subplot(111,projection= '3d')
ax.scatter(x,y,z,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

Xr, Yr = np.meshgrid(np.arange(xlim[0],xlim[1]),
                   np.arange(ylim[0],ylim[1]))

Zr = -(coeff[0] * Xr + coeff[1] * Yr + coeff[3]) / coeff[2]

ax.plot_wireframe(Xr, Yr, Zr, alpha=0.4, color='g') 
ax.set_title("ransac for pc1")
plt.show()

# pc2.csv
coeff2 = ransac(data2,1000,2)

fig= plt.figure()
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(x2,y2,z2,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

X2r, Y2r = np.meshgrid(np.arange(xlim[0],xlim[1]),
                   np.arange(ylim[0],ylim[1]))

Z2r = -(coeff[0] * X2r + coeff[1] * Y2r + coeff[3]) / coeff[2]

ax.plot_wireframe(X2r, Y2r, Z2r, alpha=0.4, color='g') 
ax.set_title("ransac for pc2")
plt.show()

fig= plt.figure()
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(x,y,z,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

#OLS
ax.plot_wireframe(X,Y,Z, alpha=0.5, color='b')

#TLS
ax.plot_wireframe(Xt,Yt,Zt, color='g', alpha=0.4)

# RANSAC
ax.plot_wireframe(Xr, Yr, Zr, alpha=0.7, color='k') 
ax.legend(["RAW DATA","OLS","TLS","RANSAC"])
fig= plt.figure()
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(x,y,z,color='r')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

#OLS
ax.plot_wireframe(X,Y,Z, alpha=0.5, color='b')

#TLS
ax.plot_wireframe(Xt,Yt,Zt, color='g', alpha=0.4)

# RANSAC
ax.plot_wireframe(Xr, Yr, Zr, alpha=0.7, color='k') 
ax.legend(["RAW DATA","OLS","TLS","RANSAC"])
ax.set_title("Comparing for pc1")
plt.show()

fig= plt.figure()
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(x2,y2,z2,color='r')

#OLS
ax.plot_wireframe(X2,Y2,Z2, alpha=0.5, color='b')

#TLS
ax.plot_wireframe(Xt2,Yt2,Zt2, color='g', alpha=0.4)

# RANSAC
ax.plot_wireframe(X2r, Y2r, Z2r, alpha=0.7, color='k') 
ax.legend(["RAW DATA","OLS","TLS","RANSAC"])
ax.set_title("Comparing for pc2")
plt.show()

'''
References:
- [Plot surface for least square fitting and RANSAC](https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points/2306029#2306029)
- [OpenCV RANSAC](https://github.com/rajatsaxena/OpenCV/blob/master/ransac.py)
'''