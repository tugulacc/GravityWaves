import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import funcs
from scipy import optimize
import wakeMcCue
import NKwakeMcCue
import wakeMcCueLoops
from matplotlib import cm


### USING THE METHOD IN THE McCue PAPER ###

#### CAN HAVE CASES IF WE WANT TO USE NEWTON-KRYLOV AND FSOLVE!!! ####

### DECLARING CONSTANTS ###
epsi = 1.
epsiMax = 1.
amp = 1.
N = 51
M = 25
F = 1.2
U = 1.
L = 1.
dX = 0.3
dY = 0.3
n = 0.05
x1 = -4.0
y1 = M/2.*dY
###

### DEFINING THE DOMAIN ###
x = sc.r_[0:N]*dX + x1 # row vector
y = np.reshape((sc.r_[0:M]*dY - M/2.*dY),(M,1)) + y1 # column vector

### INITIAL GUESS ###
phi1 = x1*np.ones((M,1))
phix = np.ones((M,N))
zeta1 = np.zeros((M,1))
zetax = np.zeros((M,N))

# stacking the guess into one vector
phiInit =  np.vstack((phi1.T, phix.T))
zetaInit = np.vstack((zeta1.T, zetax.T))

phiInit = np.reshape(phiInit.T, (M*(N+1),1))
zetaInit = np.reshape(zetaInit.T, (M*(N+1),1))

uInit = np.vstack((phiInit, zetaInit))

### OPTIONS FOR FSOLVE ###

### MAIN ITERATION ###
while epsi <= epsiMax:
	
	### CALLING THE MAIN FUNCTION ###
	uNew = optimize.fsolve(wakeMcCue.wake, uInit, args=(x,y,dX,dY,N,M,n,F,L,epsi))
	#uNew = optimize.newton_krylov(NKwakeMcCue.wake, uInit)

	# checking error
	err  = wakeMcCue.wake(uNew,x,y,dX,dY,N,M,n,F,L,epsi)
	err = np.max(err)
	
	### marching forward ### <- do we rescale amplitude?

	epsi = amp + epsi

	phiAll = uNew[0:M+N*M] # all values of phi and phix
	phiInd = (N+1)*np.arange(M)
	phi1 = phiAll[phiInd] # only values of phi at the boundary of y
	phixV = np.delete(phiAll,phiInd) # taking out the vals of phi
	phix = np.reshape(phixV,(M,N)) # reshaping the vector into a matrix

	zetaAll = uNew[M+N*M:] # all values of zeta and zetax
	zetaInd = (N+1)*np.arange(M)
	zeta1 = zetaAll[zetaInd] # only values of zeta at the boundary of y
	zetaxV = np.delete(zetaAll,zetaInd) # taking out the vals of zeta
	zetax = np.reshape(zetaxV,(M,N))

	zeta = funcs.allVals(zeta1,zetax,dX,M,N)

	# stacking the guess into one vector
	phiInit =  np.vstack((phi1.T, phix.T))
	zetaInit = np.vstack((zeta1.T, zetax.T))

	phiInit = np.reshape(phiInit.T, (M*(N+1),1))
	zetaInit = np.reshape(zetaInit.T, (M*(N+1),1))

	uInit = np.vstack((phiInit, zetaInit))


	print epsi
	print err
	

#### PLOTTING THE RESULTING PROFILE ONLY ####	
fig = plt.figure(1)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
#surf = ax.plot_surface(X,Y,zeta)
surf = ax.plot_surface(X, Y, zeta, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

plt.show()

	
