import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate

''' THIS FILE SHOULD CONTAINS THE AUXILIARY FUNCTIONS THAT ALLOW US TO COMPUTE ALL THE VARIABLES THAT GO INTO THE SYSTEM WE WISH TO SOLVE '''

def reshapingUnknowns(u,M,N):
	# extracting values of u and then phi from a long vector
	# THIS ORDERING DOES NOT MATTER, SO I CAN CHANGE TO 
	phiAll = u[0:M+N*M] # all values of phi and phix
	phiInd = (N+1)*np.arange(M)
	phi1 = phiAll[phiInd] # only values of phi at the boundary of x1
	phixV = np.delete(phiAll,phiInd) # taking out the vals of phi
	phix = np.reshape(phixV,(M,N)) # reshaping the vector into a matrix - DO I NEED THIS!?
	
	zetaAll = u[M+N*M:] # all values of zeta and zetax
	zetaInd = (N+1)*np.arange(M)
	zeta1 = zetaAll[zetaInd] # only values of zeta at the boundary of x1
	zetaxV = np.delete(zetaAll,zetaInd) # taking out the vals of zeta
	zetax = np.reshape(zetaxV,(M,N)) # reshaping the vector into a matrix - DO I NEED THIS!?

	return zetax, zeta1, phix, phi1

def allVals(v1,vx,dX,M,N):
	# does the size of v1 change when I call a function?!
	# now, using trapezoidal rule integration, obtain all other values
	v1 = np.reshape(v1,(M,1))
	v = np.hstack((v1,np.zeros((M,N-1))))
	# computing points on the domain upstream of the boundary
	for jj in range(M): # we already have the first y entry
		for ii in range(N-1):
			v[jj,ii+1] = v[jj,ii]+1./2.*dX*(vx[jj,ii] + vx[jj,ii+1])

	return v

def allValsBack(vN,vx,dX,M,N):
	vN = np.reshape(vN,(M,1))
	v = np.hstack((np.zeros((M,N-1)),vN))
	# computing points on the domain starting on the right boundary
	for jj in range(M): # we already have the first y entry
		for ii in range(N-2,0,-1):
			v[jj,ii] = v[jj,ii+1]-1./2.*dX*(vx[jj,ii] + vx[jj,ii+1])

	return v

def halfMesh(v,dX,N):
	# uses 2 point interpolation to get half mesh points in x direction
	# we can adjust to do half mesh in y direction as well?
	vHalf = (v[:,1:N] + v[:,0:N-1])/2.

	return vHalf
	

def yDerivs(v,y,M,N):
	# are these on the grid or half mesh!? Does it matter? This changes N to N-1
	# This function uses cubic interpolation between points along the y direction to compute the derivative - should this be pieacewise!?
	# this should be made more precise, since cubic interpolation might not be sufficient for things like solitary waves!
	ii = 0
	vy = np.zeros((M,N))

	for ii in range(N):
		#vInter = interp1d(y, v[ii], kind='cubic')
		vInter = interpolate.splrep(y, v[:,ii], k=3)
		vy[:,ii] = interpolate.splev(y[:,0], vInter, der=1)
	
	'''# forward difference
	vy[0,:] = (v[1,:] - v[0,:])/y[1]
	# backward difference
	vy[-1,:] = (v[-1,:] - v[-2,:])/y[1]
	# centered difference
	vy[1:-1,:] = (v[2:,:]-v[0:-2,:])/(2.*y[1])'''
	return vy


''' this function computes the derivs on the left domain using a forward scheme'''
def xDerivsForw(v,dX):
	# only first order forward differentiation
	#vx = (v[:,1] - v[:,0])/dX	
	# second order forward differentiation
	vx = -(v[:,2] - 4.*v[:,1] + 3.*v[:,0])/(2.*dX)

	return vx

def xDerivsBack(v,dX):
	# only first order forward differentiation
	vx = (v[:,-2] - v[:,-1])/dX	
	# second order forward differentiation
	#vx = (v[:,-3] - 4.*v[:,-2] + 3.*v[:,-1])/(2.*dX)

	return vx

def xDerivs(v,x,M,N):
	# are these on the grid or half mesh!? Does it matter? This changes N to N-1
	# This function uses cubic interpolation between points along the y direction to compute the derivative - should this be pieacewise!?
	# this should be made more precise, since cubic interpolation might not be sufficient for things like solitary waves!
	ii = 0
	vx = np.zeros((M,N))

	for ii in range(M):
		#vInter = interp1d(y, v[ii], kind='cubic')
		vInter = interpolate.splrep(x, v[ii,:], k=3)
		vx[ii,:] = interpolate.splev(x[:], vInter, der=1)
	
	'''# centered difference
	vx[:,1:-1] = (v[:,2:]-v[:,0:-2])/(2.*x[1])'''
	return vx








