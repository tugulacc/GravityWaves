import scipy as sc
import numpy as np
import funcs
import sys



def wake(u,x,y,dX,dY,N,M,n,F,L,epsi):

	# computing all needed quantities
	[zetax, zeta1, phix, phi1] = funcs.reshapingUnknowns(u,M,N)
	phi = funcs.allVals(phi1,phix,dX,M,N)
	zeta = funcs.allVals(zeta1,zetax,dX,M,N)
	phiy = funcs.yDerivs(phi,y,M,N)
	zetay = funcs.yDerivs(zeta,y,M,N)
	phixx1 = funcs.xDerivsForw(phix,dX) # <- make sure I use the lefthand points
	zetaxx1 = funcs.xDerivsForw(zetax,dX) # <- make sure I use the lefthand points 

	# computing the quantities at half mesh points
	phiH = funcs.halfMesh(phi,dX,N)
	zetaH = funcs.halfMesh(zeta,dX,N)
	phixH = funcs.halfMesh(phix,dX,N)
	zetaxH = funcs.halfMesh(zetax,dX,N)
	phiyH = funcs.halfMesh(phiy,dX,N)
	zetayH = funcs.halfMesh(zetay,dX,N)

	xH = (x[1:]+x[0:-1])/2.
	
	# declaring all needed arrays
	eqns = np.zeros(np.shape(u))
	K2denom1 = np.zeros((M,N-1))
	K2denom2 = np.zeros((M,N-1))
	K1num1 = np.zeros((M,N-1))
	K1num2 = np.zeros((M,N-1))
	K1 = np.zeros((M,N-1))
	K2 = np.zeros((M,N-1))
	I1 = np.zeros((M,N-1))
	I2p = np.zeros((M,N-1))
	I2pp = np.zeros((M,N-1))
	eqnsInt = np.zeros((M,N-1))
	eqnsBound = np.zeros((M,1))

	P = np.zeros((M,N-1))


	# enforcing the condition at the surface (half mesh)
	
	################################################
	### SURFACE CONDITION WITH THE PRESSURE TERM ###
	'''for jj in range(M):
		for ii in range(N-1):
			if np.abs(xH[ii])<1.:
				if np.abs(y[jj])<1.:
					P[jj,ii] = np.exp(1./(xH[ii]**2.-1.)+1./(y[jj]**2.-1.))

	#print np.max(P)'''
	xInd = np.array((np.abs(xH)<1),'double')
	yInd = np.array((np.abs(y)<1),'double')
	Pind = np.outer(xInd,yInd)
	P = np.exp(1./(xH**2.-1.)+1./(y**2.-1.))*Pind.T


	eqnsSurf = 1./2.*((1+zetaxH**2.)*phiyH**2.+(1+zetayH**2.)*phixH**2.-2.*zetaxH*zetayH*phixH*phiyH)/(1.+zetaxH**2.+zetayH**2.) + zetaH/F**2.+epsi*P-1./2.
	################################################

	'''
	###################################################
	### SURFACE CONDITION WITHOUT THE PRESSURE TERM ###
	eqnsSurf = 1./2.*((1+zetaxH**2.)*phiyH**2.+(1+zetayH**2.)*phixH**2.-2.*zetaxH*zetayH*phixH*phiyH)/(1.+phixH**2.+phiyH**2.) + zetaH/F**2.-1./2.
	###################################################
	'''
	
	if np.isnan(eqnsSurf).any():
		print 'Surface Condition contain NAN'
		sys.exit()

	#### enforcing the boundary integral (half mesh) ####
	# making sure to get rid of singularities

	# SHOULD VECTORIZE THIS LATER AND GET RID OF FOR LOOPS AND MAKE 4d ARRAYS#

	
	for jj in range(M):
		for ii in range(N-1):
			A = 1. + zetaxH[jj,ii]**2.
			B = 2.*zetaxH[jj,ii]*zetayH[jj,ii]
			C = 1. + zetayH[jj,ii]**2.
			S2denom1 = A*(x-xH[ii])**2. + B*(x-xH[ii])*(y-y[jj])+C*(y-y[jj])**2.
			S2denom2 = A*(x-xH[ii])**2. - B*(x-xH[ii])*(y+y[jj])+C*(y+y[jj])**2.
			S2 = 1./S2denom1**(1./2.) + 1./S2denom2**(1./2.)
			if np.isnan(S2).any():
				print 'S2 Denoms are ZEROS!!!'
				sys.exit()
			# don't need these for loops since python does broadcasting
			K2denom1 = (x-xH[ii])**2.+(y-y[jj])**2. + (zeta-zetaH[jj,ii])**2.
			K2denom2 = (x-xH[ii])**2.+(y+y[jj])**2. + (zeta-zetaH[jj,ii])**2.
			K1num1 = zeta - zetaH[jj,ii] - (x-xH[ii])*zetax - (y-y[jj])*zetay 
			K1num2 = zeta - zetaH[jj,ii] - (x-xH[ii])*zetax - (y+y[jj])*zetay
			K1 = K1num1/K2denom1**(3./2.) + K1num2/K2denom2**(3./2.)
			K2 = 1./K2denom1**(1./2.) + 1./K2denom2**(1./2.)
			if np.isnan(K2).any():
				print 'K2 Denoms are ZEROS!!!'
				sys.exit()

			I1in = (phi-phiH[jj,ii] - x + xH[ii])*K1 # still need to integrate
			I1[jj,ii] = np.trapz(np.trapz(I1in,x).T,y.T)
			
			I2pIn = (zetax*K2 - zetaxH[jj,ii]*S2) 	# still need to integrate
			I2p[jj,ii] = np.trapz(np.trapz(I2pIn,x).T,y.T)

			I2pp1 = lambda sIn,tIn : tIn/np.sqrt(A)*np.log(2.*A*sIn+B*tIn+2.*np.sqrt(A*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))
			I2pp2 = lambda sIn,tIn : sIn/np.sqrt(C)*np.log(2.*C*tIn+B*sIn+2.*np.sqrt(C*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))

			sN = x[-1]-xH[ii]
			tN = y[-1]-y[jj]
			s1 = x[0]-xH[ii]
			t1 = y[0]-y[jj]

			I2pp[jj,ii] = I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
			# there are four times where this is a problem for S2pp1
			if t1!=0:
				I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(sN,t1) + I2pp1(s1,t1)
			if tN!=0:
				I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(s1,tN) + I2pp1(sN,tN)

			sN = x[-1]-xH[ii]
			tN = y[-1]+y[jj]
			s1 = x[0]-xH[ii]
			t1 = y[0]+y[jj]
			B = -B

			
			I2pp[jj,ii] = I2pp[jj,ii] + I2pp1(sN,tN) - I2pp1(s1,tN) + I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
			# there are two times where this is a problem for S2pp1
			if t1!=0:
				I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(sN,t1) + I2pp1(s1,t1)


			I2pp[jj,ii] =  zetaxH[jj,ii]*(I2pp[jj,ii])
			
			'''#### WITH THE SOURCE TERM ####
			eqnsInt[jj,ii] = I1[jj,ii] + I2p[jj,ii] + I2pp[jj,ii] - 2.*np.pi*(phiH[jj,ii] - xH[ii]) - epsi/(xH[ii]**2. + y[jj]**2. + (zetaH[jj,ii]+1.)**2.)**(1./2.)
			###############################'''
		
			#### WITHOUT THE SOURCE ####
			eqnsInt[jj,ii] = I1[jj,ii] + I2p[jj,ii] + I2pp[jj,ii] - 2.*np.pi*(phiH[jj,ii] - xH[ii])

			#eqnsInt[jj,ii] = I1[jj,ii]

	if np.isnan(eqnsInt).any():
		print 'Integral Equations contain NAN'
		sys.exit()
			
	# enforcing the boundary condition -> will do the decay one!
	bc1 = x[0]*(phix[:,0]-1.)+n*(phi[:,0]-x[0])
	bc2 = x[0]*(phixx1)+n*(phix[:,0]-1.)
	bc3 = x[0]*(zetax[:,0])+n*(zeta[:,0])
	bc4 = x[0]*(zetaxx1)+n*(zetax[:,0])
	eqnsBound = np.vstack((bc1, bc2, bc3, bc4))
	if np.isnan(eqnsBound).any():
		print 'Boundary Conditions contain NAN'
		sys.exit()
	
	eqns = np.vstack((np.reshape(eqnsInt,(M*(N-1),1)), np.reshape(eqnsSurf,(M*(N-1),1)), np.reshape(eqnsBound,(4*M,1))))
	eqns = eqns[:,0]

	return eqns

