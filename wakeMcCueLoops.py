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
	K2denom1 = np.zeros((M,N))
	K2denom2 = np.zeros((M,N))
	S2denom1 = np.zeros((M,N))
	S2denom2 = np.zeros((M,N))
	S2 = np.zeros((M,N))
	K1num1 = np.zeros((M,N))
	K1num2 = np.zeros((M,N))
	K1 = np.zeros((M,N))
	K2 = np.zeros((M,N))
	I1 = np.zeros((M,N-1))
	I1in = np.zeros((M,N))
	I2pIn = np.zeros((M,N))
	I1temp = np.zeros((M,1))
	I2ptemp = np.zeros((M,1))
	I2p = np.zeros((M,N-1))
	I2pp = np.zeros((M,N-1))
	eqnsInt = np.zeros((M,N-1))
	eqnsBound = np.zeros((M,1))

	P = np.zeros((M,N-1))


	# enforcing the condition at the surface (half mesh)
	
	################################################
	### SURFACE CONDITION WITH THE PRESSURE TERM ###
	for jj in range(M):
		for ii in range(N-1):
			if np.abs(xH[ii])<1.:
				if np.abs(y[jj])<1.:
					P[jj,ii] = np.exp(1./(xH[ii]**2.-1.)+1./(y[jj]**2.-1.))

	#print np.max(P)

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
	A = 1. + zetaxH**2.
	B = 2.*zetaxH*zetayH
	C = 1. + zetayH**2.

	I2pp2 = lambda sIn,tIn,m,n,sign : tIn/np.sqrt(A[m,n])*np.log(2.*A[m,n]*sIn+sign*B[m,n]*tIn+2.*np.sqrt(A[m,n]*(A[m,n]*sIn**2.+sign*B[m,n]*sIn*tIn+C[m,n]*tIn**2.)))
	
	I2pp1 = lambda sIn,tIn,m,n,sign : sIn/np.sqrt(C[m,n])*np.log(2.*C[m,n]*tIn+sign*B[m,n]*sIn+2.*np.sqrt(C[m,n]*(A[m,n]*sIn**2.+sign*B[m,n]*sIn*tIn+C[m,n]*tIn**2.)))

	
	for jj in range(M):
		for ii in range(N-1):
			for mm in range(M):
				for nn in range(N):
					S2denom1[mm,nn] = A[jj,ii]*(x[nn]-xH[ii])**2. + B[jj,ii]*(x[nn]-xH[ii])*(y[mm]-y[jj])+C[jj,ii]*(y[mm]-y[jj])**2.
					S2denom1[mm,nn] = A[jj,ii]*(x[nn]-xH[ii])**2. + B[jj,ii]*(x[nn]-xH[ii])*(y[mm]-y[jj])+C[jj,ii]*(y[mm]-y[jj])**2.
					S2denom2[mm,nn] = A[jj,ii]*(x[nn]-xH[ii])**2. - B[jj,ii]*(x[nn]-xH[ii])*(y[mm]+y[jj])+C[jj,ii]*(y[mm]+y[jj])**2.
					S2[mm,nn] = 1./S2denom1[mm,nn]**(1./2.) + 1./S2denom2[mm,nn]**(1./2.)
					# don't need these for loops since python does broadcasting
					K2denom1[mm,nn] = (x[nn]-xH[ii])**2.+(y[mm]-y[jj])**2. + (zeta[mm,nn]-zetaH[jj,ii])**2.
					K2denom2[mm,nn] = (x[nn]-xH[ii])**2.+(y[mm]+y[jj])**2. + (zeta[mm,nn]-zetaH[jj,ii])**2.
					K1num1[mm,nn] = zeta[mm,nn] - zetaH[jj,ii] - (x[nn]-xH[ii])*zetax[mm,nn] - (y[mm]-y[jj])*zetay[mm,nn] 
					K1num2[mm,nn] = zeta[mm,nn] - zetaH[jj,ii] - (x[nn]-xH[ii])*zetax[mm,nn] - (y[mm]+y[jj])*zetay[mm,nn]
					K1[mm,nn] = K1num1[mm,nn]/K2denom1[mm,nn]**(3./2.) + K1num2[mm,nn]/K2denom2[mm,nn]**(3./2.)
					K2[mm,nn] = 1./K2denom1[mm,nn]**(1./2.) + 1./K2denom2[mm,nn]**(1./2.)

					I1in[mm,nn] = (phi[mm,nn]-phiH[jj,ii] - x[nn] + xH[ii])*K1[mm,nn]
					I2pIn[mm,nn] = (zetax[mm,nn]*K2[mm,nn] - zetaxH[jj,ii]*S2[mm,nn])

					# these need to be evaluated only at the ends or beginning of loops
					s = x[nn]-xH[ii]
					tp = y[mm]-y[jj]
					tn = y[mm]+y[jj]

					if nn == 0:
						if mm == 0:
							if jj==0:
								I2pp[jj,ii] = I2pp1(s,tp,jj,ii,1.) + I2pp1(s,tn,jj,ii,-1.)
								#print 'here i am!'
								#print s, tp, tn, I2pp[jj,ii]
							else:
								I2pp[jj,ii] = I2pp[jj,ii] + I2pp1(s,tp,jj,ii,1.) + I2pp1(s,tn,jj,ii,-1.) + I2pp2(s,tp,jj,ii,1.) + I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]
						elif mm == M-1:
							if jj==M-1:
								I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(s,tp,jj,ii,1.) - I2pp1(s,tn,jj,ii,-1.) - I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]
							else:
								I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(s,tp,jj,ii,1.) - I2pp2(s,tp,jj,ii,1.) - I2pp1(s,tn,jj,ii,-1.) - I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]

					if nn == N-2:
						if mm == 0:
							if jj==0:
								I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(s,tp,jj,ii,1.) - I2pp1(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]
							else:
								I2pp[jj,ii] = I2pp[jj,ii] - I2pp1(s,tp,jj,ii,1.) - I2pp1(s,tn,jj,ii,-1.) - I2pp2(s,tp,jj,ii,1.) - I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]
						elif mm == M-1:
							if jj==M-1:
								I2pp[jj,ii] = I2pp[jj,ii] + I2pp1(s,tp,jj,ii,1.) + I2pp1(s,tn,jj,ii,-1.) + I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]
							else:
								I2pp[jj,ii] = I2pp[jj,ii] + I2pp1(s,tp,jj,ii,1.) + I2pp2(s,tp,jj,ii,1.) + I2pp1(s,tn,jj,ii,-1.) + I2pp2(s,tn,jj,ii,-1.)
								#print I2pp[jj,ii]

				
				I1temp[mm] = np.trapz(x,I1in[mm,:])
				I2ptemp[mm] = np.trapz(x,I2pIn[mm,:])

			I2p[jj,ii] = np.trapz(y.T, I2ptemp.T)
			I2pp[jj,ii] = zetaxH[jj,ii]*(I2pp[jj,ii])
			I1[jj,ii] = np.trapz(y.T, I1temp.T)
						
			
			
			#### WITHOUT THE SOURCE ####
			eqnsInt[jj,ii] = I1[jj,ii] + I2p[jj,ii] + I2pp[jj,ii] - 2.*np.pi*(phiH[jj,ii] - xH[ii])

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

	return eqns, I1, I2p, I2pp



#############################################################################
## SUPPLEMENTARY ROUTINES ###
'''for nn in range(M):
				for mm in range(N-1):
					print nn
					print mm
					print ii
					print jj
					K2denom1[nn,mm] = (xH[nn]-xH[ii])**2.+(y[mm]-y[jj])**2.+(zetaH[nn,mm]-zetaH[ii,jj])**2.
					K2denom2[nn,mm] = (xH[nn]-xH[ii])**2.+(y[mm]+y[jj])**2.+(zetaH[nn,mm]-zetaH[ii,jj])**2.
					K1num1[nn,mm] = zetaH[nn,mm] - zetaH[ii,jj] - (x[nn]-x[ii])*zetaxH[nn,mm] - (y[mm]-y[jj])*zetayH[nn,mm] 
					K1num2[nn,mm] = zetaH[nn,mm] - zetaH[ii,jj] - (x[nn]-x[ii])*zetaxH[nn,mm] - (y[mm]+y[jj])*zetayH[nn,mm]
					K1[nn,mm] = K1num1[nn,mm]/K2denom1[nn,mm]**(3./2.) + K1num2[nn,mm]/K2denom2[nn,mm]**(3./2.)
					K2[nn,mm] = 1./K2denom1[nn,mm] + 1./K2denom2[nn,mm]'''
