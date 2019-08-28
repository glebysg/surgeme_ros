#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Python script to callbirate robot and kinect frames, i.e. find Kinect to robot
# 	transform. Also computes kinect to world, world to robot base.
# Uses a claibration pattern(chesboard pattern) to estimate Kinect to world tf
# Ensure Kinect point cloud and kinect images are being published..
# 	- i.e. run roslaunch kinect2_bridge kinect2_bridge from iai_kienct2
# -------
# First computes the pattern corners and uses Opencv functions to refine corners
# Some corners are obtained manually by user by clicking on image.
# - Then the correspondences are found and refined using nonlinear least_squares
# The arm calibration is obtained using ArmCalib from calib_arm.py
# < Ensure order of pattern points: Pink is next to vertical points for Kin,
#	and red next to vertical for arm camera >
#	Date 			Author			Ver. (changes)
#	11/16/18			Mythra			1.0		Initial coding to get arm tf
#-------------------------------------------------------------------------------
import rospy, cv2, time
from sensor_msgs.msg import Image, PointCloud2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from scipy.optimize import least_squares as ls
from calib_arm import ArmCalib

class CamCalib():
	def __init__(self, n):
		self.nimgs = n
		self.nx = 6	# no. of interior points across a row
		self.ny = 9 # no. of interior points along a col
		self.np2 = [6,8]	# no. of corners row x col in pattern 2
		self.kill = False
		self.dchk  = 0.03   # distance between points in m
		self.start = time.time()
		self.end = time.time()
		# Source corners in real world
		self.scrs = np.array([[(int(i/self.nx)+1)*self.dchk, int(i%self.nx)*self.dchk,0.0 ,1] for i in range(self.nx*self.ny)])
		zref = 0.03 # z offset of vertical points in m
		vrefs = self.scrs[0:self.nx,:] + [-self.dchk,0,zref,0]
		self.scrs = np.vstack((vrefs, self.scrs))
		self.bpath = 'src/sb/cclib/resources/'
		self.cams = {1: 'larm_head_camera', 2: 'KinectV2', 3: 'Kin_depth'}
		# Kinect vars
		self.kpts = np.ones([self.nx*(self.ny+1),4])	# Kinect detected checkered points
		# Kinect if corner detected in RGB flag and Corners vals in RGB frame
		self.kcrs = [False, np.zeros([self.nx*self.ny])]
		self.c = list()
		self.lmi = 0	# LM function call count
		self.sb_head_pmat  = np.array([])
		# Arm 2 world tf vars
		self.arm_tfv = ['/io/internal_camera/head_camera','image_rect_color'\
		,'head_camera']
		self.done = False
		# Reverse flags
		self.kinrev = False
		self.robrev = True

# ------------------------- CV functionalities
# Function to view image
	def viewimg(self, img1, i=1):
		img = np.copy(img1)
		# if not self.acrs[0]:
		cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Image', 1000, 1000)
		key = 1
		while key != ord('q'):
			cv2.imshow('Image',img)
			key = cv2.waitKey(i)
		cv2.destroyAllWindows()

# Function to get mouse click
	def mouse_cb(self, event, x, y, flags, img):
		if event ==cv2.EVENT_MBUTTONDOWN:
			self.c.append([x,y])
			cv2.circle(img, (x,y), 0, (0,0,255),1)
			cv2.imshow('Image',img)

	def get_refs(self,img):
		print('Use middle click to select points')
		cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Image', 1000, 1000)
		cv2.setMouseCallback('Image', self.mouse_cb, img)
		key = 1
		cv2.imshow('Image',img)
		while key !=ord('q'):       # Quit on pressing q
			key = cv2.waitKey(1)
			if len(self.c) == 5:
				break
		if key == ord('q'):
			rospy.signal_shutdown('Exiting')
		cv2.destroyAllWindows()

# function to compute the rotation matrix from rodigues
	def q_from_rod(self,r):
		th = np.linalg.norm(r)
		r = r/th
		k = np.cos(th)*np.eye(3) + (1 - np.cos(th))*np.matmul(r,r.T) \
		+ np.sin(th)*np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
		return(k)

# Function to compute homography
	def computeH(self, p0, p1):
		#----------------------------------------------------------------------
		# Computes Trsnformation to obtain p1 from p0 i.e. return H which does
		# 	H*p0 = p1
		# Inputs: p0 - src points(npts, size:nxm), p1 - dest points(nx1)
		# m-1 is dmension of space (m=3 for 2D and 4 for 3D)
		# The points should be in homogenous coordinates
		# Outputs: H - Transformation matrix
		# ----------------------------------------------------------------------
		# The dimensions of h (h1-h16 elements of H) is 16x1, but assume
		# last row is [0,0,0,1]
		# We rewrite Hp0 = p1 as Ah = y
		# Construct A for Ah = y
		#----------------------------------------------------------------------
		nh = p0.shape[1] -1  # dimensions of H nxn
		A = np.zeros([p0.shape[0]*nh,nh*(nh+1)]) # Based on size of H (12 elements)
		Nx = np.reshape(range(0,nh*(nh+1)),[nh+1,nh])
		b = np.zeros(A.shape[0])
		for i in range(A.shape[0]):
			k =(nh+1)*(i%nh)
			for j in range(k,k+nh+1):
				A[i,j] = p0[int(i/nh),j%(nh+1)]
			b[i] = p1[int(i/nh),i%nh]
		# Find solution of Ah = y, using h = (A^T.A)^(-1).A^T.y
		pin = np.linalg.pinv(np.matmul(A.T,A))
		solx = np.matmul(pin, np.matmul(A.T,b))
		h = np.vstack((solx.reshape(nh,-1),[0,0,0,1]))
		h0 = h.reshape(-1)
		# LM estimate
		lmsol = ls(self.lmfun, h0[0:6], method='lm', args=[nh+1,self.kpts,self.scrs],ftol=1e-10,max_nfev=10**6)
		# Compute initial estimages
		p01 = np.dot(h,p0.T).T
		p01 = (p01.T/p01[:,-1]).T
		print('Before lm ', np.linalg.norm(p01 - p1))
		#
		print(lmsol.nfev, lmsol.cost)
		r = lmsol.x
		rmat = self.q_from_rod(r[0:3])
		h = np.vstack((np.hstack((rmat,r[3:][:,None])),[0,0,0,1]))
		# Compute point estimates after LM
		p02 = np.dot(h,p0.T).T
		p02 = (p02.T/p02[:,-1]).T
		print('After lm ', np.linalg.norm(p02 - p1))
		# print(h.reshape(nh,-1))		# H after LM
		opts = (np.hstack((p1[:,0:3],p01[:,0:3],p02[:,0:3]))*1000).astype(int)
		print(opts)	#print point estimates before and after in mm
		return h

# Function to compute homography
	def computeHSvd(self, p0, p1):
		#----------------------------------------------------------------------
		# Computes Trsnformation to obtain p1 from p0 i.e. return H which does
		# 	H*p0 = p1
		# Inputs: p0 - src points(npts, size:nxm), p1 - dest points(nx1)
		# m-1 is dmension of space (m=3 for 2D and 4 for 3D)
		# The points should be in homogenous coordinates
		# Outputs: H - Transformation matrix
		# ----------------------------------------------------------------------
		# The dimensions of h (h1-h9 elements of H) is 9x1
		# We rewrite Hp0 = p1 as Ah = y
		# Construct A for Ah = y
		#----------------------------------------------------------------------
		nh = p0.shape[1]  # dimensions of H nxn
		A = np.zeros([p0.shape[0]*p0.shape[1],nh**2])   # Based on size of H (9 elements)
		Nx = np.reshape(range(0,nh**2),[nh,nh])
		for i in range(0,len(p0[:,0])):
			for j in range(0,nh**2):
			    r_ind = nh*i + np.where(Nx==j)[0][0]
			    A[r_ind,j] = p0[i,j%nh]
		# Find solution of Ah = y, using h = (A^T.A)^(-1).A^T.y
		for i in range(0,int(A.shape[0]/nh)):
			for j in range(nh-1):
				A[nh*i+j,:] = A[nh*i+j,:] - p1[i,j]*A[nh*i+nh-1,:]
		r_ind = [x for x in range(A.shape[0]) if (x+1)%nh != 0]
		A = A[r_ind,:]
		# SVD to solve for H
		u, s, v = np.linalg.svd(A)
		null_space = v.T[:,-1]
		h = np.reshape(null_space,(nh,nh))
		h = h/h[nh-1,nh-1]
		# Setup for nonlinear least squares
		h0 = h.reshape(-1)
		# LM estimate
		lmsol = ls(self.lmfun, h0, method='lm', args=[nh,self.kpts,self.scrs],ftol=1e-8,max_nfev=10**6)
		# Compute initial estimages
		p01 = np.dot(h,p0.T).T
		p01 = (p01.T/p01[:,-1]).T
		print('Before lm ', np.linalg.norm(p01 - p1))
		#
		print(lmsol.nfev, lmsol.cost)
		h = lmsol.x;
		# print(h0.reshape(nh,-1))	# Original H
		# Compute point estimates after LM
		# h = np.hstack((h,[0,0,0,1]))
		h /= h[-1];
		h = h.reshape(nh,nh)
		p02 = np.dot(h,p0.T).T
		p02 = (p02.T/p02[:,-1]).T
		print('After lm ', np.linalg.norm(p02 - p1))
		# print(h.reshape(nh,-1))		# H after LM
		# opts = (np.hstack((p1[:,0:3],p01[:,0:3],p02[:,0:3]))*1000).astype(int)
		# print(opts)	#print point estimates before and after in mm
		return h
#----------------------------------------------------------------------------
# Cost function for lm least squares
	def lmfun(self, h, nh, p0,p1):
		self.lmi+=1
		if self.lmi%100000==0:
			print(self.lmi)
		rmat = self.q_from_rod(h[0:3])
		hmat = np.vstack((np.hstack((rmat,h[3:][:,None])),[0,0,0,1]))
		p01 = np.dot(hmat,p0.T).T
		p01 = (p01.T/p01[:,-1]).T
		return(np.sqrt(np.sum((p1 - p01)**2,1)))


# Function to read point cloud data
	def getpts(self, data, camid):
		h,w = data.height, data.width
		pts = np.frombuffer(data.data,dtype=np.float32)
		n = int(pts.shape[0]/(h*w))
		pts = pts.reshape(h,w,n)
		if self.kcrs[0] and not self.done:	# If corners detected get xyz
			self.done = True
			# print(self.kpts[780,645,2],self.kpts[770,740,2])
			for i, pix in enumerate(self.kcrs[1]):
				self.kpts[i,0:3] = [pts[pix[1],pix[0],0], pts[pix[1],pix[0],1]\
				,pts[pix[1],pix[0],2]]
			print('Got corner XYZ..')
			print('Calculating transfrom from '+self.cams[camid]+' to world')
			dif = self.kpts[1:,:] - self.kpts[0:-1,:]
			print('Norm is ', np.linalg.norm(dif,axis=1)[0:3])
			# Get tf from kinect points to world points i.e. T for T*p_kin = p_w
			k2w_tf = self.computeH(self.kpts,self.scrs)
			# k2w_tf = self.computeHSvd(self.kpts,self.scrs)
			print('Kinect to world')
			print(k2w_tf)
			p = np.matmul(k2w_tf,self.kpts[0,0:4])
			print(p/p[-1])
			# Get arm to world tf
			cc = ArmCalib(self.arm_tfv[0],self.arm_tfv[1],self.arm_tfv[2],self.robrev)
			w2b_tf, acrs, rob_img = cc.get_arm_tf()
			print('World to robot base ')
			print(w2b_tf)
			# Compute kienct to robot base transform i.e. T*p_kin = p_base
			k2b_tf = np.matmul(w2b_tf,k2w_tf)
			print('Kinect to robot base')
			print(k2b_tf)
			try:
				np.save(self.bpath+self.cams[camid]+'_crxyz.npy',self.kpts) # Save to file
				np.savez(self.bpath+'tfs.npz',k2b_tf, k2w_tf, w2b_tf)
			except:
				print('Check if you are running from root of ws')
				print('Check if path in __init__ is correct ')
			rospy.signal_shutdown('Exit')

# Function to read image and decode
	def rdimg_kin(self,data, camid):
		# Compute frame rate
		self.start = self.end
		self.end = time.time()
		# Get image dimensinos and current image ..
		h, w = data.height, data.width
		img = np.frombuffer(data.data,dtype=np.uint8)
		nch = img.shape[0]/(h*w) 				# no. of channels
		img = img.reshape(h,w,nch)	# Reshape from 1d to image dims
		n = 2*int((img.shape[2]+1)/2)-1
		img = np.copy(img[:,:,0:n])
		if img.shape[2]>1 and not self.kcrs[0]:	# If color image from camera
			# Find chess board corners
			flgs = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK \
			+ cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
			pf, crs = cv2.findChessboardCorners(img,(self.nx,self.ny),flags=flgs)
			print('Camera: '+ self.cams[camid] + ' Frame rate: '\
			+str(int(1/(self.end - self.start))) + ' Pattern: ' + str(pf))
			if pf:	# If pattern found
				# Get gray image
				img_gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.5)
				# Refine points using sub pixel
				cv2.cornerSubPix(img_gr, crs, (11, 11), (-1, -1), term)
				np.save(self.bpath+self.cams[camid]+'_cpts.npy',crs) # Save to file
				cv2.drawChessboardCorners(img, (self.nx,self.ny), crs, pf)	 # draw
				crs = crs[:,0].astype(int)
				print('Click 5 ref points on box or Press \'q\' to quit')
				self.get_refs(img)	# Get vertical refs
				# Combine vertical refs and points
				if np.array(self.c).shape[0]>self.nx:
					print('Too many poitns given')
					rospy.signal_shutdown('Too many points ')
				# Reverse crs and append vrefs- because of order of points
				if self.kinrev:
					crs = crs[::-1]	#reverse
				crs = np.vstack((np.array(self.c),crs))
				# self.viewimg(img,0)
				self.kcrs = [True, crs]	# Set corner found flags

# Main function
	def calib(self):
		rospy.init_node('camcalib',anonymous = True)	# init node
		# Subscribe to Kinect rgb image
		rospy.Subscriber('/camera/color/image_raw', Image, self.rdimg_kin, 2)
		rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.getpts, 2)
		print('Press \'s\' to stop')
		rospy.spin()

if __name__=='__main__':
	cc = CamCalib(1)
	cc.calib()
