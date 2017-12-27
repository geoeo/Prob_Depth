#!/usr/bin/env python
# import freenect
import cv2
import numpy as np
import libs.write_pcd as write_pcd
import libs.PointCloudViewer as PointCloudViewer
import time


Rx=lambda x: np.matrix([[1,0,0],[0,np.cos(x),np.sin(x)],[0,-np.sin(x),np.cos(x)]])
Ry=lambda x: np.matrix([[np.cos(x),0,-np.sin(x)],[0,1,0],[np.sin(x),0,np.cos(x)]])
Rz=lambda x: np.matrix([[np.cos(x),np.sin(x),0],[-np.sin(x),np.cos(x),0],[0,0,1]])
Rxyz=lambda x,y,z: Rx(x)*Ry(y)*Rz(z)
polyArea = lambda x,y: 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

DO = -0.006

class PointCloud:
   uv_xyz_rgb = None
   

   
   def __init__(self, uv_xyz_rgb):
      self.uv_xyz_rgb = uv_xyz_rgb

   def __add__(self,other):
      return PointCloud(np.hstack((self.uv_xyz_rgb,other.uv_xyz_rgb)))

   def __radd__(self, other):
      if other == 0:
         return self
      else:
         return self.__add__(other)
            
   # returns list with x,y,z coordinates that correspond to (u,v)-pixel
   def find_xyz(self, u, v):
      xyz_list = [] # list of [x,y,z]-arrays

      row_hits = np.argwhere(self.uv_xyz_rgb[1] == v)
      column_hits = np.argwhere(self.uv_xyz_rgb[0, row_hits] == u)

      if column_hits.size > 0:
         index_list = row_hits[column_hits[:, 0], 0]
         xyz_list = self.uv_xyz_rgb[:, index_list][2:5] # shape = [3, column_hits.size]

      return xyz_list


   # returns x,y,z coordinate with least Euclidean distance
   def filter_distance(self, xyz_list):
      norms = np.linalg.norm(xyz_list, axis=0)
      least_norm_index = norms.argmin()
      xyz_nearest = xyz_list[:, least_norm_index]
      return xyz_nearest # shape = [x,y,z]


   # image space filter
   # returns keypoint that is averaged over (u,v)-pixel neighborhood
   def filter_uv_neighborhood(self, u, v):
      radius = 5

      low_u = u - radius
      if low_u < 0:
         low_u = 0
      high_u = u + radius
      if high_u > 640:
         high_u = 640
      low_v = v - radius
      if low_v < 0:
         low_v = 0
      high_v = v + radius
      if high_v > 480:
         high_v = 480

      uv_mask = (self.uv_xyz_rgb[0] >= low_u) & (self.uv_xyz_rgb[0] <= high_u) & \
             (self.uv_xyz_rgb[1] >= low_v) & (self.uv_xyz_rgb[1] <= high_v)

      neighborhood_xyz_points = self.uv_xyz_rgb[2:5, uv_mask]

      norms = np.linalg.norm(neighborhood_xyz_points, axis=0)

      p = np.percentile(norms, [15, 75])

      p_mask = (norms > p[0]) & (norms < p[1])

      new_keypoint = neighborhood_xyz_points[:, p_mask].mean(axis=1)

      return new_keypoint


   # 3d space filter
   # returns keypoint that is averaged over (x,y,z)-point neighborhood
   def filter_xyz_neighborhood(self, keypoint):
      radius = 0.01 # in meter
      keypoint = np.mat(keypoint).T
      p0_minus_pi = self.uv_xyz_rgb[2:5] - keypoint
      p0_minus_pi_norms = np.linalg.norm(p0_minus_pi, axis=0)

      idx=np.argwhere(p0_minus_pi_norms <= radius)

      new_keypoint = self.uv_xyz_rgb[2:5,idx].mean(axis=1)

      return new_keypoint.T[0]

   def calcRotFromVec(self,b):
      '''calculates a rotation matrix from a normal vector b
         the rotation matrix R represents the rotation that coverts (0,0,1) to b
      '''   
      a  = np.array((0.,0.,1.))
      b=np.array(b)
      c=b/np.linalg.norm(b)
      v=np.cross(a,c)
      cc=np.dot(a,c)

      R=np.eye(3)      
      if 1-cc<=1e-12:
         R = np.eye(3)
      if 1+cc<=1e-12:
         R = -np.eye(3)

      if 1+cc>1e-12 and 1-cc>1e-12:
         vx = np.matrix([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]],dtype=float)
         R=np.eye(3)+vx+vx*vx*1/(1+cc)
      
      return R
      
   def sliceAnalysis(self,npa,center,axis,step,func):
      ''' slice analysis takes planar slices orthogonal to a given axis and returns
          func(points) where points are points from said slice. step gives the 
          slice height.
          
          to achieve this, the rotation required to transform a z oriented normal
          vector to axis is applied to the point cloud. then binning is carried 
          out along the z axis (now equivallent to input axis). the binned data
          is then analyzed by func, appended into a list and returned.
      '''
      R=self.calcRotFromVec(axis).T # get rotation and store its inverse.
      cen=np.matrix(center).reshape(3,1)
      npa2=npa.copy()
      npa2[2:5] = np.matrix(R)*(np.matrix(npa2[2:5])-cen)+cen
      #self.show(npa2,showAxes=True)
      steps_l=np.arange(npa2[4].min(),npa2[4].max(),step)
      steps_u=np.arange(npa2[4].min()+step,npa2[4].max()+step,step)
      res = []
      for l,u in zip(steps_l,steps_u):
         pts=self.segmentHalfspace("z",l,+1,npa2.copy())
         pts=self.segmentHalfspace("z",u,-1,pts)
         res.append([(u+l)/2.0,func(pts,cen)])
      return res

   def segmentHalfspace(self,axis,value,sign,npa):
      ''' remove all points in one half-space along one of the cartesian axes.
      '''
      if npa is None:
         npa=self.uv_xyz_rgb.copy()
         
      if type(axis) is str and axis == "x":
        axis = [1,0,0]
      elif type(axis) is str and axis == "y":
        axis = [0,1,0]
      elif type(axis) is str and axis == "z":
        axis = [0,0,1]
      
      return self._segmentHalfspace(axis,value,sign,npa)

   def _segmentHalfspace(self,axis,value,sign,npa):
      axis = np.array(axis)
      surfNorm = axis/np.linalg.norm(axis)
      
      # the half-space is intersected by the plane p defined by surfNorm and the intersection
      # value*surfNorm. thus the projection of the difference of all points in the cloud
      # to value*surfnorm onto surfnorm with either positive or negative sign will be kept
      proj = np.matrix(surfNorm).reshape(1,3)*(npa[2:5] - value*np.matrix(surfNorm).reshape(3,1))
      
      if sign<0:
         idx=np.where(proj <= 0)[1]
      else:
         idx=np.where(proj > 0)[1]

      return npa[:,idx]

   def _cubeSegment(self,center,size,uv_xyz_rgb):
      npa=uv_xyz_rgb
      cx,cy,cz=center
      dx,dy,dz=size
      npa=self.segmentHalfspace("x",cx+dx/2.0,-1,npa)
      npa=self.segmentHalfspace("x",cx-dx/2.0,1,npa)
      npa=self.segmentHalfspace("y",cy+dy/2.0,-1,npa)
      npa=self.segmentHalfspace("y",cy-dy/2.0,1,npa)
      npa=self.segmentHalfspace("z",cz+dz/2.0,-1,npa)
      npa=self.segmentHalfspace("z",cz-dz/2.0,1,npa)
      return npa

   def cubeSegment(self,center,size):
      ''' segments a pointcloud npa with a cuboid given by center and size. all
          points outside the cubiod are removed!
      '''
      return self._cubeSegment(center,size,self.uv_xyz_rgb)
      
      
   def cylinderSegment_z(self,center,radius,height,npa=None,sign=-1):
      ''' segments a pointcloud npa with a cylinder given by center radius and 
          height. with a sign of -1 all points outside the cylinder are removed
          while a sign of +1 removes all points inside the cylinder.
      '''
      if npa is None:
         npa=self.uv_xyz_rgb.copy()
         
      cx,cy,cz = center
      npa=self.segmentHalfspace("z",cz+height/2.0,-1,npa)
      npa=self.segmentHalfspace("z",cz-height/2.0, 1,npa)
      # distance from center in xy-plane
      dist = np.linalg.norm((npa[2:5,:] - np.matrix(center).reshape((3,1)) )[:2],2,0)
      if sign < 0:
         idx=np.argwhere(dist <= radius)
      else:
         idx=np.argwhere(dist > radius)
      npa = npa[:,idx]
      npa = npa.reshape(npa.shape[:2])
      return npa

   def cylinderSegment(self,center,radius,height,npa=None,sign=-1, axis = [0,0,1]):
      ''' segments a pointcloud npa with a cylinder given by center radius and 
          height. with a sign of -1 all points outside the cylinder are removed
          while a sign of +1 removes all points inside the cylinder.
      '''
      if npa is None:
         npa=self.uv_xyz_rgb.copy()
         
      return self._cylinderSegment(center,axis,radius,height,npa,sign)

   def _cylinderSegment(self,center,axis,radius,height,npa=None,sign=-1):
      ''' segments a pointcloud npa with a cylinder given by center, radius, axis and 
          height. with a sign of -1 all points outside the cylinder are removed
          while a sign of +1 removes all points inside the cylinder. in this case
          it is a general cylinder with z aligned along axis.
      '''
      if npa is None:
         npa=self.uv_xyz_rgb.copy()
         
      # move cloud to center
      npa[2:5] -= np.matrix(center).reshape(3,1)
      cx,cy,cz = center
      npa=self.segmentHalfspace(axis,+height/2.0,-1,npa)
      npa=self.segmentHalfspace(axis,-height/2.0, 1,npa)
      # distance from center in xy-plane
      axis = np.matrix(axis).reshape(3,1)
      axis = axis/np.linalg.norm(axis)
      dist = np.linalg.norm(npa[2:5,:] - axis*(axis.T*npa[2:5,:]) ,2,0)
      if sign < 0:
         idx=np.argwhere(dist <= radius)
      else:
         idx=np.argwhere(dist > radius)
      npa = npa[:,idx]
      npa = npa.reshape(npa.shape[:2])
      npa[2:5] += np.matrix(center).reshape(3,1)

      return npa

   def removeNans(self,npa):
      idx = np.where(~np.isnan(npa[2])&~np.isnan(npa[3])&~np.isnan(npa[4]))[0]
      return npa[:,idx]
      

   
   def binProfile(self,x,y,N):
      ''' binx x and y into N bins. each bin contains the average of x and y as array x0 and y0
          x0 any y0 are returned average profiles
      '''
      M=len(y)
      N0 = np.mod(M,N)
      x0 = x[N0:].reshape((N,(M-N0)/N)).mean(1)
      y0 = y[N0:].reshape((N,(M-N0)/N)).mean(1)
      return x0,y0
   
      
   def find_keypoint(self, u, v):
      # u,v should be integers
      xyz_list = self.find_xyz(u, v)

      if not (len(xyz_list) > 0):
          # implicates that u,v is invalid
          return False, None

      xyz_nearest = self.filter_distance(xyz_list)

      return True, xyz_nearest
        
   def transform(self, R, t):
      self.uv_xyz_rgb[2:5] = np.array(R * np.matrix(self.uv_xyz_rgb[2:5]) + np.matrix(t).reshape((3,1)) )
      
   def angularProfile(self,center,npa,Nbins=90):
      x=npa[2]-center[0]
      y=npa[3]-center[1]
      phi=np.arctan2(y,x) #angle of all ring points
      hist=np.histogram(phi,Nbins)
      HX,HY=self._centerHistArrays(hist)
      return (HX,HY)

   def angularBinnedMedian(self,center,npa,Nbins=90,funcs=(np.median,np.median), minPts=0,fullCircle=False,filterNans=True):
      x=npa[2]-center[0]
      y=npa[3]-center[1]
      z=npa[4]-center[2]
      phi=np.arctan2(y,x) #angle of all ring points
      rad=np.sqrt(x**2+y**2)
      if fullCircle:
         bins=np.linspace(-np.pi,np.pi,Nbins+1)
      else:
         bins=np.linspace(phi.min(),phi.max(),Nbins+1)
      dHX=np.diff(bins).mean()
      P=np.linspace(bins[0]+dHX/2.0,bins[-1]-dHX/2.0,len(bins)-1)
      
      R=[]
      Z=[]
      for phi_l,phi_h in zip(bins[0:-1],bins[1:]):
         idx=((phi>phi_l) & (phi<=phi_h))
         if len(rad[idx])>minPts:
            if type(funcs) in (tuple,list):
               R.append(funcs[0](rad[idx]))
               Z.append(funcs[1](z[idx]))
            else:
               R.append(funcs(rad[idx]))
               Z.append(funcs(z[idx]))
         else:
            R.append(np.nan)
            Z.append(np.nan)
      Z=np.array(Z)+center[2]
      R=np.array(R)
      X=R*np.cos(P)+center[0]
      Y=R*np.sin(P)+center[1]
      
      if filterNans:
         idx = np.where((~np.isnan(X)|~np.isnan(Y)|~np.isnan(Z)|~np.isnan(R)))[0]
         X=X[idx]
         Y=Y[idx]
         Z=Z[idx]
         R=R[idx]
         P=P[idx]
      
      return (X,Y,Z,R,P)


   def crossLines(self,center,size=0.003,color=(255,0,0)):
      lines = []
      lines.append((color,np.array([center-[size,0,0],center+[size,0,0]])))
      lines.append((color,np.array([center-[0,size,0],center+[0,size,0]])))
      lines.append((color,np.array([center-[0,0,size],center+[0,0,size]])))
      return lines




   def argSpherical(self, center, radius, npa=None):
      if npa is None:
         npa = self.uv_xyz_rgb.copy()
      else:
         npa=npa.copy()
      
      rad = np.linalg.norm(npa[2:5] - np.matrix(center).reshape(3,1), 2, 0)
      
      return np.where(rad<radius)[0]

   def argBox(self, box, npa=None):
      if npa is None:
         npa = self.uv_xyz_rgb.copy()
      else:
         npa=npa.copy()
      
      # box is defined by rmin and rmax where rmin=(xmin,ymin,zmin) and rmax is the max equiv.
      rmin = box[0]
      rmax = box[1]
      
      idx = np.where((npa[2]>rmin[0])&(npa[2]<=rmax[0])&
                     (npa[3]>rmin[1])&(npa[3]<=rmax[1])&
                     (npa[4]>rmin[2])&(npa[4]<=rmax[2]))
      return idx[0]
      

   def _centerHistArrays(self,hist):
      HY=hist[0]
      dHX=np.diff(hist[1]).mean()
      HX=np.linspace(hist[1][0]+dHX/2.0,hist[1][-1]-dHX/2.0,len(HY))
      return HX,HY

   def argAng(self,pha,phaRange):
      idx1 = []
      idx2 = []
      idx = []
      if phaRange[0] < -np.pi and phaRange[1] > -np.pi:
         idx1=np.where(pha>phaRange[0]+np.pi*2)[0]
         idx2=np.where(pha>-np.pi)[0]
         idx=np.concatenate([idx1,idx2])
      elif phaRange[0] < -np.pi and phaRange[1] < -np.pi:
         idx = np.where((pha>phaRange[0]+np.pi*2)&(pha<phaRange[1]+np.pi*2))[0]
      elif phaRange[1] > np.pi and phaRange[0]< np.pi:
         idx1=np.where(pha<phaRange[1]-np.pi*2)[0]
         idx2=np.where(pha>phaRange[0])[0]
         idx=np.concatenate([idx1,idx2])
      elif phaRange[0] > np.pi and phaRange[1] > np.pi:
         idx = np.where((pha>phaRange[0]-np.pi*2)&(pha<phaRange[1]-np.pi*2))[0]
      else:
         idx = np.where((pha>phaRange[0])&(pha<phaRange[1]))[0]
      return idx
        

   def fit3DLineSVD(self,npa):
      data = npa[2:5].copy().T
      return self._fit3DLineSVD(data)

   def _fit3DLineSVD(self,data):
      ''' fit a 3d line with svd. this works well for lines without outliers
      '''
      # src: http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
      #data = npa[2:5].copy().T
      dmean = data.mean(0)

      # Do an SVD on the mean-centered data.
      uu, dd, vv = np.linalg.svd(data - dmean)
      
      return {"dir":vv[0],"cen":dmean}

   def fit3DLineRansac(self,npa):
      data = npa[2:5].copy().T
      model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=.001, max_trials=1000)
      #outliers = (inliers == False)
      idx = np.where(inliers)[0]
      return self._fit3DLineSVD(data[idx,:])

   def fitPlane(self,npa=None,residual_threshold=.002, max_trials=1000):
      data = npa[2:5].copy().T
      model_robust, inliers = ransac(data, PlaneModel3D, min_samples=10,
                               residual_threshold=residual_threshold, max_trials=max_trials)
      #outliers = (inliers == False)
      #idx = np.where(inliers)[0]
      #return self.fitPlaneLLSQ(npa[:,idx])
      return model_robust.params, 0,0
      
   def fitPlaneLLSQ(self,npa=None):
      if npa is None:
         npa = self.uv_xyz_rgb.copy()      
      
      XYZ = npa[2:5].T
      p0 = [0,0,1,1]
      PM = PlaneModel3D()
      PM.estimate(XYZ)
      sol=PM.params
      res0 = (PM.residuals(XYZ,p0)**2).sum()
      res = (PM.residuals(XYZ,sol)**2).sum()
      
      return sol,res0,res
         


   def resamplePC(self,voxelSize,npa=None,minPts=0,snormFilter=False,snormAx=[0,0,1],snormThresh=0.9,calcSnorm=True): #TODO: make this more efficient!!!
      ''' a simple voxel downsampler. this can be made way more efficient if i had time!!!
      '''
      if npa is None:
         npa = self.uv_xyz_rgb.copy()

      rNmin = npa[2:5].min(1)
      rNmax = npa[2:5].max(1)
      xNmin,yNmin,zNmin = rNmin
      xNmax,yNmax,zNmax = rNmax
      
      medPts = []
      surfNorms = []
      for u in np.linspace(xNmin,xNmax,(xNmax-xNmin)/voxelSize+1):
         for v in np.linspace(yNmin,yNmax,(yNmax-yNmin)/voxelSize+1):
            for w in np.linspace(zNmin,zNmax,(zNmax-zNmin)/voxelSize+1):
               rmin = (u,v,w)
               rmax = (u+voxelSize,v+voxelSize,w+voxelSize)
               idx = self.argBox((rmin,rmax),npa)
               if len(idx)>minPts:
                  if calcSnorm:
                     pPar,res0,res1=self.fitPlane(npa[:,idx],residual_threshold=0.001,max_trials=20)
                     sNorm = pPar[:3]/np.linalg.norm(pPar[:3],2)
                  else:
                     sNorm = 1
                     
                  if snormFilter:
                     if np.dot(sNorm,snormAx)>snormThresh:
                        surfNorms.append(sNorm)
                        medPts.append(npa[:,idx].mean(1))
                  else:
                     surfNorms.append(sNorm)
                     medPts.append(npa[:,idx].mean(1))
      medPts = np.array(medPts).T
      surfNorms = np.array(surfNorms).T
      return medPts, surfNorms

   def show(self,npa=None,lines=[],showAxes=False):
      ''' show a point cloud. if desired, axes can be showed or additional lines can be plotted
          npa   ... 8xN array of point cloud data uv xyz rgb vertically stacked
          lines ... an array of (color,line) tuples. color is an rgb tuple while 
                    line is a list of 3d coords
          showAxes ... shows the xyz axes if true 
      '''
      if npa is None:
         npa = self.uv_xyz_rgb
         
      PCV=PointCloudViewer.VtkPointCloud()
      PCV.fromNumpyArray(npa)
      
      for color,line in lines:
         PCV.addLine(line,color)
         
      PCV.show(showAxes)
      
   def writeNPY(self,fn):
      np.save(fn, self.uv_xyz_rgb)

   def loadNPY(self,fn):
      self.uv_xyz_rgb = np.load(fn)
 
   def writePCD(self,fn):
      write_pcd._write_pcd(fn, self.uv_xyz_rgb[2:])

   def writePLY(self,fn):
      write_pcd._write_ply(fn, self.uv_xyz_rgb[2:])
      
   def loadFromPLY(self,fn):
      with open(fn, "r") as f:
         dstr=f.read()
         D=dstr.split("\n")
         sIdx=D.index("end_header")+1
         N=len(D[sIdx:])
         npd=np.zeros((8,N))
         for col,line in enumerate(D[sIdx:]):
             if line == "":
                 break
             dl = line.strip().split(" ")
             for row,d in enumerate(dl):
                 if row>=6:
                     break
                 npd[row+2,col]=float(d)
         self.uv_xyz_rgb=npd

if __name__ == "__main__":
   mat=np.random.rand(8,100)
   pc = PointCloud(mat)
   pc.show()



      
