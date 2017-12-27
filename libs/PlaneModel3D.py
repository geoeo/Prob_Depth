import skimage.measure
from scipy.optimize import leastsq
from skimage.measure import LineModelND, ransac
import skimage.measure.fit
import numpy as np

class FP:      
   def fitPlane(self,npa=None):
      if npa is None:
         npa = self.uv_xyz_rgb.copy()      
      XYZ = npa[2:5]
      p0 = [0,0,1,1]
      sol,res0,res=self._fitPlane(XYZ,p0)
      return sol,res0,res

   def _fitPlane(self,XYZ,p0=None):
      if p0 is None:
         p0=[1,1,1,XYZ[2,:].mean()]
      f_min = lambda X,p: ((p[0:3]*X.T).sum(axis=1) + p[3]) / np.linalg.norm(p)
      residuals = lambda params, signal, X: f_min(X, params)
      sol = leastsq(residuals, p0, args=(None, XYZ))[0]
      residual0 = (f_min(XYZ, p0)**2).sum()
      residual  = (f_min(XYZ, sol)**2).sum()
      return sol, residual0, residual

class PlaneModel3D(skimage.measure.fit.BaseModel):
   def f_min(self,X,p):
      return (np.dot(X,p[0:3]) + p[3]) / np.linalg.norm(p)

   def estimate(self, data):
      if self.params is None:
         self.params=[1,1,1,data[2,:].mean()]      

      params = self.params
      
      residuals = lambda params, signal, X: self.f_min(X, params)
      
      params = leastsq(residuals, params, args=(None, data))[0]
      
      self.params = params
      return True

   def residuals(self, data, params=None):
      if params is None:
          params = self.params
      assert params is not None
      if len(params) != 4:
          raise ValueError('Parameters are defined by 4 sets.')
      
      res = self.f_min(data, params)
      return res

'''
   def predict(self, x, axis=0, params=None):
      if params is None:
          params = self.params
      assert params is not None
      if len(params) != 2:
         raise ValueError('Parameters are defined by 2 sets.')

      origin, direction = params

      if direction[axis] == 0:
         # line parallel to axis
         raise ValueError('Line parallel to axis %s' % axis)

      l = (x - origin[axis]) / direction[axis]
      data = origin + l[..., np.newaxis] * direction
      return data

'''
