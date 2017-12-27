# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 11 17:53:12 2013
#
# @author: Sukhbinder
# """
#
# import pyvtk
# import numpy as np
#
# class VtkPointCloud:
#     def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e7):
#
#         self.maxNumPoints = maxNumPoints
#         self.vtkPolyData = vtk.PolyData()
#         self.clearPoints()
#         mapper = vtk.PolyDataMapper()
#         mapper.SetInputData(self.vtkPolyData)
#         mapper.SetColorModeToDefault()
#         mapper.SetScalarRange(zMin, zMax)
#         mapper.SetScalarVisibility(1)
#         self.vtkActor = vtk.Actor()
#         self.vtkActor.SetMapper(mapper)
#
#     def addPoint(self, point,color):
#         if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
#             pointId = self.vtkPoints.InsertNextPoint(point[:])
#             self.vtkDepth.InsertNextValue(point[2])
#             self.vtkColor.InsertNextTupleValue(color)
#             self.vtkCells.InsertNextCell(1)
#             self.vtkCells.InsertCellPoint(pointId)
#         self.vtkCells.Modified()
#         self.vtkPoints.Modified()
#         self.vtkColor.Modified()
#         self.vtkDepth.Modified()
#
#     def addLine(self, lineIn,color):
#         points = vtk.vtkPoints()
#         lines = vtk.vtkCellArray()
#         points.SetNumberOfPoints(lineIn.shape[0])
#         lines.InsertNextCell(lineIn.shape[0])
#         colors = vtk.vtkUnsignedCharArray()
#         colors.SetNumberOfComponents(3)
#         for k,point in enumerate(lineIn):
#            points.SetPoint(k,*point)
#            lines.InsertCellPoint(k)
#            colors.InsertNextTupleValue(color)
#         polygon = vtk.vtkPolyData()
#         polygon.SetPoints(points)
#         polygon.SetLines(lines)
#         polygon.GetCellData().SetScalars(colors)
#
#         polygonMapper = vtk.vtkPolyDataMapper()
#         polygonMapper.SetInputData(polygon)
#         polygonMapper.Update()
#
#         polygonActor = vtk.vtkActor()
#         polygonActor.SetMapper(polygonMapper)
#         self.PolyLineActors.append(polygonActor)
#
#     def clearPoints(self):
#         self.vtkPoints = vtk.vtkPoints()
#         self.vtkCells = vtk.vtkCellArray()
#         self.vtkDepth = vtk.vtkDoubleArray()
#         self.vtkDepth.SetName('DepthArray')
#         self.vtkColor = vtk.vtkUnsignedCharArray()
#         self.vtkColor.SetNumberOfComponents(3)
#         self.vtkColor.SetName('ColorArray')
#         self.vtkPolyData.SetPoints(self.vtkPoints)
#         self.vtkPolyData.SetVerts(self.vtkCells)
#         self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
#         self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
#         self.vtkPolyData.GetPointData().SetScalars(self.vtkColor)
#         self.vtkPolyData.GetPointData().SetActiveScalars('ColorArray')
#         self.PolyLineActors = []
#
#     def show(self,showAxes=False):
#        # Renderer
#        renderer = vtk.vtkRenderer()
#        renderer.AddActor(self.vtkActor)
#
#        #lines
#        for lactor in self.PolyLineActors:
#          renderer.AddActor(lactor)
#
#        #renderer.SetBackground(.2, .3, .4)
#        renderer.SetBackground(0.0, 0.0, 0.0)
#
#        # axes
#        if showAxes:
#           axes = vtk.vtkAxesActor()
#           #axes.SetTotalLength(.1, .1, .1)
#           #axes.SetTotalLength(33.0, 33.0, 33.0)
#           axes.SetNormalizedShaftLength(1.0, 1.0, 1.0)
#           axes.SetNormalizedTipLength(0.05, 0.05, 0.05)
#           renderer.AddActor(axes)
#
#        renderer.ResetCamera()
#
#        # Render Window
#        renderWindow = vtk.vtkRenderWindow()
#        renderWindow.AddRenderer(renderer)
#
#        # Interactor
#        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
#        renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
#        renderWindowInteractor.SetRenderWindow(renderWindow)
#
#        # Begin Interaction
#        renderWindow.Render()
#        renderWindow.SetWindowName("PointCloud")
#        renderWindowInteractor.Start()
#
#
#     def fromNumpyArray(self,uv_xyz_rgb):
#        for point in uv_xyz_rgb.T:
#          self.addPoint(point[2:5],point[5:8].astype(int))
#
#     def loadFromNPFile(self,filename):
#        data = np.load(filename) #genfromtxt(filename,dtype=float,skiprows=2,usecols=[0,1,2])
#        self.fromNumpyArray(data)
#
#    def getXYZ(dimg, fx, fy, cx, cy):
#        ifx, ify = 1.0 / fx, 1.0 / fy
#        rows, cols = dimg.shape
#        # build row and col factors for uv to xy mapping. needs only to be
#        # calculated once.
#        c = np.arange(cols)
#        r = np.arange(rows)
#        c_ = (c - cx) * ifx
#        r_ = (r - cy) * ify
#
#        Z = dimg.astype(float)
#        X = Z * c_
#        Y = (Z.T * r_).T
#
#        X = X.flatten()
#        Y = Y.flatten()
#        Z = Z.flatten()
#
#        v, u = np.meshgrid(c, r)
#
#        # filteres all Z outside 0.5 and 3m away
#        idx = (dimg.flatten() >= 0.0)
#
#        XYZ = np.matrix([X[idx], Y[idx], Z[idx]])  # [3, r*c] = [X, Y, Z].T
#        uv = np.matrix([u.flatten()[idx], v.flatten()[idx]])
#        rgb = np.full_like(XYZ, 255)
#        uv_xyz_rgb = np.vstack((uv, XYZ, rgb))
#        return uv_xyz_rgb
#
#
# if __name__ == '__main__':
#     import sys
#
#
#     if len(sys.argv) < 2:
#          print('Usage: xyzviewer.py itemfile')
#          sys.exit()
#     pointCloud = VtkPointCloud()
#     pointCloud.loadFromNPFile(sys.argv[1])
#     pointCloud.show()
#
#
