import numpy as np
import subprocess, os, sys
import meshio
from PyEFVLib.simulation.Saver import Saver
from PyEFVLib.geometry.Shape import Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid

class MeshioSaver(Saver):
	# line triangle quad tetra pyramid wedge hexahedron
	# Formats:
	# msh mdpa ply stl vtk vtu xdmf xmf cgns h5m med inp mesh meshb bdf fem nas obj off post post.gz dato dato.gz su2 svg dat tec ugrid wkt 
	def __init__(self, grid, outputPath, basePath, extension, fileName="Results", **kwargs): 
		Saver.__init__(self, grid, outputPath, basePath, extension, fileName)

	def finalize(self):
		points = np.array( [v.getCoordinates() for v in self.grid.vertices] )
		
		meshioShapes   = ["triangle", "quad", "tetra", "pyramid", "wedge", "hexahedron"]
		pyEFVLibShapes = [Triangle, Quadrilateral, Tetrahedron, Pyramid, Prism, Hexahedron]

		cells = [ ( shape , np.array([[vertex.handle for vertex in element.vertices] for element in self.grid.elements if element.shape.__class__ == shapeClass], dtype=np.uint64) ) for shape, shapeClass in zip(meshioShapes, pyEFVLibShapes) ]
		cells = [ cell for cell in cells if cell[1].size ]

		data  = { fieldName : self.fields[fieldName][-1] for fieldName in self.fields }
		
		meshioMesh = meshio.Mesh( points, cells, point_data=data )
		meshioMesh.write( self.outputPath )

		self.finalized = True

