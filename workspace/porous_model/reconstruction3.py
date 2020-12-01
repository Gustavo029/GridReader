import sys,os
pyEFVLibPath = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
workspacePath = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path += [pyEFVLibPath, workspacePath]

import PyEFVLib
from colorplot import colorplot
from matplotlib import pyplot as plt
import numpy as np

def reconstruct2D(grid,Fx,Fy,Fxx,Fyy):
	X,Y = zip(*[v.getCoordinates()[:-1] for v in grid.vertices])
	Xip,Yip = zip( *[innerFace.centroid.getCoordinates()[:-1] for element in grid.elements for innerFace in element.innerFaces] )

	X,Y=np.array(X),np.array(Y)
	Xip,Yip=np.array(Xip),np.array(Yip)

	fieldXAtVertices = Fx(X,Y)
	fieldYAtVertices = Fy(X,Y)
	fieldXAtIP = Fx(Xip,Yip)
	fieldYAtIP = Fy(Xip,Yip)

	divFieldAtVertices = Fxx(X,Y) + Fyy(X,Y)	
	r1DivFieldAtVertices = grid.vertices.size * [0,]

	i = 0
	for element in grid.elements:
		elementXFieldVector = np.array( [fieldXAtVertices[vertex.handle] for vertex in element.vertices] )
		elementYFieldVector = np.array( [fieldYAtVertices[vertex.handle] for vertex in element.vertices] )
		
		for innerFaceIndex, innerFace in enumerate(element.innerFaces):
			backwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][0] ]
			forwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][1] ]

			shapeFunctionValues = element.shape.innerFaceShapeFunctionValues[innerFaceIndex]

			xValAtIP = np.dot(elementXFieldVector, shapeFunctionValues)
			yValAtIP = np.dot(elementYFieldVector, shapeFunctionValues)
			area = innerFace.area.getCoordinates()[:-1]

			r1DivFieldAtVertices[backwardVertex.handle] += np.dot( (xValAtIP, yValAtIP), area )
			r1DivFieldAtVertices[forwardVertex.handle] -= np.dot( (xValAtIP, yValAtIP), area )

			i += 1
	r1DivFieldAtVertices = [ div/vertex.volume for vertex, div in zip( grid.vertices, r1DivFieldAtVertices ) ]

	boundaryVertices = [ vertex.handle for boundary in grid.boundaries for vertex in boundary.vertices ]
	divFieldAtVertices, r1DivFieldAtVertices = zip(*[ (div, r1div) for div,r1div,vertex in zip(divFieldAtVertices,r1DivFieldAtVertices,grid.vertices) if vertex.handle not in boundaryVertices ])

	divFieldAtVertices, r1DivFieldAtVertices = np.array(divFieldAtVertices), np.array(r1DivFieldAtVertices)

	print("\nReconstructed values at integration points")
	print(f"Max difference = {max(abs(divFieldAtVertices-r1DivFieldAtVertices)) :.4f}, Field range = [{min(divFieldAtVertices) :.4f}, {max(divFieldAtVertices) :.4f}]\t| {100*(max(abs(divFieldAtVertices-r1DivFieldAtVertices)))/(max(abs(divFieldAtVertices))):.2f}%")


	divFieldAtVertices = Fxx(X,Y) + Fyy(X,Y)
	r2DivFieldAtVertices = grid.vertices.size * [0,]

	i = 0
	for element in grid.elements:
		for innerFaceIndex, innerFace in enumerate(element.innerFaces):
			backwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][0] ]
			forwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][1] ]

			xValAtIP = fieldXAtIP[i]
			yValAtIP = fieldYAtIP[i]
			area = innerFace.area.getCoordinates()[:-1]

			r2DivFieldAtVertices[backwardVertex.handle] += np.dot( (xValAtIP, yValAtIP), area )
			r2DivFieldAtVertices[forwardVertex.handle] -= np.dot( (xValAtIP, yValAtIP), area )

			i += 1
	r2DivFieldAtVertices = [ div/vertex.volume for vertex, div in zip( grid.vertices, r2DivFieldAtVertices ) ]

	boundaryVertices = [ vertex.handle for boundary in grid.boundaries for vertex in boundary.vertices ]
	divFieldAtVertices, r2DivFieldAtVertices = zip(*[ (div, r2div) for div,r2div,vertex in zip(divFieldAtVertices,r2DivFieldAtVertices,grid.vertices) if vertex.handle not in boundaryVertices ])

	divFieldAtVertices, r2DivFieldAtVertices = np.array(divFieldAtVertices), np.array(r2DivFieldAtVertices)

	print("\nAnalytical values at integration points")
	print(f"Max difference = {max(abs(divFieldAtVertices-r2DivFieldAtVertices)) :.4f}, Field range = [{min(divFieldAtVertices) :.4f}, {max(divFieldAtVertices) :.4f}]\t| {100*(max(abs(divFieldAtVertices-r2DivFieldAtVertices)))/(max(abs(divFieldAtVertices))):.2f}%")

def reconstruct3D(grid,Fx,Fy,Fz,Fxx,Fyy,Fzz):
	X,Y,Z = zip(*[v.getCoordinates() for v in grid.vertices])
	Xip,Yip,Zip = zip( *[innerFace.centroid.getCoordinates() for element in grid.elements for innerFace in element.innerFaces] )

	X,Y,Z=np.array(X),np.array(Y),np.array(Z)
	Xip,Yip,Zip=np.array(Xip),np.array(Yip),np.array(Zip)

	fieldXAtVertices = Fx(X,Y,Z)
	fieldYAtVertices = Fy(X,Y,Z)
	fieldZAtVertices = Fz(X,Y,Z)
	fieldXAtIP = Fx(Xip,Yip,Zip)
	fieldYAtIP = Fy(Xip,Yip,Zip)
	fieldZAtIP = Fz(Xip,Yip,Zip)

	divFieldAtVertices = [Fxx(x,y,z) + Fyy(x,y,z) + Fzz(x,y,z) for x,y,z in zip(X,Y,Z)]
	r1DivFieldAtVertices = grid.vertices.size * [0,]

	i = 0
	for element in grid.elements:
		elementXFieldVector = np.array( [fieldXAtVertices[vertex.handle] for vertex in element.vertices] )
		elementYFieldVector = np.array( [fieldYAtVertices[vertex.handle] for vertex in element.vertices] )
		elementZFieldVector = np.array( [fieldZAtVertices[vertex.handle] for vertex in element.vertices] )
		
		for innerFaceIndex, innerFace in enumerate(element.innerFaces):
			backwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][0] ]
			forwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][1] ]

			shapeFunctionValues = element.shape.innerFaceShapeFunctionValues[innerFaceIndex]

			xValAtIP = np.dot(elementXFieldVector, shapeFunctionValues)
			yValAtIP = np.dot(elementYFieldVector, shapeFunctionValues)
			zValAtIP = np.dot(elementZFieldVector, shapeFunctionValues)
			area = innerFace.area.getCoordinates()

			r1DivFieldAtVertices[backwardVertex.handle] += np.dot( (xValAtIP, yValAtIP, zValAtIP), area )
			r1DivFieldAtVertices[forwardVertex.handle] -= np.dot( (xValAtIP, yValAtIP, zValAtIP), area )

			i += 1
	r1DivFieldAtVertices = [ div/vertex.volume for vertex, div in zip( grid.vertices, r1DivFieldAtVertices ) ]

	boundaryVertices = [ vertex.handle for boundary in grid.boundaries for vertex in boundary.vertices ]
	divFieldAtVertices, r1DivFieldAtVertices = zip(*[ (div, r1div) for div,r1div,vertex in zip(divFieldAtVertices,r1DivFieldAtVertices,grid.vertices) if vertex.handle not in boundaryVertices ])

	divFieldAtVertices, r1DivFieldAtVertices = np.array(divFieldAtVertices), np.array(r1DivFieldAtVertices)

	print("\nReconstructed values at integration points")
	print(f"Max difference = {max(abs(divFieldAtVertices-r1DivFieldAtVertices)) :.4f}, Field range = [{min(divFieldAtVertices) :.4f}, {max(divFieldAtVertices) :.4f}]\t| {100*(max(abs(divFieldAtVertices-r1DivFieldAtVertices)))/(max(abs(divFieldAtVertices))):.2f}%")


	divFieldAtVertices = [Fxx(x,y,z) + Fyy(x,y,z) + Fzz(x,y,z) for x,y,z in zip(X,Y,Z)]
	r2DivFieldAtVertices = grid.vertices.size * [0,]

	i = 0
	for element in grid.elements:
		for innerFaceIndex, innerFace in enumerate(element.innerFaces):
			backwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][0] ]
			forwardVertex = element.vertices[ element.shape.innerFaceNeighborVertices[innerFaceIndex][1] ]

			xValAtIP = fieldXAtIP[i]
			yValAtIP = fieldYAtIP[i]
			zValAtIP = fieldZAtIP[i]
			area = innerFace.area.getCoordinates()

			r2DivFieldAtVertices[backwardVertex.handle] += np.dot( (xValAtIP, yValAtIP, zValAtIP), area )
			r2DivFieldAtVertices[forwardVertex.handle] -= np.dot( (xValAtIP, yValAtIP, zValAtIP), area )

			i += 1
	r2DivFieldAtVertices = [ div/vertex.volume for vertex, div in zip( grid.vertices, r2DivFieldAtVertices ) ]

	boundaryVertices = [ vertex.handle for boundary in grid.boundaries for vertex in boundary.vertices ]
	divFieldAtVertices, r2DivFieldAtVertices = zip(*[ (div, r2div) for div,r2div,vertex in zip(divFieldAtVertices,r2DivFieldAtVertices,grid.vertices) if vertex.handle not in boundaryVertices ])

	divFieldAtVertices, r2DivFieldAtVertices = np.array(divFieldAtVertices), np.array(r2DivFieldAtVertices)

	print("\nAnalytical values at integration points")
	print(f"Max difference = {max(abs(divFieldAtVertices-r2DivFieldAtVertices)) :.4f}, Field range = [{min(divFieldAtVertices) :.4f}, {max(divFieldAtVertices) :.4f}]\t| {100*(max(abs(divFieldAtVertices-r2DivFieldAtVertices)))/(max(abs(divFieldAtVertices))):.2f}%")


if __name__ == "__main__":
	for meshName in ["Fine.msh", "10x10.msh"]:
		grid = PyEFVLib.read( os.path.join(pyEFVLibPath, "meshes", "msh", "2D", meshName) )
		print("------------------------------------\n", meshName)

		Fx = lambda x,y: x+y
		Fy = lambda x,y: x*y
		Fxx = lambda x,y: 1
		Fyy = lambda x,y: x
		print("\n__________________")
		print("(x+y, x*y)")
		reconstruct2D(grid,Fx,Fy,Fxx,Fyy)

		Fx = lambda x,y: x**2 + y**2
		Fy = lambda x,y: np.log(y+1) + 1/(x+1)
		Fxx = lambda x,y: 2*x
		Fyy = lambda x,y: 1/(y+1)
		print("\n__________________")
		print("(x^2 + y^2, ln(y) + 1/x)")
		reconstruct2D(grid,Fx,Fy,Fxx,Fyy)

		Fx = lambda x,y: -np.exp(-x) * np.sin(y) + x
		Fy = lambda x,y: np.exp(-x) * np.cos(y)
		Fxx = lambda x,y: np.exp(-x) * np.sin(y) + 1
		Fyy = lambda x,y: -np.exp(-x) * np.sin(y)
		print("\n__________________")
		print("(-exp(-x) * sin(y) + x, exp(-x) * cos(y))")
		reconstruct2D(grid,Fx,Fy,Fxx,Fyy)

	for meshName in ["Hexas.msh", "Pyrams.msh"]:
		grid = PyEFVLib.read( os.path.join(pyEFVLibPath, "meshes", "msh", "3D", meshName) )
		print("------------------------------------\n", meshName)
		
		Fx = lambda x,y,z: 2*x
		Fy = lambda x,y,z: 2*y
		Fz = lambda x,y,z: 2*z
		Fxx = lambda x,y,z: 2
		Fyy = lambda x,y,z: 2
		Fzz = lambda x,y,z: 2
		print("\n__________________")
		print("(2x, 2y, 2z)")
		reconstruct3D(grid,Fx,Fy,Fz,Fxx,Fyy,Fzz)

		Fx = lambda x,y,z: y/(z+1)
		Fy = lambda x,y,z: x/(z+1)
		Fz = lambda x,y,z: -x*y/(z+1)**2
		Fxx = lambda x,y,z: 0
		Fyy = lambda x,y,z: 0
		Fzz = lambda x,y,z: (2*x*y)/(z+1)**3
		print("\n__________________")
		print("(y/(z+1), x/(z+1), -xy/(z+1)^2)")
		reconstruct3D(grid,Fx,Fy,Fz,Fxx,Fyy,Fzz)