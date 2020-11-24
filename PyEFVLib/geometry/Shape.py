import numpy as np
from PyEFVLib.geometry.Point import Point

def areCoplanar(p1,p2,p3,p4):
	return np.dot( (p2-p1), np.cross((p3-p1), (p4-p1)) ) == 0

class Triangle:
	dimension						   = 2
	numberOfInnerFaces				   = 3
	numberOfFacets					   = 3
	subelementTransformedVolumes	   = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
	innerFaceShapeFunctionValues 	   = np.array([[5.0/12.0, 5.0/12.0, 1.0/6.0],[1.0/6.0, 5.0/12.0, 5.0/12.0],[5.0/12.0, 1.0/6.0, 5.0/12.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1],[1, 2],[2, 0]])
	subelementShapeFunctionValues 	   = None
	subelementShapeFunctionDerivatives = np.array([[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0]]])
	facetVerticesIndices 			   = np.array([[1, 0],[2, 1],[0, 2]])
	outerFaceShapeFunctionValues 	   = np.array([[[1.0/4.0, 3.0/4.0, 0.0/1.0],[3.0/4.0, 1.0/4.0, 0.0/1.0]],[[0.0/1.0, 1.0/4.0, 3.0/4.0],[0.0/1.0, 3.0/4.0, 1.0/4.0]],[[3.0/4.0, 0.0/1.0, 1.0/4.0],[1.0/4.0, 0.0/1.0, 3.0/4.0]]])
	vertexShapeFunctionDerivatives	   = np.array([[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]],[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]],[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]]])

	def __init__(self, element):
		self.element = element

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 3:
			return True
		else:
			return False

	@staticmethod
	def getInnerFaceAreaVector(local, elementCentroid, elementVertices):
		vertex1 = elementVertices[Triangle.innerFaceNeighborVertices[local][0]]
		vertex2 = elementVertices[Triangle.innerFaceNeighborVertices[local][1]]
		areaVectorCoords = ( elementCentroid - (vertex1 + vertex2)/2.0 ).getCoordinates()
		return Point(areaVectorCoords[1], -areaVectorCoords[0], 0.0)

class Quadrilateral:
	dimension						   = 2
	numberOfInnerFaces				   = 4
	numberOfFacets 					   = 4
	subelementTransformedVolumes	   = np.array([1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0])
	innerFaceShapeFunctionValues	   = np.array([[3.0/8.0, 3.0/8.0, 1.0/8.0, 1.0/8.0], [1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0], [1.0/8.0, 1.0/8.0, 3.0/8.0, 3.0/8.0], [3.0/8.0, 1.0/8.0, 1.0/8.0, 3.0/8.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-3.0/4.0, -1.0/2.0], [3.0/4.0, -1.0/2.0], [1.0/4.0, 1.0/2.0], [-1.0/4.0, 1.0/2.0]],[[-1.0/2.0, -1.0/4.0], [1.0/2.0, -3.0/4.0], [1.0/2.0, 3.0/4.0], [-1.0/2.0, 1.0/4.0]],[[-1.0/4.0, -1.0/2.0], [1.0/4.0, -1.0/2.0], [3.0/4.0, 1.0/2.0], [-3.0/4.0, 1.0/2.0]],[[-1.0/2.0, -3.0/4.0], [1.0/2.0, -1.0/4.0], [1.0/2.0, 1.0/4.0], [-1.0/2.0, 3.0/4.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1],[1, 2],[2, 3],[3, 0]])
	subelementShapeFunctionValues	   = np.array([[9.0/16.0, 3.0/16.0, 1.0/16.0, 3.0/16.0],[3.0/16.0, 9.0/16.0, 3.0/16.0, 1.0/16.0],[1.0/16.0, 3.0/16.0, 9.0/16.0, 3.0/16.0],[3.0/16.0, 1.0/16.0, 3.0/16.0, 9.0/16.0]])
	subelementShapeFunctionDerivatives = np.array([[[-3.0/4.0, -3.0/4.0], [3.0/4.0, -1.0/4.0], [1.0/4.0, 1.0/4.0], [-1.0/4.0, 3.0/4.0]],[[-3.0/4.0, -1.0/4.0], [3.0/4.0, -3.0/4.0], [1.0/4.0, 3.0/4.0], [-1.0/4.0, 1.0/4.0]],[[-1.0/4.0, -1.0/4.0], [1.0/4.0, -3.0/4.0], [3.0/4.0, 3.0/4.0], [-3.0/4.0, 1.0/4.0]],[[-1.0/4.0, -3.0/4.0], [1.0/4.0, -1.0/4.0], [3.0/4.0, 1.0/4.0], [-3.0/4.0, 3.0/4.0]]])
	facetVerticesIndices 			   = np.array([[1, 0],[2, 1],[3, 2],[0, 3]])
	outerFaceShapeFunctionValues	   = np.array([[[1.0/4.0, 3.0/4.0, 0.0/1.0, 0.0/1.0],[3.0/4.0, 1.0/4.0, 0.0/1.0, 0.0/1.0]],[[0.0/1.0, 1.0/4.0, 3.0/4.0, 0.0/1.0],[0.0/1.0, 3.0/4.0, 1.0/4.0, 0.0/1.0]],[[0.0/1.0, 0.0/1.0, 1.0/4.0, 3.0/4.0],[0.0/1.0, 0.0/1.0, 3.0/4.0, 1.0/4.0]],[[3.0/4.0, 0.0/1.0, 0.0/1.0, 1.0/4.0],[1.0/4.0, 0.0/1.0, 0.0/1.0, 3.0/4.0]]])
	vertexShapeFunctionDerivatives	   = np.array([[[-1.0,-1.0],[1.0,0.0],[0.0,0.0],[0.0,1.0]],[[-1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]],[[0.0,0.0],[0.0,-1.0],[1.0,1.0],[-1.0,0.0]],[[0.0,-1.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]]])

	def __init__(self, element):
		self.element = element			

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 4 and areCoplanar(*[v.getCoordinates() for v in elem.vertices]):
			return True
		else:
			return False

	@staticmethod
	def getInnerFaceAreaVector(local, elementCentroid, elementVertices):
		vertex1 = elementVertices[Quadrilateral.innerFaceNeighborVertices[local][0]]
		vertex2 = elementVertices[Quadrilateral.innerFaceNeighborVertices[local][1]]
		areaVectorCoords = ( elementCentroid - (vertex1 + vertex2)/2.0 ).getCoordinates()
		return Point(areaVectorCoords[1], -areaVectorCoords[0], 0.0)

class Tetrahedron:
	dimension						   = 3
	numberOfInnerFaces				   = 6
	numberOfFacets 					   = 4
	subelementTransformedVolumes	   = np.array([1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0])
	innerFaceShapeFunctionValues	   = np.array([[17.0/48.0, 17.0/48.0, 7.0/48.0, 7.0/48.0],[7.0/48.0, 17.0/48.0, 17.0/48.0, 7.0/48.0],[17.0/48.0, 7.0/48.0, 17.0/48.0, 7.0/48.0],[17.0/48.0, 7.0/48.0, 7.0/48.0, 17.0/48.0],[7.0/48.0, 7.0/48.0, 17.0/48.0, 17.0/48.0],[7.0/48.0, 17.0/48.0, 7.0/48.0, 17.0/48.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1, 3, 2],[1, 2, 3, 0],[2, 0, 3, 1],[0, 3, 2, 1],[1, 3, 0, 2],[2, 3, 1, 0]])
	subelementShapeFunctionValues	   = np.array([[15.0/32.0, 17.0/96.0, 17.0/96.0, 17.0/96.0],[17.0/96.0, 15.0/32.0, 17.0/96.0, 17.0/96.0],[17.0/96.0, 17.0/96.0, 15.0/32.0, 17.0/96.0],[17.0/96.0, 17.0/96.0, 17.0/96.0, 15.0/32.0]])
	subelementShapeFunctionDerivatives = np.array([[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/1.0, -1.0/1.0, -1.0/1.0], [1.0/1.0, 0.0/1.0, 0.0/1.0], [0.0/1.0, 1.0/1.0, 0.0/1.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]]])
	facetVerticesIndices 			   = np.array([[0, 2, 1],[0, 3, 2],[0, 1, 3],[1, 2, 3]])
	outerFaceShapeFunctionValues	   = np.array([[[7.0/12.0, 5.0/24.0, 5.0/24.0, 0.0/1.0],[5.0/24.0, 5.0/24.0, 7.0/12.0, 0.0/1.0],[5.0/24.0, 7.0/12.0, 5.0/24.0, 0.0/1.0]],[[7.0/12.0, 0.0/1.0, 5.0/24.0, 5.0/24.0],[5.0/24.0, 0.0/1.0, 5.0/24.0, 7.0/12.0],[5.0/24.0, 0.0/1.0, 7.0/12.0, 5.0/24.0]],[[7.0/12.0, 5.0/24.0, 0.0/1.0, 5.0/24.0],[5.0/24.0, 7.0/12.0, 0.0/1.0, 5.0/24.0],[5.0/24.0, 5.0/24.0, 0.0/1.0, 7.0/12.0]],[[0.0/1.0, 7.0/12.0, 5.0/24.0, 5.0/24.0],[0.0/1.0, 5.0/24.0, 7.0/12.0, 5.0/24.0],[0.0/1.0, 5.0/24.0, 5.0/24.0, 7.0/12.0]]])

	def __init__(self, element):
		self.element = element			

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 4 and not areCoplanar(*[v.getCoordinates() for v in elem.vertices]):
			return True
		else:
			return False

	def getInnerFaceAreaVector(self, local, elementCentroid, elementVertices):
		f,b,n1,n2 = [ elementVertices[index].getCoordinates() for index in self.innerFaceNeighborVertices[local] ]
		fc1=(f+b+n1)/3.0
		fc2=(f+b+n2)/3.0
		mp=(f+b)/2.0
		ec=elementCentroid.getCoordinates()
		t1=np.cross(n1-mp, n2-mp)/2.0
		t2=np.cross(n1-ec, n2-ec)/2.0
		p=Point(*(t1+t2))
		return p

class Hexahedron:
	dimension						   = 3
	numberOfInnerFaces				   = 12
	numberOfFacets 					   = 6
	subelementTransformedVolumes	   = np.array([1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0])
	innerFaceShapeFunctionValues	   = np.array([[9.0/32.0, 9.0/32.0, 3.0/32.0, 3.0/32.0, 3.0/32.0, 3.0/32.0, 1.0/32.0, 1.0/32.0],[3.0/32.0, 9.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 3.0/32.0, 1.0/32.0],[3.0/32.0, 3.0/32.0, 9.0/32.0, 9.0/32.0, 1.0/32.0, 1.0/32.0, 3.0/32.0, 3.0/32.0],[9.0/32.0, 3.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 1.0/32.0, 3.0/32.0],[3.0/32.0, 3.0/32.0, 1.0/32.0, 1.0/32.0, 9.0/32.0, 9.0/32.0, 3.0/32.0, 3.0/32.0],[1.0/32.0, 3.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 9.0/32.0, 3.0/32.0],[1.0/32.0, 1.0/32.0, 3.0/32.0, 3.0/32.0, 3.0/32.0, 3.0/32.0, 9.0/32.0, 9.0/32.0],[3.0/32.0, 1.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 3.0/32.0, 9.0/32.0],[9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0],[3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0],[1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0],[3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0, 3.0/32.0, 1.0/32.0, 3.0/32.0, 9.0/32.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-9.0/16.0, -3.0/8.0, -3.0/8.0], [9.0/16.0, -3.0/8.0, -3.0/8.0], [3.0/16.0, 3.0/8.0, -1.0/8.0], [-3.0/16.0, 3.0/8.0, -1.0/8.0], [-3.0/16.0, -1.0/8.0, 3.0/8.0], [3.0/16.0, -1.0/8.0, 3.0/8.0], [1.0/16.0, 1.0/8.0, 1.0/8.0], [-1.0/16.0, 1.0/8.0, 1.0/8.0]],[[-3.0/8.0, -3.0/16.0, -1.0/8.0], [3.0/8.0, -9.0/16.0, -3.0/8.0], [3.0/8.0, 9.0/16.0, -3.0/8.0], [-3.0/8.0, 3.0/16.0, -1.0/8.0], [-1.0/8.0, -1.0/16.0, 1.0/8.0], [1.0/8.0, -3.0/16.0, 3.0/8.0], [1.0/8.0, 3.0/16.0, 3.0/8.0], [-1.0/8.0, 1.0/16.0, 1.0/8.0]],[[-3.0/16.0, -3.0/8.0, -1.0/8.0], [3.0/16.0, -3.0/8.0, -1.0/8.0], [9.0/16.0, 3.0/8.0, -3.0/8.0], [-9.0/16.0, 3.0/8.0, -3.0/8.0], [-1.0/16.0, -1.0/8.0, 1.0/8.0], [1.0/16.0, -1.0/8.0, 1.0/8.0], [3.0/16.0, 1.0/8.0, 3.0/8.0], [-3.0/16.0, 1.0/8.0, 3.0/8.0]],[[-3.0/8.0, -9.0/16.0, -3.0/8.0], [3.0/8.0, -3.0/16.0, -1.0/8.0], [3.0/8.0, 3.0/16.0, -1.0/8.0], [-3.0/8.0, 9.0/16.0, -3.0/8.0], [-1.0/8.0, -3.0/16.0, 3.0/8.0], [1.0/8.0, -1.0/16.0, 1.0/8.0], [1.0/8.0, 1.0/16.0, 1.0/8.0], [-1.0/8.0, 3.0/16.0, 3.0/8.0]],[[-3.0/16.0, -1.0/8.0, -3.0/8.0], [3.0/16.0, -1.0/8.0, -3.0/8.0], [1.0/16.0, 1.0/8.0, -1.0/8.0], [-1.0/16.0, 1.0/8.0, -1.0/8.0], [-9.0/16.0, -3.0/8.0, 3.0/8.0], [9.0/16.0, -3.0/8.0, 3.0/8.0], [3.0/16.0, 3.0/8.0, 1.0/8.0], [-3.0/16.0, 3.0/8.0, 1.0/8.0]],[[-1.0/8.0, -1.0/16.0, -1.0/8.0], [1.0/8.0, -3.0/16.0, -3.0/8.0], [1.0/8.0, 3.0/16.0, -3.0/8.0], [-1.0/8.0, 1.0/16.0, -1.0/8.0], [-3.0/8.0, -3.0/16.0, 1.0/8.0], [3.0/8.0, -9.0/16.0, 3.0/8.0], [3.0/8.0, 9.0/16.0, 3.0/8.0], [-3.0/8.0, 3.0/16.0, 1.0/8.0]],[[-1.0/16.0, -1.0/8.0, -1.0/8.0], [1.0/16.0, -1.0/8.0, -1.0/8.0], [3.0/16.0, 1.0/8.0, -3.0/8.0], [-3.0/16.0, 1.0/8.0, -3.0/8.0], [-3.0/16.0, -3.0/8.0, 1.0/8.0], [3.0/16.0, -3.0/8.0, 1.0/8.0], [9.0/16.0, 3.0/8.0, 3.0/8.0], [-9.0/16.0, 3.0/8.0, 3.0/8.0]],[[-1.0/8.0, -3.0/16.0, -3.0/8.0], [1.0/8.0, -1.0/16.0, -1.0/8.0], [1.0/8.0, 1.0/16.0, -1.0/8.0], [-1.0/8.0, 3.0/16.0, -3.0/8.0], [-3.0/8.0, -9.0/16.0, 3.0/8.0], [3.0/8.0, -3.0/16.0, 1.0/8.0], [3.0/8.0, 3.0/16.0, 1.0/8.0], [-3.0/8.0, 9.0/16.0, 3.0/8.0]],[[-3.0/8.0, -3.0/8.0, -9.0/16.0], [3.0/8.0, -1.0/8.0, -3.0/16.0], [1.0/8.0, 1.0/8.0, -1.0/16.0], [-1.0/8.0, 3.0/8.0, -3.0/16.0], [-3.0/8.0, -3.0/8.0, 9.0/16.0], [3.0/8.0, -1.0/8.0, 3.0/16.0], [1.0/8.0, 1.0/8.0, 1.0/16.0], [-1.0/8.0, 3.0/8.0, 3.0/16.0]],[[-3.0/8.0, -1.0/8.0, -3.0/16.0], [3.0/8.0, -3.0/8.0, -9.0/16.0], [1.0/8.0, 3.0/8.0, -3.0/16.0], [-1.0/8.0, 1.0/8.0, -1.0/16.0], [-3.0/8.0, -1.0/8.0, 3.0/16.0], [3.0/8.0, -3.0/8.0, 9.0/16.0], [1.0/8.0, 3.0/8.0, 3.0/16.0], [-1.0/8.0, 1.0/8.0, 1.0/16.0]],[[-1.0/8.0, -1.0/8.0, -1.0/16.0], [1.0/8.0, -3.0/8.0, -3.0/16.0], [3.0/8.0, 3.0/8.0, -9.0/16.0], [-3.0/8.0, 1.0/8.0, -3.0/16.0], [-1.0/8.0, -1.0/8.0, 1.0/16.0], [1.0/8.0, -3.0/8.0, 3.0/16.0], [3.0/8.0, 3.0/8.0, 9.0/16.0], [-3.0/8.0, 1.0/8.0, 3.0/16.0]],[[-1.0/8.0, -3.0/8.0, -3.0/16.0], [1.0/8.0, -1.0/8.0, -1.0/16.0], [3.0/8.0, 1.0/8.0, -3.0/16.0], [-3.0/8.0, 3.0/8.0, -9.0/16.0], [-1.0/8.0, -3.0/8.0, 3.0/16.0], [1.0/8.0, -1.0/8.0, 1.0/16.0], [3.0/8.0, 1.0/8.0, 3.0/16.0], [-3.0/8.0, 3.0/8.0, 9.0/16.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1, 4, 5, 2, 3],[1, 2, 5, 6, 3, 0],[2, 3, 6, 7, 0, 1],[3, 0, 7, 4, 1, 2],[4, 5, 6, 7, 0, 1],[5, 6, 7, 4, 1, 2],[6, 7, 4, 5, 2, 3],[7, 4, 5, 6, 3, 0],[4, 0, 1, 5, 3, 7],[5, 1, 2, 6, 4, 0],[6, 2, 3, 7, 5, 1],[7, 3, 4, 0, 6, 2]])
	subelementShapeFunctionValues	   = np.array([[27.0/64.0, 9.0/64.0, 3.0/64.0, 9.0/64.0, 9.0/64.0, 3.0/64.0, 1.0/64.0, 3.0/64.0],[9.0/64.0, 27.0/64.0, 9.0/64.0, 3.0/64.0, 3.0/64.0, 9.0/64.0, 3.0/64.0, 1.0/64.0],[3.0/64.0, 9.0/64.0, 27.0/64.0, 9.0/64.0, 1.0/64.0, 3.0/64.0, 9.0/64.0, 3.0/64.0],[9.0/64.0, 3.0/64.0, 9.0/64.0, 27.0/64.0, 3.0/64.0, 1.0/64.0, 3.0/64.0, 9.0/64.0],[9.0/64.0, 3.0/64.0, 1.0/64.0, 3.0/64.0, 27.0/64.0, 9.0/64.0, 3.0/64.0, 9.0/64.0],[3.0/64.0, 9.0/64.0, 3.0/64.0, 1.0/64.0, 9.0/64.0, 27.0/64.0, 9.0/64.0, 3.0/64.0],[1.0/64.0, 3.0/64.0, 9.0/64.0, 3.0/64.0, 3.0/64.0, 9.0/64.0, 27.0/64.0, 9.0/64.0],[3.0/64.0, 1.0/64.0, 3.0/64.0, 9.0/64.0, 9.0/64.0, 3.0/64.0, 9.0/64.0, 27.0/64.0]])
	subelementShapeFunctionDerivatives = np.array([[[-9.0/16.0, -9.0/16.0, -9.0/16.0], [9.0/16.0, -3.0/16.0, -3.0/16.0], [3.0/16.0, 3.0/16.0, -1.0/16.0], [-3.0/16.0, 9.0/16.0, -3.0/16.0], [-3.0/16.0, -3.0/16.0, 9.0/16.0], [3.0/16.0, -1.0/16.0, 3.0/16.0], [1.0/16.0, 1.0/16.0, 1.0/16.0], [-1.0/16.0, 3.0/16.0, 3.0/16.0]],[[-9.0/16.0, -3.0/16.0, -3.0/16.0], [9.0/16.0, -9.0/16.0, -9.0/16.0], [3.0/16.0, 9.0/16.0, -3.0/16.0], [-3.0/16.0, 3.0/16.0, -1.0/16.0], [-3.0/16.0, -1.0/16.0, 3.0/16.0], [3.0/16.0, -3.0/16.0, 9.0/16.0], [1.0/16.0, 3.0/16.0, 3.0/16.0], [-1.0/16.0, 1.0/16.0, 1.0/16.0]],[[-3.0/16.0, -3.0/16.0, -1.0/16.0], [3.0/16.0, -9.0/16.0, -3.0/16.0], [9.0/16.0, 9.0/16.0, -9.0/16.0], [-9.0/16.0, 3.0/16.0, -3.0/16.0], [-1.0/16.0, -1.0/16.0, 1.0/16.0], [1.0/16.0, -3.0/16.0, 3.0/16.0], [3.0/16.0, 3.0/16.0, 9.0/16.0], [-3.0/16.0, 1.0/16.0, 3.0/16.0]],[[-3.0/16.0, -9.0/16.0, -3.0/16.0], [3.0/16.0, -3.0/16.0, -1.0/16.0], [9.0/16.0, 3.0/16.0, -3.0/16.0], [-9.0/16.0, 9.0/16.0, -9.0/16.0], [-1.0/16.0, -3.0/16.0, 3.0/16.0], [1.0/16.0, -1.0/16.0, 1.0/16.0], [3.0/16.0, 1.0/16.0, 3.0/16.0], [-3.0/16.0, 3.0/16.0, 9.0/16.0]],[[-3.0/16.0, -3.0/16.0, -9.0/16.0], [3.0/16.0, -1.0/16.0, -3.0/16.0], [1.0/16.0, 1.0/16.0, -1.0/16.0], [-1.0/16.0, 3.0/16.0, -3.0/16.0], [-9.0/16.0, -9.0/16.0, 9.0/16.0], [9.0/16.0, -3.0/16.0, 3.0/16.0], [3.0/16.0, 3.0/16.0, 1.0/16.0], [-3.0/16.0, 9.0/16.0, 3.0/16.0]],[[-3.0/16.0, -1.0/16.0, -3.0/16.0], [3.0/16.0, -3.0/16.0, -9.0/16.0], [1.0/16.0, 3.0/16.0, -3.0/16.0], [-1.0/16.0, 1.0/16.0, -1.0/16.0], [-9.0/16.0, -3.0/16.0, 3.0/16.0], [9.0/16.0, -9.0/16.0, 9.0/16.0], [3.0/16.0, 9.0/16.0, 3.0/16.0], [-3.0/16.0, 3.0/16.0, 1.0/16.0]],[[-1.0/16.0, -1.0/16.0, -1.0/16.0], [1.0/16.0, -3.0/16.0, -3.0/16.0], [3.0/16.0, 3.0/16.0, -9.0/16.0], [-3.0/16.0, 1.0/16.0, -3.0/16.0], [-3.0/16.0, -3.0/16.0, 1.0/16.0], [3.0/16.0, -9.0/16.0, 3.0/16.0], [9.0/16.0, 9.0/16.0, 9.0/16.0], [-9.0/16.0, 3.0/16.0, 3.0/16.0]],[[-1.0/16.0, -3.0/16.0, -3.0/16.0], [1.0/16.0, -1.0/16.0, -1.0/16.0], [3.0/16.0, 1.0/16.0, -3.0/16.0], [-3.0/16.0, 3.0/16.0, -9.0/16.0], [-3.0/16.0, -9.0/16.0, 3.0/16.0], [3.0/16.0, -3.0/16.0, 1.0/16.0], [9.0/16.0, 3.0/16.0, 3.0/16.0], [-9.0/16.0, 9.0/16.0, 9.0/16.0]]])
	facetVerticesIndices 			   = np.array([[0, 3, 2, 1],[0, 4, 7, 3],[0, 1, 5, 4],[4, 5, 6, 7],[1, 2, 6, 5],[2, 3, 7, 6]])
	outerFaceShapeFunctionValues	   = np.array([[[9.0/16.0, 3.0/16.0, 1.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0],[3.0/16.0, 1.0/16.0, 3.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0],[1.0/16.0, 3.0/16.0, 9.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0],[3.0/16.0, 9.0/16.0, 3.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0]],[[9.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 1.0/16.0],[3.0/16.0, 0.0/1.0, 0.0/1.0, 1.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0],[1.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 9.0/16.0],[3.0/16.0, 0.0/1.0, 0.0/1.0, 9.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0]],[[9.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0],[3.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0],[1.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0],[3.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0]],[[0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 9.0/16.0, 3.0/16.0, 1.0/16.0, 3.0/16.0],[0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 9.0/16.0, 3.0/16.0, 1.0/16.0],[0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 1.0/16.0, 3.0/16.0, 9.0/16.0, 3.0/16.0],[0.0/1.0, 0.0/1.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 1.0/16.0, 3.0/16.0, 9.0/16.0]],[[0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0],[0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0],[0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0],[0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0]],[[0.0/1.0, 0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 1.0/16.0],[0.0/1.0, 0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0, 0.0/1.0, 1.0/16.0, 3.0/16.0],[0.0/1.0, 0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0, 0.0/1.0, 3.0/16.0, 9.0/16.0],[0.0/1.0, 0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0, 0.0/1.0, 9.0/16.0, 3.0/16.0]]])

	def __init__(self, element):
		self.element = element			

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 8:
			return True
		else:
			return False

	def getInnerFaceAreaVector(self, local, elementCentroid, elementVertices):
		b = elementVertices[ (self.innerFaceNeighborVertices[local])[0] ]
		f = elementVertices[ (self.innerFaceNeighborVertices[local])[1] ]
		q = elementVertices[ (self.innerFaceNeighborVertices[local])[2] ]
		w = elementVertices[ (self.innerFaceNeighborVertices[local])[3] ]
		e = elementVertices[ (self.innerFaceNeighborVertices[local])[4] ]
		r = elementVertices[ (self.innerFaceNeighborVertices[local])[5] ]

		# Element centroid
		x0, y0, z0 = elementCentroid.getCoordinates()
		# Facet [f-b-q-w] centroid
		x1, y1, z1 = ( (b + f + q + w) / 4.0 ).getCoordinates()
		# Edge [f-b] midpoint
		x2, y2, z2 = ( (b + f) / 2.0 ).getCoordinates()
		# Facet [f-b-e-r] centroid
		x3, y3, z3 = ( (b + f + e + r) / 4.0 ).getCoordinates()
		# Face area vector components
		x = 0.5 * ((y1-y0)*(z3-z0) - (y3-y0)*(z1-z0) + (y3-y2)*(z1-z2) - (y1-y2)*(z3-z2))
		y = 0.5 * ((x3-x0)*(z1-z0) - (x1-x0)*(z3-z0) + (x1-x2)*(z3-z2) - (x3-x2)*(z1-z2))
		z = 0.5 * ((x1-x0)*(y3-y0) - (x3-x0)*(y1-y0) + (x3-x2)*(y1-y2) - (x1-x2)*(y3-y2))
		return Point(x, y, z)

class Prism:
	dimension						   = 3
	numberOfInnerFaces				   = 9
	numberOfFacets 					   = 5
	subelementTransformedVolumes	   = np.array([1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0])
	innerFaceShapeFunctionValues	   = np.array([[5.0/16.0, 5.0/16.0, 1.0/8.0, 5.0/48.0, 5.0/48.0, 1.0/24.0],[1.0/8.0, 5.0/16.0, 5.0/16.0, 1.0/24.0, 5.0/48.0, 5.0/48.0],[5.0/16.0, 1.0/8.0, 5.0/16.0, 5.0/48.0, 1.0/24.0, 5.0/48.0],[5.0/48.0, 5.0/48.0, 1.0/24.0, 5.0/16.0, 5.0/16.0, 1.0/8.0],[1.0/24.0, 5.0/48.0, 5.0/48.0, 1.0/8.0, 5.0/16.0, 5.0/16.0],[5.0/48.0, 1.0/24.0, 5.0/48.0, 5.0/16.0, 1.0/8.0, 5.0/16.0],[7.0/24.0, 5.0/48.0, 5.0/48.0, 7.0/24.0, 5.0/48.0, 5.0/48.0],[5.0/48.0, 7.0/24.0, 5.0/48.0, 5.0/48.0, 7.0/24.0, 5.0/48.0],[5.0/48.0, 5.0/48.0, 7.0/24.0, 5.0/48.0, 5.0/48.0, 7.0/24.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-3.0/4.0, -3.0/4.0, -5.0/12.0], [3.0/4.0, 0.0/1.0, -5.0/12.0], [0.0/1.0, 3.0/4.0, -1.0/6.0], [-1.0/4.0, -1.0/4.0, 5.0/12.0], [1.0/4.0, 0.0/1.0, 5.0/12.0], [0.0/1.0, 1.0/4.0, 1.0/6.0]],[[-3.0/4.0, -3.0/4.0, -1.0/6.0], [3.0/4.0, 0.0/1.0, -5.0/12.0], [0.0/1.0, 3.0/4.0, -5.0/12.0], [-1.0/4.0, -1.0/4.0, 1.0/6.0], [1.0/4.0, 0.0/1.0, 5.0/12.0], [0.0/1.0, 1.0/4.0, 5.0/12.0]],[[-3.0/4.0, -3.0/4.0, -5.0/12.0], [3.0/4.0, 0.0/1.0, -1.0/6.0], [0.0/1.0, 3.0/4.0, -5.0/12.0], [-1.0/4.0, -1.0/4.0, 5.0/12.0], [1.0/4.0, 0.0/1.0, 1.0/6.0], [0.0/1.0, 1.0/4.0, 5.0/12.0]],[[-1.0/4.0, -1.0/4.0, -5.0/12.0], [1.0/4.0, 0.0/1.0, -5.0/12.0], [0.0/1.0, 1.0/4.0, -1.0/6.0], [-3.0/4.0, -3.0/4.0, 5.0/12.0], [3.0/4.0, 0.0/1.0, 5.0/12.0], [0.0/1.0, 3.0/4.0, 1.0/6.0]],[[-1.0/4.0, -1.0/4.0, -1.0/6.0], [1.0/4.0, 0.0/1.0, -5.0/12.0], [0.0/1.0, 1.0/4.0, -5.0/12.0], [-3.0/4.0, -3.0/4.0, 1.0/6.0], [3.0/4.0, 0.0/1.0, 5.0/12.0], [0.0/1.0, 3.0/4.0, 5.0/12.0]],[[-1.0/4.0, -1.0/4.0, -5.0/12.0], [1.0/4.0, 0.0/1.0, -1.0/6.0], [0.0/1.0, 1.0/4.0, -5.0/12.0], [-3.0/4.0, -3.0/4.0, 5.0/12.0], [3.0/4.0, 0.0/1.0, 1.0/6.0], [0.0/1.0, 3.0/4.0, 5.0/12.0]],[[-1.0/2.0, -1.0/2.0, -7.0/12.0], [1.0/2.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 1.0/2.0, -5.0/24.0], [-1.0/2.0, -1.0/2.0, 7.0/12.0], [1.0/2.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 1.0/2.0, 5.0/24.0]],[[-1.0/2.0, -1.0/2.0, -5.0/24.0], [1.0/2.0, 0.0/1.0, -7.0/12.0], [0.0/1.0, 1.0/2.0, -5.0/24.0], [-1.0/2.0, -1.0/2.0, 5.0/24.0], [1.0/2.0, 0.0/1.0, 7.0/12.0], [0.0/1.0, 1.0/2.0, 5.0/24.0]],[[-1.0/2.0, -1.0/2.0, -5.0/24.0], [1.0/2.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 1.0/2.0, -7.0/12.0], [-1.0/2.0, -1.0/2.0, 5.0/24.0], [1.0/2.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 1.0/2.0, 7.0/12.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1, 3, 4, 2, 0],[1, 2, 4, 5, 0, 0],[2, 0, 5, 3, 1, 0],[4, 3, 0, 1, 5, 0],[5, 4, 1, 2, 3, 0],[3, 5, 2, 0, 4, 0],[3, 0, 1, 4, 2, 5],[4, 1, 2, 5, 3, 0],[5, 2, 3, 0, 4, 1]])
	subelementShapeFunctionValues	   = np.array([[7.0/16.0, 5.0/32.0, 5.0/32.0, 7.0/48.0, 5.0/96.0, 5.0/96.0],[5.0/32.0, 7.0/16.0, 5.0/32.0, 5.0/96.0, 7.0/48.0, 5.0/96.0],[5.0/32.0, 5.0/32.0, 7.0/16.0, 5.0/96.0, 5.0/96.0, 7.0/48.0],[7.0/48.0, 5.0/96.0, 5.0/96.0, 7.0/16.0, 5.0/32.0, 5.0/32.0],[5.0/96.0, 7.0/48.0, 5.0/96.0, 5.0/32.0, 7.0/16.0, 5.0/32.0],[5.0/96.0, 5.0/96.0, 7.0/48.0, 5.0/32.0, 5.0/32.0, 7.0/16.0]])
	subelementShapeFunctionDerivatives = np.array([[[-3.0/4.0, -3.0/4.0, -7.0/12.0], [3.0/4.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 3.0/4.0, -5.0/24.0], [-1.0/4.0, -1.0/4.0, 7.0/12.0], [1.0/4.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 1.0/4.0, 5.0/24.0]],[[-3.0/4.0, -3.0/4.0, -5.0/24.0], [3.0/4.0, 0.0/1.0, -7.0/12.0], [0.0/1.0, 3.0/4.0, -5.0/24.0], [-1.0/4.0, -1.0/4.0, 5.0/24.0], [1.0/4.0, 0.0/1.0, 7.0/12.0], [0.0/1.0, 1.0/4.0, 5.0/24.0]],[[-3.0/4.0, -3.0/4.0, -5.0/24.0], [3.0/4.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 3.0/4.0, -7.0/12.0], [-1.0/4.0, -1.0/4.0, 5.0/24.0], [1.0/4.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 1.0/4.0, 7.0/12.0]],[[-1.0/4.0, -1.0/4.0, -7.0/12.0], [1.0/4.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 1.0/4.0, -5.0/24.0], [-3.0/4.0, -3.0/4.0, 7.0/12.0], [3.0/4.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 3.0/4.0, 5.0/24.0]],[[-1.0/4.0, -1.0/4.0, -5.0/24.0], [1.0/4.0, 0.0/1.0, -7.0/12.0], [0.0/1.0, 1.0/4.0, -5.0/24.0], [-3.0/4.0, -3.0/4.0, 5.0/24.0], [3.0/4.0, 0.0/1.0, 7.0/12.0], [0.0/1.0, 3.0/4.0, 5.0/24.0]],[[-1.0/4.0, -1.0/4.0, -5.0/24.0], [1.0/4.0, 0.0/1.0, -5.0/24.0], [0.0/1.0, 1.0/4.0, -7.0/12.0], [-3.0/4.0, -3.0/4.0, 5.0/24.0], [3.0/4.0, 0.0/1.0, 5.0/24.0], [0.0/1.0, 3.0/4.0, 7.0/12.0]]])
	facetVerticesIndices 			   = np.array([[0, 2, 1],[3, 4, 5],[0, 3, 5, 2],[0, 1, 4, 3],[1, 2, 5, 4]], dtype=np.object)
	outerFaceShapeFunctionValues	   = np.array([[[7.0/12.0, 5.0/24.0, 5.0/24.0, 0.0/1.0, 0.0/1.0, 0.0/1.0],[5.0/24.0, 5.0/24.0, 7.0/12.0, 0.0/1.0, 0.0/1.0, 0.0/1.0],[5.0/24.0, 7.0/12.0, 5.0/24.0, 0.0/1.0, 0.0/1.0, 0.0/1.0]],[[0.0/1.0, 0.0/1.0, 0.0/1.0, 7.0/12.0, 5.0/24.0, 5.0/24.0],[0.0/1.0, 0.0/1.0, 0.0/1.0, 5.0/24.0, 7.0/12.0, 5.0/24.0],[0.0/1.0, 0.0/1.0, 0.0/1.0, 5.0/24.0, 5.0/24.0, 7.0/12.0]],[[9.0/16.0, 0.0/1.0, 3.0/16.0, 3.0/16.0, 0.0/1.0, 1.0/16.0],[3.0/16.0, 0.0/1.0, 1.0/16.0, 9.0/16.0, 0.0/1.0, 3.0/16.0],[1.0/16.0, 0.0/1.0, 3.0/16.0, 3.0/16.0, 0.0/1.0, 9.0/16.0],[3.0/16.0, 0.0/1.0, 9.0/16.0, 1.0/16.0, 0.0/1.0, 3.0/16.0]],[[9.0/16.0, 3.0/16.0, 0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0],[3.0/16.0, 9.0/16.0, 0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0],[1.0/16.0, 3.0/16.0, 0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0],[3.0/16.0, 1.0/16.0, 0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0]],[[0.0/1.0, 9.0/16.0, 3.0/16.0, 0.0/1.0, 3.0/16.0, 1.0/16.0],[0.0/1.0, 3.0/16.0, 9.0/16.0, 0.0/1.0, 1.0/16.0, 3.0/16.0],[0.0/1.0, 1.0/16.0, 3.0/16.0, 0.0/1.0, 3.0/16.0, 9.0/16.0],[0.0/1.0, 3.0/16.0, 1.0/16.0, 0.0/1.0, 9.0/16.0, 3.0/16.0]]], dtype=np.object)

	def __init__(self, element):
		self.element = element			

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 6:
			return True
		else:
			return False

	def getInnerFaceAreaVector(self, local, elementCentroid, elementVertices):
		# Vertices indices
		b = elementVertices[ (self.innerFaceNeighborVertices[local])[0] ]
		f = elementVertices[ (self.innerFaceNeighborVertices[local])[1] ]
		q = elementVertices[ (self.innerFaceNeighborVertices[local])[2] ]
		w = elementVertices[ (self.innerFaceNeighborVertices[local])[3] ]
		e = elementVertices[ (self.innerFaceNeighborVertices[local])[4] ]
		r = elementVertices[ (self.innerFaceNeighborVertices[local])[5] ]
		# Element centroid
		x0, y0, z0 = elementCentroid.getCoordinates()
		# Facet [f-b-q-w] centroid
		x1, y1, z1 = ( (b + f + q + w)/4.0 ).getCoordinates()
		# Edge [f-b] midpoint
		x2, y2, z2 = ( (b + f)/2.0 ).getCoordinates()
		# Facet [f-b-e] or [f-b-e-r] centroid
		if local < 6:
			x3, y3, z3 = ( (b + f + e)/3.0 ).getCoordinates()
		else:
			x3, y3, z3 = ( (b + f + e + r)/4.0 ).getCoordinates()
		# Face area vector components
		x = 0.5 * ((y1-y0)*(z3-z0) - (y3-y0)*(z1-z0) + (y3-y2)*(z1-z2) - (y1-y2)*(z3-z2))
		y = 0.5 * ((x3-x0)*(z1-z0) - (x1-x0)*(z3-z0) + (x1-x2)*(z3-z2) - (x3-x2)*(z1-z2))
		z = 0.5 * ((x1-x0)*(y3-y0) - (x3-x0)*(y1-y0) + (x3-x2)*(y1-y2) - (x1-x2)*(y3-y2))
		return Point(x, y, z)

class Pyramid:
	dimension						   = 3
	numberOfInnerFaces				   = 8
	numberOfFacets 					   = 5
	subelementTransformedVolumes	   = np.array([1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/9.0])
	innerFaceShapeFunctionValues	   = np.array([[13.0/36.0, 13.0/36.0, 1.0/12.0, 1.0/12.0, 1.0/9.0],[1.0/12.0, 13.0/36.0, 13.0/36.0, 1.0/12.0, 1.0/9.0],[1.0/12.0, 1.0/12.0, 13.0/36.0, 13.0/36.0, 1.0/9.0],[13.0/36.0, 1.0/12.0, 1.0/12.0, 13.0/36.0, 1.0/9.0],[6.0/17.0, 5.0/34.0, 25.0/408.0, 5.0/34.0, 7.0/24.0],[5.0/34.0, 6.0/17.0, 5.0/34.0, 25.0/408.0, 7.0/24.0],[25.0/408.0, 5.0/34.0, 6.0/17.0, 5.0/34.0, 7.0/24.0],[5.0/34.0, 25.0/408.0, 5.0/34.0, 6.0/17.0, 7.0/24.0]])
	innerFaceShapeFunctionDerivatives  = np.array([[[-13.0/16.0, -1.0/2.0, -1.0/4.0], [13.0/16.0, -1.0/2.0, -1.0/4.0], [3.0/16.0, 1.0/2.0, -1.0/4.0], [-3.0/16.0, 1.0/2.0, -1.0/4.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/2.0, -3.0/16.0, -1.0/4.0], [1.0/2.0, -13.0/16.0, -1.0/4.0], [1.0/2.0, 13.0/16.0, -1.0/4.0], [-1.0/2.0, 3.0/16.0, -1.0/4.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-3.0/16.0, -1.0/2.0, -1.0/4.0], [3.0/16.0, -1.0/2.0, -1.0/4.0], [13.0/16.0, 1.0/2.0, -1.0/4.0], [-13.0/16.0, 1.0/2.0, -1.0/4.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/2.0, -13.0/16.0, -1.0/4.0], [1.0/2.0, -3.0/16.0, -1.0/4.0], [1.0/2.0, 3.0/16.0, -1.0/4.0], [-1.0/2.0, 13.0/16.0, -1.0/4.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-12.0/17.0, -12.0/17.0, -60.0/289.0], [12.0/17.0, -5.0/17.0, -169.0/578.0], [5.0/17.0, 5.0/17.0, -60.0/289.0], [-5.0/17.0, 12.0/17.0, -169.0/578.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-12.0/17.0, -5.0/17.0, -169.0/578.0], [12.0/17.0, -12.0/17.0, -60.0/289.0], [5.0/17.0, 12.0/17.0, -169.0/578.0], [-5.0/17.0, 5.0/17.0, -60.0/289.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-5.0/17.0, -5.0/17.0, -60.0/289.0], [5.0/17.0, -12.0/17.0, -169.0/578.0], [12.0/17.0, 12.0/17.0, -60.0/289.0], [-12.0/17.0, 5.0/17.0, -169.0/578.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-5.0/17.0, -12.0/17.0, -169.0/578.0], [5.0/17.0, -5.0/17.0, -60.0/289.0], [12.0/17.0, 5.0/17.0, -169.0/578.0], [-12.0/17.0, 12.0/17.0, -60.0/289.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]]])
	innerFaceNeighborVertices		   = np.array([[0, 1],[1, 2],[2, 3],[3, 0],[0, 4, 3, 1],[1, 4, 0, 2],[2, 4, 1, 3],[3, 4, 2, 0]], dtype=np.object)
	subelementShapeFunctionValues	   = np.array([[75.0/164.0, 55.0/328.0, 121.0/1968.0, 55.0/328.0, 7.0/48.0],[55.0/328.0, 75.0/164.0, 55.0/328.0, 121.0/1968.0, 7.0/48.0],[121.0/1968.0, 55.0/328.0, 75.0/164.0, 55.0/328.0, 7.0/48.0],[55.0/328.0, 121.0/1968.0, 55.0/328.0, 75.0/164.0, 7.0/48.0],[11.0/96.0, 11.0/96.0, 11.0/96.0, 11.0/96.0, 13.0/24.0]])
	subelementShapeFunctionDerivatives = np.array([[[-30.0/41.0, -30.0/41.0, -330.0/1681.0], [30.0/41.0, -11.0/41.0, -1021.0/3362.0], [11.0/41.0, 11.0/41.0, -330.0/1681.0], [-11.0/41.0, 30.0/41.0, -1021.0/3362.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-30.0/41.0, -11.0/41.0, -1021.0/3362.0], [30.0/41.0, -30.0/41.0, -330.0/1681.0], [11.0/41.0, 30.0/41.0, -1021.0/3362.0], [-11.0/41.0, 11.0/41.0, -330.0/1681.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-11.0/41.0, -11.0/41.0, -330.0/1681.0], [11.0/41.0, -30.0/41.0, -1021.0/3362.0], [30.0/41.0, 30.0/41.0, -330.0/1681.0], [-30.0/41.0, 11.0/41.0, -1021.0/3362.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-11.0/41.0, -30.0/41.0, -1021.0/3362.0], [11.0/41.0, -11.0/41.0, -330.0/1681.0], [30.0/41.0, 11.0/41.0, -1021.0/3362.0], [-30.0/41.0, 30.0/41.0, -330.0/1681.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]],[[-1.0/2.0, -1.0/2.0, -1.0/4.0], [1.0/2.0, -1.0/2.0, -1.0/4.0], [1.0/2.0, 1.0/2.0, -1.0/4.0], [-1.0/2.0, 1.0/2.0, -1.0/4.0], [0.0/1.0, 0.0/1.0, 1.0/1.0]]])
	facetVerticesIndices 			   = np.array([[0, 4, 3],[0, 1, 4],[1, 2, 4],[2, 3, 4],[0, 3, 2, 1]], dtype=np.object)
	outerFaceShapeFunctionValues	   = np.array([[[7.0/12.0, 0.0/1.0, 0.0/1.0, 5.0/24.0, 5.0/24.0],[5.0/24.0, 0.0/1.0, 0.0/1.0, 5.0/24.0, 7.0/12.0],[5.0/24.0, 0.0/1.0, 0.0/1.0, 7.0/12.0, 5.0/24.0]],[[7.0/12.0, 5.0/24.0, 0.0/1.0, 0.0/1.0, 5.0/24.0],[5.0/24.0, 7.0/12.0, 0.0/1.0, 0.0/1.0, 5.0/24.0],[5.0/24.0, 5.0/24.0, 0.0/1.0, 0.0/1.0, 7.0/12.0]],[[0.0/1.0, 7.0/12.0, 5.0/24.0, 0.0/1.0, 5.0/24.0],[0.0/1.0, 5.0/24.0, 7.0/12.0, 0.0/1.0, 5.0/24.0],[0.0/1.0, 5.0/24.0, 5.0/24.0, 0.0/1.0, 7.0/12.0]],[[0.0/1.0, 0.0/1.0, 7.0/12.0, 5.0/24.0, 5.0/24.0],[0.0/1.0, 0.0/1.0, 5.0/24.0, 7.0/12.0, 5.0/24.0],[0.0/1.0, 0.0/1.0, 5.0/24.0, 5.0/24.0, 7.0/12.0]],[[9.0/16.0, 3.0/16.0, 1.0/16.0, 3.0/16.0, 0.0/1.0],[3.0/16.0, 1.0/16.0, 3.0/16.0, 9.0/16.0, 0.0/1.0],[1.0/16.0, 3.0/16.0, 9.0/16.0, 3.0/16.0, 0.0/1.0],[3.0/16.0, 9.0/16.0, 3.0/16.0, 1.0/16.0, 0.0/1.0]]], dtype=np.object)

	def __init__(self, element):
		self.element = element			

	@staticmethod
	def _is(elem):
		if len(elem.vertices) == 5:
			return True
		else:
			return False

	def getInnerFaceAreaVector(self, local, elementCentroid, elementVertices):
		# Vertices indices
		b = elementVertices[ (self.innerFaceNeighborVertices[local])[0] ]
		f = elementVertices[ (self.innerFaceNeighborVertices[local])[1] ]
		# Base centroid
		x0, y0, z0 = ( (elementVertices[1] + elementVertices[2] + elementVertices[4] + elementVertices[3])/4.0 ).getCoordinates()
		# Edge [f-b] midpoint
		x2, y2, z2 = ( (f + b)/2.0 ).getCoordinates()

		if local < 4:
			# Facet [f-b-4] centroid
			x1, y1, z1 = ( (b + f + elementVertices[4])/3.0 ).getCoordinates()
			# Face area vector components
			x = 0.5 * ((y1-y0)*(z2-z0) - (y2-y0)*(z1-z0))
			y = 0.5 * ((x2-x0)*(z1-z0) - (x1-x0)*(z2-z0))
			z = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
		else:
			# Auxiliar vertices
			q = elementVertices[ (self.innerFaceNeighborVertices[local])[2] ]
			w = elementVertices[ (self.innerFaceNeighborVertices[local])[3] ]
			# Facet [f-b-q] centroid
			x1, y1, z1 = ( (f + b + q)/3.0 ).getCoordinates()
			# Facet [f-b-w] centroid
			x3, y3, z3 = ( (f + b + w)/3.0 ).getCoordinates()
			# Face area vector components
			x = 0.5 * ((y1-y0)*(z3-z0) - (y3-y0)*(z1-z0) + (y3-y2)*(z1-z2) - (y1-y2)*(z3-z2))
			y = 0.5 * ((x3-x0)*(z1-z0) - (x1-x0)*(z3-z0) + (x1-x2)*(z3-z2) - (x3-x2)*(z1-z2))
			z = 0.5 * ((x1-x0)*(y3-y0) - (x3-x0)*(y1-y0) + (x3-x2)*(y1-y2) - (x1-x2)*(y3-y2))

		return Point(x, y, z)