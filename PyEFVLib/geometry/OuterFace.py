import numpy as np

class OuterFace:
	def __init__(self, vertex, facet, local, handle):
		self.vertex = vertex
		self.facet = facet
		self.local = local
		self.handle = handle
		self.vertexLocalIndex = vertex.getLocal( facet.element )

		self.facet.element.addOuterFace(self)

	def getShapeFunctionAtCentroid(self):
		return self.facet.element.shape.outerFaceShapeFunctionValues[ self.vertexLocalIndex ]