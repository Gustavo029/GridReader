import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import PyEFVLib
from PyEFVLib import Solver
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import time

class GeomechanicsSolver(Solver):
	def __init__(self, workspaceDirectory, gravity=False, verbosity=True, **kwargs):
		# kwargs -> outputFileName, extension, transient, verbosity
		Solver.__init__(self, workspaceDirectory, verbosity=True, **kwargs)
		self.gravity = gravity

	getPoissonsRatio		= lambda self, element: self.propertyData.get(element.region.handle, "PoissonsRatio")
	getShearModulus			= lambda self, element: self.propertyData.get(element.region.handle, "ShearModulus")
	getSolidCompressibility	= lambda self, element: self.propertyData.get(element.region.handle, "SolidCompressibility")
	getPorosity				= lambda self, element: self.propertyData.get(element.region.handle, "Porosity")
	getPermeability			= lambda self, element: self.propertyData.get(element.region.handle, "Permeability")
	getFluidCompressibility	= lambda self, element: self.propertyData.get(element.region.handle, "FluidCompressibility")
	getViscosity			= lambda self, element: self.propertyData.get(element.region.handle, "Viscosity")
	getSolidDensity			= lambda self, element: self.propertyData.get(element.region.handle, "SolidDensity")
	getFluidDensity			= lambda self, element: self.propertyData.get(element.region.handle, "FluidDensity")

	def getBiotCoefficient(self, element):
	    # α = 1 - cs / cb
		return 1 - self.getSolidCompressibility(element) / self.getBulkCompressibility(element)
	def getBulkCompressibility(self, element):
		# cb = 1 / K
		return 1 / self.getBulkModulus(element)
	def getBulkModulus(self, element):
		# K = 2G(1 + ν) / 3(1-2ν)
		return 2*self.getShearModulus(element)*(1 + self.getPoissonsRatio(element)) / ( 3*(1-2*self.getPoissonsRatio(element)) )
	def getBulkDensity(self, element):
		# ρ = Φ * ρf + (1-Φ) * ρs
		return self.getPorosity(element) * self.getFluidDensity(element) + (1-self.getPorosity(element)) * self.getSolidDensity(element)
	def getBiotModulus(self, element):
		# M = 1 / (Φ * cf + (α-Φ) * cs)
		return 1 / (self.getPorosity(element) * self.getFluidCompressibility(element) + (self.getBiotCoefficient(element)-self.getPorosity(element)) * self.getSolidCompressibility(element))


	def init(self):
		self.gravityVector = -9.81 * np.array([0.0, 0.0, 0.0])[:self.dimension]

		self.prevPressureField = self.problemData.initialValues["pressure"].copy()		# p_old
		self.pressureField = np.repeat(0.0, self.grid.numberOfVertices)					# p_k
		self.nextPressureField = np.repeat(0.0, self.grid.numberOfVertices)				# p_(k+1)

		self.prevDisplacements = np.repeat(0.0, self.dimension*self.numberOfVertices)	# u_old
		self.displacements = np.repeat(0.0, self.dimension*self.numberOfVertices)		# u_k
		self.nextDisplacements = np.repeat(0.0, self.dimension*self.numberOfVertices)	# u_(k+1)

		self.massIndependentVector = np.zeros( self.numberOfVertices )
		self.geoIndependentVector = np.zeros( 3*self.numberOfVertices )

		self.massMatrixVals, self.massCoords = [], []
		self.geoMatrixVals,  self.geoCoords  = [], []

	def mainloop(self):
		self.assembleGlobalMatrixMass()
		self.assembleGlobalMatrixGeo()

		self.iteration = 0
		self.difference = 2*self.tolerance

		# Temporal Loop
		while not self.converged:
			self.pressureField = self.prevPressureField.copy()
			self.displacements = self.prevDisplacements.copy()

			self.iterativeDifference = 2*self.tolerance

			# Iterative Loop
			self.iteration = 0
			while self.iteration <= 1 or (self.iterativeDifference >= self.tolerance):
				self.assembleMassVector()
				self.solveIterativePressureField()
				self.assembleGeoVector()
				self.solveIterativeDisplacementsField()

				self.iterativeDifference = max( abs(self.nextPressureField - self.pressureField) )
	
				self.pressureField = self.nextPressureField.copy()
				self.displacements = self.nextDisplacements.copy()

				self.iteration += 1

				if self.iteration >= self.problemData.maxNumberOfIterations:
					break

			self.difference = max( abs(self.nextPressureField - self.prevPressureField) )

			self.prevPressureField = self.nextPressureField.copy()
			self.prevDisplacements = self.nextDisplacements.copy()

			self.printIterationData()

			self.saveTemporalResults()
			self.currentTime += self.timeStep

			self.converged = ( self.difference <= self.tolerance ) or ( self.currentTime >= self.problemData.finalTime ) or ( self.iteration >= self.problemData.maxNumberOfIterations )

	# -------------- HELPER FUNCTIONS --------------

	def massAdd(self, i, j, val):
		self.massMatrixVals.append(val)
		self.massCoords.append((i,j))

	def geoAdd(self, i, j, val):
		self.geoMatrixVals.append(val)
		self.geoCoords.append((i,j))

	def getTransposedVoigtArea(self, face):
		Sx, Sy, Sz = face.area.getCoordinates()
		if self.dimension == 2:
			return np.array([[Sx,0,Sy],[0,Sy,Sx]])
		elif self.dimension == 3:
			return np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])

	def getConstitutiveMatrix(self, region):
		shearModulus = self.propertyData.get(region.handle, "ShearModulus")
		poissonsRatio = self.propertyData.get(region.handle, "PoissonsRatio")

		lameParameter=2*shearModulus*poissonsRatio/(1-2*poissonsRatio)

		if self.dimension == 2:
			constitutiveMatrix = np.array([[2*shearModulus+lameParameter ,lameParameter 				,0			 ],
										   [lameParameter				 ,2*shearModulus+lameParameter 	,0			 ],
										   [0			 				 ,0								,shearModulus]])

		elif self.dimension == 3:
			constitutiveMatrix = np.array([[2*shearModulus+lameParameter	,lameParameter				 ,lameParameter				  ,0		,0	 ,0],
										   [lameParameter					,2*shearModulus+lameParameter,lameParameter				  ,0		,0	 ,0],
										   [lameParameter					,lameParameter				 ,2*shearModulus+lameParameter,0		,0	 ,0],
										   [0								,0							 ,0							  ,shearModulus,0,0],
										   [0								,0							 ,0							  ,0,shearModulus,0],
										   [0								,0							 ,0							  ,0,0,shearModulus]])

		return constitutiveMatrix

	@staticmethod
	def getVoigtGradientOperator(globalDerivatives):
		if len(globalDerivatives) == 2:
			Nx,Ny = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero],[zero,Ny],[Ny,Nx]])

		if len(globalDerivatives) == 3:
			Nx,Ny,Nz = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero,zero],[zero,Ny,zero],[zero,zero,Nz],[Ny,Nx,zero],[zero,Nz,Ny],[Nz,zero,Nx]])

	def assembleLocalMatrixMass(self, element, localMatrix):
		for i in range(element.vertices.size):
			for j in range(element.vertices.size):
				self.massAdd( element.vertices[i].handle, element.vertices[j].handle, localMatrix[i][j] )
	
	def assembleLocalMatrixGeo(self, element, localMatrix):
		for i in range(element.vertices.size):
			for j in range(element.vertices.size):
				self.geoAdd( element.vertices[i].handle + 0*self.numberOfVertices, element.vertices[j].handle + 0*self.numberOfVertices, localMatrix[i][j] )
				self.geoAdd( element.vertices[i].handle + 1*self.numberOfVertices, element.vertices[j].handle + 1*self.numberOfVertices, localMatrix[i][j] )
				if self.dimension == 3:
					self.geoAdd( element.vertices[i].handle + 2*self.numberOfVertices, element.vertices[j].handle + 2*self.numberOfVertices, localMatrix[i][j] )

	def computeLocalMatrixA(self, element):
		# Accumulation Term
		localMatrixA = np.zeros( (element.vertices.size, element.vertices.size) )

		
		biotModulus = self.getBiotModulus(element)

		for local in range(element.vertices.size):
			coefficient = element.subelementVolumes[local] / ( biotModulus * self.timeStep )

			localMatrixA[local][local] += coefficient												# ΔΩ / (M * Δt)

		return localMatrixA

	def computeLocalMatrixR(self, element):
		localMatrixR = np.zeros( (element.vertices.size, element.vertices.size) )
		
		biotCoefficient = self.getBiotCoefficient(element)
		bulkModulus = self.getBulkModulus(element)

		for local in range(element.vertices.size):
			coefficient = (biotCoefficient**2) * element.subelementVolumes[local] / (bulkModulus * self.timeStep)
			
			localMatrixR[local][local] += coefficient

		return localMatrixR

	def computeLocalMatrixH(self, element):
		localMatrixH = np.zeros( (element.vertices.size, element.vertices.size) )

		permeability = self.getPermeability(element)
		viscosity = self.getViscosity(element)

		# Pressure gradient term
		for innerFace in element.innerFaces:
			area = innerFace.area.getCoordinates()[:self.dimension]
			coefficientsVector = -(permeability/viscosity)*np.matmul( area.T, innerFace.globalDerivatives )
	
			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for local in range(element.vertices.size):
				localMatrixH[backwardVertexLocal][local] += coefficientsVector[local]
				localMatrixH[forwardVertexLocal][local] -= coefficientsVector[local]

		return localMatrixH

	def computeLocalMatrixQ(self, element):
		# Volumetric Deformation Term
		localMatrixQ = np.zeros( ( element.vertices.size, element.vertices.size * self.dimension ) )
		
		biotCoefficient = self.getBiotCoefficient(element)

		for innerFace in element.innerFaces:
			shapeFunctionValues = element.shape.innerFaceShapeFunctionValues[innerFace.local]
			area = innerFace.area.getCoordinates()[:self.dimension]
			coefficients = -(biotCoefficient/self.timeStep) * np.array([sj*Ni for sj in area for Ni in shapeFunctionValues])

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for c in range( self.dimension ):
				for local in range( element.vertices.size ):
					localMatrixQ[backwardVertexLocal][local+c*self.dimension] += coefficients[c+local*self.dimension]
					localMatrixQ[forwardVertexLocal][local+c*self.dimension]  -= coefficients[c+local*self.dimension]

		for outerFace in element.outerFaces:
			shapeFunctionValues = element.shape.outerFaceShapeFunctionValues[outerFace.facet.elementLocalIndex][outerFace.local]
			coefficients = -(biotCoefficient/self.timeStep) * np.array([sj*Ni for Ni in shapeFunctionValues for sj in outerFace.area.getCoordinates()[:self.dimension]])

			vertexLocal = outerFace.vertex.getLocal(element)
			for c in range( self.dimension ):
				for local in range( element.vertices.size ):
					localMatrixQ[vertexLocal][local+c*self.dimension] += coefficients[c+local*self.dimension]

		return localMatrixQ

	def computeLocalMatrixG(self, element):
		localMatrixG = np.zeros( (element.vertices.size, self.dimension) )

		permeability = self.getPermeability(element)
		fluidDensity = self.getFluidDensity(element)
		viscosity = self.getViscosity(element)

		for innerFace in element.innerFaces:
			coefficients = (fluidDensity*permeability/viscosity) * innerFace.area.getCoordinates()[:self.dimension]

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for c in range(self.dimension):
				localMatrixG[backwardVertexLocal][c] += coefficients[c]
				localMatrixG[forwardVertexLocal][c] -= coefficients[c]

		return localMatrixG

	def computeLocalMatrixK(self, element):
		# Effective Stress Term
		m = element.vertices.size
		localMatrixK = np.zeros( (self.dimension*m, self.dimension*m) )

		constitutiveMatrix = self.getConstitutiveMatrix(element.region)
		for innerFace in element.innerFaces:
			transposedVoigtArea = self.getTransposedVoigtArea(innerFace)
			voigtGradientOperator = self.getVoigtGradientOperator(innerFace.globalDerivatives)

			matrixCoefficient = np.einsum("ij,jk,kmn->imn", transposedVoigtArea, constitutiveMatrix, voigtGradientOperator)

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]
			
			for local in range(element.vertices.size):
				for i in range(self.dimension):
					for j in range(self.dimension):
						localMatrixK[backwardVertexLocal + m*i][local + m*j] += matrixCoefficient[i][j][local]
						localMatrixK[forwardVertexLocal + m*i][local + m*j] -= matrixCoefficient[i][j][local]

		return localMatrixK

	def computeLocalMatrixL(self, element):
		localMatrixL = np.zeros( ( element.vertices.size * self.dimension, element.vertices.size ) )

		biotCoefficient = self.getBiotCoefficient(element)

		for innerFace in element.innerFaces:
			transposedVoigtArea = self.getTransposedVoigtArea(innerFace)

			shapeFunctionValues = element.shape.innerFaceShapeFunctionValues[innerFace.local]
			zeros = np.zeros(element.vertices.size)
			identityShapeFunctionMatrix = np.array([ shapeFunctionValues, shapeFunctionValues, shapeFunctionValues, zeros, zeros, zeros ]) if self.dimension == 3 else np.array([shapeFunctionValues, shapeFunctionValues, zeros])

			coefficients = -biotCoefficient * np.matmul( transposedVoigtArea, identityShapeFunctionMatrix )

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for c in range( self.dimension ):
				for local in range( element.vertices.size ):
					localMatrixL[backwardVertexLocal+c*element.vertices.size][local] += coefficients[c][local]
					localMatrixL[forwardVertexLocal+c*element.vertices.size][local] -= coefficients[c][local]

		return localMatrixL

	# ----------- END OF HELPER FUNCTIONS -----------

	def assembleGlobalMatrixMass(self):
		self.massMatrixVals, self.massCoords = [], []

		for element in self.grid.elements:
			localMatrixA = self.computeLocalMatrixA(element)
			localMatrixR = self.computeLocalMatrixR(element)
			localMatrixH = self.computeLocalMatrixH(element)

			self.assembleLocalMatrixMass(element, localMatrixA)
			self.assembleLocalMatrixMass(element, localMatrixR)
			self.assembleLocalMatrixMass(element, localMatrixH)

		# Boundary Conditions
		for bCondition in self.problemData.dirichletBoundaries["pressure"]:
			for vertex in bCondition.boundary.vertices:
				self.massMatrixVals, self.massCoords = zip( *[(val, coord) for coord, val in zip(self.massCoords, self.massMatrixVals) if coord[0] != vertex.handle] )
				self.massMatrixVals, self.massCoords = list(self.massMatrixVals), list(self.massCoords)
				self.massAdd(vertex.handle, vertex.handle, 1.0 )


		# Invert Mass Matrix
		self.massMatrix = sparse.csc_matrix( (self.massMatrixVals, zip(*self.massCoords)) )#, shape=(self.numberOfVertices, self.numberOfVertices) )
		self.inverseMassMatrix = sparse.linalg.inv( self.massMatrix )

	def assembleGlobalMatrixGeo(self):
		self.geoMatrixVals,  self.geoCoords  = [], []

		for element in self.grid.elements:
			self.assembleLocalMatrixGeo(element, self.computeLocalMatrixK(element))

		# Boundary Conditions
		U = lambda handle: handle + self.numberOfVertices * 0
		V = lambda handle: handle + self.numberOfVertices * 1
		W = lambda handle: handle + self.numberOfVertices * 2

		for bc in self.problemData.boundaryConditions:
			boundary=bc["u"].boundary
			uBoundaryType = bc["u"].__type__
			vBoundaryType = bc["v"].__type__
			wBoundaryType = bc["w"].__type__ if "w" in bc.keys() else None

			# Dirichlet Boundary Conditions
			if uBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoMatrixVals = [val for coord, val in zip(self.geoCoords, self.geoMatrixVals) if coord[0] != U(vertex.handle)]
					self.geoCoords 	   = [coord for coord in self.geoCoords if coord[0] != U(vertex.handle)]
					self.geoAdd(U(vertex.handle), U(vertex.handle), 1.0 )

			if vBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoMatrixVals = [val for coord, val in zip(self.geoCoords, self.geoMatrixVals) if coord[0] != V(vertex.handle)]
					self.geoCoords 	   = [coord for coord in self.geoCoords if coord[0] != V(vertex.handle)]
					self.geoAdd(V(vertex.handle), V(vertex.handle), 1.0 )

			if wBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoMatrixVals = [val for coord, val in zip(self.geoCoords, self.geoMatrixVals) if coord[0] != W(vertex.handle)]
					self.geoCoords 	   = [coord for coord in self.geoCoords if coord[0] != W(vertex.handle)]
					self.geoAdd(W(vertex.handle), W(vertex.handle), 1.0 )


		# Invert Geo Matrix
		self.geoMatrix = sparse.csc_matrix( (self.geoMatrixVals, zip(*self.geoCoords)), shape=(self.dimension * self.numberOfVertices, self.dimension * self.numberOfVertices) )
		self.inverseGeoMatrix = sparse.linalg.inv( self.geoMatrix )

	def assembleMassVector(self):
		self.massIndependentVector = np.zeros( self.numberOfVertices )

		for element in self.grid.elements:
			localMatrixQ = self.computeLocalMatrixQ(element)
			localMatrixA = self.computeLocalMatrixA(element)
			localMatrixR = self.computeLocalMatrixR(element)
			localMatrixG = self.computeLocalMatrixG(element)

			# Get pressures and displacements
			prevElementPressures 	 = np.array([ self.prevPressureField[vertex.handle] for vertex in element.vertices ])
			elementPressures 		 = np.array([ self.pressureField[vertex.handle] for vertex in element.vertices ])
			prevElementDisplacements = np.array([ self.prevDisplacements[m+c*self.numberOfVertices] for c in range(self.dimension) for m in range(element.vertices.size) ])
			elementDisplacements 	 = np.array([ self.displacements[m+c*self.numberOfVertices] for c in range(self.dimension) for m in range(element.vertices.size) ])

			coefficients = ( 
				np.matmul(localMatrixA, prevElementPressures) + 
				np.matmul(localMatrixR, elementPressures) + 
				np.matmul(localMatrixQ, prevElementDisplacements - elementDisplacements) + 
				np.matmul(localMatrixG, self.gravityVector) 
			)

			local = 0
			for vertex in element.vertices:
				self.massIndependentVector[vertex.handle] = coefficients[local]
				local += 1

		# Dirichlet Boundary Condition
		for bCondition in self.problemData.dirichletBoundaries["pressure"]:
			for vertex in bCondition.boundary.vertices:
				self.massIndependentVector[vertex.handle] = bCondition.getValue(vertex.handle)

		# Neumann Boundary Condition
		for bCondition in self.problemData.neumannBoundaries["pressure"]:
			for facet in bCondition.boundary.facets:
				for outerFace in facet.outerFaces:
					self.massIndependentVector[outerFace.vertex.handle] += bCondition.getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

	def assembleGeoVector(self):
		self.geoIndependentVector = np.zeros( self.dimension * self.numberOfVertices )

		for element in self.grid.elements:
			density = self.getBulkDensity(element)

			localMatrixL = self.computeLocalMatrixL(element)

			nextElementPressures = [ self.nextPressureField[vertex.handle] for vertex in element.vertices ]
			coefficients = np.matmul( localMatrixL, nextElementPressures )

			local = 0
			for vertex in element.vertices:
				subelementVolume = element.subelementVolumes[local]
				for c in range(self.dimension):
					# self.geoIndependentVector[ vertex.handle+c*self.numberOfVertices ] = -density * subelementVolume * self.gravityVector[c]
					self.geoIndependentVector[ vertex.handle+c*self.numberOfVertices ] = -density * subelementVolume * self.gravityVector[c] + coefficients[local+c*self.dimension]

				local += 1

		# Boundary Conditions
		U = lambda handle: handle + self.numberOfVertices * 0
		V = lambda handle: handle + self.numberOfVertices * 1
		W = lambda handle: handle + self.numberOfVertices * 2

		for bc in self.problemData.boundaryConditions:
			boundary=bc["u"].boundary
			uBoundaryType = bc["u"].__type__
			vBoundaryType = bc["v"].__type__
			wBoundaryType = bc["w"].__type__ if "w" in bc.keys() else None

			# Neumann Boundary Conditions
			if uBoundaryType == "NEUMANN":
				for facet in boundary.facets:
					for outerFace in facet.outerFaces:
						self.geoIndependentVector[U(outerFace.vertex.handle)] -= bc["u"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

			if vBoundaryType == "NEUMANN":
				for facet in boundary.facets:
					for outerFace in facet.outerFaces:
						self.geoIndependentVector[V(outerFace.vertex.handle)] -= bc["v"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

			if wBoundaryType == "NEUMANN":
				for facet in boundary.facets:
					for outerFace in facet.outerFaces:
						self.geoIndependentVector[W(outerFace.vertex.handle)] -= bc["w"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

			# Dirichlet Boundary Conditions
			if uBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoIndependentVector[U(vertex.handle)] = bc["u"].getValue(vertex.handle)

			if vBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoIndependentVector[V(vertex.handle)] = bc["v"].getValue(vertex.handle)

			if wBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.geoIndependentVector[W(vertex.handle)] = bc["w"].getValue(vertex.handle)

	def solveIterativePressureField(self):
		self.nextPressureField = np.matmul(self.inverseMassMatrix.toarray(), self.massIndependentVector)

	def solveIterativeDisplacementsField(self):
		self.nextDisplacements = np.matmul(self.inverseGeoMatrix.toarray(), self.geoIndependentVector)

	def saveTemporalResults(self):
		self.saver.save("pressure", self.nextPressureField, self.currentTime)
		self.saver.save("u", self.nextDisplacements[0*self.numberOfVertices:1*self.numberOfVertices], self.currentTime)
		self.saver.save("v", self.nextDisplacements[1*self.numberOfVertices:2*self.numberOfVertices], self.currentTime)
		if self.dimension == 3:
			self.saver.save("w", self.nextDisplacements[2*self.numberOfVertices:3*self.numberOfVertices], self.currentTime)


def geomechanics(problemData, solve=True, extension="csv", saverType="default", transient=True, verbosity=True):
	solver = GeomechanicsSolver(problemData, extension=extension, saverType=saverType, transient=transient, verbosity=verbosity)
	if solve:
		solver.solve()
	return solver

if __name__ == "__main__":
	if "--help" in sys.argv: print("Usage: python apps/geomechanics.py\n\t-p\t:Transient Solution\n\t-v\t:Verbosity\n\t--extension=EXT\t:Saves the results in EXT extension (msh, csv, xdmf, vtk, vtu, cgns (requires instalation), xmf, h5m, stl, obj, post, post.gz, dato, dato.gz, dat, fem, ugrid, wkt, ...)"); exit()
	extension = "xdmf" if not [1 for arg in sys.argv if "--extension" in arg] else [arg.split('=')[1] for arg in sys.argv if "--extension" in arg][0]
	saverType = "default" if not [1 for arg in sys.argv if "--saver" in arg] else [arg.split('=')[1] for arg in sys.argv if "--saver" in arg][0]

	problemData = PyEFVLib.ProblemData(
		meshFilePath = "{MESHES}/msh/2D/Square.msh",
		outputFilePath = "{RESULTS}/geomechanics/linear",
		numericalSettings = PyEFVLib.NumericalSettings( timeStep = 1e-02, finalTime = np.inf, tolerance = 1e-06, maxNumberOfIterations = 300 ),
		propertyData = PyEFVLib.PropertyData({
		    "Body":
		    {
		        "PoissonsRatio"			: 2.0e-1,			# ν
		        "ShearModulus"			: 6.0e+9,			# G
		        "SolidCompressibility"	: 0.0,				# cs
		        "Porosity"				: 1.9e-1,			# Φ
		        "Permeability"			: 1.9e-15,			# k
		        "FluidCompressibility"	: 3.0303e-10,		# cf
		        "Viscosity"				: 1.0e-03,			# μ
		        "SolidDensity"			: 2.7e+3,			# ρs
		        "FluidDensity"			: 1.0e+3			# ρf
		    }
		    # α = 1 - cs / cb 					# Biot Coefficient
		    # cb = 1 / K 						# Bulk Compressibility
		    # K = 2G(1 + ν) / 3(1-2ν)			# Bulk Modulus
		    # ρ = Φ * ρf + (1-Φ) * ρs 			# Bulk Density
		    # M = 1 / (Φ * cf + (α-Φ) * cs) 	# Biot Modulus
		}),
		boundaryConditions = PyEFVLib.BoundaryConditions({
			"pressure": {
				"InitialValue": 0.0,
				"West":	 { "condition" : PyEFVLib.Neumann,   "type" : PyEFVLib.Constant,"value" : 0.0 },
				"East":	 { "condition" : PyEFVLib.Neumann,   "type" : PyEFVLib.Constant,"value" : 0.0 },
				"South": { "condition" : PyEFVLib.Neumann,   "type" : PyEFVLib.Constant,"value" : 0.0 },
				"North": { "condition" : PyEFVLib.Neumann,   "type" : PyEFVLib.Constant,"value" : 0.0 },
			},
			"u": {
			    "InitialValue": 0.0,
			    "West":  { "condition": PyEFVLib.Dirichlet, "type": PyEFVLib.Constant, "value": 0.0 },
			    "East":  { "condition": PyEFVLib.Dirichlet, "type": PyEFVLib.Constant, "value": 0.0 },
			    "South": { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": 0.0 },
			    "North": { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": 0.0 }
			},
			"v": {
			    "InitialValue": 0.0,
			    "West":  { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": 0.0 },
			    "East":  { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": 0.0 },
			    "South": { "condition": PyEFVLib.Dirichlet, "type": PyEFVLib.Constant, "value": 0.0 },
			    "North": { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": -1e4 }
			},
		}),
	)

	geomechanics(problemData, extension=extension, saverType=saverType, transient=not "-p" in sys.argv, verbosity="-v" in sys.argv)