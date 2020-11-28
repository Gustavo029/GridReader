import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from PyEFVLib import Solver
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import time

class PorousModelSolver(Solver):
	def __init__(self, workspaceDirectory, gravity=False, **kwargs):
		# kwargs -> outputFileName, outputFormat, transient, verbosity
		Solver.__init__(self, workspaceDirectory, **kwargs)
		self.gravity = gravity

	def init(self):
		self.gravityVector = -9.81 * np.array([0.0, 1.0, 0.0])

		self.prevPressureField = self.problemData.initialValues["pressure"]				# p_old
		self.pressureField = np.repeat(0.0, self.grid.vertices.size)					# p_k
		self.nextPressureField = np.repeat(0.0, self.grid.vertices.size)				# p_(k+1)

		self.prevDisplacements = np.repeat(0.0, self.dimension*self.numberOfVertices)	# u_old
		self.displacements = np.repeat(0.0, self.dimension*self.numberOfVertices)		# u_k
		self.nextDisplacements = np.repeat(0.0, self.dimension*self.numberOfVertices)	# u_(k+1)

		# self.saver.save("pressure", self.prevPressureField, self.currentTime)

		# self.coords,self.matrixVals = [], []
		# self.difference=0.0

		self.massIndependentVector = np.zeros( self.numberOfVertices )
		self.geoIndependentVector = np.zeros( 3*self.numberOfVertices )

		self.massMatrixVals, self.massCoords = [], []
		self.geoMatrixVals,  self.geoCoords  = [], []

		self.matricesDict = {
			"mass": ( self.massMatrixVals, self.massCoords ),
			"geo" : ( self.geoMatrixVals,  self.geoCoords ),
		}

	def mainloop(self):
		self.assembleGlobalMatrixMass()
		self.assembleGlobalMatrixGeo()

		# Temporal Loop
		print("aaaaaaaaaaaaaaaaaa")
		while not self.converged:
			print("bbbbbbbbbbbbbbbbbb")
			# Rever essa linha, eu duvido um pouco...
			self.difference = 2*self.tolerance

			self.pressureField = self.prevPressureField.copy()
			self.displacements = self.prevDisplacements.copy()

			print("cccccccccccccccccc")
			# Iterative Loop
			while self.difference >= self.tolerance:
				print("ddddddddddddddddd")
				self.assembleMassVector()
				self.solveIterativePressureField()
				self.assembleGeoVector()
				self.solveIterativeDisplacementsField()

				self.checkIterativeConvergence()
				# fazer aq mesmo, não chamar uma função...
				# self.updateIterativeFields2()

				self.iteration += 1
				break

			# fazer aq mesmo, não chamar uma função...
			# self.updateTemporalFields()
			self.saveTemporalResults()

			self.currentTime += self.timeStep
			break

	# -------------- HELPER FUNCTIONS --------------

	def add(self, i, j, val, matrixName):
		self.matricesDict[matrixName][0].append(val)
		self.matricesDict[matrixName][1].append((i,j))

	def getTransposedVoigtArea(self, face):
		Sx, Sy, Sz = face.area.getCoordinates()
		if self.dimension == 2:
			return np.array([[Sx,0,Sy],[0,Sy,Sx]])
		elif self.dimension == 3:
			return np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])

	def getConstitutiveMatrix(self, region):
		shearModulus = self.propertyData[region.handle]["ShearModulus"]
		poissonsRatio = self.propertyData[region.handle]["PoissonsRatio"]

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
				self.add( element.vertices[i].handle, element.vertices[j].handle, localMatrix[i][j], "mass" )
	
	def assembleLocalMatrixGeo(self, element, localMatrix):
		for i in range(element.vertices.size):
			for j in range(element.vertices.size):
				self.add( element.vertices[i].handle + 0*self.numberOfVertices, element.vertices[j].handle + 0*self.numberOfVertices, localMatrix[i][j], "geo" )
				self.add( element.vertices[i].handle + 1*self.numberOfVertices, element.vertices[j].handle + 1*self.numberOfVertices, localMatrix[i][j], "geo" )
				self.add( element.vertices[i].handle + 2*self.numberOfVertices, element.vertices[j].handle + 2*self.numberOfVertices, localMatrix[i][j], "geo" )

	def computeLocalMatrixA(self, element):
		# Accumulation Term
		localMatrixA = np.zeros( (element.vertices.size, element.vertices.size) )

		biotCoefficient = self.propertyData[element.region.handle]["BiotCoefficient"]

		for local in range(element.vertices.size):
			coefficient = element.subelementVolumes[local] / ( biotCoefficient * self.timeStep )
	
			localMatrixA[local][local] += coefficient

		return localMatrixA

	def computeLocalMatrixR(self, element):
		localMatrixR = np.zeros( (element.vertices.size, element.vertices.size) )
		
		biotCoefficient = self.propertyData[element.region.handle]["BiotCoefficient"]
		bulkModulus = self.propertyData[element.region.handle]["BulkModulus"]

		for local in range(element.vertices.size):
			coefficient = (biotCoefficient**2) * element.subelementVolumes[local] / (bulkModulus * self.timeStep)
			
			localMatrixR[local][local] += coefficient

		return localMatrixR

	def computeLocalMatrixH(self, element):
		localMatrixH = np.zeros( (element.vertices.size, element.vertices.size) )

		permeability = self.propertyData[element.region.handle]["Permeability"]
		viscosity = self.propertyData[element.region.handle]["Viscosity"]

		# Pressure gradient term
		for innerFace in element.innerFaces:
			coefficientsVector = -(permeability/viscosity)*np.matmul( np.transpose(innerFace.area.getCoordinates()[:self.dimension]), innerFace.globalDerivatives )
	
			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for local in range(element.vertices.size):
				localMatrixH[backwardVertexLocal][local] += coefficientsVector[local]
				localMatrixH[forwardVertexLocal][local] -= coefficientsVector[local]

		return localMatrixH

	def computeLocalMatrixQ(self, element):
		# Volumetric Deformation Term
		localMatrixQ = np.zeros( ( element.vertices.size, element.vertices.size * self.dimension ) )

		for innerFace in element.innerFaces:
			biotCoefficient = self.propertyData[element.region.handle]["BiotCoefficient"]

			coefficients = -(biotCoefficient/self.timeStep) * np.transpose(innerFace.area.getCoordinates()[self.dimension]) * innerFace.globalDerivatives

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for c in range( self.dimension ):
				for local in range( element.vertices.size ):
					localMatrixQ[backwardVertexLocal][local+c*self.dimension] += coefficients[c+local*self.dimension]
					localMatrixQ[forwardVertexLocal][local+c*self.dimension]  -= coefficients[c+local*self.dimension]

		return localMatrixQ

	def computeLocalMatrixG(self, element):
		localMatrixG = np.zeros( (element.vertices.size, element.vertices.size) )

		permeability = self.propertyData[element.region.handle]["Permeability"]
		fluidDensity = self.propertyData[element.region.handle]["FluidDensity"]
		viscosity = self.propertyData[element.region.handle]["Viscosity"]

		for innerFace in element.innerFaces:
			Gpi = (fluidDensity*permeability/viscosity) * innerFace.area.getCoordinates()[:self.dimension]

			backwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][0]
			forwardVertexLocal = element.shape.innerFaceNeighborVertices[innerFace.local][1]

			for local in range(element.vertices.size):
				localMatrixG[backwardVertexLocal][local] += Gpi[local]
				localMatrixG[forwardVertexLocal][local] -= Gpi[local]

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

	# ----------- END OF HELPER FUNCTIONS -----------

	def assembleGlobalMatrixMass(self):
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
				self.add(vertex.handle, vertex.handle, 1.0, "mass")

	def assembleGlobalMatrixGeo(self):
		U = lambda handle: handle + self.numberOfVertices * 0
		V = lambda handle: handle + self.numberOfVertices * 1
		W = lambda handle: handle + self.numberOfVertices * 2

		for element in self.grid.elements:
			self.assembleLocalMatrixGeo(element, self.computeLocalMatrixK(element))

		# Boundary Conditions
		for bc in self.problemData.boundaryConditions:
			boundary=bc["u"].boundary
			uBoundaryType = bc["u"].__type__
			vBoundaryType = bc["v"].__type__
			wBoundaryType = bc["w"].__type__ if "w" in bc.keys() else None

			# Dirichlet Boundary Conditions
			if uBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.independent[U(vertex.handle)] = bc["u"].getValue(vertex.handle)
					self.GeoMatrixVals = [val for coord, val in zip(self.GeoCoords, self.GeoMatrixVals) if coord[0] != U(vertex.handle)]
					self.GeoCoords 	   = [coord for coord in self.GeoCoords if coord[0] != U(vertex.handle)]
					self.add(U(vertex.handle), U(vertex.handle), 1.0, "geo")

			if vBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.independent[V(vertex.handle)] = bc["v"].getValue(vertex.handle)
					self.GeoMatrixVals = [val for coord, val in zip(self.GeoCoords, self.GeoMatrixVals) if coord[0] != V(vertex.handle)]
					self.GeoCoords 	   = [coord for coord in self.GeoCoords if coord[0] != V(vertex.handle)]
					self.add(V(vertex.handle), V(vertex.handle), 1.0, "geo")

			if wBoundaryType == "DIRICHLET":
				for vertex in boundary.vertices:
					self.independent[W(vertex.handle)] = bc["w"].getValue(vertex.handle)
					self.GeoMatrixVals = [val for coord, val in zip(self.GeoCoords, self.GeoMatrixVals) if coord[0] != W(vertex.handle)]
					self.GeoCoords 	   = [coord for coord in self.GeoCoords if coord[0] != W(vertex.handle)]
					self.add(W(vertex.handle), W(vertex.handle), 1.0, "geo")

	def assembleMassVector(self):
		self.massIndependentVector = np.zeros( self.numberOfVertices )

		for element in self.grid.elements:
			localMatrixQ = self.computeLocalMatrixQ(element)
			localMatrixA = self.computeLocalMatrixA(element)
			localMatrixR = self.computeLocalMatrixR(element)
			localMatrixG = self.computeLocalMatrixG(element)

			# Get pressures and displacements
			prevElementPressures 	 = [ prevPressureField[vertex.handle] for vertex in element.vertices ]
			elementPressures 		 = [ pressureField[vertex.handle] for vertex in element.vertices ]
			prevElementDisplacements = [ prevDisplacements[m+c*self.numberOfVertices] for c in range(self.dimension) for m in range(element.vertices.size) ]
			elementDisplacements 	 = [ displacements[m+c*self.numberOfVertices] for c in range(self.dimension) for m in range(element.vertices.size) ]

			coefficients = np.matmul(localMatrixA, prevElementPressures) + np.matmul(localMatrixR, elementPressures - prevElementPressures) + np.matmul(localMatrixQ, prevElementDisplacements - elementDisplacements) + np.matmul(localMatrixG, self.gravityVector)

			local = 0
			for vertex in element.vertices:
				self.massIndependentVector[vertex.handle] = coefficients[local]
				local += 1

		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()
		# applyBounndaryConditionsToMassVector()

	def assembleGeoVector(self):
		self.geoIndependentVector = np.zeros( self.dimension * self.numberOfVertices )

		for element in self.grid.elements:
			solidDensity = self.propertyData[element.region.handle]["SolidDensity"]
			fluidDensity = self.propertyData[element.region.handle]["FluidDensity"]
			porosity = self.propertyData[element.region.handle]["Porosity"]
			density = porosity * fluidDensity + (1 - porosity) * solidDensity

			localMatrixL = self.computeLocalMatrixL(element)

			nextElementPressures = [ nextPressureField[vertex.handle] for vertex in element.vertices ]
			coefficients = np.matmul( localMatrixL, nextElementPressures )

			local = 0
			for vertex in element.vertices:
				subelementVolume = element.subelementVolumes[local]

				for c in range(self.dimension):
					self.geoIndependentVector[ vertex.handle+c*self.numberOfVertices ] = density * subelementVolume * self.gravityVector[c] + coefficients[c+local*element.vertices.size]

				local += 1

		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()
		# applyBounndaryConditionsToGeoVector()


	def solveIterativePressureField(self):
		pass

	def solveIterativeDisplacementsField(self):
		pass

	def checkIterativeConvergence(self):
		pass

	def saveTemporalResults(self):
		pass

def porousModel(workspaceDirectory,solve=True,outputFileName="Results",outputFormat="csv",gravity=False,verbosity=False):
	solver = PorousModelSolver(workspaceDirectory,outputFileName=outputFileName,outputFormat=outputFormat,gravity=gravity,verbosity=verbosity)
	if solve:
		solver.solve()
	return solver

if __name__ == "__main__":
	model = "workspace/porous_model/linear"
	if len(sys.argv)>1 and not "-" in sys.argv[1]: model=sys.argv[1]
	extension = "csv" if not [1 for arg in sys.argv if "--extension" in arg] else [arg.split('=')[1] for arg in sys.argv if "--extension" in arg][0]

	solver=porousModel(model,outputFormat=extension,gravity="-G" in sys.argv)