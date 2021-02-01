import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np
import PyEFVLib

problemData = PyEFVLib.ProblemData(
	meshFilePath = "{MESHES}/msh/2D/Square.msh",
	outputFilePath = "{RESULTS}/geomechanics/linear",
	numericalSettings = PyEFVLib.NumericalSettings( timeStep = 1e-02, finalTime = None, tolerance = 1e-06, maxNumberOfIterations = 300 ),
	propertyData = PyEFVLib.PropertyData({
	    "Body":
	    {
	        "PoissonsRatio"			: 2.0e-1,			# ν
	        "ShearModulus"			: 6.0e+9,			# G
	        "SolidCompressibility"	: 2.777777e-11,		# cs
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
		    "North": { "condition": PyEFVLib.Neumann,   "type": PyEFVLib.Constant, "value": -490.0 }
		},
	}),
)

propertyData 	 = problemData.propertyData
timeStep 		 = problemData.timeStep
grid 			 = problemData.grid
numberOfVertices = grid.numberOfVertices
dimension 		 = grid.dimension

saver = PyEFVLib.MeshioSaver(grid, problemData.outputFilePath, problemData.libraryPath, extension="xdmf")

prevPressureField = problemData.initialValues["pressure"].copy()
pressureField 	  = np.repeat(0.0, numberOfVertices)
nextPressureField = np.repeat(0.0, numberOfVertices)

U = np.array(numberOfVertices*[0.0] + [-1e6*v.getCoordinates()[1] for v in grid.vertices])
prevDisplacements = np.repeat(0.0, dimension * numberOfVertices)
displacements 	  = np.repeat(0.0, dimension * numberOfVertices)
nextDisplacements = np.repeat(0.0, dimension * numberOfVertices)
# prevDisplacements = U.copy()
displacements 	  = U.copy()
nextDisplacements = U.copy()

gravityVector	  = np.array([0.0, 0.0, 0.0])[:dimension]

massMatrix 		= np.zeros((numberOfVertices, numberOfVertices))
massIndependent = np.zeros(numberOfVertices)

geoMatrix 		= np.zeros((dimension * numberOfVertices, dimension * numberOfVertices))
geoIndependent  = np.zeros(dimension * numberOfVertices)

poissonsRatio			= propertyData.get(0, "PoissonsRatio")
shearModulus			= propertyData.get(0, "ShearModulus")
solidCompressibility	= propertyData.get(0, "SolidCompressibility")
porosity				= propertyData.get(0, "Porosity")
permeability			= propertyData.get(0, "Permeability")
fluidCompressibility	= propertyData.get(0, "FluidCompressibility")
viscosity				= propertyData.get(0, "Viscosity")
solidDensity			= propertyData.get(0, "SolidDensity")
fluidDensity			= propertyData.get(0, "FluidDensity")
bulkModulus				= 2*shearModulus*(1 + poissonsRatio) / ( 3*(1-2*poissonsRatio) )
bulkCompressibility		= 1 / bulkModulus
biotCoefficient			= 1 - solidCompressibility / bulkCompressibility
bulkDensity				= porosity * fluidDensity + (1-porosity) * solidDensity
biotModulus				= 1 / (porosity * fluidCompressibility + (biotCoefficient-porosity) * solidCompressibility)
lameParameter=2*shearModulus*poissonsRatio/(1-2*poissonsRatio)
constitutiveMatrix = np.array([[2*shearModulus+lameParameter,lameParameter,0],[lameParameter,2*shearModulus+lameParameter,0],[0,0,shearModulus]]) if dimension==2 else np.array([[2*shearModulus+lameParameter,lameParameter,lameParameter,0,0,0],[lameParameter,2*shearModulus+lameParameter,lameParameter,0,0,0],[lameParameter,lameParameter,2*shearModulus+lameParameter,0,0,0],[0,0,0,shearModulus,0,0],[0,0,0,0,shearModulus,0],[0,0,0,0,0,shearModulus]])

def assembleMassMatrix():
	global massMatrix, inverseMassMatrix
	massMatrix 		= np.zeros((numberOfVertices, numberOfVertices))

	# Accumulation Term (1/M)(∂p/∂t)
	for vertex in grid.vertices:
		massMatrix[vertex.handle][vertex.handle] += vertex.volume / (biotModulus * timeStep)
		massMatrix[vertex.handle][vertex.handle] += vertex.volume * (biotCoefficient**2) / (bulkModulus * timeStep)

	# Darcy's velocity grad(p) contribution
	for element in grid.elements:
		for innerFace in element.innerFaces:
			globalDerivatives = innerFace.globalDerivatives
			area = innerFace.area.getCoordinates()[:dimension]
			coefficients = -(permeability/viscosity) * np.matmul(area.T, innerFace.globalDerivatives)

			backwardsHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
			forwardHandle   = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle

			for local, vertex in enumerate(element.vertices):
				massMatrix[backwardsHandle][vertex.handle] += coefficients[local]
				massMatrix[forwardHandle][vertex.handle]   -= coefficients[local]

	# Apply Boundary Conditions
	for bCondition in problemData.dirichletBoundaries["pressure"]:
		for vertex in bCondition.boundary.vertices:
			massMatrix[vertex.handle] = np.zeros(numberOfVertices)
			massMatrix[vertex.handle][vertex.handle] = 1.0

	inverseMassMatrix = np.linalg.inv(massMatrix)

def assembleMassVector():
	global massIndependent
	massIndependent = np.zeros(numberOfVertices)

	# Accumulation Term (1/M)(∂p/∂t)
	for vertex in grid.vertices:
		massIndependent[vertex.handle] += vertex.volume * prevPressureField[vertex.handle] / (biotModulus * timeStep)
		massIndependent[vertex.handle] += vertex.volume * pressureField[vertex.handle] * (biotCoefficient**2) / (bulkModulus * timeStep)

	# Gravity Term ∇∙[ (ρf*k/μ) g ]
	for element in grid.elements:
		for innerFace in element.innerFaces:
			globalDerivatives = innerFace.globalDerivatives
			area = innerFace.area.getCoordinates()[:dimension]
			coefficient = -(fluidDensity*permeability/viscosity) * np.matmul(area.T, gravityVector)

			backwardsHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
			forwardHandle   = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle

			massIndependent[backwardsHandle] += coefficient
			massIndependent[forwardHandle]   -= coefficient

	# Volumetric Deformation Rate Term ∂/∂t(∇∙u)
	for element in grid.elements:
		for face in [*element.innerFaces, *element.outerFaces]:
			area = face.area.getCoordinates()[:dimension]
			shapeFunctions = face.getShapeFunctions()
			# deltaU = U_old - U_k
			deltaU = np.array([prevDisplacements[vertex.handle + coord * numberOfVertices] - displacements[vertex.handle + coord * numberOfVertices] for coord in range(dimension) for vertex in element.vertices])
			op = np.array([Ni*sj for sj in area for Ni in shapeFunctions])

			coefficient = (biotCoefficient/timeStep) * np.matmul(op, deltaU)

			if type(face) == PyEFVLib.InnerFace:
				backwardsHandle, forwardHandle = face.getNeighborVerticesHandles()

				massIndependent[backwardsHandle] += coefficient
				massIndependent[forwardHandle] 	 -= coefficient

			elif type(face) == PyEFVLib.OuterFace:
				massIndependent[face.vertex.handle] += coefficient
				pass

	# Apply Boundary Conditions
	for bCondition in problemData.dirichletBoundaries["pressure"]:
		for vertex in bCondition.boundary.vertices:
			massIndependent[vertex.handle] = bCondition.getValue(vertex.handle)

	for bCondition in problemData.neumannBoundaries["pressure"]:
		for facet in bCondition.boundary.facets:
			for outerFace in facet.outerFaces:
				massIndependent[outerFace.vertex.handle] += bCondition.getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

# Effective Stress Term
def assembleGeoMatrix():
	global geoMatrix, inverseGeoMatrix
	geoMatrix 		= np.zeros((dimension * numberOfVertices, dimension * numberOfVertices))

	def getTransposedVoigtArea(innerFace):
		Sx, Sy, Sz = innerFace.area.getCoordinates()
		return np.array([[Sx,0,Sy],[0,Sy,Sx]]) if dimension==2 else np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])
	def getVoigtGradientOperator(globalDerivatives):
		if len(globalDerivatives) == 2:
			Nx,Ny = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero],[zero,Ny],[Ny,Nx]])
		if len(globalDerivatives) == 3:
			Nx,Ny,Nz = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero,zero],[zero,Ny,zero],[zero,zero,Nz],[Ny,Nx,zero],[zero,Nz,Ny],[Nz,zero,Nx]])

	# Effective Stress Term
	for element in grid.elements:
		for innerFace in element.innerFaces:
			transposedVoigtArea = getTransposedVoigtArea(innerFace)
			voigtGradientOperator = getVoigtGradientOperator(innerFace.globalDerivatives)

			coefficients = np.einsum("ij,jk,kmn->imn", transposedVoigtArea, constitutiveMatrix, voigtGradientOperator)

			backwardsHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
			forwardHandle   = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle
			
			for local in range(element.vertices.size):
				for i in range(dimension):
					for j in range(dimension):
						geoMatrix[backwardsHandle + numberOfVertices*i][local + numberOfVertices*j] += coefficients[i][j][local]
						geoMatrix[forwardHandle   + numberOfVertices*i][local + numberOfVertices*j] -= coefficients[i][j][local]

def assembleGeoVector():
	def getTransposedVoigtArea(innerFace):
		Sx, Sy, Sz = innerFace.area.getCoordinates()
		return np.array([[Sx,0,Sy],[0,Sy,Sx]]) if dimension==2 else np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])

	global geoIndependent

	for element in grid.elements:
		for innerFace in element.innerFaces:
			transposedVoigtArea = getTransposedVoigtArea(innerFace)

			shapeFunctionValues = element.shape.innerFaceShapeFunctionValues[innerFace.local]
			zeros = np.zeros(element.vertices.size)
			identityShapeFunctionMatrix = np.array([ shapeFunctionValues, shapeFunctionValues, shapeFunctionValues, zeros, zeros, zeros ]) if self.dimension == 3 else np.array([shapeFunctionValues, shapeFunctionValues, zeros])

			coefficients = -biotCoefficient * np.matmul( transposedVoigtArea, identityShapeFunctionMatrix )

			backwardsHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
			forwardHandle   = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle

			for coord in range( dimension ):
				for local in range( element.vertices.size ):
					localMatrixL[backwardsHandle+coord*element.vertices.size][local] += coefficients[coord][local]
					localMatrixL[forwardHandle+coord*element.vertices.size][local] -= coefficients[coord][local]


# Pressure Term

# Gravity Term


def main():
	global nextPressureField
	assembleMassMatrix()
	assembleMassVector()

	nextPressureField = np.matmul(inverseMassMatrix, massIndependent)
	saver.save("pressure", nextPressureField, 0.0)

	# assembleGeoMatrix()

	saver.finalize()

if __name__ == '__main__':
	main()

	print(max(abs(massIndependent)), max(abs(massMatrix.reshape(numberOfVertices**2))), max(abs(nextPressureField)))
	print(nextPressureField)

	import matplotlib
	from matplotlib import pyplot as plt
	from matplotlib import cm
	from matplotlib.colors import ListedColormap as CM, Normalize
	from scipy.interpolate import griddata

	X,Y = zip(*[v.getCoordinates()[:-1] for v in grid.vertices])

	Xi, Yi = np.meshgrid( np.linspace(min(X), max(X), len(X)), np.linspace(min(Y), max(Y), len(Y)) )
	nTi = griddata((X,Y), nextPressureField, (Xi,Yi), method="linear")

	plt.pcolor(Xi,Yi,nTi, shading="auto", cmap=CM( cm.get_cmap("RdBu",64)(np.linspace(1,0,64)) )) # Makes BuRd instead of RdBu
	plt.title("Numerical Temperature")
	plt.colorbar()
	plt.show()
