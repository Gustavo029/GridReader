import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from PyEFVLib import MSHReader, Grid, ProblemData, CsvSaver, CgnsSaver
import numpy as np
from scipy import sparse
import scipy.sparse.linalg

#-------------------------SETTINGS----------------------------------------------
def stressEquilibrium(
		libraryPath,			# PyEFVLib path
		outputPath,				# Results directory path (Ex.: "results/heat_transfer_2d/...")
		extension,				# Extension type. Either "csv" or "cgns"

		grid,					# Object of class Grid
		propertyData,			# List of dictionaries containing the properties

		neumannBoundaries,		# Dictionary whose keys are the field names, and values are objects of the class NeumannBoundaryCondition
		dirichletBoundaries,	# Dictionary whose keys are the field names, and values are objects of the class DirichletBoundaryCondition
		boundaryConditions,		# List of dictionaries whose keys are field names and values are BoundaryCondition objects

		fileName="Results",		# File name
		verbosity=True 			# If False does not print iteration info
	):

	# from PyEFVLib.boundaryConditionPrinter import stressEquilibriumBoundaryConditionsPrinter
	# stressEquilibriumBoundaryConditionsPrinter(problemData.boundaryConditions)

	savers = {"csv": CsvSaver, "cgns": CgnsSaver}
	saver = savers[extension](grid, outputPath, libraryPath, fileName=fileName)

	currentTime = 0.0
	numberOfVertices = grid.vertices.size
	displacements = np.repeat(0.0, grid.dimension*numberOfVertices)

	coords,matrixVals = [], []
	#---------------------------HELPER FUNCTIONS------------------------------------
	def add(i, j, val):
		coords.append((i,j))
		matrixVals.append(val)

	def getConstitutiveMatrix(region):
		shearModulus = propertyData[region.handle]["ShearModulus"]
		poissonsRatio = propertyData[region.handle]["PoissonsRatio"]

		lameParameter=2*shearModulus*poissonsRatio/(1-2*poissonsRatio)

		if region.grid.dimension == 2:
			constitutiveMatrix = np.array([[lameParameter*(1-poissonsRatio)/poissonsRatio,lameParameter 							   ,0			],
										   [lameParameter								 ,lameParameter*(1-poissonsRatio)/poissonsRatio,0			],
										   [0			 								 ,0											   ,shearModulus]])
		elif region.grid.dimension == 3:
			constitutiveMatrix = np.array([[2*shearModulus+lameParameter	,lameParameter				 ,lameParameter				  ,0		,0	 ,0],
										   [lameParameter					,2*shearModulus+lameParameter,lameParameter				  ,0		,0	 ,0],
										   [lameParameter					,lameParameter				 ,2*shearModulus+lameParameter,0		,0	 ,0],
										   [0								,0							 ,0							  ,shearModulus,0,0],
										   [0								,0							 ,0							  ,0,shearModulus,0],
										   [0								,0							 ,0							  ,0,0,shearModulus]])

		return constitutiveMatrix

	def getTransposedVoigtArea(face):
		Sx, Sy, Sz = face.area.getCoordinates()
		if face.element.grid.dimension == 2:
			return np.array([[Sx,0,Sy],[0,Sy,Sx]])
		elif face.element.grid.dimension == 3:
			return np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])

	def getVoigtGradientOperator(globalDerivatives):
		if len(globalDerivatives) == 2:
			Nx,Ny = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero],[zero,Ny],[Ny,Nx]])

		if len(globalDerivatives) == 3:
			Nx,Ny,Nz = globalDerivatives
			zero=np.zeros(Nx.size)
			return np.array([[Nx,zero,zero],[zero,Ny,zero],[zero,zero,Nz],[Ny,Nx,zero],[zero,Nz,Ny],[Nz,zero,Nx]])


	def getOuterFaceGlobalDerivatives(outerFace):
		localDerivatives = outerFace.facet.element.shape.vertexShapeFunctionDerivatives[ outerFace.vertexLocalIndex ]
		return outerFace.facet.element.getGlobalDerivatives(localDerivatives)

	#-------------------------ADD TO LINEAR SYSTEM------------------------------
	independent = np.zeros(grid.dimension*numberOfVertices)

	U = lambda handle: handle + numberOfVertices * 0
	V = lambda handle: handle + numberOfVertices * 1
	W = lambda handle: handle + numberOfVertices * 2

	# Gravity Term
	for region in grid.regions:
		density = propertyData[region.handle]["Density"]
		gravity = propertyData[region.handle]["Gravity"]
		for element in region.elements:
			local = 0
			for vertex in element.vertices:
				independent[V(vertex.handle)] += - density * gravity * element.subelementVolumes[local]
				local += 1

	# Stress Term
	for region in grid.regions:
		constitutiveMatrix = getConstitutiveMatrix(region)
		for element in region.elements:
			for innerFace in element.innerFaces:
				transposedVoigtArea = getTransposedVoigtArea(innerFace)
				voigtGradientOperator = getVoigtGradientOperator(innerFace.globalDerivatives)
				matrixCoefficient = np.einsum("ij,jk,kmn->imn", transposedVoigtArea, constitutiveMatrix, voigtGradientOperator)

				backwardVertexHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
				forwardVertexHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle
				
				local=0
				for vertex in element.vertices:
					for neighborVertex in [backwardVertexHandle, forwardVertexHandle]:
						add( U(neighborVertex), U(vertex.handle), matrixCoefficient[0][0][local] )
						add( U(neighborVertex), V(vertex.handle), matrixCoefficient[0][1][local] )
						add( V(neighborVertex), U(vertex.handle), matrixCoefficient[1][0][local] )
						add( V(neighborVertex), V(vertex.handle), matrixCoefficient[1][1][local] )
						if grid.dimension == 3:
							add( W(neighborVertex), W(vertex.handle), matrixCoefficient[2][2][local] )
							add( U(neighborVertex), W(vertex.handle), matrixCoefficient[0][2][local] )
							add( V(neighborVertex), W(vertex.handle), matrixCoefficient[1][2][local] )
							add( W(neighborVertex), U(vertex.handle), matrixCoefficient[2][0][local] )
							add( W(neighborVertex), V(vertex.handle), matrixCoefficient[2][1][local] )

						matrixCoefficient *= -1
					local+=1

	# Boundary Conditions
	for bc in boundaryConditions:
		boundary=bc["u"].boundary
		uBoundaryType = bc["u"].__type__
		vBoundaryType = bc["v"].__type__
		wBoundaryType = bc["w"].__type__ if "w" in bc.keys() else None


		if uBoundaryType == "NEUMANN":
			for facet in boundary.facets:
				for outerFace in facet.outerFaces:
					independent[U(outerFace.vertex.handle)] -= bc["u"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

		if vBoundaryType == "NEUMANN":
			for facet in boundary.facets:
				for outerFace in facet.outerFaces:
					independent[V(outerFace.vertex.handle)] -= bc["v"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

		if wBoundaryType == "NEUMANN":
			for facet in boundary.facets:
				for outerFace in facet.outerFaces:
					independent[W(outerFace.vertex.handle)] -= bc["w"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())


		if uBoundaryType == "DIRICHLET":
			for vertex in boundary.vertices:
				independent[U(vertex.handle)] = bc["u"].getValue(vertex.handle)
				matrixVals = [val for coord, val in zip(coords, matrixVals) if coord[0] != U(vertex.handle)]
				coords 	   = [coord for coord in coords if coord[0] != U(vertex.handle)]
				add(U(vertex.handle), U(vertex.handle), 1.0)


		if vBoundaryType == "DIRICHLET":
			for vertex in boundary.vertices:
				independent[V(vertex.handle)] = bc["v"].getValue(vertex.handle)
				matrixVals = [val for coord, val in zip(coords, matrixVals) if coord[0] != V(vertex.handle)]
				coords 	   = [coord for coord in coords if coord[0] != V(vertex.handle)]
				add(V(vertex.handle), V(vertex.handle), 1.0)

		if wBoundaryType == "DIRICHLET":
			for vertex in boundary.vertices:
				independent[W(vertex.handle)] = bc["w"].getValue(vertex.handle)
				matrixVals = [val for coord, val in zip(coords, matrixVals) if coord[0] != W(vertex.handle)]
				coords 	   = [coord for coord in coords if coord[0] != W(vertex.handle)]
				add(W(vertex.handle), W(vertex.handle), 1.0)



	#-------------------------SOLVE LINEAR SYSTEM-------------------------------
	matrix = sparse.coo_matrix( (matrixVals, zip(*coords)) )
	matrix = sparse.csc_matrix( matrix )
	displacements = scipy.sparse.linalg.spsolve(matrix, independent)

	#-------------------------SAVE RESULTS--------------------------------------
	print(displacements.shape)
	saver.save('u', displacements[0*numberOfVertices:1*numberOfVertices], currentTime)
	saver.save('v', displacements[1*numberOfVertices:2*numberOfVertices], currentTime)
	if grid.dimension == 3:
		saver.save('w', displacements[2*numberOfVertices:3*numberOfVertices], currentTime)
	
	saver.finalize()

	print("\n\t\033[1;35mresult:\033[0m", saver.outputPath, '\n')

	return displacements


if __name__ == "__main__":
	model = "workspace/stress_equilibrium_3d/linear"

	problemData = ProblemData(model)

	reader = MSHReader(problemData.paths["Grid"])
	grid = Grid(reader.getData())
	problemData.setGrid(grid)
	problemData.read()

	displacements = stressEquilibrium(
		libraryPath = problemData.libraryPath,
		outputPath = problemData.paths["Output"],
		extension = "csv" if not "--extension=cgns" in sys.argv else "cgns",
		
		grid 	  = grid,
		propertyData = problemData.propertyData,
		
		neumannBoundaries = problemData.neumannBoundaries,
		dirichletBoundaries = problemData.dirichletBoundaries,
		boundaryConditions = problemData.boundaryConditions,

		verbosity = not "-s" in sys.argv
	)

	#-------------------------------------------------------------------------------
	#-------------------------SHOW RESULTS GRAPHICALY-------------------------------
	#-------------------------------------------------------------------------------
	from matplotlib import pyplot as plt, colors, cm
	def show_1d(fieldValues, name):
		top_stress = problemData.boundaryConditionData["v"]["North"]["value"]
		shearModulus = problemData.propertyData[0]["ShearModulus"]
		poissonsRatio = problemData.propertyData[0]["PoissonsRatio"]
		lameParameter = 2*shearModulus*poissonsRatio/(1-2*poissonsRatio)
		density = problemData.propertyData[0]["Density"]
		gravity = problemData.propertyData[0]["Gravity"]
		height = 1.0

		y, vals = zip(*[ (vertex.getCoordinates()[1], val) for vertex, val in zip(grid.vertices, fieldValues) if 0.1 > np.abs(vertex.getCoordinates()[0]-0.5)])
		y, vals = zip(*( sorted( zip(y, vals), key=lambda p:p[0] ) ))
		y, vals = np.array(y), np.array(vals)
		
		a_vals=y*(top_stress+density*gravity*(height-y/2))/(2*shearModulus+lameParameter)
		plt.figure()
		plt.scatter(y,1000*vals, marker='.', color='k', label="Resultados Numéricos")
		plt.plot(y,1000*a_vals, label="Solução Analítica")
		plt.xlabel("X (m)")
		plt.ylabel("v (mm)")
		plt.legend()	
		plt.title(name)

	show_1d(displacements[grid.vertices.size:], "Deslocamento em y")
	plt.show()
