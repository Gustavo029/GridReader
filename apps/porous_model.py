import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from PyEFVLib import MSHReader, Grid, ProblemData, CgnsSaver, CsvSaver, VtuSaver, VtmSaver
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import time

def porousModel(
		libraryPath,			# PyEFVLib path
		outputPath,				# Results directory path (Ex.: "results/heat_transfer_2d/...")
		extension,				# Extension type. Either "csv" or "cgns"

		grid,					# Object of class Grid
		propertyData,			# List of dictionaries containing the properties

		initialValues,			# Dictionary whose keys are the field names, and values are the field values
		neumannBoundaries,		# Dictionary whose keys are the field names, and values are objects of the class NeumannBoundaryCondition
		dirichletBoundaries,	# Dictionary whose keys are the field names, and values are objects of the class DirichletBoundaryCondition

		timeStep,				# Floating point number indicating the timeStep used in the simulation (constant)
		finalTime,				# The time at which, if reached, the simulation stops. If None, then it is not used.
		maxNumberOfIterations,	# Number of iterations at which, if reached, the simulation stops. If None, then it is not used.
		tolerance,				# The value at which, if the maximum difference between field values reach, the simulation stops. If None, then it is not used.
		
		fileName="Results",		# File name
		transient=True,			# If False, the transient term is not added to the equation, and it's solved in one iteration
		verbosity=True, 			# If False does not print iteration info
	):
	#-------------------------SETTINGS----------------------------------------------
	initialTime = time.time()

	dimension = grid.dimension
	currentTime = 0.0

	savers = {"cgns": CgnsSaver, "csv": CsvSaver, "vtu": VtuSaver, "vtm": VtmSaver}
	saver = savers[extension](grid, outputPath, libraryPath, fileName=fileName)

	pressureField = np.repeat(0.0, grid.vertices.size)
	prevPressureField = initialValues["pressure"].copy()

	coords,matrixVals = [], []
	difference = 0.0
	iteration = 0
	converged = False

	def add(i, j, val):
		coords.append((i,j))
		matrixVals.append(val)

	def getConstitutiveMatrix(region):
		shearModulus = propertyData[region.handle]["ShearModulus"]
		poissonsRatio = propertyData[region.handle]["PoissonsRatio"]

		lameParameter=2*shearModulus*poissonsRatio/(1-2*poissonsRatio)

		if region.grid.dimension == 2:
			constitutiveMatrix = np.array([[2*shearModulus+lameParameter ,lameParameter 				,0			 ],
										   [lameParameter				 ,2*shearModulus+lameParameter 	,0			 ],
										   [0			 				 ,0								,shearModulus]])

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



 

	#-------------------------------------------------------------------------------
	#-------------------------AFTER END OF MAIN LOOP ITERATION------------------------
	#-------------------------------------------------------------------------------
	finalSimulationTime = time.time()
	if verbosity:
		print("Ended Simultaion, elapsed {:.2f}s".format(finalSimulationTime-initialTime))

	saver.finalize()
	if verbosity:
		print("Saved file: elapsed {:.2f}s".format(time.time()-finalSimulationTime))

		print("\n\tresult: ", end="")
		print(os.path.realpath(saver.outputPath), "\n")

if __name__ == "__main__":
	if "--help" in sys.argv:
		print("\npython apps/porous_model.py workspace_file for opening a described model in workspace\n")
		print("-p\t for permanent regime (without the accumulation term)")
		print("-g\t for show results graphicaly")
		print("-s\t for verbosity 0")
		print("--extension=csv for output file in csv extension\n")
		print("--extension=cgns for output file in cgns extension\n")
		print("--extension=vtu for output file in vtu extension\n")
		print("--extension=vtm for output file in vtm extension\n")
		exit(0)
	
	model = "workspace/porous_model_2d/linear"
	if len(sys.argv)>1 and not "-" in sys.argv[1]: model=sys.argv[1]

	problemData = ProblemData(model)

	reader = MSHReader(problemData.paths["Grid"])
	grid = Grid(reader.getData())
	problemData.setGrid(grid)
	problemData.read()

	if not "-s" in sys.argv:
		for key,path in zip( ["input", "output", "grids"] , [os.path.join(problemData.libraryPath,"workspace",model) , problemData.paths["Output"], problemData.paths["Grid"]] ):
			print("\t{}\n\t\t{}\n".format(key, path))
		print("\tsolid")
		for region in grid.regions:
			print("\t\t{}".format(region.name))
			for _property in problemData.propertyData[region.handle].keys():
				print("\t\t\t{}   : {}".format(_property, problemData.propertyData[region.handle][_property]))
			print("")
		print("\n{:>9}\t{:>14}\t{:>14}\t{:>14}".format("Iteration", "CurrentTime", "TimeStep", "Difference"))

	porous_model(
		libraryPath = problemData.libraryPath,
		outputPath = problemData.paths["Output"],
		extension = "csv" if not [1 for arg in sys.argv if "--extension" in arg] else [arg.split('=')[1] for arg in sys.argv if "--extension" in arg][0],
		
		grid 	  = grid,
		propertyData = problemData.propertyData,
		
		# initialValues = problemData.initialValues,
		initialValues = { "pressure": np.repeat( problemData.initialValues["pressure"], grid.vertices.size ) },
		neumannBoundaries = problemData.neumannBoundaries,
		dirichletBoundaries = problemData.dirichletBoundaries,

		timeStep  = problemData.timeStep,
		finalTime = problemData.finalTime,
		maxNumberOfIterations = problemData.maxNumberOfIterations,
		tolerance = problemData.tolerance,
		
		transient = not "-p" in sys.argv,
		verbosity = not "-s" in sys.argv,
	)