from PyEFVLib import ( 
	Grid, ProblemData,
	CgnsSaver, CsvSaver, VtuSaver, VtmSaver,
	MSHReader,
)
import os
import time


class Solver:
	def __init__(self, workspaceDirectory, outputFileName="Results", outputFormat="csv", transient=True, verbosity=False, **kwargs):
		self.workspaceDirectory = workspaceDirectory
		self.outputFileName = outputFileName
		self.outputFormat = outputFormat
		self.transient = transient
		self.verbosity = verbosity
		
		self.problemData = ProblemData(workspaceDirectory)
		self.reader = MSHReader( self.problemData.paths["Grid"] )
		self.grid = Grid( self.reader.getData() )
		self.problemData.setGrid(self.grid)
		self.problemData.read()


	def solve(self):
		self.settings()			# DEFINED HERE 
		self.printHeader()		# DEFINED HERE 
		self.init()				# USER DEFINED
		self.mainloop()			# USER DEFINED
		self.save()				# DEFINED HERE

	def settings(self):
		self.initialTime = time.time()

		self.propertyData = self.problemData.propertyData
		self.outputPath = self.problemData.paths["Output"]

		savers = {"cgns": CgnsSaver, "csv": CsvSaver}

		self.saver = savers[self.outputFormat](self.grid, self.outputPath, self.problemData.libraryPath, fileName=self.outputFileName)

		self.dimension = self.grid.dimension
		self.currentTime = 0.0
		self.timeStep = self.problemData.timeStep
		self.tolerance = self.problemData.tolerance

		self.iteration = 0
		self.converged = False

	def printHeader(self):
		if self.verbosity:
			for key,path in zip( ["input", "output", "grids"] , [os.path.join(self.problemData.libraryPath,"workspace",self.workspaceDirectory) , self.problemData.paths["Output"], self.problemData.paths["Grid"]] ):
				print("\t{}\n\t\t{}\n".format(key, path))
			print("\tsolid")
			for region in self.grid.regions:
				print("\t\t{}".format(region.name))
				for _property in self.problemData.propertyData[region.handle].keys():
					print("\t\t\t{}   : {}".format(_property, self.problemData.propertyData[region.handle][_property]))
				print("")
			print("\n{:>9}\t{:>14}\t{:>14}\t{:>14}".format("Iteration", "CurrentTime", "TimeStep", "Difference"))

	def printFooter(self):
		if self.verbosity:
			print("Ended Simultaion, elapsed {:.2f}s".format(self.finalSimulationTime-self.initialTime))
			print("Saved file: elapsed {:.2f}s".format(time.time()-self.finalSimulationTime))
			print("\n\tresult: ")
			print(os.path.realpath(self.saver.outputPath), "\n")

	def printIterationData(self):
		if self.verbosity:
			print("{:>9}\t{:>14.2e}\t{:>14.2e}\t{:>14.2e}".format(self.iteration, self.currentTime, self.timeStep, self.difference))

	def save(self):
		self.saver.finalize()

"""
class SomeSolver(Solver):
	def __init__(self, workspaceDirectory, **kwargs):
		Solver.__init__(self, workspaceDirectory, **kwargs)
	def mainloop(self);
	def assembleMatrix(self);
	def addToIndependentVector(self);
	def solveLinearSystem(self);
	def saveIterationResults(self);
	def checkConvergence(self);
	def save(self);
"""