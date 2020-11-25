import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from PyEFVLib import Solver
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import time

class StressEquilibriumSolver(Solver):
	def __init__(self, workspaceDirectory, gravity=False, **kwargs):
		# kwargs -> outputFileName, outputFormat, transient, verbosity
		Solver.__init__(self, workspaceDirectory, **kwargs)
		self.gravity = gravity

	def init(self):
		self.displacements = np.repeat(0.0, self.dimension*self.numberOfVertices)
		
		self.coords,self.matrixVals = [], []

	def mainloop(self):
		self.assembleLinearSystem()
		self.solveLinearSystem()
		self.saveIterationResults()

	def add(self, i, j, val):
		self.coords.append((i,j))
		self.matrixVals.append(val)

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

	def getTransposedVoigtArea(self, face):
		Sx, Sy, Sz = face.area.getCoordinates()
		if self.dimension == 2:
			return np.array([[Sx,0,Sy],[0,Sy,Sx]])
		elif self.dimension == 3:
			return np.array([[Sx,0,0,Sy,0,Sz],[0,Sy,0,Sx,Sz,0],[0,0,Sz,0,Sy,Sx]])

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

	@staticmethod
	def getOuterFaceGlobalDerivatives(outerFace):
		localDerivatives = outerFace.facet.element.shape.vertexShapeFunctionDerivatives[ outerFace.vertexLocalIndex ]
		return outerFace.facet.element.getGlobalDerivatives(localDerivatives)

	def assembleLinearSystem(self):
		self.independent = np.zeros(self.dimension*self.numberOfVertices)

		U = lambda handle: handle + self.numberOfVertices * 0
		V = lambda handle: handle + self.numberOfVertices * 1
		W = lambda handle: handle + self.numberOfVertices * 2

		def gravityTerm():
			# Gravity Term
			for region in self.grid.regions:
				density = self.propertyData[region.handle]["Density"]
				gravity = self.propertyData[region.handle]["Gravity"]
				for element in region.elements:
					local = 0
					for vertex in element.vertices:
						self.independent[V(vertex.handle)] += - density * gravity * element.subelementVolumes[local]
						local += 1

		def stressTerm():
			# Stress Term
			for region in self.grid.regions:
				constitutiveMatrix = self.getConstitutiveMatrix(region)
				for element in region.elements:
					for innerFace in element.innerFaces:
						transposedVoigtArea = self.getTransposedVoigtArea(innerFace)
						voigtGradientOperator = self.getVoigtGradientOperator(innerFace.globalDerivatives)
						matrixCoefficient = np.einsum("ij,jk,kmn->imn", transposedVoigtArea, constitutiveMatrix, voigtGradientOperator)

						backwardVertexHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][0]].handle
						forwardVertexHandle = element.vertices[element.shape.innerFaceNeighborVertices[innerFace.local][1]].handle

						for local, vertex in enumerate(element.vertices):
							for neighborVertex in [backwardVertexHandle, forwardVertexHandle]:
								self.add( U(neighborVertex), U(vertex.handle), matrixCoefficient[0][0][local] )
								self.add( U(neighborVertex), V(vertex.handle), matrixCoefficient[0][1][local] )
								self.add( V(neighborVertex), U(vertex.handle), matrixCoefficient[1][0][local] )
								self.add( V(neighborVertex), V(vertex.handle), matrixCoefficient[1][1][local] )
								if self.dimension == 3:
									self.add( W(neighborVertex), W(vertex.handle), matrixCoefficient[2][2][local] )
									self.add( U(neighborVertex), W(vertex.handle), matrixCoefficient[0][2][local] )
									self.add( V(neighborVertex), W(vertex.handle), matrixCoefficient[1][2][local] )
									self.add( W(neighborVertex), U(vertex.handle), matrixCoefficient[2][0][local] )
									self.add( W(neighborVertex), V(vertex.handle), matrixCoefficient[2][1][local] )
								matrixCoefficient *= -1


		def boundaryConditions():
			# Boundary Conditions
			for bc in self.problemData.boundaryConditions:
				boundary=bc["u"].boundary
				uBoundaryType = bc["u"].__type__
				vBoundaryType = bc["v"].__type__
				wBoundaryType = bc["w"].__type__ if "w" in bc.keys() else None

				SÓ PRA RECAPITULAR, PARECEMOS QUE ESTAMOS COM UM CÓGIDO EM STRESS EQUILIBRIUM2
				IGUAL AO STRESS EQUILIBRIUM QUE FAZEM COISAS DIFERENTES

				A DIFERENÇA ESTÁ SÓ NAS CONDIÇÕES DE FRONTEIRA
				ESSE ESTÁ ADICIONANDO ALGUNS TERMOS QUE O SE2 NÃO ADICIONA, MAS O SE2 TB
				ADICIONA UNS TERMOS QUE NÃO TEM AQUI

				EM SE2 TEM UMA FUNÇÃOZINHA QUE SALVA A MATIZ ESPARSA EM UMA PLANILHA DO EXCEL
				E AQUI TEM 5 LINHAS QUE LEEM ISSO E COMPARAM

				PRECISAMOS ACHAR A DIFERENÇA ENTRE OS CÓDIGOS

				-------------

				DEPOIS QUE ESSE CÓDIGO ESTIVER FUNCIONADO TEMOS QUE VALIDAR O ESQUEMA DE
				SOMATÓRIO DE DIVERGENTE E GRADIENTE, E ÁREA, E VOLUME, E INTEGRAL, E CAMPO ANALÍTICO,
				TODA ESSA PARADA TEM QUE SER VALIDADA

				DEPOIS AINDA QUEREMOS IMPLEMENTAR O MODELO GEOMECÂNICO COM EQUAÇÃO DA CONSERVAÇÃO DA MASSA

				def neumannBoundaryConditions():
					if uBoundaryType == "NEUMANN":
						for facet in boundary.facets:
							for outerFace in facet.outerFaces:
								self.independent[U(outerFace.vertex.handle)] -= bc["u"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

					if vBoundaryType == "NEUMANN":
						for facet in boundary.facets:
							for outerFace in facet.outerFaces:
								self.independent[V(outerFace.vertex.handle)] -= bc["v"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

					if wBoundaryType == "NEUMANN":
						for facet in boundary.facets:
							for outerFace in facet.outerFaces:
								self.independent[W(outerFace.vertex.handle)] -= bc["w"].getValue(outerFace.handle) * np.linalg.norm(outerFace.area.getCoordinates())

				def dirichletBoundaryConditions():
					if uBoundaryType == "DIRICHLET":
						for vertex in boundary.vertices:
							self.independent[U(vertex.handle)] = bc["u"].getValue(vertex.handle)
							self.matrixVals = [val for coord, val in zip(self.coords, self.matrixVals) if coord[0] != U(vertex.handle)]
							self.coords 	   = [coord for coord in self.coords if coord[0] != U(vertex.handle)]
							self.add(U(vertex.handle), U(vertex.handle), 1.0)

					if vBoundaryType == "DIRICHLET":
						for vertex in boundary.vertices:
							self.independent[V(vertex.handle)] = bc["v"].getValue(vertex.handle)
							self.matrixVals = [val for coord, val in zip(self.coords, self.matrixVals) if coord[0] != V(vertex.handle)]
							self.coords 	   = [coord for coord in self.coords if coord[0] != V(vertex.handle)]
							self.add(V(vertex.handle), V(vertex.handle), 1.0)

					if wBoundaryType == "DIRICHLET":
						for vertex in boundary.vertices:
							self.independent[W(vertex.handle)] = bc["w"].getValue(vertex.handle)
							self.matrixVals = [val for coord, val in zip(self.coords, self.matrixVals) if coord[0] != W(vertex.handle)]
							self.coords 	   = [coord for coord in self.coords if coord[0] != W(vertex.handle)]
							self.add(W(vertex.handle), W(vertex.handle), 1.0)

			neumannBoundaryConditions()
			dirichletBoundaryConditions()

		if self.gravity:
			gravityTerm()
		stressTerm()
		boundaryConditions()

		import pandas as pd
		data = pd.read_csv("matrix.csv")
		self.dcoords = list(zip( list(np.array(data["I"])), list(np.array(data["J"])) ))
		self.dmvals = list(np.array(data["V"]))
		self.ccoords = self.coords.copy()
		self.cmvals = self.matrixVals.copy()


	def solveLinearSystem(self):
		self.matrix = sparse.csc_matrix( (self.matrixVals, zip(*self.coords)) )
		self.inverseMatrix = sparse.linalg.inv( self.matrix )
		self.displacements = self.inverseMatrix * self.independent

	def saveIterationResults(self):
		self.saver.save('u', self.displacements[0*self.numberOfVertices:1*self.numberOfVertices], self.currentTime)
		self.saver.save('v', self.displacements[1*self.numberOfVertices:2*self.numberOfVertices], self.currentTime)
		if self.dimension == 3:
			self.saver.save('w', self.displacements[2*self.numberOfVertices:3*self.numberOfVertices], self.currentTime)

def stressEquilibrium(workspaceDirectory,solve=True,outputFileName="Results",outputFormat="csv",gravity=False,verbosity=False):
	solver = StressEquilibriumSolver(workspaceDirectory,outputFileName=outputFileName,outputFormat=outputFormat,gravity=gravity,verbosity=verbosity)
	if solve:
		solver.solve()
	return solver

if __name__ == "__main__":
	model = "workspace/stress_equilibrium_2d/linear"
	if len(sys.argv)>1 and not "-" in sys.argv[1]: model=sys.argv[1]
	extension = "csv" if not [1 for arg in sys.argv if "--extension" in arg] else [arg.split('=')[1] for arg in sys.argv if "--extension" in arg][0]

	solver=stressEquilibrium(model,outputFormat=extension,gravity="-G" in sys.argv)