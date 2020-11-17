import matplotlib.pyplot as plt
import pandas as pd


def show(file):
	dData = pd.read_csv(file)
	X,Y,Z = dData["X"],dData["Y"],dData["Z"]
	u,v,w = dData["TimeStep1 - u"],dData["TimeStep1 - v"],dData["TimeStep1 - w"]
	plt.scatter(X,u,label='u')
	plt.scatter(Y,v,label='v')
	plt.scatter(Z,w,label='w')
	plt.xlabel("x,y,z (m)")
	plt.ylabel("displacement (mm)")
	plt.legend()	
	plt.title(file.replace('.csv',''))
	plt.show()


# Verify DIRICHLET
dFileName = "compression - prescribed displacement (dirichlet).csv"
show(dFileName)
# Verify Neumann
nFileName = "compression - prescribed stress (neumann).csv"
show(nFileName)