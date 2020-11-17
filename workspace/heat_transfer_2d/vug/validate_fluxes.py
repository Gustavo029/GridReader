import numpy as np
import pandas as pd
import os

def validate_fluxes(fileName, baseDir):
	data = pd.read_csv(fileName)

	X = sorted(list(set([ round(x,5) for x in data["X1"] ])))
	x1, x2 = X[1], X[-2]

	s1 = sum( [ qx*(x1/2) for qx, x in zip(data["time_step_1 - q''x"], data["X1"]) if abs(x-x1)<(X[2]-X[0])/8 ] )
	s2 = sum( [ qx*(x1/2) for qx, x in zip(data["time_step_1 - q''x"], data["X1"]) if abs(x-x2)<(X[-1]-X[-3])/8 ] )

	print(f"{ fileName.replace(baseDir,'')[1:] } - Σq(x={x1})={s1} - Σq(x={x2})={s2} - (Σq(x2)-Σq(x1))/Σq(x2)={(s2-s1)/s2:.5e}")

if __name__ == "__main__":
	baseDir = "fluxos - vec\\results"
	for fileName in os.listdir(baseDir):
		validate_fluxes(os.path.join(baseDir, fileName), baseDir=baseDir)
