from PyEFVLib.geometry.Vertex import *
from PyEFVLib.geometry.OuterFace import *
from PyEFVLib.geometry.GridData import *
from PyEFVLib.geometry.InnerFace import *
from PyEFVLib.geometry.Point import *
from PyEFVLib.geometry.Shape import *
from PyEFVLib.geometry.Region import *
from PyEFVLib.geometry.Facet import *
from PyEFVLib.geometry.MSHReader import *
from PyEFVLib.geometry.XDMFReader import *
from PyEFVLib.geometry.Boundary import *
from PyEFVLib.geometry.Element import *
from PyEFVLib.geometry.Grid import *
from PyEFVLib.simulation.BoundaryConditions import *
from PyEFVLib.simulation.CgnsSaver import *
from PyEFVLib.simulation.CsvSaver import *
from PyEFVLib.simulation.VtuSaver import *
from PyEFVLib.simulation.VtmSaver import *
from PyEFVLib.simulation.ProblemData import *
# from PyEFVLib.simulation.LinearSystem import *

INPUT_EXTENSIONS = ["msh"]
READERS_DICTIONARY = {
	"msh": MSHReader
}

def read(filePath):
	extension = filePath.split('.')[-1]

	if extension not in INPUT_EXTENSIONS:
		raise Exception("File extension not supported yet! Input your extension sugestion at https://github.com/GustavoExel/PyEFVLib")
	else:
		return Grid( READERS_DICTIONARY[ extension ]( filePath ).getData() )