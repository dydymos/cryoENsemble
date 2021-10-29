import sys
from bioEN_functions import *
from cryoEM_methods import *
from reference_map_ADK import *


nx,ny,nz,VOX_,em_origin = reference_map(sys.argv[1])
