import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import ariadne_cf as A
from para import para_build
A.build_conditions = lambda df, n, seed=0: ({"matched": dict(zip(df.Image, df.Description))}, set(), "n/a")
A.slots.build = para_build
A.main()
