import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reclmis_slots as S
from para import para_build
S.build = para_build          # reuse the slot runner: same loader, padding, Dice, paired bootstrap
S.main()
