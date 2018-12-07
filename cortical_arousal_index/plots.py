import sys, pathlib, os, json
import numpy as np
import matplotlib.pylab as plt
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from graphs.my_graph import *

curdir=os.path.abspath(__file__).replace(os.path.basename(__file__),'')
datadir= '../../sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'+os.path.sep
s1 = 'sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'
s2 = 'cortical_arousal_index'+os.path.sep+'cortical_arousal_index'

cell_colormap = get_linear_colormap(Blue, Red)


fig, ax = figure()
for i in np.linspace(0, 1, 10):
    ax.plot(np.random.randn(10), np.random.randn(10), 'o', color=cell_colormap(i), ms=3)

show()
