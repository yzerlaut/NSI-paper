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

def show_raw_data_Vm_Vext(data,
                          tstart=12, twidth=168,
                          highlighted_episodes=[],
                          Vext_loc=0., # in mV
                          Vm_scale=10, # in mV
                          Vext_scale=0.1, # in mV
                          Vm_color='k', Vext_color=Grey,
                          bar_loc = None,
                          Tscale=5.,
                          subsampling=10):
    
    fig, ax = figure(axes_extents=[[[5,1]]])
    
    cond = (data['sbsmpl_t']>tstart) & (data['sbsmpl_t']<tstart+twidth)

    ######### Vext plot ##########
    ax.plot(data['sbsmpl_t'][cond][::subsampling],
            data['sbsmpl_Extra'][cond][::subsampling]*Vm_scale/Vext_scale+Vext_loc,
            color=Vext_color, lw=1)
    ax.plot([tstart, tstart+twidth], np.zeros(2), ':', color=Vext_color, lw=1)
    
    ######### Vm plot ##########
    # subthreshold, we subsample
    ax.plot(data['sbsmpl_t'][cond][::subsampling], data['sbsmpl_Vm'][cond][::subsampling], color=Vm_color, lw=1)
    ax.plot([tstart, tstart+twidth], np.ones(2)*data['p0_Vm'], ':', color=Vm_color, lw=1)
    # now spikes without subsampling
    for tspike in data['tspikes'][(data['tspikes']>tstart) & (data['tspikes']<tstart+twidth)]:
        ispike = np.argmin((tspike-data['t'])**2)
        ax.plot(data['t'][ispike-20:ispike+20], data['Vm'][ispike-20:ispike+20], color=Vm_color, lw=1)
    ylim = ax.get_ylim()

    # bar scales with annotations
    if bar_loc is None:
        t0, y0 = tstart+twidth/10., 0.99*ylim[1]
    else:
        t0, y0 = bar_loc
        
    ax.plot([t0,t0-Tscale], np.ones(2)*y0, 'k-', lw=1)
    ax.annotate(str(int(Tscale))+'s', (t0-Tscale/1.3, y0), color=Vm_color, fontsize=FONTSIZE)
    ax.plot([t0,t0], y0-np.arange(2)*Vm_scale, 'k-', lw=1)
    ax.annotate(str(int(Vm_scale))+'mV', (t0+Tscale/10., y0-Vm_scale/3.), color=Vm_color, fontsize=FONTSIZE)
    ax.annotate(str(int(1e3*Vext_scale))+'$\mu$V',(t0+Tscale/10., y0-Vm_scale), color=Vext_color, fontsize=FONTSIZE)

    for n, interval in enumerate(highlighted_episodes):
        ax.annotate('(%i)'%(n+1), (interval[0], ylim[1]), color=Grey)
        ax.fill_between(interval, np.ones(2)*ylim[0], np.ones(2)*ylim[1], alpha=.1, color='k', lw=0)

    # ax.annotate('$V_m^0$', (tstart+twidth, data['p0_Vm']), color=Vm_color)
    # ax.annotate('0$\mu$V', (tstart+twidth, Vext_loc), color=Vext_color)
    set_plot(ax, [], xlim=[tstart, tstart+twidth], ylim=ylim)
    
    return fig, ax
