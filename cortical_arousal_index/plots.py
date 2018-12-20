import sys, pathlib, os, json
import numpy as np
import matplotlib.pylab as plt
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data_analysis.processing.signanalysis import gaussian_smoothing
from graphs.my_graph import *
# COLORS
NSI_color, Vm_color, LFP_color, pLFP_color = Brown, 'k', Grey, Brown
Rhythm_color, Asynch_color = Purple, Kaki



curdir=os.path.abspath(__file__).replace(os.path.basename(__file__),'')
datadir= '../../sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'+os.path.sep
s1 = 'sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'
s2 = 'cortical_arousal_index'+os.path.sep+'cortical_arousal_index'

cell_colormap = get_linear_colormap(Blue, Red)

def compute_relationship_smooth_and_plot(cbins, pop_hist, num_hist, ax,
                                         Nsmooth = 10):
    
    x, y, sy = [], [], []
    for xx, ph, N in zip(cbins, pop_hist, num_hist):
        if len(ph)>1:
            x.append(xx)
            y.append(np.sum(np.array(ph)*np.array(N)/np.sum(N)))
            sy.append(np.sqrt(np.sum((np.array(ph)-y[-1])**2*np.array(N)/np.sum(N))))
    
    xx, yy, ss = np.linspace(np.min(x), np.max(x), int(Nsmooth/2*len(x))), np.zeros(int(Nsmooth/2*len(x))), np.zeros(int(Nsmooth/2*len(x)))
    for i in range(len(xx)):
        i0 = np.argmin((xx[i]-x)**2)
        yy[i], ss[i] = y[i0], sy[i0]
    
    yy, ss = gaussian_smoothing(yy, Nsmooth), gaussian_smoothing(ss, Nsmooth)
    ax.plot(xx, yy, 'k-', lw=1.5)
    ax.fill_between(xx, yy+ss, yy-ss, color='k', alpha=.5, lw=0)

    
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
        ax.annotate('(%i)'%(n+1), (interval[0], ylim[1]), color=Grey, fontsize=FONTSIZE)
        ax.fill_between(interval, np.ones(2)*ylim[0], np.ones(2)*ylim[1], alpha=.1, color='k', lw=0)

    # ax.annotate('$V_m^0$', (tstart+twidth, data['p0_Vm']), color=Vm_color)
    # ax.annotate('0$\mu$V', (tstart+twidth, Vext_loc), color=Vext_color)
    ax.annotate('$V_m$', (-0.03, .15), color=Vm_color, xycoords='axes fraction')
    ax.annotate('$V_{ext}$', (-0.04, 0.65), color=Vext_color, xycoords='axes fraction')
    set_plot(ax, [], xlim=[tstart, tstart+twidth], ylim=ylim)
    
    return fig, ax

def show_raw_data_large_scale(data,
                              Zooms=[],
                              Tbar=5, Vm_bar=30, Vm_bar_loc=(0,0),
                              LFP_bar=0.7, LFP_loc=-5,
                              NSI_bar=10, NSI_bar_loc=(0,10),
                              tstart=0, twidth=np.inf,
                              fig_options={'figsize':(0.8,.1), 'hspace':0.01, 'left':.12, 'bottom':0.01, 'right':.999},
                              subsampling=400, spikes=40):
    tend = tstart+twidth
    fig, AX = figure(axes_extents=[[[1,3]], [[1,2]]], **fig_options)
    
    AX[0][0].plot(Vm_bar_loc[0]*np.ones(2)+tstart, Vm_bar_loc[1]+np.arange(2)*Vm_bar, '-', lw=1, color=Vm_color)
    AX[0][0].annotate(str(Vm_bar)+'mV', (Vm_bar_loc[0]+tstart, Vm_bar_loc[1]), fontsize=FONTSIZE, color=Vm_color)
    AX[0][0].annotate(str(round(LFP_bar,1))+'mV', (Vm_bar_loc[0]+tstart, Vm_bar_loc[1]+Vm_bar), fontsize=FONTSIZE, color=LFP_color)
    AX[0][0].annotate('$V_m$', (-0.04,0.1), fontsize=FONTSIZE, color=Vm_color, xycoords='axes fraction')
    AX[0][0].annotate('$V_{ext}$', (-0.04,0.6), fontsize=FONTSIZE, color=LFP_color, xycoords='axes fraction')
    AX[1][0].plot(NSI_bar_loc[0]*np.ones(2)+tstart, NSI_bar_loc[1]+np.arange(2)*NSI_bar, 'k-', lw=1)
    AX[1][0].annotate(str(NSI_bar)+'$\mu$V', (NSI_bar_loc[0]+tstart, NSI_bar_loc[1]), fontsize=FONTSIZE)#, color=NSI_color)
    AX[1][0].annotate('NSI', (-0.04,0.3), fontsize=FONTSIZE, color=NSI_color, xycoords='axes fraction')
    AX[0][0].plot([tstart,tstart+twidth], data['p0_Vm']*np.ones(2), ':', lw=0.5, color=Vm_color)
    AX[0][0].plot([tstart,tstart+twidth], LFP_loc*np.ones(2), ':', lw=0.5, color=LFP_color)
    
    cond = (data['t']>tstart) & (data['t']<tend)
    AX[0][0].plot(data['t'][cond][::subsampling], data['Vm'][cond][::subsampling], '-', lw=1, color=Vm_color)
    AX[0][0].plot(data['t'][cond][::subsampling],\
                  (data['Extra'][cond][::subsampling]-data['Extra'][cond].mean())/LFP_bar*Vm_bar+LFP_loc,
                  color=LFP_color, lw=1)
    ispk = np.argwhere(cond[1:] & (data['Vm'][:-1]<=spikes) & (data['Vm'][1:]>spikes) & (data['t'][1:]>tstart)).flatten()
    for i in ispk:
        Vspk = data['Vm'][i-10:i+30]
        AX[0][0].plot(data['t'][i-10:i+30][Vspk<spikes], Vspk[Vspk<spikes], '-', lw=1, color=Vm_color)
    # tstart = data['t'][cond][0]
    cond0 = (data['sbsmpl_t']>tstart) & (data['sbsmpl_t']<tend)
    AX[1][0].plot(data['sbsmpl_t'][cond0][data['NSI_validated'][cond0]],\
                  data['NSI'][cond0][data['NSI_validated'][cond0]], 'o', lw=0, ms=1, color=NSI_color)
    y20, y21 = AX[1][0].get_ylim()
    y10, y11 = AX[0][0].get_ylim()
    AX[1][0].annotate(str(Tbar)+'s', (data['t'][cond][0]+Tbar, y20), fontsize=FONTSIZE)
    AX[1][0].plot(data['t'][cond][0]+np.arange(2)*Tbar, y20*np.ones(2), 'k-', lw=1)
    for i, (z1, z2) in enumerate(Zooms):
        AX[0][0].annotate('('+str(i+1)+')', (z1, y11), fontsize=FONTSIZE, color=Grey)
        AX[1][0].fill_between([z1, z2], y20*np.ones(2), y21*np.ones(2), color='k', alpha=.1, lw=0)
        AX[0][0].fill_between([z1, z2], y10*np.ones(2), y11*np.ones(2), color='k', alpha=.1, lw=0)
    AX[1][0].plot([data['t'][cond][0],data['t'][cond][-1]], np.zeros(2), 'k:', lw=0.5)
    AX[1][0].fill_between([data['t'][cond][0],data['t'][cond][-1]], np.zeros(2), np.ones(2)*y20,
                              color=Rhythm_color, alpha=.2, lw=0)
    AX[1][0].fill_between([data['t'][cond][0],data['t'][cond][-1]], np.zeros(2), np.ones(2)*y21,
                              color=Asynch_color, alpha=.2, lw=0)
    set_plot(AX[0][0], [],xlim=[data['t'][cond][0],data['t'][cond][-1]])
    set_plot(AX[1][0], [],xlim=[data['t'][cond][0],data['t'][cond][-1]])
    return fig, AX
