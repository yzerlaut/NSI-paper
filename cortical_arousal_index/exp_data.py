import sys, pathlib, os, json
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import skew, pearsonr, ttest_rel, wilcoxon, ttest_1samp, linregress
from scipy.optimize import minimize
# specific modules
# sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data_analysis.IO.load_data import load_file
from data_analysis.freq_analysis.fourier_for_real import time_to_freq, FT
from data_analysis.processing.signanalysis import gaussian_smoothing,\
    autocorrel
from data_analysis.processing.filters import butter_lowpass_filter, butter_bandpass_filter, butter_highpass_filter
from graphs.my_graph import *
from graphs.plot_export import put_list_of_figs_to_multipage_pdf
from matplotlib.cm import viridis, copper, plasma, gray, binary
from scipy.integrate import cumtrapz

curdir=os.path.abspath(__file__).replace(os.path.basename(__file__),'')
datadir= home+'work'+os.path.sep+'sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'+os.path.sep

###############################################################
##          LOAD DATASETS #####################################
###############################################################

def get_one_dataset(directory, info='', include_only_chosen=True):
    DATASET = []
    cells = os.listdir(directory)
    for cell in cells:
        files = os.listdir(directory+os.path.sep+cell)
        FILES = []
        for f in files:
            if f.endswith('.abf'):
                with open(directory+os.path.sep+cell+os.path.sep+\
                          f.replace('abf', 'json')) as ff: props = json.load(ff)
                if props['t1']!='0' or not include_only_chosen:
                    FILES.append(f)
        if len(FILES)>0:
            DATASET.append({'cell':cell, 'info':info,
                            'folder':directory+os.path.sep+cell+os.path.sep,
                            'files':[directory+os.path.sep+cell+os.path.sep+f \
                                 for f in FILES]})
    return DATASET

def get_full_dataset(args, include_only_chosen=True):
    return get_one_dataset(datadir+'data'+os.path.sep+args.dataset,
                           info='Wild_Type', include_only_chosen=include_only_chosen)

def show_dataset(directory):
    import pprint
    pprint.pprint(get_full_dataset(args, include_only_chosen=False))
    DATASET = get_full_dataset(args, include_only_chosen=True)
    pprint.pprint(DATASET)
    TIME = np.array([0 for i in range(len(DATASET))])
    for i, cell in enumerate(DATASET):
        icounter = 0
        for f in cell['files']:
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            TIME[i] += args.sliding*len(t_muV)/60.
    print('over ', TIME)
    print('recording time: ', round(TIME.mean(),1), '+/-',\
          round(TIME.std(), 1))

###############################################################
##          LOAD DATAFILES (ABF format)                      ##
###############################################################

def load_data(fn, args,
              chosen_window_only=True):

    with open(fn.replace('abf', 'json')) as f: props = json.load(f)
    
    if chosen_window_only:
        t0, t1 = float(props['t0']), float(props['t1'])
    else:
        t0, t1 = 0, np.inf
        
    raw_data = load_file(fn, zoom=[t0, t1])
    
    # building a dictionary
    if len(fn.split('Wild_Typ'))>1:
        Vm_index = 1
    else:
        Vm_index = 0
        
    data = {'t':raw_data[0]-raw_data[0][0],
            'Vm':raw_data[1][1],
            'Extra':raw_data[1][0],
            'name':fn.split(os.path.sep)[-1],
            'filename':fn}
    data['dt'] = data['t'][1]-data['t'][0]

    if 'offset' in props:
        data['Vm'] += float(props['offset'])
    
    isubsampling = int(args.subsampling_period/data['dt'])
    
    data['sbsmpl_Vm'] = data['Vm'][::isubsampling]
    data['sbsmpl_t'] = data['t'][::isubsampling]
    data['sbsmpl_dt'] = data['sbsmpl_t'][1]-data['sbsmpl_t'][0]
    
    # low freq analysis
    if compute_low_freq_and_muV:
        print('--------> running low freq analysis')
        get_low_freq_power(data,\
                            freq_window=args.freq_window,\
                            shift=args.sliding,\
                            smoothing_for_freq=args.smoothing_for_freq,
                            debug=args.debug)
        print('--------> running muV level analysis')
        get_muV(data,\
                window=args.window,\
                freq_window=args.freq_window,\
                shift=args.sliding,\
                debug=args.debug)
        discard_light(data)
        np.save(data['filename'].replace('.abf', '_low_freq_and_muV.npy'),\
                [data['t_muV'][data['t_nolight']],
                 data['power_low_freq'][data['t_nolight']],
                 data['smooth_power_low_freq'][data['t_nolight']],
                 data['muV'][data['t_nolight']]])
    else:
        data['t_muV'], data['power_low_freq'],\
            data['smooth_power_low_freq'], data['muV'] = np.load(\
                data['filename'].replace('.abf', '_low_freq_and_muV.npy'))
        
    return data

###############################################################
##          EXTRACT TIME_VARYING LOW FREQUENCY CONTENT       ##
###############################################################
from data_analysis.freq_analysis.fourier_for_real import time_to_freq, FT
from data_analysis.processing.signanalysis import gaussian_smoothing

def get_low_freq_power(data,
                        freq_window=1., shift=.05,
                        Fmax=10., Fmin=2.,
                        spike_threshold=-45.,
                        smoothing_for_freq=0.1,
                        debug=True):
    if debug:
        t_key, Vm_key = 'sbsmpl_t', 'sbsmpl_Vm'
    else:
        t_key, Vm_key = 't', 'Vm'
    t, Vm = data[t_key].copy(), data[Vm_key].copy()

    # removing spikes
    Vm[Vm>=spike_threshold] = spike_threshold
    
    TIME, LOW_FREQS = [], []
    middle = freq_window/2.
    while middle<t[-1]-freq_window/2.:
        cond = (t>middle-freq_window/2.) & (t<middle+freq_window/2.)
        tt = t[cond]
        freqs = time_to_freq(len(tt), tt[1]-tt[0])
        TF = FT(Vm[cond], len(tt), tt[1]-tt[0])
        TIME.append(middle)
        freq_cond = (freqs>Fmin) & (freqs<Fmax)
        LOW_FREQS.append(np.array(np.abs(TF)**2)[freq_cond].max())
        middle+=shift
        
    data['t_low_freq'] = np.array(TIME)
    data['power_low_freq'] = np.array(LOW_FREQS)
    data['smooth_power_low_freq'] = gaussian_smoothing(\
                             data['power_low_freq'],\
                             smoothing_for_freq/shift)
    
def get_freq_threshold(args):
    try:
        percentiles = np.load(curdir+'data'+os.path.sep+args.dataset+\
                                 '_freq_percentiles.npy')
        freq_threshold = percentiles[\
                        int(100.*args.percentile_for_freq_threshold)]
        return freq_threshold
    except FileNotFoundError:
        print('need to compute the threshold')
        return 1.

def compute_low_freq_and_muV(args):
    
    DATASET = get_full_dataset(args)

    # plot full dataset
    for i, cell in enumerate(DATASET):
        for fn in cell['files']:
            print(fn)
            data = load_data(fn, args,
                             compute_low_freq_and_muV=True,
                             compute_spikes=True)
    
def plot_low_freq_and_muV(args, CELLS=[0,9,21]):

    DATASET = get_full_dataset(args)

    fig1, ax1 = plt.subplots(1, figsize=(4,3))
    plt.subplots_adjust(left=.3, bottom=.3)
    fig2, ax2 = plt.subplots(1, figsize=(4,3))
    plt.subplots_adjust(left=.3, bottom=.3)

    freq_threshold = get_freq_threshold(args)


    # need to classify by mean muV first for color
    POW_LOW_FREQS = [np.empty(0) for i in range(len(DATASET))]
    FULL_POW_LOW_FREQS = np.empty(0)
    MUV = [np.empty(0) for i in range(len(DATASET))]
    FULL_MUV = np.empty(0)
    for i, cell in enumerate(DATASET):
        icounter = 0
        for f in cell['files']:
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            POW_LOW_FREQS[i] = np.concatenate([POW_LOW_FREQS[i], pow_lf])
            POW_LOW_FREQS[i][POW_LOW_FREQS[i]<args.low_bound_for_freq] = \
                                    args.low_bound_for_freq
            cond1 = (muV>args.muV_min) & (muV<=args.muV_max)
            cond = cond1 & (pow_lf<freq_threshold)
            MUV[i] = np.concatenate([MUV[i], muV[cond]])
            icounter += len(muV[cond1])
    means = np.array([np.mean(m) for m in MUV])
    COLOR = np.argsort(means)
    imax = np.argmax(means)
    print(COLOR[imax])
    print(means[COLOR[0]], means[COLOR[-1]])
    # end sorting by muV level
    
    POW_LOW_FREQS = [np.empty(0) for i in range(len(DATASET))]
    FULL_POW_LOW_FREQS = np.empty(0)
    MUV = [np.empty(0) for i in range(len(DATASET))]
    FULL_MUV = np.empty(0)
    for i, k in enumerate(COLOR):
        cell = DATASET[k]

        icounter = 0
        for f in cell['files']:
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            POW_LOW_FREQS[i] = np.concatenate([POW_LOW_FREQS[i], pow_lf])
            POW_LOW_FREQS[i][POW_LOW_FREQS[i]<args.low_bound_for_freq] = \
                                    args.low_bound_for_freq
            cond1 = (muV>args.muV_min) & (muV<=args.muV_max)
            cond = cond1 & (pow_lf<freq_threshold)
            MUV[i] = np.concatenate([MUV[i], muV[cond]])
            icounter += len(muV[cond1])

        # plotting single-cell low freq 
        hist,be=np.histogram(np.log(POW_LOW_FREQS[i])/np.log(10),
                             bins=50)
        ax1.plot(be[1:], hist/icounter,
                 color=viridis(i/len(DATASET)))
        
        # plotting single-cell muV
        hist, be=np.histogram(MUV[i], bins=30)
        n_hist = 100.*hist/icounter
        ax2.bar([-84.+i*5./len(DATASET)], [100.-n_hist.sum()],
                 color=viridis(i/len(DATASET)),
                bottom=0.1, width=5./len(DATASET))
        n_hist[n_hist<0.1] = 0.108
        x, dx = be[1:], be[1]-be[0]
        if x.min()>args.muV_min+2.*dx:
            x = np.concatenate([[args.muV_min+dx, x.min()], x])
            n_hist = np.concatenate([[0.108, 0.108], n_hist])
        if x.max()<args.muV_max-2.*dx:
            x = np.concatenate([x, [x.max()+dx, args.muV_max+dx]])
            n_hist = np.concatenate([n_hist, [0.108, 0.108]])
        if i in CELLS:
            ax2.semilogy(x, n_hist,
                         color=viridis(i/len(DATASET)), lw=3,
                         label='Cell '+str(i))
            print(f)
        else:
            ax2.semilogy(x, n_hist,
                         color=viridis(i/len(DATASET)), lw=.3)
        
        FULL_MUV = np.concatenate([FULL_MUV, MUV[i]])
        FULL_POW_LOW_FREQS = np.concatenate([FULL_POW_LOW_FREQS,\
                                             POW_LOW_FREQS[i]])
        
        
    hist, be= np.histogram(FULL_MUV, bins=50)
    n_hist = 100.*hist/len(FULL_POW_LOW_FREQS)
    # showing ensemble data
    ax2.bar(be[1:], n_hist, width=be[:-1]-be[1:],\
            color='k', alpha=.2, label='pooled data',
            bottom=0.1)
    # ax2.plot(be[1:], n_hist, color='k', alpha=.2, label='pooled data',
    #          lw=3)
    ax2.bar([-82], [100.*(1-len(FULL_MUV)/len(FULL_POW_LOW_FREQS))],
                    width=6, color='k', alpha=.2, bottom=0.1)
    
    hist,be=np.histogram(np.log(FULL_POW_LOW_FREQS)/np.log(10),
                         bins=40, normed=True)

    percentiles = [np.percentile(FULL_POW_LOW_FREQS,i)\
                   for i in range(1, 100)]
    # saving percentiles for threshold
    np.save(curdir+'data'+os.path.sep+args.dataset+\
            '_freq_percentiles.npy', np.array(percentiles))

    # plotting threshold
    # freq_threshold = get_freq_threshold(args)
    # ax1.plot(np.ones(2)*np.log(freq_threshold)/np.log(10),\
    #          plt.gca().get_ylim(), 'k--',
    #          label='threshold: '+\
    #          str(int(100.*args.percentile_for_freq_threshold))+
    #          'th percentile')

    ax1.legend(frameon=False, prop={'size':'xx-small'})
    set_plot(ax1, xlabel='low. freq. power: $P^{max}_{[2,10]Hz}$',\
             ylabel='occurence \n (in '+\
             str(int(1e3*args.freq_window))+'ms windows \n shifted by '+\
             str(int(1e3*args.sliding))+'ms)',
            xticks=[np.floor(np.log(args.low_bound_for_freq)/np.log(10)),\
                     -1, 0, 1],
             xticks_labels=['<'+str(round(args.low_bound_for_freq,2)),
                            '0.1', '1', '10'])

    # showing ensemble data
    # plt.bar(be[1:], hist*plt.gca().get_ylim()[1],
    #         width=be[:-1]-be[1:],\
    #         color='k', alpha=.2, label='ensemble data (rescaled)')
    c = plt.axes([.5, .5, .2, .03])
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(viridis(np.linspace(0,1,\
                            len(DATASET))))
    cb = mpl.colorbar.ColorbarBase(c, cmap=cmap,
                                   orientation='horizontal')
    cb.set_label(str(len(DATASET))+' cells')
    cb.set_ticks([])

    ax2.legend(frameon=False, prop={'size':'xx-small'})
    set_plot(ax2, xlabel='$\mu_V$ (mV)',\
             ylabel='occurence \n  (% of rec. time)',
             xticks=[-82, -76, -64, -52],
             xticks_labels=['', '-76', '-64', '-52'],
             yticks=[0.1, 1., 10.],
             yticks_labels=['<0.1', '1', '10'])

    return fig1, fig2


def compute_spikes(args):
    
    DATASET = get_full_dataset(args)

    # plot full dataset
    for i, cell in enumerate(DATASET):
        for fn in cell['files']:
            print(fn)
            data = load_data(fn, args,
                             compute_spikes=True)
    
def plot_spikes(args):

    
    DATASET = get_full_dataset(args)

    POW_LOW_FREQS = [np.empty(0) for i in range(len(DATASET))]
    FULL_POW_LOW_FREQS = np.empty(0)
    MUV = [np.empty(0) for i in range(len(DATASET))]
    FULL_MUV = np.empty(0)
    
    freq_threshold = get_freq_threshold(args)

    MUV = np.linspace(args.muV_min, args.muV_max, args.Npoints)
    ANALYSIS = {}
    ANALYSIS['FR'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['COUNTER'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['SPK_COUNT'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['SPK_TIMES'] = [[[[] \
                       for k in range(len(DATASET[j]['files']))]\
                       for j in range(len(DATASET))]\
                       for i in range(args.Npoints-1)]
    
    for i, cell in enumerate(DATASET):

        firing_rate, icounter = 0., 0
        for k, f in enumerate(cell['files']):
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            tspikes = np.load(\
                     f.replace('.abf', '_spikes.npy')).flatten()
            cond = (muV>args.muV_min) & (muV<=args.muV_max) &\
                   (pow_lf<freq_threshold)
            for muVl, tt in zip(muV[cond], t_muV[cond]):
                imuV = np.argwhere(\
                    (muVl>MUV[:-1]) &\
                    (muVl<=MUV[1:])).flatten()
                if len(imuV)>0:
                    cond = (tspikes>tt-args.sliding/2.) &\
                           (tspikes<tt+args.sliding/2.)
                    ANALYSIS['SPK_COUNT'][imuV[0]][i] +=\
                                    len(tspikes[cond])
                    ANALYSIS['COUNTER'][imuV[0]][i] +=1
                    if len(tspikes[cond])>0:
                        ANALYSIS['SPK_TIMES'][imuV[0]][i][k].append(tt)
                    
        for j in range(args.Npoints-1):
            if ANALYSIS['COUNTER'][j,i]>0:
                ANALYSIS['FR'][j,i] = ANALYSIS['SPK_COUNT'][j,i]/\
                        ANALYSIS['COUNTER'][j,i]/args.sliding

    ################################################################
    ####### DO NOT DELETE, it's to visualize single spikes
    ################################################################
    # # # for j, col in zip([0, 4, 9], [Blue, Green, Orange]):
    # for j, col in zip([4], [Blue]):
    #     for i, cell in enumerate(DATASET):
    #         for k, fn in enumerate(cell['files']):
    #             print(fn)
    #             data = load_data(fn, args)
    #             for l, tt in enumerate(\
    #                         ANALYSIS['SPK_TIMES'][j][i][k]):
    #                 print(j, i, k, l, tt)
    #                 fig, ax = plt.subplots(1, figsize=(3,2.5))
    #                 cond = (data['t']>tt-0.2) & (data['t']<tt+0.2)
    #                 ax.plot(data['t'][cond][::3]-data['t'][cond][0],
    #                         data['Vm'][cond][::3],\
    #                         color=col)
    #                 set_plot(ax, xlabel='$\mu_V$ (mV)',
    #                          ylabel='$V_m$ (mV)')
    #                 show()
    # fig, ax = plt.subplots(1, figsize=(2,2.5))
    # for j, i, k, l, col, shift in zip(
    #         [0, 4, 9],
    #         [3,4,2],
    #         [1,1,1],
    #         [3,32,1],
    #         [Blue, Green, Orange],
    #         [0.05, -0.08, 0]):
    #     data = load_data(DATASET[i]['files'][k], args)
    #     tt = ANALYSIS['SPK_TIMES'][j][i][k][l]
    #     print(tt)
    #     cond = (data['t']>tt-0.15+shift) & (data['t']<tt+0.15+shift)\
    #            & (data['Vm']<15)
    #     ax.plot(data['t'][cond]-data['t'][cond][0],
    #             data['Vm'][cond],\
    #             color=col, lw=1)
    # x0, x1 = ax.get_xlim() 
    # ax.plot([x0, x0+0.04], [-80, -80], '-', lw=2, color='gray')
    # ax.annotate('40ms', (x0, -80))
    # ax.plot([x0, x0], [-80, -80+20], '-', lw=2, color='gray')
    # ax.annotate('20mV', (x0, -60))
    # ax.plot(ax.get_xlim(), [-80, -80], 'k:', lw=.5)
    # set_plot(ax, [], xticks=[], yticks=[])
    
    fig1, ax1 = plt.subplots(1, figsize=(3.3, 2.7))
    plt.subplots_adjust(left=.3, bottom=.3)
    m_fr, s_fr= [np.zeros(args.Npoints-1) for i in range(2)]
    for j in range(args.Npoints-1):
        weight_array = ANALYSIS['COUNTER'][j,:]/\
                       np.mean(ANALYSIS['COUNTER'][j,:])
        m_fr[j] = np.mean(ANALYSIS['FR'][j,:]*weight_array)
        s_fr[j]=np.sqrt(np.mean((ANALYSIS['FR'][j,:]-m_fr[j])**2\
                                *weight_array))
        m_fr[j] += args.fr_low_bound
        
    fig2, ax2 = plt.subplots(1, figsize=(3.3, 2.7))
    plt.subplots_adjust(left=.3, bottom=.3)
    for k in range(len(DATASET)):
        ax2.plot(.5*(MUV[1:]+MUV[:-1]),
                 np.log(ANALYSIS['FR'][:,k]+args.fr_low_bound)/np.log(10), 'o',
                 color=viridis(k/ANALYSIS['FR'].shape[0]), ms=4)
        
    x = .5*(MUV[1:]+MUV[:-1])
    
    FULL_MUV, FULL_FR = [], []
    for j in range(args.Npoints-1):
        for k in range(len(DATASET)):
            FULL_FR += [ANALYSIS['FR'][j,k]+args.fr_low_bound for l in range(int(ANALYSIS['COUNTER'][j,k]))]
            FULL_MUV += [x[j] for l in range(int(ANALYSIS['COUNTER'][j,k]))]
    stat = linregress(FULL_MUV, np.log(FULL_FR)/np.log(10))
    ax1.annotate("c=%1.1f \np=%.0e" % (stat.rvalue, stat.pvalue), (-60, 0.05), color='r')

    pol = np.polyfit(x, np.log(m_fr+args.fr_low_bound)/np.log(10), 1)
    ax2.plot(x, np.polyval(pol, x), 'r--')
    ax1.plot(x, 10**np.polyval(pol, x), 'r--')
            
    ax1.plot(x, m_fr, 'k-', lw=3, label='1')
    minus = m_fr-s_fr
    minus[minus<args.fr_low_bound] = args.fr_low_bound
    ax1.fill_between(x,minus,m_fr+s_fr, color='k', alpha=.2, lw=0)
    ax1.annotate('n='+str(len(DATASET))+' cells', (-75, 0.02))
    
        
    ax1.set_yscale('log')
    set_plot(ax1, xlabel='$\mu_V$ (mV)',
             ylabel='$\\nu_e$ (Hz)',
             yticks=[0.01, 0.1, 1., 10.],
             yticks_labels=['<0.01', '0.1', '1', '10'],
             ylim=[args.fr_low_bound, 17.],
             xticks=[-76,-64,-52])

    set_plot(ax2, xlabel='$\mu_V$ (mV)',
             ylabel='log'+r'$_{10}(\nu_e)$',
             yticks=[-2, -1, 0., 1., 2.],
             yticks_labels=['<-2', '-1', '0', '1', '2'],
             xticks=[-76,-64,-52])

    return fig2, fig1

def get_high_freq_LFP_power(data,
                            window=0.3,
                            Fmax=100., Fmin=20.,
                            debug=True):
    if debug:
        t_key, LFP_key = 'sbsmpl_t', 'sbsmpl_LFP'
    else:
        t_key, LFP_key = 't', 'LFP'
        
    t, LFP = data[t_key].copy(), data[LFP_key].copy()

    HIGH_FREQS = []
    for t0 in data['t_muV']:
        cond = (t>t0-window/2.) & (t<t0+window/2.)
        tt = t[cond]
        freqs = time_to_freq(len(tt), tt[1]-tt[0])
        TF = FT(LFP[cond], len(tt), tt[1]-tt[0])
        freq_cond = (freqs>Fmin) & (freqs<Fmax)
        HIGH_FREQS.append(np.array(np.abs(TF)**2)[freq_cond].max())
        
    data['power_high_freq_LFP'] = np.array(HIGH_FREQS)
    
def compute_high_freq_LFP(args):
    
    DATASET = get_full_dataset(args)

    # plot full dataset
    for i, cell in enumerate(DATASET):
        for fn in cell['files']:
            print(fn)
            data = load_data(fn, args,
                             compute_high_freq_LFP=True)
            
def plot_high_freq_LFP(args):

    
    DATASET = get_full_dataset(args)

    POW_HIGH_FREQS = [np.empty(0) for i in range(len(DATASET))]
    FULL_POW_HIGH_FREQS = np.empty(0)
    MUV = [np.empty(0) for i in range(len(DATASET))]
    FULL_MUV = np.empty(0)
    
    freq_threshold = get_freq_threshold(args)

    freq_threshold = get_freq_threshold(args)

    MUV = np.linspace(args.muV_min, args.muV_max, args.Npoints)
    ANALYSIS = {}
    ANALYSIS['HF_0'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['HF'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['COUNTER'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    
    fig1, ax1 = plt.subplots(1, figsize=(3.3, 2.7))
    plt.subplots_adjust(left=.3, bottom=.3)
    
    for i, cell in enumerate(DATASET):

        firing_rate, icounter = 0., 0
        for k, f in enumerate(cell['files']):
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            pow_hf = np.load(\
                    f.replace('.abf', '_high_freq_LFP.npy'))
            cond = (muV>args.muV_min) & (muV<=args.muV_max) &\
                   (pow_lf<freq_threshold)
            
            for muVl, it in zip(muV[cond], np.arange(len(muV))[cond]):
                imuV = np.argwhere(\
                    (muVl>MUV[:-1]) &\
                    (muVl<=MUV[1:])).flatten()
                if len(imuV)>0:
                    ANALYSIS['HF_0'][imuV[0]][i] += pow_hf[it]
                    ANALYSIS['COUNTER'][imuV[0]][i] +=1
                    
        for j in range(args.Npoints-1):
            if ANALYSIS['COUNTER'][j,i]>0:
                ANALYSIS['HF'][j,i] = ANALYSIS['HF_0'][j,i]/\
                                      ANALYSIS['COUNTER'][j,i]

    m_fr, s_fr= [np.zeros(args.Npoints-1) for i in range(2)]
    for j in range(args.Npoints-1):
        weight_array = ANALYSIS['COUNTER'][j,:]/\
                       np.mean(ANALYSIS['COUNTER'][j,:])
        m_fr[j] = np.mean(ANALYSIS['HF'][j,:]*weight_array)
        s_fr[j]=np.sqrt(np.mean((ANALYSIS['HF'][j,:]-m_fr[i])**2\
                                *weight_array))

    ax1.plot(.5*(MUV[1:]+MUV[:-1]), m_fr, 'k-', lw=3, label='1')
    # minus = m_fr-s_fr
    # minus[minus<0.04] = 0.04
    # ax1.fill_between(.5*(MUV[1:]+MUV[:-1]),minus,m_fr+s_fr,
    #                  color='k', alpha=.2, label='n=9 cells', lw=0)
    ax1.legend()
    ax1.set_yscale('log')
    set_plot(ax1, xlabel='$\mu_V$ (mV)',
             # yticks=[0.1, 1., 10.],
             # yticks_labels=['0.1', '1', '10'],
             # ylim=[0.04, 12.],
             xticks=[-76,-64,-52])
    return fig1

def get_mean_MUA(data, args,
            debug=True):
    
    data['MUA'] = gaussian_smoothing(\
        np.abs(butter_bandpass_filter(data['Extra'],
                                      args.MUA_band[0], args.MUA_band[1], 1./data['dt'], order=5)),\
                                        int(args.MUA_smoothing/data['dt']))
    
    mean_MUA = []
    for t0 in data['t_muV']:
        cond = (data['t']>t0-args.window/2.) &\
                     (data['t']<t0+args.window/2.)
        mean_MUA.append(np.mean(data['MUA'][cond]))
        
    data['mean_MUA'] = np.array(mean_MUA)
    
def compute_MUA(args):
    
    DATASET = get_full_dataset(args)

    # plot full dataset
    for i, cell in enumerate(DATASET):
        for fn in cell['files']:
            print(fn)
            data = load_data(fn, args,
                             compute_MUA=True)
def plot_MUA(args):

    DATASET = get_full_dataset(args)

    freq_threshold = get_freq_threshold(args)

    MUV = np.linspace(args.muV_min, args.muV_max, args.Npoints)
    ANALYSIS = {}
    ANALYSIS['MUA_0'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['MUA'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    ANALYSIS['COUNTER'] = np.array([np.zeros(len(DATASET))\
                              for i in range(args.Npoints-1)])
    
    fig1, ax1 = plt.subplots(1, figsize=(3.3, 2.9))
    plt.subplots_adjust(left=.3, bottom=.3)
    fig2, ax2 = plt.subplots(1, figsize=(3.3, 2.9))
    plt.subplots_adjust(left=.3, bottom=.3)
    
    for i, cell in enumerate(DATASET):

        firing_rate, icounter = 0., 0
        for k, f in enumerate(cell['files']):
            t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                        f.replace('.abf', '_low_freq_and_muV.npy'))
            mean_MUA = np.load(\
                    f.replace('.abf', '_mean_MUA.npy'))
            cond = (muV>args.muV_min) & (muV<=args.muV_max) &\
                   (pow_lf<freq_threshold)
            
            for muVl, it in zip(muV[cond], np.arange(len(muV))[cond]):
                imuV = np.argwhere(\
                    (muVl>MUV[:-1]) &\
                    (muVl<=MUV[1:])).flatten()
                if len(imuV)>0:
                    ANALYSIS['MUA_0'][imuV[0]][i] += mean_MUA[it]
                    ANALYSIS['COUNTER'][imuV[0]][i] +=1
                    
        for j in range(args.Npoints-1):
            if ANALYSIS['COUNTER'][j,i]>0:
                ANALYSIS['MUA'][j,i] = 1e3*ANALYSIS['MUA_0'][j,i]/\
                                       ANALYSIS['COUNTER'][j,i]
            else:
                ANALYSIS['MUA'][j,i] = -1

    x = .5*(MUV[1:]+MUV[:-1])
    for k in range(len(DATASET)):
        cond = ANALYSIS['MUA'][:,k]>=0
        ax1.plot(x[cond],
                 ANALYSIS['MUA'][:,k][cond], '-',
                 color=viridis(k/ANALYSIS['MUA'].shape[0]), ms=4)
        # ax1.plot(x[cond],
        #          np.polyval(np.polyfit(x[cond], ANALYSIS['MUA'][:,k][cond], 1),x[cond]),
        #          '--', color=viridis(k/ANALYSIS['MUA'].shape[0]), lw=1)
                
    FULL_MUV, FULL_FR = [], []
    for j in range(args.Npoints-1):
        for k in range(len(DATASET)):
            if ANALYSIS['MUA'][j,k]>0:
                FULL_FR += [ANALYSIS['MUA'][j,k]]
                FULL_MUV += [x[j]]
    stat = linregress(FULL_MUV, np.log(FULL_FR))
    print(stat)
    ax2.annotate("c=%1.1f p=%.0e" % (stat.rvalue, stat.pvalue), (-60, 5), color='r')
    
    m_fr, s_fr= [np.zeros(args.Npoints-1) for i in range(2)]
    for j in range(args.Npoints-1):
        cond = (ANALYSIS['MUA'][j,:]>=0)
        m_fr[j] = np.mean(ANALYSIS['MUA'][j,:][cond])
        s_fr[j]= np.std(ANALYSIS['MUA'][j,:][cond])

    ax2.plot(x, m_fr, 'k-', lw=3, label='1')
    ax2.fill_between(x,m_fr-s_fr,m_fr+s_fr,
                     color='k', alpha=.2, lw=0)
    
    pol = np.polyfit(x, np.log(m_fr)/np.log(10), 1)
    ax2.plot(x, 10**np.polyval(pol, x), 'r--')
    
    set_plot(ax1, xlabel='$\mu_V$ (mV)',
             ylabel=r'$\langle$MUA$\rangle$ ($\mu$V)',
             xticks=[-76,-64,-52])
    ax2.set_yscale('log')
    set_plot(ax2, xlabel='$\mu_V$ (mV)',
             ylabel=r'$\langle$MUA$\rangle$ ($\mu$V)',
             yticks=[4., 6., 8, 10.],
             yticks_labels=['4', '6', '8', '10'],
             xticks=[-76,-64,-52])
    return [fig1, fig2]

###############################################################
##          EXTRACT TIME_VARYING MUV                         ##
###############################################################

def get_muV(data,
            window=0.3, freq_window=0.3, shift=.05,
            spike_threshold=-40.,
            debug=True):
        
    if debug:
        t_key, Vm_key = 'sbsmpl_t', 'sbsmpl_Vm'
    else:
        t_key, Vm_key = 't', 'Vm'
    t, Vm = data[t_key].copy(), data[Vm_key].copy()
    
    # removing spikes
    Vm[Vm>=spike_threshold] = spike_threshold
    
    TIME, MUV = [], []
    
    middle = freq_window/2.
    while middle<t[-1]-freq_window/2.:
        cond = (t>middle-freq_window/2.) & (t<middle+freq_window/2.)
        if len(t[cond])>10:
            MUV.append(Vm[cond].mean())
        else:
            MUV.append(0) # will be dicarded anyway
        TIME.append(middle)
        middle+=shift
        
    data['t_muV'] = np.array(TIME)
    data['muV'] = np.array(MUV)


####################################################################
######## GET MEMBRANE POTENTIAL SIGNATURES #########################
####################################################################

def single_cell_analysis(args, analysis_prefix=''):

    cell = get_full_dataset(args)[args.cell_index]
    ANALYSIS = {}
    
    # to insure non-rhythmic:
    ANALYSIS['freq_threshold'] = get_freq_threshold(args) 
    # discretize muV to classify the episodes
    ANALYSIS['muV'] = np.linspace(args.muV_min, args.muV_max, args.Npoints)
    #counter for episode number:
    ANALYSIS['Ncount'] = np.zeros(args.Npoints-1,dtype=int)
    ANALYSIS['Ncount_Tv'] = np.zeros(args.Npoints-1,dtype=int)

    ANALYSIS['ACF'] = []
    ANALYSIS['VM_DISTRIB'] = [np.empty(0) for i in range(args.Npoints-1)]

    ANALYSIS['spike_threshold'] = args.spike_threshold
    ANALYSIS['window'] = args.window
    ANALYSIS['max_time_for_Tv'] = args.max_time_for_Tv
    
    if args.file_index>0:
        FILES = [cell['files'][args.file_index-1]]
    else:
        FILES = cell['files']
        
    for fn in FILES:
        print(fn)
        data = load_data(fn, args)
        get_Vm_prop_during_non_rhythmic(data, ANALYSIS,
                                        muV_min=ANALYSIS['muV'][0],
                                        muV_max=ANALYSIS['muV'][-1],
                                        debug=args.debug)

    print('saving the data [...]')
    print(cell['folder']+analysis_prefix+'analyzed.npz')
    
    np.savez(cell['folder']+analysis_prefix+'analyzed.npz', **ANALYSIS)

    
def get_Vm_prop_during_non_rhythmic(data, ANALYSIS,
                                    muV_min=-80., muV_max=-50.,
                                    debug=False):

    # loading Vm data
    if debug:
        t_key, Vm_key = 'sbsmpl_t', 'sbsmpl_Vm'
    else:
        t_key, Vm_key = 't', 'Vm'
    t, Vm = data[t_key].copy(), data[Vm_key].copy()
    
    dt = t[1]-t[0]

    # initializing ACF if not done before
    if len(ANALYSIS['ACF'])==0:
        ANALYSIS['ACF'] = [\
        np.zeros(int(ANALYSIS['max_time_for_Tv']/dt)+1)\
                           for i in range(len(ANALYSIS['Ncount']))]
        
    cond = (data['muV']>muV_min) & (data['muV']<=muV_max) & \
           (data['power_low_freq']<=ANALYSIS['freq_threshold']) 
    iwindow = int(ANALYSIS['window']/dt)

    for i, tt in enumerate(data['t_muV'][cond]):
        i0 = int((tt-ANALYSIS['window']/2.)/dt) # translated
        # then the segment of interest is: Vm[i0:i0+iwindow]

        # find the muV level
        imuV = np.argwhere(\
                (data['muV'][cond][i]>ANALYSIS['muV'][:-1]) &\
                (data['muV'][cond][i]<=ANALYSIS['muV'][1:])).flatten()

        if (len(imuV)>0) and (len(Vm[i0:i0+iwindow])>10):
            no_spk_cond = Vm[i0:i0+iwindow]<=ANALYSIS['spike_threshold']
            ANALYSIS['Ncount'][imuV[0]] += 1
            ANALYSIS['VM_DISTRIB'][imuV[0]] = np.concatenate(\
                   [ANALYSIS['VM_DISTRIB'][imuV[0]],\
                    Vm[i0:i0+iwindow][no_spk_cond]])
            # compute autocorrelation
            acf, shift = autocorrel(Vm[i0:i0+iwindow],\
                           ANALYSIS['max_time_for_Tv'], dt)
            if len(acf)==len(ANALYSIS['ACF'][imuV[0]]):
                ANALYSIS['ACF'][imuV[0]] += acf # add autocorrelation
                ANALYSIS['Ncount_Tv'][imuV[0]] += 1

    ANALYSIS['shift'] = np.arange(int(ANALYSIS['max_time_for_Tv']/dt)+1)*dt
    
def plot_cell(args):

    cell = get_full_dataset(args)[args.cell_index]
    print('loading the data [...]')
    print(cell['folder'])
    ANALYSIS = dict(np.load(cell['folder']+'analyzed.npz'))

    ANALYSIS['sV'] = np.array([np.std(vv) if (ANALYSIS['Ncount'][l]>args.Ncount_min) & (len(vv)>1)\
                               else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
    ANALYSIS['gV'] = np.array([skew(vv) if ANALYSIS['Ncount'][l]>args.Ncount_min \
                               else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
    my_map = get_linear_colormap(Blue, Orange)
    
    fig1, ax1 = plt.subplots(1, figsize=(6, 2.5))
    plt.subplots_adjust(bottom=.3, left=.4)
    fig2, ax2 = plt.subplots(1, figsize=(3, 2.5))
    plt.subplots_adjust(bottom=.3, left=.4)
    for i, vv in enumerate(ANALYSIS['VM_DISTRIB']):
        if (ANALYSIS['Ncount'][i]>args.Ncount_min):
            hist, be = np.histogram(vv, bins=25, normed=True)
            ax1.plot(.5*(be[1:]+be[:-1]), hist,\
                       color=my_map(i/len(ANALYSIS['ACF'])))
    for i in range(len(ANALYSIS['ACF']))[::-1]:
        if (ANALYSIS['Ncount'][i]>args.Ncount_min):
            ax2.plot(1e3*ANALYSIS['shift'],\
                 ANALYSIS['ACF'][i][:len(ANALYSIS['shift'])]/ANALYSIS['Ncount_Tv'][i],\
                   color=my_map(i/len(ANALYSIS['ACF'])))
            
    set_plot(ax1, xlabel='$V_m$ (mV)', ylabel='n. occurence', yticks=[])
    set_plot(ax2, xlabel='$\\tau_V$ (ms)', ylabel='norm. ACF',
             xlim=[0,75], xticks=[0,30,60])

    muV = .5*(ANALYSIS['muV'][1:]+ANALYSIS['muV'][:-1])

    # estimating Tv
    Tv = 0.*muV
    for i in range(len(ANALYSIS['ACF'])):
        if (ANALYSIS['Ncount'][i]>args.Ncount_min):
            Tv[i] = cumtrapz(
                np.array(ANALYSIS['ACF'][i][:len(ANALYSIS['shift'])]/ANALYSIS['Ncount'][i]),\
                np.array(ANALYSIS['shift']))[-1]

    cond = (ANALYSIS['Ncount']>args.Ncount_min)
    fig3, AX = plt.subplots(1, 4, figsize=(11, 2.5))
    plt.subplots_adjust(wspace=.5, bottom=.25)
    AX[0].plot(muV[cond], np.array(ANALYSIS['sV'])[cond], 'ko-', lw=1)
    set_plot(AX[0], xlabel='$\mu_V$ (mV)', ylabel='$\sigma_V$ (mV)',
             num_yticks=3, num_xticks=4)
    AX[1].plot(muV[cond], np.array(ANALYSIS['gV'])[cond], 'ko-', lw=1)
    set_plot(AX[1], xlabel='$\mu_V$ (mV)', ylabel='$\gamma_V$',
             num_yticks=3, num_xticks=4)
    AX[2].plot(muV[cond], 1e3*Tv[cond], 'ko-', lw=1)
    set_plot(AX[2], xlabel='$\mu_V$ (mV)', ylabel='$\\tau_V$ (ms)',
             num_yticks=3, num_xticks=4)
    AX[3].plot(muV, np.array(ANALYSIS['Ncount']), 'ro-', lw=1,
               label='removed', ms=3)
    AX[3].plot(muV[cond], np.array(ANALYSIS['Ncount'])[cond], 'ko-', lw=1)
    AX[3].legend(frameon=False, prop={'size':'xx-small'})
    set_plot(AX[3], xlabel='$\mu_V$ (mV)', ylabel='Ncount',
             num_yticks=3, num_xticks=4)
    
    c = plt.axes([.5, .5, .1, .06])
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(my_map(np.linspace(0,1,20)))
    cb = mpl.colorbar.ColorbarBase(c, cmap=cmap,
                                   orientation='horizontal')
    cb.set_label('$\mu_V$ (mV)')
    cb.set_ticks([np.linspace(0,1,len(muV))[cond].min(),
                  np.linspace(0,1,len(muV))[cond].max()])
    cb.set_ticklabels([str(round(muV[cond].min())),
                       str(round(muV[cond].max()))])
    
    return fig1, fig2, fig3

##########################################################
## Dataset analysis #######################################
##########################################################

def dataset_analysis(args):

    # start with the low freq and muV
    compute_low_freq_and_muV(args)
    # just to compute the threhosld
    plot_low_freq_and_muV(args)

    # then loop over cells for spikes-freqs-analysis
    for args.cell_index in range(len(get_full_dataset(args))):
        single_cell_analysis(args)
        

def final_plot(args):
    
    CELLS = get_full_dataset(args)
    
    fig1, AX = plt.subplots(1, 3, figsize=(9, 2.5))
    plt.subplots_adjust(wspace=.5, bottom=.25)
    fig2, AX2 = plt.subplots(1, 3, figsize=(9, 2.5))
    plt.subplots_adjust(wspace=.5, bottom=.25)

    muV_Tv, Tv = np.empty(0), np.empty(0)
    muV_gV, gV = np.empty(0), np.empty(0)
    muV_sV, sV = np.empty(0), np.empty(0)

    mm = []
    
    for i, cell in enumerate(get_full_dataset(args)):
        
        cell = CELLS[i]
        print(cell['folder'])
        print('analysis of cell', i, ' [...]')
        ANALYSIS = dict(np.load(cell['folder']+'analyzed.npz'))
        ANALYSIS['Ncount'] = np.array(ANALYSIS['Ncount'])

        ANALYSIS['sV'] = np.array([np.std(vv) if ANALYSIS['Ncount'][l]>args.Ncount_min \
                                   else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
        ANALYSIS['gV'] = np.array([skew(vv) if ANALYSIS['Ncount'][l]>args.Ncount_min \
                                   else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
        muV = .5*(ANALYSIS['muV'][1:]+ANALYSIS['muV'][:-1])
        # estimating Tv
        ANALYSIS['Tv'] = 0.*muV
        for j in range(len(ANALYSIS['ACF'])):
            if ANALYSIS['Ncount'][j]>args.Ncount_min:
                ANALYSIS['Tv'][j] = 1e3*cumtrapz(
                    np.array(ANALYSIS['ACF'][j][:len(ANALYSIS['shift'])]/ANALYSIS['Ncount'][j]),\
                    np.array(ANALYSIS['shift']))[-1]

        cond = ANALYSIS['Ncount']>args.Ncount_min
        for k, key in enumerate(['sV', 'gV', 'Tv']):
            AX[k].plot(muV[cond], np.array(ANALYSIS[key])[cond], 'o', color=viridis(1.0*i/len(CELLS)), ms=4)

        cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['Tv']>0) & (muV<=args.high_muV)
        muV_Tv, Tv = np.concatenate([muV_Tv, muV[cond]]), np.concatenate([Tv, ANALYSIS['Tv'][cond]])
        cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['sV']>0) & (muV<=args.high_muV)
        muV_sV, sV = np.concatenate([muV_sV, muV[cond]]), np.concatenate([sV, ANALYSIS['sV'][cond]])
        cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['sV']>0) & (muV<=args.high_muV)
        muV_gV, gV = np.concatenate([muV_gV, muV[cond]]), np.concatenate([gV, ANALYSIS['gV'][cond]])

        mm.append(muV[cond].max()-muV[cond].min())
    # for x, y, ax1, ax2 in zip([muV_sV, muV_gV, muV_Tv], [sV, gV, Tv], AX, AX2):
    for x, y, ax1, ax2 in zip([muV_sV], [sV], [AX[0]], [AX2[0]]):
        x, y = np.array(x), np.array(y)
        i0 = np.digitize(x, muV)
        mean, std = np.zeros(len(muV)), np.zeros(len(muV))
        for ii in np.unique(i0):
            cond = (i0==ii)
            if len(x[cond])>0:
                mean[ii-1], std[ii-1] = y[i0==ii].mean(), y[i0==ii].std()
        ax2.plot(muV[std>0], mean[std>0], 'k-', lw=3)
        ax2.fill_between(muV[std>0], mean[std>0]-std[std>0], mean[std>0]+std[std>0],
                         color='k', alpha=.2, lw=0)
        pol = np.polyfit(x[x<args.inter_muV], y[x<args.inter_muV], 1)
        stat = linregress(x[x<args.inter_muV], y[x<args.inter_muV])
        for ax in [ax1, ax2]:
            ax.plot([x.min(),args.inter_muV], np.polyval(pol,[x.min(),args.inter_muV]), 'r--', lw=1)
            ax.annotate("c=%1.1f \np=%.0e" % (stat.rvalue, stat.pvalue), (-75, ax.get_ylim()[1]), color='r')
        pol = np.polyfit(x[x>args.inter_muV], y[x>args.inter_muV], 1)
        stat = linregress(x[x>args.inter_muV], y[x>args.inter_muV])
        for ax in [ax1, ax2]:
            ax.plot([args.high_muV,args.inter_muV], np.polyval(pol,[args.high_muV,args.inter_muV]), 'r--', lw=1)
            ax.annotate("c=%1.1f \np=%.0e" % (stat.rvalue, stat.pvalue), (-55, ax.get_ylim()[1]), color='r')
    
    for x, y, ax1, ax2 in zip([muV_gV, muV_Tv], [gV, Tv], [AX[1], AX[2]], [AX2[1], AX2[2]]):
        x, y = np.array(x), np.array(y)
        pol = np.polyfit(x, y, 1)
        stat = linregress(x, y)
        for ax in [ax1, ax2]:
            ax.plot([x.min(),args.high_muV], np.polyval(pol,[x.min(),args.high_muV]), 'r--', lw=1)
            ax.annotate("c=%1.1f \np=%.0e" % (stat.rvalue, stat.pvalue), (-60, ax.get_ylim()[1]), color='r')
        i0 = np.digitize(x, muV)
        for ii in np.unique(i0):
            cond = (i0==ii)
            if len(x[cond])>0:
                mean[ii-1], std[ii-1] = y[i0==ii].mean(), y[i0==ii].std()
        ax2.plot(muV[std>0], mean[std>0], 'k-', lw=3)
        ax2.fill_between(muV[std>0], mean[std>0]-std[std>0], mean[std>0]+std[std>0],
                         color='k', alpha=.2, lw=0)

    for ax in [AX[0], AX2[0]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\sigma_V$ (mV)',
                 num_yticks=3, xticks=[-76,-64,-52])
    for ax in [AX[1], AX2[1]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\gamma_V$',
             num_yticks=3, xticks=[-76,-64,-52])
    for ax in [AX[2], AX2[2]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\\tau_V$ (ms)',
             yticks=[10,20,30], xticks=[-76,-64,-52])

    mm = np.array(mm)
    print(mm)
    print('n=', len(mm[mm>20]), ' display more than 20mV variations')
    print('for remaining, they display', round(mm[mm<21].mean(),1),
          '+/-', round(mm[mm<21].std(),1), 'mV range')
    return fig1, fig2
    
##########################################################
## Varying rhytmicity threshold ##########################
##########################################################

def varying_rhythmicity_threshold(args):

    # assuming that the low freq power and muV classification have already been performed
    for args.percentile_for_freq_threshold in [0.2, 0.3, 0.4, 0.5]:
        for args.cell_index in range(len(get_full_dataset(args))):
            single_cell_analysis(args, analysis_prefix=\
                                 'R_thresh_'+str(args.percentile_for_freq_threshold)+'_')
            
def plot_various_Rthreshold(args):
    
    DATASET = get_full_dataset(args)
    
    fig1, AX = plt.subplots(1, 4, figsize=(9, 2.5))
    plt.subplots_adjust(wspace=.5, bottom=.25)

    c = plt.axes([.5, .5, .2, .06])
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(copper(np.linspace(0,1,4)))
    cb = mpl.colorbar.ColorbarBase(c, cmap=cmap,
                                   orientation='horizontal')
    cb.set_label('fraction of discarded data \n because classified as \"rhythmic\"')
    cb.set_ticks([.1,.35,.65,.9])
    cb.set_ticklabels(['80%', '70%', '60%', '50%'])
    
    MUV = np.linspace(args.muV_min, args.muV_max, args.Npoints)

    # assuming that the low freq power and muV classification have already been performed
    for t, args.percentile_for_freq_threshold in enumerate([0.2, 0.3, 0.4, 0.5]):
        print(args.percentile_for_freq_threshold)
        freq_threshold = get_freq_threshold(args)

        muV_Tv, Tv = np.empty(0), np.empty(0)
        muV_gV, gV = np.empty(0), np.empty(0)
        muV_sV, sV = np.empty(0), np.empty(0)
    
        for args.cell_index, cell in enumerate(get_full_dataset(args)):
            ANALYSIS = dict(np.load(cell['folder']+\
                                    'R_thresh_'+str(args.percentile_for_freq_threshold)+'_analyzed.npz'))
            ANALYSIS['Ncount'] = np.array(ANALYSIS['Ncount'])

            ANALYSIS['sV'] = np.array([np.std(vv) if ANALYSIS['Ncount'][l]>args.Ncount_min \
                                       else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
            ANALYSIS['gV'] = np.array([skew(vv) if ANALYSIS['Ncount'][l]>args.Ncount_min \
                                       else 0 for l, vv in enumerate(ANALYSIS['VM_DISTRIB'])])
            muV = .5*(ANALYSIS['muV'][1:]+ANALYSIS['muV'][:-1])
            # estimating Tv
            ANALYSIS['Tv'] = 0.*muV
            for j in range(len(ANALYSIS['ACF'])):
                if ANALYSIS['Ncount'][j]>args.Ncount_min:
                    ANALYSIS['Tv'][j] = 1e3*cumtrapz(
                        np.array(ANALYSIS['ACF'][j][:len(ANALYSIS['shift'])]/ANALYSIS['Ncount'][j]),\
                        np.array(ANALYSIS['shift']))[-1]

            cond = ANALYSIS['Ncount']>args.Ncount_min

            cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['Tv']>0) & (muV<=args.high_muV)
            muV_Tv, Tv = np.concatenate([muV_Tv, muV[cond]]), np.concatenate([Tv, ANALYSIS['Tv'][cond]])
            cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['sV']>0) & (muV<=args.high_muV)
            muV_sV, sV = np.concatenate([muV_sV, muV[cond]]), np.concatenate([sV, ANALYSIS['sV'][cond]])
            cond = (ANALYSIS['Ncount']>args.Ncount_min) & (ANALYSIS['sV']>0) & (muV<=args.high_muV)
            muV_gV, gV = np.concatenate([muV_gV, muV[cond]]), np.concatenate([gV, ANALYSIS['gV'][cond]])

        for x, y, ax1 in zip([muV_sV], [sV], [AX[1]]):
            x, y = np.array(x), np.array(y)
            i0 = np.digitize(x, muV)
            mean, std = np.zeros(len(muV)), np.zeros(len(muV))
            for ii in np.unique(i0):
                cond = (i0==ii)
                if len(x[cond])>0:
                    mean[ii-1], std[ii-1] = y[i0==ii].mean(), y[i0==ii].std()
            pol = np.polyfit(x[x<args.inter_muV], y[x<args.inter_muV], 1)
            stat = linregress(x[x<args.inter_muV], y[x<args.inter_muV])
            ax1.plot([x.min(),args.inter_muV], np.polyval(pol,[x.min(),args.inter_muV]), '-', lw=1,\
                     color=copper(t/4.))
            ax1.annotate("c=%1.1f, p=%.0e" % (stat.rvalue, stat.pvalue), (-75, ax1.get_ylim()[1]),\
                     color=copper(t/4.))
            pol = np.polyfit(x[x>args.inter_muV], y[x>args.inter_muV], 1)
            stat = linregress(x[x>args.inter_muV], y[x>args.inter_muV])
            ax1.plot([args.high_muV,args.inter_muV], np.polyval(pol,[args.high_muV,args.inter_muV]), '-', lw=1,\
                     color=copper(t/4.))
            ax1.annotate("c=%1.1f, p=%.0e" % (stat.rvalue, stat.pvalue), (-65, ax1.get_ylim()[1]),\
                     color=copper(t/4.))

        for x, y, ax1 in zip([muV_gV, muV_Tv], [gV, Tv], [AX[2], AX[3]]):
            x, y = np.array(x), np.array(y)
            pol = np.polyfit(x, y, 1)
            stat = linregress(x, y)
            ax1.plot([x.min(),args.high_muV], np.polyval(pol,[x.min(),args.high_muV]), '-', lw=1,\
                         color=copper(t/4.))
            ax1.annotate("c=%1.1f p=%.0e" % (stat.rvalue, stat.pvalue), (-60, ax1.get_ylim()[1]),\
                     color=copper(t/4.))
            i0 = np.digitize(x, muV)
            for ii in np.unique(i0):
                cond = (i0==ii)
                if len(x[cond])>0:
                    mean[ii-1], std[ii-1] = y[i0==ii].mean(), y[i0==ii].std()
                    
        ANALYSIS['FR'] = np.array([np.zeros(len(DATASET))\
                                  for i in range(args.Npoints-1)])
        ANALYSIS['COUNTER'] = np.array([np.zeros(len(DATASET))\
                                  for i in range(args.Npoints-1)])
        ANALYSIS['SPK_COUNT'] = np.array([np.zeros(len(DATASET))\
                                  for i in range(args.Npoints-1)])
        ANALYSIS['SPK_TIMES'] = [[[[] \
                           for k in range(len(DATASET[j]['files']))]\
                           for j in range(len(DATASET))]\
                           for i in range(args.Npoints-1)]

        for i, cell in enumerate(DATASET):

            firing_rate, icounter = 0., 0
            for k, f in enumerate(cell['files']):
                t_muV, pow_lf, smooth_pow_lf, muV = np.load(\
                            f.replace('.abf', '_low_freq_and_muV.npy'))
                tspikes = np.load(\
                         f.replace('.abf', '_spikes.npy')).flatten()
                cond = (muV>args.muV_min) & (muV<=args.muV_max) &\
                       (pow_lf<freq_threshold)
                for muVl, tt in zip(muV[cond], t_muV[cond]):
                    imuV = np.argwhere(\
                        (muVl>MUV[:-1]) &\
                        (muVl<=MUV[1:])).flatten()
                    if len(imuV)>0:
                        cond = (tspikes>tt-args.sliding/2.) &\
                               (tspikes<tt+args.sliding/2.)
                        ANALYSIS['SPK_COUNT'][imuV[0]][i] +=\
                                        len(tspikes[cond])
                        ANALYSIS['COUNTER'][imuV[0]][i] +=1
                        if len(tspikes[cond])>0:
                            ANALYSIS['SPK_TIMES'][imuV[0]][i][k].append(tt)

            for j in range(args.Npoints-1):
                if ANALYSIS['COUNTER'][j,i]>0:
                    ANALYSIS['FR'][j,i] = ANALYSIS['SPK_COUNT'][j,i]/\
                            ANALYSIS['COUNTER'][j,i]/args.sliding


        x = .5*(MUV[1:]+MUV[:-1])

        FULL_MUV, FULL_FR = [], []
        for j in range(args.Npoints-1):
            for k in range(len(DATASET)):
                FULL_FR += [ANALYSIS['FR'][j,k]+args.fr_low_bound for l in range(int(ANALYSIS['COUNTER'][j,k]))]
                FULL_MUV += [x[j] for l in range(int(ANALYSIS['COUNTER'][j,k]))]
        stat = linregress(FULL_MUV, np.log(FULL_FR)/np.log(10))
        AX[0].annotate("c=%1.1f p=%.0e" % (stat.rvalue, stat.pvalue), (-60, 0.05),\
                     color=copper(t/4.))

        m_fr, s_fr= [np.zeros(args.Npoints-1) for i in range(2)]
        for j in range(args.Npoints-1):
            weight_array = ANALYSIS['COUNTER'][j,:]/\
                           np.mean(ANALYSIS['COUNTER'][j,:])
            m_fr[j] = np.mean(ANALYSIS['FR'][j,:]*weight_array)
            s_fr[j]=np.sqrt(np.mean((ANALYSIS['FR'][j,:]-m_fr[j])**2\
                                    *weight_array))
            m_fr[j] += args.fr_low_bound
            
        pol = np.polyfit(x, np.log(m_fr+args.fr_low_bound)/np.log(10), 1)
        AX[0].plot(x, 10**np.polyval(pol, x), color=copper(t/4.))

    AX[0].set_yscale('log')
    for ax in [AX[0]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\\nu_e$ (Hz)',
                 yticks=[0.01, 0.1, 1, 10],
                 yticks_labels=['0.01', '0.1', '1', '10'],
                 xticks=[-76,-64,-52])
        
    for ax in [AX[1]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\sigma_V$ (mV)',
                 num_yticks=3, xticks=[-76,-64,-52])
    for ax in [AX[2]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\gamma_V$',
             num_yticks=3, xticks=[-76,-64,-52])
    for ax in [AX[3]]:
        set_plot(ax, xlabel='$\mu_V$ (mV)', ylabel='$\\tau_V$ (ms)',
             num_yticks=3, xticks=[-76,-64,-52])

    return [fig1]
        
##########################################################
## Plotting function #####################################
##########################################################

def visualize_full_trace(data, ax=None, with_spikes=True):
    
    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,3.))

    ax.plot(data['sbsmpl_t'], data['sbsmpl_Vm'], 'k-', lw=.3)

    # if 'ispikes' in data:
    #     for i in data['ispikes']:
    #         cond = (data['t']>data['t'][i]-0.005) &\
    #                (data['t']<data['t'][i]+0.01)
    #         ax.plot(data['t'][cond], data['Vm'][cond], 'k-', lw=.3)

    if 't_nolight' in data:
        for tt, nolight in zip(data['t_muV'], data['t_nolight']):
            if not nolight:
                ax.fill_between([tt-args.sliding/2.,
                                 tt+args.sliding/2.],\
                                -80.*np.ones(2), 0*np.ones(2),
                                color='k', alpha=.02)
        
    ax.plot([0, 5], [-40, -40], lw=4, color='gray')
    ax.annotate('5s', (2, -35))
    set_plot(ax, ylabel='Vm (mV)',
                 xlim=[data['t'][0], data['t'][-1]],
                 ylim=[data['Vm'].min(), data['Vm'].max()])
    return ax

def plot_raw_data(args):
    
    DATASET = get_full_dataset(args, include_only_chosen=False)
    
    args.subsampling_period = 5e-3
    
    FIGS = []
    # plot full dataset
    for i, cell in enumerate(DATASET):
        fig, AX = plt.subplots(len(cell['files']),
                               figsize=(9,3.*len(cell['files'])))
        plt.subplots_adjust(hspace=.6, top=.92, bottom=.12)
        fig.suptitle(cell['info']+' '+cell['cell']+'                                                                     ')
        if len(cell['files'])==1: AX = [AX]
        for ax, fn in zip(AX, cell['files']):
            print(fn)
            data = load_data(fn, args,\
                             chosen_window_only=False,
                             compute_low_freq_and_muV=True,
                             compute_spikes=False)
            print('--------> plotting trace')
            visualize_full_trace(data, ax=ax)
            with open(fn.replace('abf', 'json')) as f:
                props = json.load(f)
            ax.fill_between([float(props['t0']), float(props['t1'])],
                            ax.get_ylim()[0]*np.ones(2),
                            ax.get_ylim()[1]*np.ones(2),
                            color='r', alpha=.1)
            ax.set_title(data['name'])
            data = None
        FIGS.append(fig)
        fig.savefig('data'+os.path.sep+\
                    cell['info']+'_'+cell['cell']+'.pdf')
    put_list_of_figs_to_multipage_pdf(FIGS,
                  pdf_name='data'+os.path.sep+cell['info']+'.pdf',
                  pdf_title='raw DATASET: '+cell['info'])


if __name__=='__main__':
    
    import argparse
    parser=argparse.ArgumentParser(\
                    description='Model parameters',
                    formatter_class=argparse.RawTextHelpFormatter)
    # type of analysis or plotting
    parser.add_argument("--show_dataset", help="",action="store_true")
    parser.add_argument("--plot_raw_data", help="",action="store_true")
    parser.add_argument("--compute_low_freq_and_muV", action="store_true")
    parser.add_argument("--plot_low_freq_and_muV", action="store_true")
    parser.add_argument("--compute_spikes", action="store_true")
    parser.add_argument("--plot_spikes", action="store_true")
    parser.add_argument("--compute_high_freq_LFP", action="store_true")
    parser.add_argument("--plot_high_freq_LFP", action="store_true")
    parser.add_argument("--compute_MUA", action="store_true")
    parser.add_argument("--plot_MUA", action="store_true")
    parser.add_argument("--single_cell", action="store_true")
    parser.add_argument("--plot_cell", help="",action="store_true")
    parser.add_argument("--full", help="full analysis",action="store_true")
    parser.add_argument("--final_plot", help="",action="store_true")
    parser.add_argument("-vrt", "--varying_rhythmicity_threshold", help="",action="store_true")
    parser.add_argument("-pvrt", "--plot_various_Rthreshold", help="",action="store_true")
    # parser.add_argument("-d", "--debug", help="debug", action="store_true")
    parser.add_argument("-d", "--debug", help="debug",
                        default=True, action="store_true")
    # type of dataset
    parser.add_argument('--dataset', type=str, default='Wild_Type')    
    parser.add_argument('--cell_index', help='starts at 0',
                        type=int, default=0)    
    parser.add_argument('--file_index', help='0 means full data',
                        type=int, default=0)    
    # parameters of the analysis
    parser.add_argument('--subsampling_period', type=float,default=5e-4)    
    parser.add_argument('--sliding', type=float, default=0.025)    
    parser.add_argument('--window', type=float, default=0.3)    
    parser.add_argument('--freq_window', type=float, default=0.5)    
    parser.add_argument('--smoothing_for_freq', type=float, default=0.1)    
    parser.add_argument('--percentile_for_freq_threshold',\
                        type=float, default=0.5)
    parser.add_argument('--low_bound_for_freq', type=float, default=1e-2)
    parser.add_argument('--spike_threshold', type=float, default=-40.)
    parser.add_argument('--Ncount_min', type=int, default=200)    
    # muV classification parameters
    parser.add_argument('--Npoints', type=int, default=20)    
    parser.add_argument('--muV_min', type=float, default=-80.)    
    # parser.add_argument('--muV_max', type=float, default=-46.)    
    parser.add_argument('--muV_max', type=float, default=-48.)    
    # parser.add_argument('--Nsmoothing', type=int, default=2)    
    # TauV determination parameters
    parser.add_argument('--max_time_for_Tv', type=float,
                        default=100e-3)    
    # Intervals for statistical differences
    parser.add_argument('--low_muV', type=float, default=-77)
    parser.add_argument('--inter_muV', type=float, default=-66)
    parser.add_argument('--high_muV', type=float, default=-50.)
    # for firing rate lower bound
    parser.add_argument('--fr_low_bound', type=float, default=0.01)
    # parameters of Multi-Unit-Activity (MUA)
    parser.add_argument('--MUA_band', nargs='+', type=float, default=[300., 3000.])
    parser.add_argument('--MUA_smoothing', type=float, default=20e-3)
    # parameters of LFP
    parser.add_argument('--LFP_band', nargs='+', type=float, default=[0.1, 300.])


    args = parser.parse_args()

    FIGS = []
    # dataset
    if args.show_dataset:
        show_dataset(args)
    elif args.plot_raw_data:
        plot_raw_data(args)
    # low freq analysis
    elif args.compute_low_freq_and_muV:
        compute_low_freq_and_muV(args)
    elif args.plot_low_freq_and_muV:
        FIGS = plot_low_freq_and_muV(args)
    # firing analysis
    elif args.compute_spikes:
        compute_spikes(args)
    elif args.plot_spikes:
        FIGS = plot_spikes(args)
    # high freq LFP analysis
    elif args.compute_high_freq_LFP:
        compute_high_freq_LFP(args)
    elif args.plot_high_freq_LFP:
        FIGS = plot_high_freq_LFP(args)
    elif args.compute_MUA:
        compute_MUA(args)
    elif args.plot_MUA:
        FIGS = plot_MUA(args)
    # Vm comodulation analysis
    elif args.single_cell:
        single_cell_analysis(args)
    elif args.plot_cell:
        FIGS = plot_cell(args)
    elif args.final_plot:
        FIGS = final_plot(args)
    # varying rhytmicity threshold
    elif args.varying_rhythmicity_threshold:
        varying_rhythmicity_threshold(args)
    elif args.plot_various_Rthreshold:
        FIGS = plot_various_Rthreshold(args)
    # full dataset analysis
    elif args.full:
        dataset_analysis(args)
    else:
        print('full analysis by default')
        dataset_analysis(args)

    if len(FIGS)>0:
        show()
    for i, fig in enumerate(FIGS):
        fig.savefig('/Users/yzerlaut/Desktop/temp'+str(i)+'.svg')

        
