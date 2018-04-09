import sys, pathlib, os, json
import numpy as np
import matplotlib.pylab as plt
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data_analysis.IO.load_data import load_file
from graphs.my_graph import *
from graphs.plot_export import put_list_of_figs_to_multipage_pdf
from matplotlib.cm import viridis, copper, plasma, gray, binary
import multiprocessing as mp
from itertools import product
import functions

curdir=os.path.abspath(__file__).replace(os.path.basename(__file__),'')
datadir= '../../sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'+os.path.sep
s1 = 'sparse_vs_balanced'+os.path.sep+'sparse_vs_balanced'
s2 = 'cortical_arousal_index'+os.path.sep+'cortical_arousal_index'

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
                          f.replace('abf', 'json').replace(s1, s2)) as ff: props = json.load(ff)
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

###############################################################
##          LOAD DATAFILES (ABF format)                      ##
###############################################################

def load_data(fn, args,
              chosen_window_only=True,
              full_processing=False,
              with_Vm_low_freq=False):

    with open(fn.replace(s1, s2).replace('abf', 'json')) as f: props = json.load(f)
    
    if chosen_window_only:
        t0, t1 = np.float(props['t0']), np.float(props['t1'])
    else:
        t0, t1 = 0, np.inf
        
    raw_data = load_file(fn, zoom=[t0, t1])
    
    data = {'t':raw_data[0]-raw_data[0][0],
            'Vm':raw_data[1][1],
            'Extra':raw_data[1][0],
            'name':fn.split(os.path.sep)[-1],
            'filename':fn}
    data['dt'] = data['t'][1]-data['t'][0]

    if 'offset' in props:
        data['Vm'] += float(props['offset'])
    
    isubsampling = int(args.subsampling_period/data['dt'])
    data['sbsmpl_Vm'] = data['Vm'][::isubsampling][:-1]
    data['sbsmpl_Extra'] = data['Extra'][::isubsampling][:-1]
    data['sbsmpl_t'] = data['t'][::isubsampling][:-1]
    data['sbsmpl_dt'] = data['dt']*isubsampling

    if full_processing:
        # compute the pLFP
        functions.preprocess_LFP(data,
                                 freqs=np.linspace(args.f0/args.w0, args.f0*args.w0, args.wavelet_number),
                                 new_dt=args.subsampling_period,
                                 percentile_for_p0=args.percentile_for_p0,
                                 smoothing=args.T0)
        # compute the Network State Index
        functions.compute_Network_State_Index(data,
                                              Tstate=args.Tstate,
                                              Var_criteria=data['p0'], # HERE TAKING NOISE AS CRITERIA !!!
                                              alpha=args.alpha,
                                              T_sliding_mean=args.T_sliding_mean,
                                              with_Vm_low_freq=with_Vm_low_freq)
    
    return data

###############################################################
##          Find the good wavelet packet                     ##
###############################################################

def test_different_wavelets(args):
    
    CENTER_FREQUENCIES = np.logspace(np.log(args.center_wavelet_freq_min)/np.log(10),
                                     np.log(args.center_wavelet_freq_max)/np.log(10),
                                     args.discretization)
    BAND_LENGTH_FACTOR = np.linspace(args.factor_wavelet_freq_min,
                                     args.factor_wavelet_freq_max,
                                     args.discretization)
    

    DATASET = get_full_dataset(args)
    DATA = []
    CROSS_CORRELS = np.zeros((len(CENTER_FREQUENCIES), len(BAND_LENGTH_FACTOR), len(DATASET)))
    
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        CROSS_CORRELS0 = np.zeros((len(CENTER_FREQUENCIES), len(BAND_LENGTH_FACTOR)))
        for icf, ibl in product(range(len(CENTER_FREQUENCIES)),
                                range(len(BAND_LENGTH_FACTOR))):
            cf = CENTER_FREQUENCIES[icf]
            blf = BAND_LENGTH_FACTOR[ibl]
            print('running ', icf, ibl, 'on cell', icell, '[...]')
            functions.preprocess_LFP(DATA[icell],
                                     freqs=np.linspace(cf/blf, cf*blf, args.wavelet_number),
                                     new_dt=args.subsampling_period,
                                     smoothing=0.) # HERE NO SMOOTHING YET !!
            cc = np.abs(np.corrcoef(DATA[icell]['sbsmpl_Vm'].flatten(), DATA[icell]['pLFP'].flatten()))[0,1]
            CROSS_CORRELS0[icf, ibl] = cc
        np.save(DATASET[icell]['files'][0].replace('.abf', '_wavelet_scan.npy'),
                CROSS_CORRELS0)
        print('=================================================')
            
    for icell, cell in enumerate(DATASET):
        print('Cell '+str(icell+1)+' :', cell['files'][0])
        DATA.append(load_data(cell['files'][0], args))
        
        if args.parallelize:
            PROCESSES.append(mp.Process(target=run_func, args=(icell, output)))
        else:
            run_func(icell, 0)

    if args.parallelize:
        # Run processes
        for p in PROCESSES:
            p.start()
        # # Exit the completed processes
        for p in PROCESSES:
            p.join()
    
    for i, cell in enumerate(DATASET):
        CROSS_CORRELS[:, :, i] = np.load(cell['files'][0].replace('.abf', '_wavelet_scan.npy'))
                
    OUTPUT = {'Params':args,
              'CENTER_FREQUENCIES' : CENTER_FREQUENCIES,
              'BAND_LENGTH_FACTOR' : BAND_LENGTH_FACTOR,
              'CROSS_CORRELS' : CROSS_CORRELS}

    print(OUTPUT)
    np.savez(args.datafile_output, **OUTPUT)

def plot_test_different_wavelets(args, colormap=plasma):

    OUTPUT = dict(np.load(args.datafile_input))

    print('frequencies :', OUTPUT['CENTER_FREQUENCIES'])
    print('length :', OUTPUT['BAND_LENGTH_FACTOR'])
    
    pCC = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)

    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              pCC.shape)
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]
    
    print('== OPTIMIZATION RESULTS == ')
    print(f0, w0)
    print(f0/w0, f0*w0)
    
    fig_optimum, ax = figure(figsize=(.3, .16), right=.7, top=.9, bottom=1.2, left=.9)
    p = plt.contourf(OUTPUT['BAND_LENGTH_FACTOR'],
                     OUTPUT['CENTER_FREQUENCIES'],
                     pCC,
                     levels=np.linspace(pCC.min(),pCC.max(),30),
                     cmap=colormap)
    ax.scatter([w0], [f0], color=Brown, facecolor='None', label='($f_{opt}, w_{opt}$)')
    ax.set_yscale('log')
    ax.set_title('wavelet packets in bands: [$f/w$, $f\cdot w$]', fontsize=FONTSIZE)
    set_plot(ax, xlabel=' $w$, width factor\n for freq. band extent',
             ylabel=' $f$, root freq. (Hz)    ',
             yticks=[2, 20, 200], yticks_labels=['2', '20', '200'], xticks=np.arange(1, 5))
    ax.legend(loc=(1.05, .8), prop={'size':'x-small'})
    acb = plt.axes([.71,.35,.02,.4])
    build_bar_legend(np.unique(np.round(np.linspace(pCC.min(),pCC.max(),10),1)),
                     acb, colormap,
                     color_discretization=30,
                     label='cc $V_m$-pLFP \n (n='+str(OUTPUT['CROSS_CORRELS'].shape[-1])+')')
    return [fig_optimum]

###############################################################
##          Find the good wavelet packet                     ##
###############################################################

def test_different_smoothing(args):
    
    T_SMOOTH = np.concatenate([\
                               [0],
                               # np.logspace(np.log(args.smoothing_min)/np.log(10),
                               #             np.log(args.smoothing_max)/np.log(10),
                               #             args.discretization)
                               np.linspace(args.smoothing_min,
                                           args.smoothing_max,
                                           args.discretization)
                               ])

    ## NEED TO GRAB THE OPTIMAL FREQUENCY !
    OUTPUT = dict(np.load(args.datafile_input))
    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              OUTPUT['CROSS_CORRELS'].shape[:2])
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]
    
    DATASET = get_full_dataset(args)
    DATA = []
    CROSS_CORRELS = np.zeros((len(T_SMOOTH), len(DATASET)))
    
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        CROSS_CORRELS0 = np.zeros(len(T_SMOOTH))
        for it in range(len(T_SMOOTH)):
            print('running ', T_SMOOTH[it], 'on cell', icell, '[...]')
            functions.preprocess_LFP(DATA[icell],
                                     freqs=np.linspace(f0/w0, f0*w0, args.wavelet_number),
                                     new_dt = args.subsampling_period,
                                     smoothing=T_SMOOTH[it]) # SMOOTHING !!
            cc = np.abs(np.corrcoef(DATA[icell]['sbsmpl_Vm'].flatten(), DATA[icell]['pLFP'].flatten()))[0,1]
            CROSS_CORRELS0[it] = cc
        np.save(DATASET[icell]['files'][0].replace('.abf', '_wavelet_scan.npy'),
                CROSS_CORRELS0)
        print('=================================================')
            
    for icell, cell in enumerate(DATASET):
        print('Cell '+str(icell+1)+' :', cell['files'][0])
        DATA.append(load_data(cell['files'][0], args))
        
        if args.parallelize:
            PROCESSES.append(mp.Process(target=run_func, args=(icell, output)))
        else:
            run_func(icell, 0)

    if args.parallelize:
        # Run processes
        for p in PROCESSES:
            p.start()
        # # Exit the completed processes
        for p in PROCESSES:
            p.join()
    
    for i, cell in enumerate(DATASET):
        CROSS_CORRELS[:, i] = np.load(cell['files'][0].replace('.abf', '_wavelet_scan.npy'))
                
    OUTPUT = {'Params':args,
              'T_SMOOTH' : T_SMOOTH,
              'CROSS_CORRELS' : CROSS_CORRELS}
    
    np.savez(args.datafile_output, **OUTPUT)
    
def plot_test_different_smoothing(args):

    OUTPUT = dict(np.load(args.datafile_input))

    fig_optimum, [[ax, ax1]] = figure(figsize=(.5, .16),
                                      right=0.85, top=0.9, bottom=1.2, left=.6, wspace=1.8,
                                      axes=(1,2))
    
    i0 = np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1))

    mean_Output = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)
    
    Tsmooth = 1e3*OUTPUT['T_SMOOTH']
    ax.plot(Tsmooth, mean_Output, color='k', lw=2)
    ax.scatter([1e3*OUTPUT['T_SMOOTH'][i0]], [np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)[i0]],
               marker='o', color=Brown, facecolor='None')
    ax.annotate('$T_{opt}$', (Tsmooth[i0]+4, ax.get_ylim()[0]), color=Brown, fontsize=FONTSIZE)
    ax.plot(np.array([Tsmooth[i0], Tsmooth[i0]]),
            [mean_Output[i0], ax.get_ylim()[0]], '--', color=Brown, lw=1)
    
    order = np.argsort(np.mean(OUTPUT['CROSS_CORRELS'], axis=0))
    for i in range(len(order)):
        ax1.plot(Tsmooth, OUTPUT['CROSS_CORRELS'][:,order[i]], color=viridis(i/(len(order)-1)))
    ax1.plot(Tsmooth, mean_Output, '-', color='k', lw=0.5)
    ax1.fill_between(Tsmooth,\
                     mean_Output+np.std(OUTPUT['CROSS_CORRELS'], axis=-1),
                     mean_Output-np.std(OUTPUT['CROSS_CORRELS'], axis=-1),
                     lw=0, color='k', alpha=.2)

    set_plot(ax, xlabel=' $T_{smoothing}$ (ms)',
             ylabel='cc $V_m$-pLFP')
    set_plot(ax1, xlabel=' $T_{smoothing}$ (ms)',
             ylabel='cc $V_m$-pLFP')
    acb = plt.axes([.86,.4,.02,.4])
    cb = build_bar_legend(np.arange(len(order)),
                          acb, viridis,
                          no_ticks=True,
                          label='cell index \n (n='+str(len(order))+'cells)')
    return [fig_optimum]

###############################################################
##          Find the good low-freq criteria (alpha)          ##
###############################################################

def test_different_alpha(args):
    
    ALPHA = np.linspace(args.alpha_min,
                        args.alpha_max,
                        args.discretization)

    DATASET = get_full_dataset(args)
    DATA = []
    VM_LOW_FREQ_POWER1 = np.zeros((len(ALPHA), len(DATASET)))
    VM_LOW_FREQ_POWER_ASYNCH1 = np.zeros((len(ALPHA), len(DATASET)))
    N_LOW_FREQ1, N_ASYNCH1, N_NC1 = [np.zeros((len(ALPHA), len(DATASET))) for i in range(3)]
    
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        VM_LOW_FREQ_POWER0 = np.zeros(len(ALPHA))
        VM_LOW_FREQ_POWER_ASYNCH = np.zeros(len(ALPHA))
        N_LOW_FREQ, N_ASYNCH, N_NC = np.zeros(len(ALPHA)), np.zeros(len(ALPHA)), np.zeros(len(ALPHA))

        for it in range(len(ALPHA)):
            print('running ', ALPHA[it], 'on cell', icell, '[...]')
            # compute the Network State Index
            functions.compute_Network_State_Index(DATA[icell],
                                                  Tstate=args.Tstate,
                                                  Var_criteria=DATA[icell]['p0'],
                                                  alpha=ALPHA[it],
                                                  already_low_freqs_and_mean=True)
            # get Vm low freq power
            cond = DATA[icell]['NSI_validated'] & (DATA[icell]['NSI']<0)
            VM_LOW_FREQ_POWER0[it] = np.mean(DATA[icell]['Vm_max_low_freqs_power'][cond])
            N_LOW_FREQ[it] = len(DATA[icell]['NSI'][cond])
            cond = DATA[icell]['NSI_validated'] & (DATA[icell]['NSI']>=0.)
            VM_LOW_FREQ_POWER_ASYNCH[it] = np.mean(DATA[icell]['Vm_max_low_freqs_power'][cond])
            N_ASYNCH[it] = len(DATA[icell]['NSI'][cond])
            N_NC[it] = len(DATA[icell]['NSI'][np.invert(DATA[icell]['NSI_validated'])])
            
        np.save(DATASET[icell]['files'][0].replace('.abf', '_varying_alpha.npy'),
                [VM_LOW_FREQ_POWER0, VM_LOW_FREQ_POWER_ASYNCH,
                 N_LOW_FREQ, N_ASYNCH, N_NC])
        print('=================================================')
            
    for icell, cell in enumerate(DATASET):
        print('Cell '+str(icell+1)+' :', cell['files'][0])
        DATA.append(load_data(cell['files'][0], args,
                              full_processing=True,
                              with_Vm_low_freq=True))
        
        if args.parallelize:
            PROCESSES.append(mp.Process(target=run_func, args=(icell, output)))
        else:
            run_func(icell, 0)

    if args.parallelize:
        # Run processes
        for p in PROCESSES:
            p.start()
        # # Exit the completed processes
        for p in PROCESSES:
            p.join()
    
    for i, cell in enumerate(DATASET):
        VM_LOW_FREQ_POWER1[:, i],\
            VM_LOW_FREQ_POWER_ASYNCH1[:, i],\
            N_LOW_FREQ1[:, i], N_ASYNCH1[:, i], N_NC1[:, i] = \
                np.load(cell['files'][0].replace('.abf', '_varying_alpha.npy'))
        
    OUTPUT = {'Params':args,
              'ALPHA':ALPHA,
              'VM_LOW_FREQ_POWER':VM_LOW_FREQ_POWER1,
              'VM_LOW_FREQ_POWER_ASYNCH':VM_LOW_FREQ_POWER_ASYNCH1,
              'N_LOW_FREQ':N_LOW_FREQ1,
              'N_ASYNCH':N_ASYNCH1,
              'N_NC':N_NC1}
    
    np.savez(args.datafile_output, **OUTPUT)
    
def plot_test_different_smoothing(args):

    OUTPUT = dict(np.load(args.datafile_input))

    fig_optimum, [[ax, ax1]] = figure(figsize=(.5, .16),
                                      right=0.85, top=0.9, bottom=1.2, left=.6, wspace=1.8,
                                      axes=(1,2))
    
    i0 = np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1))

    mean_Output = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)
    
    Tsmooth = 1e3*OUTPUT['T_SMOOTH']
    ax.plot(Tsmooth, mean_Output, color='k', lw=2)
    ax.scatter([1e3*OUTPUT['T_SMOOTH'][i0]], [np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)[i0]],
               marker='o', color=Brown, facecolor='None')
    ax.annotate('$T_{opt}$', (Tsmooth[i0]+4, ax.get_ylim()[0]), color=Brown, fontsize=FONTSIZE)
    ax.plot(np.array([Tsmooth[i0], Tsmooth[i0]]),
            [mean_Output[i0], ax.get_ylim()[0]], '--', color=Brown, lw=1)
    
    order = np.argsort(np.mean(OUTPUT['CROSS_CORRELS'], axis=0))
    for i in range(len(order)):
        ax1.plot(Tsmooth, OUTPUT['CROSS_CORRELS'][:,order[i]], color=viridis(i/(len(order)-1)))
    ax1.plot(Tsmooth, mean_Output, '-', color='k', lw=0.5)
    ax1.fill_between(Tsmooth,\
                     mean_Output+np.std(OUTPUT['CROSS_CORRELS'], axis=-1),
                     mean_Output-np.std(OUTPUT['CROSS_CORRELS'], axis=-1),
                     lw=0, color='k', alpha=.2)

    set_plot(ax, xlabel=' $T_{smoothing}$ (ms)',
             ylabel='cc $V_m$-pLFP')
    set_plot(ax1, xlabel=' $T_{smoothing}$ (ms)',
             ylabel='cc $V_m$-pLFP')
    acb = plt.axes([.86,.4,.02,.4])
    cb = build_bar_legend(np.arange(len(order)),
                          acb, viridis,
                          no_ticks=True,
                          label='cell index \n (n='+str(len(order))+'cells)')
    return [fig_optimum]

def get_pLFP_parameters_from_scan(datafile1='data/final_wvl_scan.npz',
                                  datafile2='data/final_smooth.npz'):
    
    OUTPUT = dict(np.load(datafile1))
    pCC = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)
    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              pCC.shape)
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]

    
    OUTPUT = dict(np.load(datafile2))


    NORM = np.zeros(OUTPUT['CROSS_CORRELS'].shape)
    order = np.argsort(np.mean(OUTPUT['CROSS_CORRELS'], axis=0))
    for i in range(len(order)):
        NORM[:,order[i]] = (OUTPUT['CROSS_CORRELS'][:,order[i]]-OUTPUT['CROSS_CORRELS'][0,order[i]])/\
            (OUTPUT['CROSS_CORRELS'][:,order[i]].max()-OUTPUT['CROSS_CORRELS'][0,order[i]])
    i0 = np.argmax(np.mean(NORM, axis=-1))
    T0 = OUTPUT['T_SMOOTH'][i0]

    print('wavelet pack in band: [', round(f0/w0,1), ',', round(f0*w0,1), ']Hz')
    print('with smoothing time constant ', round(1e3*T0,1), 'ms')
    
    return f0, w0, T0

###############################################################
##          Show full-recording                              ##
###############################################################

def show_cell(args):
    DATASET = get_full_dataset(args, include_only_chosen=False)
    args.subsampling_period = 5e-3
    FIGS = []
    # plot full dataset
    i, cell = args.cell_index, DATASET[args.cell_index]
    fig, AX = plt.subplots(2*len(cell['files']),
                           figsize=(9,3.*len(cell['files'])))
    plt.subplots_adjust(hspace=.6, top=.92, bottom=.12)
    fig.suptitle(cell['info']+' '+cell['cell']+100*' ')
    # if len(cell['files'])==1: AX = [AX]
    for ax1, ax2, fn in zip(AX[::2], AX[1::2], cell['files']):
        print(fn)
        data = load_data(fn, args,\
                         chosen_window_only=False)
        ax1.plot(data['sbsmpl_t'], data['sbsmpl_Vm'], 'k-', lw=.3)
        ax1.plot([0, 5], [-40, -40], lw=4, color='gray')
        ax1.annotate('5s', (2, -35))
        set_plot(ax1, ylabel='Vm (mV)',
                 xlim=[data['t'][0], data['t'][-1]],
                 ylim=[data['Vm'].min(), data['Vm'].max()])
        ax2.plot(data['sbsmpl_t'], data['sbsmpl_Extra'], 'k-', lw=.3)
        set_plot(ax2, ylabel='$V_{ext}$ (mV)',
                 xlim=[data['t'][0], data['t'][-1]])
        with open(fn.replace(s1, s2).replace('abf', 'json')) as f:
            props = json.load(f)
        ax1.fill_between([float(props['t0']), float(props['t1'])],
                        ax1.get_ylim()[0]*np.ones(2),
                        ax1.get_ylim()[1]*np.ones(2),
                        color='r', alpha=.1)
        ax1.set_title(data['name'])
    return [fig]
        
###############################################################
##             COMPUTE FINAL pLFP                            ##
###############################################################

def compute_final_pLFP(args):

    #####################################################
    ## Get frequency parameters from the analysis
    #####################################################
    OUTPUT = dict(np.load('data/final_wvl_scan.npz'))
    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              OUTPUT['CROSS_CORRELS'].shape[:2])
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]
    # Smoothing time constant
    OUTPUT = dict(np.load('data/final_smooth.npz'))
    Ts = OUTPUT['T_SMOOTH'][np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1))]
    #####################################################

    DATASET = get_full_dataset(args)
    DATA = []
    
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        functions.preprocess_LFP(DATA[icell],
                                 freqs=np.linspace(f0/w0, f0*w0, args.wavelet_number),
                                 new_dt = args.subsampling_period,
                                 smoothing=Ts)
        cc = np.abs(np.corrcoef(DATA[icell]['sbsmpl_Vm'].flatten(), DATA[icell]['pLFP'].flatten()))[0,1]
        np.savez(DATASET[icell]['files'][0].replace('.abf', '_pLFP.npz'),
                 **{'t':DATA[icell]['sbsmpl_t'],
                    'Vm':DATA[icell]['sbsmpl_Vm'],
                    'Extra':DATA[icell]['sbsmpl_Extra'],
                    'pLFP':DATA[icell]['pLFP'],
                    'cc':cc})
        print('=================================================')
            
    for icell, cell in enumerate(DATASET):
        print('Cell '+str(icell+1)+' :', cell['files'][0])
        DATA.append(load_data(cell['files'][0], args))
        if args.parallelize:
            PROCESSES.append(mp.Process(target=run_func, args=(icell, output)))
        else:
            run_func(icell, 0)

    if args.parallelize:
        # Run processes
        for p in PROCESSES:
            p.start()
        # # Exit the completed processes
        for p in PROCESSES:
            p.join()
    
    
###############################################################
##          Show Sample with pLFP                            ##
###############################################################

def show_sample_with_pLFP(args):
    DATASET = get_full_dataset(args, include_only_chosen=False)
    args.subsampling_period = 5e-3
    FIGS = []
    # plot full dataset
    i, cell = args.cell_index, DATASET[args.cell_index]

    data = np.load(cell['files'][0].replace('.abf', '_pLFP.npz'))
    
    fig, AX = plt.subplots(2*len(cell['files']),
                           figsize=(9,2.*len(cell['files'])))
    plt.subplots_adjust(hspace=.2, top=.85, bottom=.2)
    fig.suptitle(cell['info']+' '+cell['cell']+100*' ', fontsize=FONTSIZE)

    V2 = data['Vm']
    V2[data['Vm']>=args.spike_threshold] = args.spike_threshold
    
    for ax1, ax2, fn in zip(AX[::2], AX[1::2], cell['files']):
        cond = (data['t']>args.t1) & (data['t']<args.t2)
        ax1.plot(data['t'][cond],
                 V2[cond], 'k-', lw=.3)
        ax1.plot([0, 5], [-40, -40], 'k-', lw=1)
        ax1.annotate('5s', (2, -35), fontsize=FONTSIZE)
        set_plot(ax1, ['left'], ylabel='Vm (mV)',
                 xlim=[data['t'][cond][0], data['t'][cond][-1]])
        ax2.plot(data['t'][cond], data['pLFP'][cond], 'k-', lw=.3)
        set_plot(ax2, ['left'], ylabel='pLFP ($\mu$V)',
                 xlim=[data['t'][cond][0], data['t'][cond][-1]])
        ax1.set_title('Vm-pLFP cc='+str(np.round(data['cc'],3)), fontsize=FONTSIZE)
    return [fig]

###############################################################
##          Compare Correl Extra and pLFP                            ##
###############################################################

def compare_correl_LFP_pLFP(args):
    DATASET = get_full_dataset(args)
    args.subsampling_period = 5e-3
    FIGS = []
    # plot full dataset
    fig, ax = figure()
    mean1, mean0 = 0, 0
    for i, cell in enumerate(DATASET):

        data = np.load(cell['files'][0].replace('.abf', '_pLFP.npz'))
        cc0 = np.abs(np.corrcoef(data['Vm'].flatten(), data['Extra'].flatten()))[0,1]
        cc1 = np.abs(np.corrcoef(data['Vm'].flatten(), data['pLFP'].flatten()))[0,1]
        ax.plot([0, 1], [cc0, cc1])
        mean1+=cc1/len(DATASET)
        mean0+=cc0/len(DATASET)
    ax.bar([0], [mean0], width=0.4)
    ax.bar([1], [mean1], width=0.4)

    return [fig]


if __name__=='__main__':
    
    import argparse
    parser=argparse.ArgumentParser(\
                    description='Model parameters',
                    formatter_class=argparse.RawTextHelpFormatter)
    # type of analysis or plotting
    parser.add_argument("--show_dataset", help="",action="store_true")
    #### TEST DIFFERENT WAVELETS
    parser.add_argument('-tdw', "--test_different_wavelets", help="",action="store_true")
    parser.add_argument('-ptdw', "--plot_test_different_wavelets", help="",action="store_true")
    parser.add_argument('--wavelet_number', type=int, default=5)    
    parser.add_argument('--discretization', type=int, default=2)    
    parser.add_argument('-cfmi', '--center_wavelet_freq_min', type=float, default=1)    
    parser.add_argument('-cfma', '--center_wavelet_freq_max', type=float, default=1000)    
    parser.add_argument('-ffmi', '--factor_wavelet_freq_min', type=float, default=1.01)    
    parser.add_argument('-ffma', '--factor_wavelet_freq_max', type=float, default=4.)    
    parser.add_argument('-f', '--datafile_output', default='data/data.npz')    
    parser.add_argument('-if', '--datafile_input', default='data/data.npz')    
    #### TEST DIFFERENT SMOOTHING
    parser.add_argument('-tds', "--test_different_smoothing", help="",action="store_true")
    parser.add_argument('-ptds', "--plot_test_different_smoothing", help="",action="store_true")
    parser.add_argument('-smi', '--smoothing_min', type=float, default=5e-3)    
    parser.add_argument('-sma', '--smoothing_max', type=float, default=0.15)    
    #### TEST DIFFERENT ALPHA (LOW-FREQ-THRESH-CRITERIA)
    parser.add_argument('-tda', "--test_different_alpha", help="",action="store_true")
    parser.add_argument('-ptda', "--plot_test_different_alpha", help="",action="store_true")
    parser.add_argument('-ami', '--alpha_min', type=float, default=1.)    
    parser.add_argument('-ama', '--alpha_max', type=float, default=4.)    
    #### SHOW A CELL
    parser.add_argument('-sc', "--show_cell", help="",action="store_true")
    #### COPMUTE FINAL pLFP
    parser.add_argument('-cfp', "--compute_final_pLFP", help="",action="store_true")
    #### show_sample_with_pLFP
    parser.add_argument('-sswp', "--show_sample_with_pLFP", help="",action="store_true")
    parser.add_argument('--t1', type=float,default=-np.inf)    
    parser.add_argument('--t2', type=float,default=np.inf)    
    #### compare_correl_LFP_pLFP
    parser.add_argument('-cclp', "--compare_correl_LFP_pLFP", help="",action="store_true")
    parser.add_argument("--parallelize",
                        help="parallelize the computation using multiprocessing",
                        action="store_true")
    
    parser.add_argument("-d", "--debug", help="debug",
                        default=True, action="store_true")
    # type of dataset
    parser.add_argument('--dataset', type=str, default='Wild_Type')    
    parser.add_argument('--cell_index', help='starts at 0',
                        type=int, default=0)    
    parser.add_argument('--file_index', help='0 means full data',
                        type=int, default=0)    
    # parameters of the analysis
    parser.add_argument('--subsampling_period', type=float,default=1e-3)    
    parser.add_argument('--spike_threshold', type=float,default=-35.)    
    parser.add_argument('--f0', type=float,default=72.79)    
    parser.add_argument('--w0', type=float, default=1.83)
    parser.add_argument('--T0', type=float, default=42.17e-3)
    parser.add_argument('--percentile_for_p0', type=float, default=0.01)
    parser.add_argument('--Tstate', type=float, default=200e-3)
    parser.add_argument('--Var_criteria', type=float, default=2.)
    parser.add_argument('--T_sliding_mean', type=float, default=500e-3)
    parser.add_argument('--alpha', type=float, default=2.)
    
    args = parser.parse_args()

    FIGS = []
    # dataset
    if args.show_dataset:
        show_dataset(args)
    elif args.test_different_wavelets:
        test_different_wavelets(args)
    elif args.plot_test_different_wavelets:
        FIGS = plot_test_different_wavelets(args)
    elif args.test_different_smoothing:
        test_different_smoothing(args)
    elif args.plot_test_different_smoothing:
        FIGS = plot_test_different_smoothing(args)
    elif args.test_different_alpha:
        test_different_alpha(args)
    elif args.plot_test_different_alpha:
        FIGS = plot_test_different_alpha(args)
    elif args.show_cell:
        FIGS = show_cell(args)
    elif args.compute_final_pLFP:
        compute_final_pLFP(args)
    elif args.show_sample_with_pLFP:
        FIGS = show_sample_with_pLFP(args)
    elif args.compare_correl_LFP_pLFP:
        FIGS = compare_correl_LFP_pLFP(args)
    else:
        pass
        
    if len(FIGS)>0:
        show()
    for i, fig in enumerate(FIGS):
        fig.savefig('/Users/yzerlaut/Desktop/temp'+str(i)+'.svg')

        
