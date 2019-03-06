import sys, pathlib, os, json
import numpy as np
import matplotlib.pylab as plt
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data_analysis.IO.load_data import load_file
from data_analysis.processing.filters import butter_bandpass_filter
from data_analysis.processing.signanalysis import gaussian_smoothing
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
              fraction_extent_of_data=[0., 1.], # for cross-validation
              verbose=False,
              with_spiking_activity=True,
              chosen_window_only=True,
              full_processing=False,
              with_Vm_low_freq=False):

    if verbose:
        print('analyzing :', fn)
    with open(fn.replace(s1, s2).replace('abf', 'json')) as f: props = json.load(f)
    
    if chosen_window_only:
        t0, t1 = np.float(props['t0']), np.float(props['t1'])
    else:
        t0, t1 = 0, np.inf
        
    raw_data = load_file(fn, zoom=[t0, t1])
    
    data_full = {'t':raw_data[0]-raw_data[0][0],
                 'Vm':raw_data[1][1],
                 'Extra':raw_data[1][0]}
    # in case
    cond = (np.arange(len(data_full['t']))>=int(len(data_full['t'])*fraction_extent_of_data[0])) &\
           (np.arange(len(data_full['t']))<=int(len(data_full['t'])*fraction_extent_of_data[1]))
    data = {'t':data_full['t'][cond], 'Vm':data_full['Vm'][cond], 'Extra':data_full['Extra'][cond],
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
        if verbose:
            print('processing LFP')
        functions.preprocess_LFP(data,
                                 freqs=np.linspace(args.f0/args.w0, args.f0*args.w0, args.wavelet_number),
                                 new_dt=args.subsampling_period,
                                 percentile_for_p0=args.percentile_for_p0,
                                 smoothing=args.T0)
        data['pLFP'] = data['pLFP'][:min([len(data['pLFP']), len(data['sbsmpl_t'])])]
        
        # compute the Network State Index
        if verbose:
            print('computing NSI')
        functions.compute_Network_State_Index(data,
                                              freqs=np.linspace(args.delta_band[0], args.delta_band[1], 8), # freqs in delta band
                                              Tstate=args.Tstate,
                                              Var_criteria=data['p0'], # HERE TAKING NOISE AS CRITERIA !!!
                                              alpha=args.alpha,
                                              T_sliding_mean=args.T_sliding_mean,
                                              with_Vm_low_freq=with_Vm_low_freq)
        
        # extract delta and gamma power from LFP
        if verbose:
            print('computing delta and gamma')
        functions.compute_delta_and_gamma(data, args)
        
        # MUA from extracellular signal
        if with_spiking_activity:
            if verbose:
                print('computing MUA, spikes and FR')
            data['MUA'] = gaussian_smoothing(\
                            np.abs(butter_bandpass_filter(data['Extra'],\
                                     args.MUA_band[0], args.MUA_band[1], 1./data['dt'], order=5)),\
                                           int(args.MUA_smoothing/data['dt']))

            data['sbsmpl_MUA'] = 1e-3*data['MUA'][::int(args.subsampling_period/data['dt'])][:-1] # in uV
            # Spike times from Vm
            data['tspikes'] = data['t'][np.argwhere((data['Vm'][:-1]<=args.spike_threshold) & (data['Vm'][1:]>args.spike_threshold)).flatten()]
            Vpeaks = []
            for tt in data['tspikes']:
                Vpeaks.append(np.max(data['Vm'][(data['t']>tt-5e-3) & (data['t']<tt+5e-3)])) # max in Vm surrouding spike time
            data['Vpeak_spikes'] = np.array(Vpeaks)
            data['sbsmpl_FR'] = np.histogram(data['tspikes'], bins=data['sbsmpl_t'])[0]/data['sbsmpl_dt']

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
            
    FILENAMES = []
    for icell, cell in enumerate(DATASET):
        FILENAMES.append(cell['files'][0])
        print('Cell '+str(icell+1)+' :', FILENAMES[-1])
        DATA.append(load_data(FILENAMES[-1], args))
        
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
    
    for i, fn in enumerate(FILENAMES):
        CROSS_CORRELS[:, i] = np.load(fn.replace('.abf', '_wavelet_scan.npy'))
                
    OUTPUT = {'Params':args,
              'FILENAMES':FILENAMES,
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
            # then unvalidated states
            N_NC[it] = len(DATA[icell]['NSI'][DATA[icell]['NSI_unvalidated']])
            
        np.save(DATASET[icell]['files'][0].replace('.abf', '_varying_alpha.npy'),
                [VM_LOW_FREQ_POWER0, VM_LOW_FREQ_POWER_ASYNCH,
                 N_LOW_FREQ, N_ASYNCH, N_NC])
        print('=================================================')

    FILENAMES = []
    for icell, cell in enumerate(DATASET):
        FILENAMES.append(cell['files'][0])
        print('Cell '+str(icell+1)+' :', FILENAMES[-1])
        DATA.append(load_data(FILENAMES[-1], args,
                              fraction_extent_of_data=args.data_fraction,
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
    
    for i, fn in enumerate(FILENAMES):
        VM_LOW_FREQ_POWER1[:, i],\
            VM_LOW_FREQ_POWER_ASYNCH1[:, i],\
            N_LOW_FREQ1[:, i], N_ASYNCH1[:, i], N_NC1[:, i] = \
                np.load(fn.replace('.abf', '_varying_alpha.npy'))
        
    OUTPUT = {'Params':args,
              'FILENAMES':FILENAMES,
              'ALPHA':ALPHA,
              'VM_LOW_FREQ_POWER':VM_LOW_FREQ_POWER1,
              'VM_LOW_FREQ_POWER_ASYNCH':VM_LOW_FREQ_POWER_ASYNCH1,
              'N_LOW_FREQ':N_LOW_FREQ1,
              'N_ASYNCH':N_ASYNCH1,
              'N_NC':N_NC1}
    
    np.savez(args.datafile_output, **OUTPUT)

###############################################################
##          Find the good low-freq criteria (alpha)          ##
###############################################################

def get_polarization_level(args):
    
    DATASET = get_full_dataset(args)
    DATA = []

    # phase_bins = np.linspace()
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        NSI_ASYNCH_LEVELS, VM_ASYNCH_LEVELS = [], []
        NSI_SYNCH_LEVELS, VM_SYNCH_LEVELS = [], []
        DVM_0PI_ASYNCH, DVM_0PI_SYNCH = [], []
        phase_bins = np.linspace(-np.pi, np.pi, 30)
        FINAL_PHASE, FINAL_HIST = [], []
        FINAL_PHASE_ASYNCH, FINAL_HIST_ASYNCH = [], []
        iTstate = int(args.Tstate/DATA[icell]['sbsmpl_dt'])
        # ====== ASYNCH COND ! =========
        cond = DATA[icell]['NSI_validated'] & (DATA[icell]['NSI']>0.) 
        HIST = [[] for jj in range(len(phase_bins))]
        phase_data = np.angle(np.mean(DATA[icell]['pLFP_W_low_freqs'], axis=0))
        for ii in np.arange(len(DATA[icell]['NSI']))[cond]:
            vm = DATA[icell]['sbsmpl_Vm'][ii-int(iTstate/2):ii+int(iTstate/2)]
            VM_ASYNCH_LEVELS.append(np.mean(vm[vm < args.spike_threshold])) # removing spikes
            NSI_ASYNCH_LEVELS.append(DATA[icell]['NSI'][ii])
            phase = phase_data[ii-int(iTstate/2):ii+int(iTstate/2)]
            for jj in range(len(phase_bins)):
                cond = (np.digitize(phase, bins=phase_bins)==jj)
                if len(vm[cond])>0:
                    HIST[jj].append(np.mean(vm[cond][vm[cond]<args.spike_threshold])) # removing spikes
            # difference between the 0 and pi phase at the single cycle level:
            cond_0_phase = np.abs(phase-0)<0.1
            cond_pi_phase = np.abs(phase-np.pi)<0.1
            DVM_0PI_ASYNCH.append(np.mean(vm[cond_0_phase])-np.mean(vm[cond_pi_phase]))
        for jj in range(len(phase_bins)):
            if len(HIST[jj])>1:
                FINAL_PHASE_ASYNCH.append(phase_bins[jj])
                FINAL_HIST_ASYNCH.append(np.mean(HIST[jj]))
        # ====== SYNCH COND ! =========
        cond = DATA[icell]['NSI_validated'] & (DATA[icell]['NSI']<0.) 
        HIST = [[] for jj in range(len(phase_bins))]
        for ii in np.arange(len(DATA[icell]['NSI']))[cond]:
            vm = DATA[icell]['sbsmpl_Vm'][ii-int(iTstate/2):ii+int(iTstate/2)]
            # VM_SYNCH_LEVELS.append(np.max(functions.gaussian_smoothing(vm, int(20e-3/DATA[icell]['sbsmpl_dt']))))
            VM_SYNCH_LEVELS.append(DATA[icell]['Vm_max_low_freqs_power'][ii]) # enveloppe of [2,10]Hz
            NSI_SYNCH_LEVELS.append(DATA[icell]['NSI'][ii])
            phase = phase_data[ii-int(iTstate/2):ii+int(iTstate/2)]
            for jj in range(len(phase_bins)):
                cond = (np.digitize(phase, bins=phase_bins)==jj)
                if len(vm[cond])>0:
                    HIST[jj].append(np.mean(vm[cond][vm[cond]<args.spike_threshold])) # removing spikes
            # difference between the 0 and pi phase at the single cycle level:
            cond_0_phase = np.abs(phase-0)<0.1
            cond_pi_phase = np.abs(phase-np.pi)<0.1
            DVM_0PI_SYNCH.append(np.mean(vm[cond_0_phase])-np.mean(vm[cond_pi_phase]))
        for jj in range(len(phase_bins)):
            if len(HIST[jj])>1:
                FINAL_PHASE.append(phase_bins[jj])
                FINAL_HIST.append(np.mean(HIST[jj]))
        # NO ACT CONDadding the Vm level where plFP<p0, as the -1
        cond = (DATA[icell]['pLFP']<DATA[icell]['p0'])
        NSI_ASYNCH_LEVELS.append(0)
        VM_ASYNCH_LEVELS.append(np.mean(DATA[icell]['sbsmpl_Vm'][cond]))
        # NSI_ASYNCH_LEVELS.append(-2) # and -2 is its variability
        # VM_ASYNCH_LEVELS.append(np.std(DATA[icell]['sbsmpl_Vm'][cond]))
        np.save(DATASET[icell]['files'][0].replace('.abf', '_depol_asynch_states.npy'),
                [np.array(FINAL_PHASE_ASYNCH), np.array(FINAL_HIST_ASYNCH),
                 np.array(NSI_ASYNCH_LEVELS), np.array(VM_ASYNCH_LEVELS),
                 np.array(FINAL_PHASE), np.array(FINAL_HIST),
                 np.array(NSI_SYNCH_LEVELS), np.array(VM_SYNCH_LEVELS),
                 np.array(DVM_0PI_ASYNCH), np.array(DVM_0PI_SYNCH)])
        print('=================================================')

    FILENAMES = []
    for icell, cell in enumerate(DATASET):
        FILENAMES.append(cell['files'][0])
        print('Cell '+str(icell+1)+' :', FILENAMES[-1])
        DATA.append(load_data(FILENAMES[-1], args,
                              with_Vm_low_freq=True,
                              full_processing=True))
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
    
    NSI_ASYNCH_LEVELS, VM_ASYNCH_LEVELS = [], []
    NSI_SYNCH_LEVELS, VM_SYNCH_LEVELS = [], []
    PHASE, VM_PHASE_LEVELS = [], []
    PHASE_ASYNCH, VM_PHASE_LEVELS_ASYNCH = [], []
    DVM_0PI_ASYNCH, DVM_0PI_SYNCH = [], []
    for i, fn in enumerate(FILENAMES):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = np.load(fn.replace('.abf', '_depol_asynch_states.npy')) 
        PHASE_ASYNCH.append(x1)
        VM_PHASE_LEVELS_ASYNCH.append(x2)
        NSI_ASYNCH_LEVELS.append(x3)
        VM_ASYNCH_LEVELS.append(x4)
        PHASE.append(x5)
        VM_PHASE_LEVELS.append(x6)
        NSI_SYNCH_LEVELS.append(x7)
        VM_SYNCH_LEVELS.append(x8)
        DVM_0PI_ASYNCH.append(x9)
        DVM_0PI_SYNCH.append(x10)
        
    OUTPUT = {'Params':args,
              'FILENAMES':FILENAMES,
              'NSI_ASYNCH_LEVELS':NSI_ASYNCH_LEVELS,
              'VM_ASYNCH_LEVELS':VM_ASYNCH_LEVELS,
              'NSI_SYNCH_LEVELS':NSI_SYNCH_LEVELS,
              'VM_SYNCH_LEVELS':VM_SYNCH_LEVELS,
              'PHASE':PHASE,
              'VM_PHASE_LEVELS':VM_PHASE_LEVELS,
              'PHASE_ASYNCH':PHASE_ASYNCH,
              'VM_PHASE_LEVELS_ASYNCH':VM_PHASE_LEVELS_ASYNCH,
              'DVM_0PI_ASYNCH':DVM_0PI_ASYNCH,
              'DVM_0PI_SYNCH':DVM_0PI_SYNCH}
    
    np.savez(args.datafile_output, **OUTPUT)

def extended_NSI_correlate_analysis(args):
    
    DATASET = get_full_dataset(args)
    DATA = []

    NSI_DISCRET = np.array([-20, -5, 0, 5, 10, 40])
    phase_bins = np.linspace(-np.pi, np.pi, 30)

    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()


    def run_func(icell, output):
        print('=================================================')
        print('running cell', icell, '[...]')
        iTstate = int(args.Tstate/DATA[icell]['sbsmpl_dt'])
        output = {'NSI_DISCRET':NSI_DISCRET,
                  'phase_bins': phase_bins,
                  'Vm0':DATA[icell]['p0_Vm'],
                  'NSI_LEVELS':[[] for i in range(len(NSI_DISCRET)-1)],
                  'FR_LEVELS':[[] for i in range(len(NSI_DISCRET)-1)],
                  'MUA_LEVELS':[[] for i in range(len(NSI_DISCRET)-1)],
                  'DEPOL_LEVELS':[[] for i in range(len(NSI_DISCRET)-1)],
                  'LINK_DEPOL_PHASE':np.ones((len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
                  'LINK_STD_DEPOL_PHASE':np.ones((len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
                  'LINK_FR_PHASE':np.ones((len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
                  'LINK_MUA_PHASE':np.ones((len(NSI_DISCRET)-1, len(phase_bins)))*np.inf}
        
        
        for iND in range(len(NSI_DISCRET)-1):
            cond = DATA[icell]['NSI_validated'] & (DATA[icell]['NSI']>NSI_DISCRET[iND]) & (DATA[icell]['NSI']<=NSI_DISCRET[iND+1])

            HISTVM = [[] for ipb in range(len(phase_bins))]
            HISTMUA = [[] for ipb in range(len(phase_bins))]
            HISTFR = [[] for ipb in range(len(phase_bins))]
            
            for iEp in np.arange(len(DATA[icell]['NSI']))[cond]:
                # store the exact NSI level
                output['NSI_LEVELS'][iND].append(DATA[icell]['NSI'][iEp])
                # compute the depol level from the Vm
                vm = DATA[icell]['sbsmpl_Vm'][iEp-int(iTstate/2):iEp+int(iTstate/2)]
                output['DEPOL_LEVELS'][iND].append(np.mean(vm[vm < args.spike_threshold])) # removing spikes
                # compute the mua
                mua = DATA[icell]['sbsmpl_MUA'][iEp-int(iTstate/2):iEp+int(iTstate/2)]
                output['MUA_LEVELS'][iND].append(np.mean(mua))
                # compute the firing rate of single cells
                firing_rate = DATA[icell]['sbsmpl_FR'][iEp-int(iTstate/2):iEp+int(iTstate/2)]
                output['FR_LEVELS'][iND].append(np.mean(firing_rate))

                
                phase = DATA[icell]['pLFP_phase_of_max_low_freqs_power'][iEp-int(iTstate/2):iEp+int(iTstate/2)]
                
                for ipb in range(len(phase_bins)):
                    phase_cond = (np.digitize(phase, bins=phase_bins)==ipb)
                    if len(vm[phase_cond])>0:
                        HISTVM[ipb].append(np.mean(vm[phase_cond][vm[phase_cond]<args.spike_threshold])) # removing spikes
                    if len(mua[phase_cond])>0:
                        HISTMUA[ipb].append(np.mean(mua[phase_cond]))
                    if len(firing_rate[phase_cond])>0:
                        HISTFR[ipb].append(np.mean(firing_rate[phase_cond]))


            for ipb in range(len(phase_bins)):
                if len(HISTVM[ipb])>0:
                     output['LINK_DEPOL_PHASE'][iND, :] = np.array([np.mean(HISTVM[ipb]) for ipb in range(len(phase_bins))])
                     output['LINK_STD_DEPOL_PHASE'][iND, :] = np.array([np.std(HISTVM[ipb]) for ipb in range(len(phase_bins))])
                if len(HISTFR[ipb])>0:
                    output['LINK_FR_PHASE'][iND, :] = np.array([np.mean(HISTFR[ipb]) for ipb in range(len(phase_bins))])
                if len(HISTMUA[ipb])>0:
                    output['LINK_MUA_PHASE'][iND, :] = np.array([np.mean(HISTMUA[ipb]) for ipb in range(len(phase_bins))])

                    
        np.savez(DATASET[icell]['files'][0].replace('.abf', '_extended_analysis.npz'), **output)
        print('=================================================')

    FILENAMES = []
    for icell, cell in enumerate(DATASET):
        FILENAMES.append(cell['files'][0])
        print('Cell '+str(icell+1)+' :', FILENAMES[-1])
        DATA.append(load_data(FILENAMES[-1], args,
                              with_Vm_low_freq=True,
                              full_processing=True))
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
    
    OUTPUT = {'NSI_DISCRET':NSI_DISCRET,
              'phase_bins': phase_bins,
              'Vm0':np.zeros(len(FILENAMES)),
              'NSI_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              # 'DELTA_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              # 'GAMMA_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              'FR_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              'MUA_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              'DEPOL_LEVELS':[np.empty(1) for i in range(len(FILENAMES))],
              'LINK_DEPOL_PHASE':np.ones((len(FILENAMES), len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
              'LINK_STD_DEPOL_PHASE':np.ones((len(FILENAMES), len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
              'LINK_FR_PHASE':np.ones((len(FILENAMES), len(NSI_DISCRET)-1, len(phase_bins)))*np.inf,
              'LINK_MUA_PHASE':np.ones((len(FILENAMES), len(NSI_DISCRET)-1, len(phase_bins)))*np.inf}
    
    for icell, fn in enumerate(FILENAMES):
        output = np.load(fn.replace('.abf', '_extended_analysis.npz'))
        OUTPUT['Vm0'][icell] = output['Vm0']
        OUTPUT['NSI_LEVELS'][icell] = np.array(output['NSI_LEVELS']).flatten()
        # OUTPUT['DELTA_LEVELS'][icell] = np.array(output['DELTA_LEVELS']).flatten()
        # OUTPUT['GAMMA_LEVELS'][icell] = np.array(output['GAMMA_LEVELS']).flatten()
        OUTPUT['FR_LEVELS'][icell] = np.array(output['FR_LEVELS']).flatten()
        OUTPUT['MUA_LEVELS'][icell] = np.array(output['MUA_LEVELS']).flatten()
        OUTPUT['DEPOL_LEVELS'][icell] = np.array(output['DEPOL_LEVELS']).flatten()
        OUTPUT['LINK_DEPOL_PHASE'][icell,:,:] = output['LINK_DEPOL_PHASE']
        OUTPUT['LINK_STD_DEPOL_PHASE'][icell,:,:] = output['LINK_STD_DEPOL_PHASE']
        OUTPUT['LINK_FR_PHASE'][icell,:,:] = output['LINK_FR_PHASE']
        OUTPUT['LINK_MUA_PHASE'][icell,:,:] = output['LINK_MUA_PHASE']
    np.savez(args.datafile_output, **OUTPUT)
    
    
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
    #### Get Polarization levels
    parser.add_argument('-gpl', "--get_polarization_level", help="",action="store_true")
    #### Extended NSI correlate analysis
    parser.add_argument('-enca', "--extended_NSI_correlate_analysis", help="",action="store_true")
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
    parser.add_argument('--alpha', type=float, default=2.87)
    parser.add_argument('--delta_band', nargs='+', type=float, default=[2., 4.])
    parser.add_argument('--gamma_band', nargs='+', type=float, default=[30., 80.])
    # parameters of Multi-Unit-Activity (MUA)
    parser.add_argument('--MUA_band', nargs='+', type=float, default=[300., 3000.])
    parser.add_argument('--MUA_smoothing', type=float, default=20e-3)
    # DATA fraction, for the cross-validation analysis
    parser.add_argument('--data_fraction', nargs='+', type=float, default=[0., 1.])
    
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
    elif args.get_polarization_level:
        get_polarization_level(args)
    elif args.extended_NSI_correlate_analysis:
        extended_NSI_correlate_analysis(args)
    else:
        pass
        
    if len(FIGS)>0:
        show()
    for i, fig in enumerate(FIGS):
        fig.savefig(desktop+'temp'+str(i)+'.svg')

        
