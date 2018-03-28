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

###############################################################
##          LOAD DATAFILES (ABF format)                      ##
###############################################################

def load_data(fn, args,
              chosen_window_only=True):

    with open(fn.replace('abf', 'json')) as f: props = json.load(f)
    
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
    
    data['sbsmpl_Vm'] = data['Vm'][::isubsampling]
    data['sbsmpl_Extra'] = data['Extra'][::isubsampling]
    data['sbsmpl_t'] = data['t'][::isubsampling]
    data['sbsmpl_dt'] = data['dt']*isubsampling
    
    return data

###############################################################
##          Find the good wavelet packet                     ##
###############################################################

def test_different_wavelets(args):
    
    CENTER_FREQUENCIES = np.logspace(np.log(args.center_wavelet_freq_min)/np.log(10),
                                     np.log(args.center_wavelet_freq_max)/np.log(10),
                                     args.wavelet_discretization)
    BAND_LENGTH_FACTOR = np.linspace(args.factor_wavelet_freq_min,
                                     args.factor_wavelet_freq_max,
                                     args.wavelet_discretization)
    

    DATASET = get_full_dataset(args)[:1]
    DATA = []
    CROSS_CORRELS = np.zeros((len(CENTER_FREQUENCIES), len(BAND_LENGTH_FACTOR), len(DATASET)))
    
    if args.parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
        
    def run_func(icell, output):
        print(DATASET[icell]['files'][0].replace('.abf', '_wavelet_scan.npy'))
        CROSS_CORRELS0 = np.zeros((len(CENTER_FREQUENCIES), len(BAND_LENGTH_FACTOR)))
        for icf, ibl in product(range(len(CENTER_FREQUENCIES)),
                                range(len(BAND_LENGTH_FACTOR))):
            cf = CENTER_FREQUENCIES[icf]
            blf = BAND_LENGTH_FACTOR[ibl]
            print('running ', icf, ibl, 'on cell', icell, '[...]')
            functions.preprocess_LFP(DATA[icell],
                                     freqs=np.linspace(cf/blf, cf*blf, args.wavelet_number),
                                     smoothing=0.) # HERE NO SMOOTHING YET !!
            cc = np.abs(np.corrcoef(DATA[icell]['new_Vm'], DATA[icell]['pLFP']))[0,1]
            CROSS_CORRELS0[icf, ibl] = cc
        np.save(DATASET[icell]['files'][0].replace('.abf', '_wavelet_scan.npy'),
                CROSS_CORRELS0)
            
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

def plot_test_different_wavelets(args):

    OUTPUT = dict(np.load('data/test_different_wavelet_analysis.npz'))

    print('frequencies :', OUTPUT['CENTER_FREQUENCIES'])
    print('length :', OUTPUT['BAND_LENGTH_FACTOR'])
    
    pCC = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)

    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              pCC.shape)
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]
    
    fig_optimum, ax = figure(figsize=(.3, .16), right=.7, top=.9, bottom=1.2, left=.9)
    p = plt.contourf(OUTPUT['BAND_LENGTH_FACTOR'],
                     OUTPUT['CENTER_FREQUENCIES'],
                     pCC,
                     levels=np.linspace(pCC.min(),pCC.max(),30),
                     cmap=viridis)
    ax.scatter([w0], [f0], color=Brown, facecolor='None', label='($f_{opt}, w_{opt}$)')
    ax.set_yscale('log')
    ax.set_title('wavelet packets in bands: [$f/w$, $f\cdot w$]', fontsize=FONTSIZE)
    set_plot(ax, xlabel=' $w$, width factor\n for freq. band extent',
             ylabel=' $f$, root freq. (Hz)    ',
             yticks=[2, 20, 200], yticks_labels=['2', '20', '200'], xticks=np.arange(1, 5))
    ax.legend(loc=(1.05, .8), prop={'size':'x-small'})
    acb = plt.axes([.71,.35,.02,.4])
    build_bar_legend(np.unique(np.round(np.linspace(pCC.min(),pCC.max(),10),1)),
                     acb, viridis,
                     color_discretization=30,
                     label='cc $V_m$-pLFP \n (n='+str(OUTPUT['CROSS_CORRELS'].shape[-1])+')')
    return [fig_optimum]

###############################################################
##          Find the good wavelet packet                     ##
###############################################################

def test_different_smoothing(args):
    
    T_SMOOTH = np.concatenate([\
                               [0],
                               np.logspace(np.log(args.smoothing_min)/np.log(10),
                                           np.log(args.smoothing_max)/np.log(10),
                                           args.smoothing_discretization)
                               ])

    ## NOEED TO GRAB THE OPTIMAL FREQUENCY !
    OUTPUT = dict(np.load('data/test_different_wavelet_analysis.npz'))
    i0, j0 = np.unravel_index(np.argmax(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1), axis=None),
                              pCC.shape)
    f0, w0 = OUTPUT['CENTER_FREQUENCIES'][i0], OUTPUT['BAND_LENGTH_FACTOR'][j0]

    
    DATASET = get_full_dataset(args)
    DATA = []
    print('================================')
    print(' loading files [...]')
    for i, cell in enumerate(DATASET):
        print('Cell '+str(i+1)+' :', cell['files'][0])
        DATA.append(load_data(cell['files'][0], args))
            
    CROSS_CORRELS = np.zeros((len(T_SMOOTH), len(DATASET)))
    
    def run_func(it, icell, CROSS_CORRELS):
        Ts = Tsmooth[it]
        print('running ', it, 'on cell', icell, '[...]')
        functions.preprocess_LFP(DATA[icell],
                                 freqs=np.linspace(cf0/blf0, cf0*blf0, args.wavelet_number),
                                 smoothing=Ts) # HERE SMOOTHING !!
        cc = np.abs(np.corrcoef(DATA[icell]['new_Vm'], DATA[icell]['pLFP']))[0,1]
        CROSS_CORRELS[icf, ibl, icell] = cc

    for icell, icf, ibl in product(range(len(DATASET)),\
                                   range(len(CENTER_FREQUENCIES)),
                                   range(len(BAND_LENGTH_FACTOR))):
        run_func(icf, ibl, icell, CROSS_CORRELS)

    OUTPUT = {'Params':args,
              'CENTER_FREQUENCIES' : CENTER_FREQUENCIES,
              'BAND_LENGTH_FACTOR' : BAND_LENGTH_FACTOR,
              'CROSS_CORRELS' : CROSS_CORRELS}

    np.savez(args.datafile_output, **OUTPUT)

def plot_test_different_smoothing(args):

    OUTPUT = dict(np.load('data/test_different_wavelet_analysis.npz'))

    pCC = np.mean(OUTPUT['CROSS_CORRELS'], axis=-1)
    # pCC = np.log(np.mean(OUTPUT['CROSS_CORRELS'], axis=-1))/np.log(10)
    
    fig_optimum, ax = figure(figsize=(.3, .16), right=.7, top=.9, bottom=1.2, left=.9)
    p = plt.contourf(OUTPUT['BAND_LENGTH_FACTOR'],
                     OUTPUT['CENTER_FREQUENCIES'],
                     pCC,
                     levels=np.linspace(pCC.min(),pCC.max(),30),
                     cmap=viridis)
    ax.set_yscale('log')
    ax.set_title('wavelet packets in bands: [$f/w$, $f\cdot w$]', fontsize=FONTSIZE)
    set_plot(ax, xlabel=' $w$, width factor\n for freq. band extent',
             ylabel=' $f$, center freq. (Hz)    ',
             yticks=[2, 20, 200], yticks_labels=['2', '20', '200'], xticks=np.arange(1, 5))
    acb = plt.axes([.71,.4,.02,.4])
    build_bar_legend(np.unique(np.round(np.linspace(pCC.min(),pCC.max(),10),1)),
                     acb, viridis,
                     color_discretization=30,
                     label='cc $V_m$-pLFP \n (n='+str(OUTPUT['CROSS_CORRELS'].shape[-1])+')')
    return [fig_optimum]

    
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
    parser.add_argument('--wavelet_discretization', type=int, default=2)    
    parser.add_argument('-cfmi', '--center_wavelet_freq_min', type=float, default=1)    
    parser.add_argument('-cfma', '--center_wavelet_freq_max', type=float, default=1000)    
    parser.add_argument('-ffmi', '--factor_wavelet_freq_min', type=float, default=1.01)    
    parser.add_argument('-ffma', '--factor_wavelet_freq_max', type=float, default=4.)    
    parser.add_argument('-f', '--datafile_output', default='data/data.npz')    
    #### TEST DIFFERENT SMOOTHING
    parser.add_argument('-tds', "--test_different_smoothing", help="",action="store_true")
    parser.add_argument('-ptds', "--plot_test_different_smoothing", help="",action="store_true")
    parser.add_argument('-smi', '--smoothing_min', type=float, default=0.)    
    parser.add_argument('-sma', '--smoothing_max', type=float, default=0.5)    
    
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
    parser.add_argument('--subsampling_period', type=float,default=5e-4)    

    args = parser.parse_args()

    FIGS = []
    # dataset
    if args.show_dataset:
        show_dataset(args)
    elif args.test_different_wavelets:
        test_different_wavelets(args)
    elif args.plot_test_different_wavelets:
        FIGS = plot_test_different_wavelets(args)

    if len(FIGS)>0:
        show()
    for i, fig in enumerate(FIGS):
        fig.savefig('/Users/yzerlaut/Desktop/temp'+str(i)+'.svg')

        
