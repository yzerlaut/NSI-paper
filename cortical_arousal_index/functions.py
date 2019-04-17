import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

##############################################
########### Wavelet Transform ################
##############################################

def my_cwt(data, frequencies, dt, w0=6.):
    """
    wavelet transform with normalization to catch the amplitude of a sinusoid
    """
    output = np.zeros([len(frequencies), len(data)], dtype=np.complex)

    for ind, freq in enumerate(frequencies):
        wavelet_data = np.conj(get_Morlet_of_right_size(freq, dt, w0=w0))
        sliding_mean = signal.convolve(data,
                                       np.ones(len(wavelet_data))/len(wavelet_data),
                                       mode='same')
        # the final convolution
        wavelet_data_norm = norm_constant_th(freq, dt, w0=w0)
        output[ind, :] = signal.convolve(data-sliding_mean+0.*1j,
                                         wavelet_data,
                                         mode='same')/wavelet_data_norm
    return output

### MORLET WAVELET, definition, properties and normalization
def Morlet_Wavelet(t, f, w0=6.):
    x = 2.*np.pi*f*t
    output = np.exp(1j * x)
    output *= np.exp(-0.5 * ((x/w0) ** 2)) # (Normalization comes later)
    return output

def Morlet_Wavelet_Decay(f, w0=6.):
    """
    Time value of the wavelet where the amplitude decays of 
    """
    return 2 ** .5 * (w0/(np.pi*f))

def from_fourier_to_morlet(freq):
    x = np.linspace(0.1/freq, 2.*freq, 1e3)
    return x[np.argmin((x-freq*(1-np.exp(-freq*x)))**2)]
    
def get_Morlet_of_right_size(f, dt, w0=6., with_t=False):
    Tmax = Morlet_Wavelet_Decay(f, w0=w0)
    t = np.arange(-int(Tmax/dt), int(Tmax/dt)+1)*dt
    if with_t:
        return t, Morlet_Wavelet(t, f, w0=w0)
    else:
        return Morlet_Wavelet(t, f, w0=w0)

def norm_constant_th(freq, dt, w0=6.):
    # from theoretical calculus:
    n = (w0/2./np.sqrt(2.*np.pi)/freq)*(1.+np.exp(-w0**2/2))
    return n/dt

##################################################
########### Processing of the LFP ################
##################################################

def gaussian_smoothing(Signal, idt_sbsmpl=10.):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(Signal, idt_sbsmpl)

def preprocess_LFP(data,
                   freqs = np.linspace(50, 300, 5), 
                   new_dt = 5e-3,
                   smoothing=0e-3,
                   percentile_for_p0=0.01,                   
                   pLFP_unit='$\mu$V'):
    """
    performs continuous wavelet transform
    """

    # if debug:
    #     Extra_key = 'sbsmpl_Extra'
    #     dt = data['sbsmpl_dt']

    Extra_key = 'Extra'
    dt = data['dt']
    
    # performing wavelet transform
    data['W'] = my_cwt(data[Extra_key].flatten(), freqs, dt) 
    data['pLFP_freqs'] = freqs # keeping track of the frequency used

    # taking the mean power over the frequency content considered
    W2 = np.abs(data['W']).mean(axis=0)
    isubsmpl = int(new_dt/dt)

    # then smoothing and subsampling
    if smoothing>0:
        data['pLFP'] = gaussian_smoothing(\
                                          np.reshape(W2[:int(len(W2)/isubsmpl)*isubsmpl],
                                                     (int(len(W2)/isubsmpl),isubsmpl)).mean(axis=1),
                                          int(smoothing/new_dt)).flatten()
    else: # smoothing=0, means no smoothing, just subsampling
        data['pLFP'] = np.reshape(W2[:int(len(W2)/isubsmpl)*isubsmpl],
                                  (int(len(W2)/isubsmpl),isubsmpl)).mean(axis=1).flatten()
    data['pLFP_unit']=pLFP_unit
    
    if pLFP_unit=='$\mu$V':
        data['pLFP'] *= 1e3

    # find p0
    data['p0'] = np.percentile(data['pLFP'], percentile_for_p0)
    # find p0 for Vm
    data['p0_Vm'] = np.percentile(data['sbsmpl_Vm'], percentile_for_p0)

def heaviside(x):
    return (np.sign(x)+1)/2


def Network_State_Index(data,
                        key='pLFP',
                        p0=0.,
                        alpha=2.):
    
    NSI=np.zeros(len(data['sbsmpl_t']))
    # where rhythmicity is matched
    X = (p0+alpha*data[key+'_max_low_freqs_power'])-data[key+'_sliding_mean']
    NSI = -2*data[key+'_max_low_freqs_power']*heaviside(X)+heaviside(-X)*(data[key+'_sliding_mean']-p0)
    return NSI

def Validate_Network_States(data, 
                            target_key='NSI',
                            Tstate=200e-3,
                            Var_criteria=2):
    
    # validate states:
    iTstate = int(Tstate/data['sbsmpl_dt'])
    # validate the transitions
    data[target_key+'_validated'] = np.zeros(len(data['sbsmpl_t']), dtype=bool)
    data[target_key+'_unvalidated'] = np.zeros(len(data['sbsmpl_t']), dtype=bool)
    for i in np.arange(len(data['sbsmpl_t']))[::iTstate][1:-1]:
        if np.array(np.abs(data[target_key][i-iTstate:i+iTstate]-data[target_key][i])<=Var_criteria).all():
            data[target_key+'_validated'][i]=True
        else:
            data[target_key+'_unvalidated'][i]=True

    data['t_'+target_key+'_validated'] = data['sbsmpl_t'][data[target_key+'_validated']]
    data['i_'+target_key+'_validated'] = np.arange(len(data['sbsmpl_t']))[data[target_key+'_validated']]

    
def compute_Network_State_Index(data,
                                key='pLFP',
                                target_key='NSI',
                                p0_key='p0',
                                freqs = np.linspace(2,5,10),
                                Tstate=200e-3,
                                alpha=2.,
                                with_Vm_low_freq=False,
                                T_sliding_mean=0.5,
                                validate=True,
                                Var_criteria=2,
                                already_low_freqs_and_mean=False):
    

    if not already_low_freqs_and_mean:
        # sliding mean
        data[key+'_sliding_mean'] = gaussian_smoothing(data[key], int(T_sliding_mean/data['sbsmpl_dt']))
        # low frequency power
        data[key+'_low_freqs'] = freqs # storing the used-freq
        data[key+'_W_low_freqs'] = my_cwt(data[key].flatten(), freqs, data['sbsmpl_dt']) # wavelet transform
        data[key+'_max_low_freqs_power'] = np.max(np.abs(data[key+'_W_low_freqs']), axis=0) # max of freq.
        imax = np.argmax(np.abs(data[key+'_W_low_freqs']), axis=0)
        data[key+'_phase_of_max_low_freqs_power'] = np.angle([data[key+'_W_low_freqs'][imax[i], i] for i in range(len(imax))])
    
    if with_Vm_low_freq:
        W = my_cwt(data['sbsmpl_Vm'].flatten(), freqs, data['sbsmpl_dt']) # wavelet transform
        data['Vm_max_low_freqs_power'] = np.max(np.abs(W), axis=0) # max of freq.
        data['Vm_sliding_mean'] = gaussian_smoothing(data['sbsmpl_Vm'], int(T_sliding_mean/data['sbsmpl_dt']))

    data[target_key]= Network_State_Index(data,
                                          key=key,
                                          p0 = data[p0_key],
                                          alpha=alpha)

    if validate:
        Validate_Network_States(data,
                                target_key=target_key,
                                Tstate=Tstate,
                                Var_criteria=Var_criteria)

def compute_delta_and_gamma(data, args,
                            target_key='Extra'):

    
    delta_freqs = np.linspace(args.delta_band[0], args.delta_band[1], 10)
    gamma_freqs = np.linspace(args.gamma_band[0], args.gamma_band[1], 20)
    
    # delta as max power in [2,10]Hz band
    data[target_key+'_delta_power'] = np.max(np.abs(my_cwt(data['sbsmpl_'+target_key].flatten(), delta_freqs, data['sbsmpl_dt'])), axis=0)
    # gamma as max power in [30,80]Hz band
    data[target_key+'_gamma_power'] = np.max(np.abs(my_cwt(data['sbsmpl_'+target_key].flatten(), gamma_freqs, data['sbsmpl_dt'])), axis=0)
        
        
