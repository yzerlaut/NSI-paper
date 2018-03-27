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

def gaussian_smoothing(Signal, idt_sbsmpl=10):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(Signal, idt_sbsmpl)

def preprocess_LFP(data,
                   freqs = np.linspace(50, 300, 5),
                   new_dt = 5e-3, smoothing=20e-3, pLFP_unit='$\mu$V'):
    """
    performs continuous wavelet transform
    """

    
    # performing wavelet transform
    data['W'] = my_cwt(data['Extra'], freqs, data['dt']) 
    data['pLFP_freqs'] = freqs # keeping track of the frequency used

    # taking the mean power over the frequency content considered
    W2 = np.abs(data['W']).mean(axis=0)
    isubsmpl = int(round(new_dt/data['dt']))

    # then smoothing and subsampling
    if smoothing>0:
        data['pLFP'] = gaussian_smoothing(\
                                          np.reshape(W2[:int(len(W2)/isubsmpl)*isubsmpl],
                                                     (int(len(W2)/isubsmpl),isubsmpl)).mean(axis=1),
                                          int(smoothing/new_dt))
    else: # smoothing=0, means no smoothing, just subsampling
        data['pLFP'] = np.reshape(W2[:int(len(W2)/isubsmpl)*isubsmpl],
                                  (int(len(W2)/isubsmpl),isubsmpl)).mean(axis=1)
    data['pLFP_unit']=pLFP_unit
    
    if pLFP_unit=='$\mu$V':
        data['pLFP'] *= 1e3
        
    data['new_t'] = np.arange(len(data['pLFP']))*new_dt+data['t'][0]
    data['new_indices'] = np.arange(len(data['pLFP']))*isubsmpl # indices of data['t'] corresponding to data['new_t']
    data['new_Vm'] = data['Vm'][data['new_indices']] # subsampled Vm corresponding to the
    data['new_Extra'] = data['Extra'][data['new_indices']]
    
    data['new_dt'] = new_dt
    
