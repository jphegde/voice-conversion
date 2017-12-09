from signal_processing import *


def mfcc(filename):
    ### Parameters ###
    fft_size = 2048 # window size for the FFT
    step_size = fft_size/16 # distance to slide along the window (in time)
    spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
    lowcut = 0 # Hz # Low cut for our butter bandpass filter
    highcut = 8000 # Hz # High cut for our butter bandpass filter # was 15000
    # For mels
    n_mel_freq_components = 100 # number of mel frequency channels
    shorten_factor = 10 # how much should we compress the x-axis (time)
    start_freq = 300 # Hz # What frequency to start sampling our melS from 
    end_freq = 8000 # Hz # What frequency to stop sampling our melS from 
    
    rate, data = wavfile.read(filename)
    data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
    # Only use a short clip for our demo
    if np.shape(data)[0]/float(rate) > 10:
        data = data[0:rate*10] 
    print ('Length in time (s): ', np.shape(data)[0]/float(rate))
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size, step_size = step_size, log = True, thresh = spec_thresh)
    mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)
    
    return mel_spec, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate

def imfcc(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor):
    return mel_to_spectrogram(mel_spec, mel_inversion_filter,
                                                spec_thresh=spec_thresh,
                                                shorten_factor=shorten_factor)