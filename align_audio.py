import numpy as np
import matplotlib.pyplot as plt
import librosa  # soundfile access library - see https://github.com/bmcfee/librosa
import librosa.display
import dpcore   # the current library

from os import listdir
from os.path import isfile, join

def align_wav(dir1, dir2):
    filelist = [f for f in listdir(dir1) if isfile(join(dir1, f))]

    for i in range(len(filelist)):
        d1, sr = librosa.load(dir1+'/'+filelist[i], sr=None)
        d2, sr = librosa.load(dir2+'/'+filelist[i], sr=None)
        # Calculate their short-time Fourier transforms
        D1C = librosa.stft(d1, n_fft=512, hop_length=128)
        D2C = librosa.stft(d2, n_fft=512, hop_length=128)
        # We'll use the magnitudes to calculate similarity (ignore phase)
        D1 = D1C#np.abs(D1C)
        D2 = D2C#np.abs(D2C)

        SM = np.array([[np.sum(a*b)/np.sqrt(np.sum(a**2)*np.sum(b**2)) for b in D2.T] for a in D1.T])

        localcost = np.array(1.0-SM, order='C', dtype=float)
        p, q, C, phi = dpcore.dp(localcost, penalty=0.1)

        iD1 = librosa.istft(D1[:, p], hop_length=128)
        iD2 = librosa.istft(D2[:, q], hop_length=128)
        print iD1.shape, iD2.shape
        librosa.output.write_wav('aligned_audio_data/SF2/'+filelist[i], iD1, sr)
        librosa.output.write_wav('aligned_audio_data/TF2/'+filelist[i], iD2, sr)
        
        
if __name__=="__main__":
    align_wav('/home/jph/fall_2017/682/project/DS_10283_2211/vcc2016_training/SF2', '/home/jph/fall_2017/682/project/DS_10283_2211/vcc2016_training/TF2')
  