# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:18:58 2018

@author: admin
"""

#!/usr/bin/env python
#coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from ProcessSignal import Recording 
from PIL import Image


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    

def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

def Spec(out_image):
    samplerate, samples = wav.read(Recording("D:/voice-classification-master/scripts/demo/nga.wav"))

    s = stft(samples, 2**10)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    #print("timebins: ", timebins)
    #print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap="jet", interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*(2**10)))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    plt.savefig(out_image, bbox_inches="tight")
    
    return out_image

def CropImage(in_file, out_file):   
    image_obj = Image.open(in_file).convert("RGB") 
        #cropped_image = image_obj.crop((57, 60,  310, 545)) # crop image ( file 2s) 
        #cropped_image = image_obj.crop((57, 60,  159, 545))   # crop image ( file 1s)
    #cropped_image = image_obj.crop((76, 10,  747, 419))
    cropped_image = image_obj.crop((70, 10,  740, 419))
    #cropped_image = image_obj.crop((57, 60,  560, 545))
    #new_width  = 800
    #new_height = 513        
    #cropped_image  = cropped_image .resize((new_width, new_height), Image.ANTIALIAS)
    #cropped_image.show()
    cropped_image.save(out_file)
    return out_file

"""
in_file = "C:/Users/admin/Desktop/DA2/vivos/train/waves/VIVOSSPK03"
n =  len(fnmatch.filter(os.listdir(in_file), '*.wav')) 

for i in range(n):
   
    samplerate, samples = wav.read("C:/Users/admin/Desktop/DA2/vivos/train/waves/VIVOSSPK03/VIVOSSPK03_R0" +str(i) +".wav")
    
    s = stft(samples, 2**10)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    #print("timebins: ", timebins)
    #print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap="jet", interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*(2**10)))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    plt.savefig("D:/VIVOS/train/VIVOSSPK03/VIVOSSPK03_R0" +str(i) +".png", bbox_inches="tight")
"""