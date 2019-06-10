# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:37:06 2018

@author: admin
"""
import pyaudio
import wave
import os
from pydub import AudioSegment 
import fnmatch
from PIL import Image


def Recording(out_file):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 4
    audio = pyaudio.PyAudio()
    
    print("start recording")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print( "finished recording")
          
    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(out_file, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return out_file
    
    
def ReNameFile(in_file):
    files = os.listdir(in_file)
    i = 0   
    
    for file in files : 
         os.rename(os.path.join(in_file, file), os.path.join(in_file, 'VIVOSDEV19_R0'+ str(i)+'.wav'))
         i = i +1 

def CropAudio(in_file):
    n =  len(fnmatch.filter(os.listdir(in_file), '*.wav')) 
    for i in range(n) :
        in_audio =  AudioSegment.from_wav("D:/t1/s"+ str(i) + ".wav")       
        newAudio = in_audio[3420:]        
        newAudio.export("D:/t1/c"+ str(i) + ".wav", format="wav")
        
def CropAudio_2(in_file):
    n =  len(fnmatch.filter(os.listdir(in_file), '*.wav')) 
    sumAudio = 0 
    a = 2120
    b = 3400
    for i in range(n) :
        in_audio =  AudioSegment.from_wav("D:/add/s"+ str(i) + ".wav")       
        newAudio1 = in_audio[500:a]        
        sumAudio += newAudio1
        newAudio2 = in_audio[b:]
        sumAudio += newAudio2
    sumAudio.export("D:/ad/g12.wav", format="wav") 
    
def Convert_Mono(in_file):
    n =  len(fnmatch.filter(os.listdir(in_file), '*.wav'))
    for i in range(n): 
        sound = AudioSegment.from_wav("D:/nga/s" +str(i) +".wav")
        sound = sound.set_channels(1)
        sound.export("D:/1s/s"+ str(i) + ".wav", format="wav")

def SumAudio(in_file):
    n =  len(fnmatch.filter(os.listdir(in_file), '*.wav')) 
    sumAudio = 0 
    
    
    for i in range(n):              
        newAudio = AudioSegment.from_wav("D:/ad/s" +str(i) +".wav")
        newAudio = newAudio[0:]
        sumAudio += newAudio    
    sumAudio.export("D:/ad/g.wav", format="wav") 
        
#def CropImage(in_file):        
def CropImage(in_file, out_file):   
    image_obj = Image.open(in_file).convert("RGB") 
    #cropped_image = image_obj.crop((57, 60,  310, 545)) # crop image ( file 2s) 
    #cropped_image = image_obj.crop((57, 60,  159, 545)) # crop image ( file 1s)
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
    n =  len(fnmatch.filter(os.listdir(in_file), '*.png')) 
   
    for i in range(n) : 
        in_image =  "D:/sp10_img/s" +str(i) +".png"
        out_image ="D:/sp10/s" +str(i) +".png"
        image_obj = Image.open(in_image).convert("RGB") 
        #cropped_image = image_obj.crop((57, 60,  310, 545)) # crop image ( file 2s) 
        #cropped_image = image_obj.crop((57, 60,  159, 545))   # crop image ( file 1s)
        #cropped_image = image_obj.crop((70, 10,  740, 419))
        #cropped_image = image_obj.crop((57, 60,  560, 545))
        cropped_image = image_obj.crop((76, 10,  747, 419)) # 2s,3s
        #new_width  = 800
        #new_height = 513      
        #cropped_image  = cropped_image .resize((new_width, new_height), Image.ANTIALIAS)
        #cropped_image.show()
        cropped_image.save(out_image)
        
    """
#ReNameFile("C:/Users/admin/Desktop/DA2/vivos/test/waves/VIVOSDEV19")    
"""
from scipy import signal
from scipy.io import wavfile
filename = 'D:/ad/g12.wav'
new_sample_rate = 8000

sample_rate, samples = wavfile.read(str(filename) )
resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))   
wavfile.write('D:/ad/ggg.wav',new_sample_rate , resampled)  

"""
#n =  len(fnmatch.filter(os.listdir(in_file), '*.wav'))
#for i in range(n): 
#CropAudio('D:/t1')
#CropAudio('D:/t1')
#####CropAudio_2('D:/add')
#SumAudio('D:/ad')       
#CropImage("D:/sp10_img", "D:/sp10")
    
#CropAudio("D:/sp1_3s")    
#in_audio =  AudioSegment.from_wav("D:/Dowload/male5.wav")       
#newAudio = in_audio[120000:135000]
#newAudio.export("D:/male5.wav", format="wav")   
#ReNameFile("D:/cc1.wav")  
#Recording("D:/recog.wav") 
#CropImage("D:/thuyChoLon.png")    
#CropImage("D:/re/s0.png", "D:/re/nga2.png")