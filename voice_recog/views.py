#from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, render_to_response
import pandas as pd
import numpy as np
import sklearn.neural_network as NN
from sklearn.model_selection import RandomizedSearchCV
from scipy.io import wavfile
import struct
import wave
from scipy.io import wavfile
import struct
import wave
import numpy as np
from praatio import tgio

clf = NN.MLPClassifier()

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Encode "+str(target_column)] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

def pre_process(data_file, target):
    x = data_file.loc[:, data_file.columns != target]
    for i in x.columns:
        if isinstance(x.at[0, i], str):
            x, col = encode_target(x, i)
            x = x.loc[:, x.columns != i]
    return x

x_test_1 = pd.DataFrame()

def process(request):
    data_file = pd.read_csv("C:/wamp/Hola10/Dataset.csv", sep = ',')
    target = "status"
    x = data_file.fillna(0)
    x = pre_process(x, target)
    y = data_file[str(target)]
    params = {'random_state': np.arange(1, 100, 5), 'hidden_layer_sizes': np.arange(50, 100, 2), 'alpha': np.arange(0.1, 1, 0.1), 'max_iter': np.arange(90, 100, 1)}
    grid_search = RandomizedSearchCV(clf, params)
    grid_search.fit(x, y)
    #print(x.shape)
    clf.fit(x, y)
    context = {}
    x_test_1 = pd.read_csv("C:/wamp/Hola10/Dataset.csv", sep = ',')#pd.read_csv("test.csv", sep=',', header = None)
    x_test_1 = data_file.fillna(0)
    x_test_1 = pre_process(x, target)
    x_test_1 = x_test_1.loc[:, x_test_1.columns != target]
    #print(x_test_1)
    context['y_pred'] = clf.predict(x_test_1)
    #print(context['y_pred'])
    return render_to_response("index.html", context, request)



sound =[]
windowSize = 2205

def isSilence(windowPosition):
    sumVal = sum( [ x*x for x in sound[windowPosition:windowPosition+windowSize+1] ] )
    avg = sumVal/(windowSize)
    if avg <= 0.0001:
        return True
    else:
        return False

def calculateJitterRatio(data):
    n = len(data)
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i>0:
            sum1 += abs(data[i-1]-data[i])
        sum2 += data[i]
    sum1 /= float(n-1)
    sum2 /= float(n)
    return float(sum1)/sum2 /1000

def calculateJitterPercent(data):
    return calculateJitterRatio(data)*10.0

def calculateRelativeAveragePerturbation(data):
    n = len(data)
    if n < 3:
        raise Exception("need at least three data points")
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i>0 and i<n-1:
            sum1 += abs((data[i-1] + data[i] + data[i+1]) / 3 - data[i])
        sum2 += data[i]
    sum1 /= float(n-2)
    sum2 /= float(n)
    return sum1 / sum2 / 100

def calculatePPQ(data):
    n = len(data)
    if n < 5:
        raise Exception("need at least three data points")
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i>1 and i<n-2:
            sum1 += (abs(data[i] - (data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2]))/5)
        sum2 += data[i]
    sum1 /= float(n-4)
    sum2 /= float(n)
    return float(sum1)/sum2 /100

def calculateDDP(data):
    n = len(data)
    if n < 3:
        raise Exception("need at least two data points")
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i > 0 and i < n - 1:
            sum1 += abs(((data[i + 1] - data[i]) - (data[i] + data[i - 1])))
        sum2 += data[i]
    sum1 /= float(n - 2)
    sum2 /= float(n)
    return sum1 / sum2 / 100



#read from wav file
def main_point(request):
    sound_file = wave.open('C:/Users/Manthan/Downloads/myRecording01', 'r')
    file_length = sound_file.getnframes()
    #data = sound_file.readframes(file_length)
    #sound_file.close()
    #data = struct.unpack("<h", data)
    #data = struct.unpack('{n}h'.format(n=file_length), data)
    #sound = np.array(data)
    fmts = (None, "=B", "=h", None, "=l")
    fmt = fmts[4]#sound_file.getsampwidth()]
    dcs  = (None, 128, 0, None, 0)
    dc = dcs[sound_file.getsampwidth()]
    for i in range(file_length-4):
        data = sound_file.readframes(1)
        #print(data)
        data = struct.unpack(fmt, data)[0]
        sound.append(data)
        data -= dc
    sound_file.close()
    #sound is now a list of values

    #detect silence and notes
    print(len(sound))
    i=0
    windowPosition = 0
    listOfLists = []
    listOfLists.append([])
    maxVal = len(sound) - windowSize
    while True:
        if windowPosition >= maxVal:
            break
        if not isSilence(windowPosition):
            #while not isSilence(windowPosition):
            for v in sound[windowPosition:windowPosition+windowSize+1]:
               listOfLists[i].append(v)
            windowPosition += windowSize
            listOfLists.append([]) #empty list
            i += 1
        #windowPosition += windowSize

    #print("List of lists:"+str(listOfLists))

    frequencies = []
    #Calculating the frequency of each detected note by using DFT
    print(len(listOfLists))
    i = 0
    for signal in listOfLists:
        #print("i:"+str(i))
        if not signal:
            break
        w = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(w))
        #print(freqs)
        l = len(signal)

        #imax = index of first peak in w
        imax = np.argmax(np.abs(w))
        fs = freqs[imax]

        i = i+1

        freq = imax*fs/l
        frequencies.append(freq)

    #print (frequencies)
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    avg_freq = sum(frequencies)/float(len(frequencies))

    jitt = []
    for f in frequencies:
        jitt.append(float(1/f))

    jitterPercent = calculateJitterPercent(jitt)
    jitterRatio = calculateJitterRatio(jitt)
    jitterRAP = calculateRelativeAveragePerturbation(jitt)
    jitterPPQ = calculatePPQ(jitt)
    jitterDDP = calculateDDP(jitt)
    print(str(jitterPercent)+" "+str(jitterRatio)+" "+str(jitterRAP)+" "+str(jitterPPQ)+" "+str(jitterDDP))
    target = "status"
    data_file = pd.read_csv("C:/wamp/Hola10/Dataset.csv",
                           sep=',')  # pd.read_csv("test.csv", sep=',', header = None)
    x_test_1 = data_file.fillna(0)
    #x_test_1 = pre_process(x_test_1, target)
    x_test_1 = x_test_1.loc[:, x_test_1.columns != target]

    my_dict = {"name": "user1", "MDVP:Fo(Hz)":avg_freq, "MDVP:Fhi(Hz)":max_freq, "MDVP:Flo(Hz)":min_freq, "MDVP:Jitter(%)":jitterPercent, \
               "MDVP:Jitter(Abs)":jitterRatio, "MDVP:RAP":jitterRAP, "MDVP:PPQ":jitterPPQ, "Jitter:DDP":jitterDDP}
    my_list =[]
    for x in x_test_1.columns.values:
        if x != target:
            my_list.append(x)
    my_test = pd.DataFrame(data=my_dict, index = [0], columns = my_list).fillna(0)
    my_test = pre_process(my_test, target)
    print(my_test)
    y_pred = clf.predict(my_test)
    print(y_pred)
    context = {}
    return render_to_response("index.html", context, request)