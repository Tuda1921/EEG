import serial
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# Close any existing open serial ports
filename = 'test.h5'

if serial.Serial:
    serial.Serial().close()

# Open the serial port
s = serial.Serial("COM5", baudrate=57600)
s_sound = serial.Serial("COM7", baudrate=57600)

def FeatureExtract(y):
    # y trong truong hop nay co do dai 15*512
    flm = 512
    L = len(y)
    Y = np.fft.fft(y)
    Y[0] = 0
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1] #hàm chứa giá tr của fft
    plt.plot(P1)
    plt.show()

    # Find the indices of the frequency values between 0.5 Hz and 4 Hz
    f1 = np.arange(len(P1)) * flm / len(P1) # co thằng x lại
    indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]
    delta = np.sum(P1[indices1])

    f1 = np.arange(len(P1)) * flm / len(P1)
    indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]
    theta = np.sum(P1[indices1])

    f1 = np.arange(len(P1)) * flm / len(P1)
    indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]
    alpha = np.sum(P1[indices1])

    f1 = np.arange(len(P1)) * flm / len(P1)
    indices1 = np.where((f1 >= 13) & (f1 <= 30))[0]
    beta = np.sum(P1[indices1])

    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (alpha + beta) / (delta + theta)
    dict = {"delta": delta,
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "abr": abr,
            "tbr": tbr,
            "dbr": dbr,
            "tar": tar,
            "dar": dar,
            "dtabr": dtabr
            }
    #print(dict)
    return dict


x = 0
y = []
z = []
feature = []
feature_names = ['delta', 'theta', 'alpha', 'beta', 'abr', 'tbr', 'dbr', 'tar', 'dar', 'dtabr']
sliding_window_start = 0
sliding_window_end = 0
k = 15 * 512
print("START!")
while x < (3600 * 512):
    noise = 0
    x += 1
    print(x)
    data = s.readline().decode('utf-8').rstrip("\r\n")  # strip removes leading/trailing whitespace
    if data:
        value = int(data)
    else:
        x -= 1
        continue
    if value < -256 and value > 256:
        x -= 1
        noise += 1
        continue
    y.append(value)
    # print(value)
        # with open("raw.txt", 'a') as f:
        #     f.writelines(value)

    if (x >= k):
        if (x % (1 * 512) == 0):
            # y = np.array(y)
            # y = y[~np.isnan(y)]
            print(x/(1*512))
            print("Noise", float(noise)/(15*512)*100)
            sliding_window_start = x - k
            sliding_window_end = x
            sliding_window = np.array(y[sliding_window_start:sliding_window_end])
            plt.plot(sliding_window)
            plt.show()

            # feature.append(FeatureExtract(sliding_window)) #abc
            model = pickle.load(open(filename, 'rb'))
            feature_test = np.array(list(FeatureExtract(sliding_window).values())).reshape(1,-1)
            print(feature_test)
            print(model.predict(feature_test))
            # print(FeatureExtract(y))
            s_sound.write(int(model.predict(feature_test)))   #output to sound
            print(int(model.predict(feature_test)))
            sliding_window = []

# Use to get data
# df = pd.DataFrame.from_dict(feature)
# df.to_csv("Banmoi.csv")

# Close the serial port
print("DONE")
s.close()