import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
import pickle
import serial

filename = 'test.h5'
#
# if serial.Serial:
#     serial.Serial().close()
#
# # Open the serial port
# s = serial.Serial("COM5", baudrate=57600)
# ĐỌC VÀ XỬ LÝ THÔNG TIN
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
    P1[1:-1] = 2 * P1[1:-1]
    save_image(2,P1)
    # Find the indices of the frequency values between 0.5 Hz and 4 Hz
    f1 = np.arange(len(P1)) * flm / len(P1)
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


# GIAO DIỆN
image_path1 = 'image1.png'
image_path2 = 'image2.png'

def save_image(label, data):
    if label == 1:
        plt.figure
        plt.plot(data)
        plt.xlabel('Raw')
        plt.savefig(image_path1)
    else:
        plt.figure
        plt.plot(data)
        plt.xlabel('FFT')
        plt.savefig(image_path2)
    plt.close()

isLoading = True

def start_test():
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
                print(x / (1 * 512))
                print("Noise", float(noise) / (15 * 512) * 100)
                sliding_window_start = x - k
                sliding_window_end = x
                sliding_window = np.array(y[sliding_window_start:sliding_window_end])
                save_image(1, sliding_window)

                # feature.append(FeatureExtract(sliding_window)) #abc
                model = pickle.load(open(filename, 'rb'))
                feature_test = np.array(list(FeatureExtract(sliding_window).values())).reshape(1, -1)
                print(feature_test)
                print(model.predict(feature_test))
                # print(FeatureExtract(y))
                s_sound.write(int(model.predict(feature_test)))  # output to sound
                print(int(model.predict(feature_test)))
                # show_image()
                sliding_window = []

def show_image():
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    photo1 = ImageTk.PhotoImage(image1)
    photo2 = ImageTk.PhotoImage(image2)
    image_frame1.configure(image=photo1)
    image_frame2.configure(image=photo2)
    window.update()
    image1.close()
    image2.close()
def start():

    while True:
        if isPause.get() == 0:
            data1 = np.random.rand(100)
            data2 = np.random.rand(100)
            save_image(1, data1)
            save_image(2, data2)
            show_image()
            time.sleep(0.01)
        elif isPause.get() == 1:
            show_image()



window = tk.Tk()

window.title("Awake Drive")


isPause = tk.IntVar()
pause_button = tk.Checkbutton(window, text="Dừng", variable=isPause, onvalue=1, offvalue=0)
pause_button.pack(pady=10)

start_button = tk.Button(window, text="Bắt đầu", command=start)
start_button.pack(pady=10)
# Tạo hai khung hình ảnh

# Raw wave
image_frame1 = tk.Label(window, width=400, height=300, bg="white")
image_frame1.pack(side=tk.LEFT, padx=10)

# FFT wave
image_frame2 = tk.Label(window, width=400, height=300, bg="white")
image_frame2.pack(side=tk.LEFT, padx=10)
show_image()
# Tạo một khung văn bản để hiển thị trạng thái buồn ngủ/tỉnh táo
status_label = tk.Label(window, text="Trạng thái", font=("Arial", 14))
status_label.pack(pady=10)
state_label = tk.Label(window, text="", font=("Arial", 20))
state_label.pack(pady=10)


# Chạy vòng lặp chính của giao diện
window.mainloop()