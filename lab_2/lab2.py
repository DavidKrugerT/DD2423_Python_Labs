# %%
import requests
import numpy as np
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt
from Functions import *

## hard coded derivaties in x and y axis
## dxmask = convovle (gaus and derivate)  = con on dxmaks and image

# %% Functions
#
# def gauss(image, s):
#     Xsize, Ysize = image.shape
#     gaussianKernel = np.zeros([Xsize, Ysize])
#     for x in range(Xsize):
#         for y in range(Ysize):
#             gaussianKernel[x][y] = 1 / (2 * np.pi * s) * np.exp(-((x-Xsize/2) ** 2 + (y-Ysize/2) ** 2) / (2 * s))
#     return gaussianKernel

## derivative in x-axis - shows all the edges in x-axis
# def deltax(image, h):
#     Xsize, Ysize = image.shape
#     deltaX = np.zeros([Xsize-(2*h), Ysize-(2*h)])
#     for x in range(Xsize):
#         for y in range(Ysize):
#             if x-h < 0 or x+h > Xsize-(2*h) or y-h < 0 or y+h > Ysize-(2*h):
#                 continue
#             deltaX[x][y] = (image[x+h][y]-image[x-h][y])/(2*h)
#     return deltaX
#
# def deltay(image, h):
#     Xsize, Ysize = image.shape
#     deltaY = np.zeros([Xsize-(2*h), Ysize-(2*h)])
#     for x in range(Xsize):
#         for y in range(Ysize):
#             if x-h < 0 or x+h > Xsize-(2*h) or y-h < 0 or y+h > Ysize-(2*h):
#                 continue
#             deltaY[x][y] = (image[x][y+h]-image[x][y-h])/(2*h)
#     return deltaY

#%% Load derivative masks
def Dymask():
    return np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1/2, 0, -1/2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

def Dxmask():
    return Dymask().T

def Dyymask():
    return np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

def Dxxmask():
    return Dyymask().T

# %% Load image
def Deltax():
    return Deltay().T

def Deltay():
    return np.array([[0, 0, 0],
                    [1/2, 0, -1/2],
                    [0, 0, 0]])


def Deltaxx():
    return Deltayy().T

def Deltayy():
    return np.array([[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]])


def Lx(image,shape = 'same'):
    #conv1 = discgaussfft(image, s)
    return convolve2d(image, Deltax(), shape=shape)

def Ly(image, shape = 'same'):
    #conv1 = discgaussfft(image, s)
    return convolve2d(image, Deltay(), shape=shape)

def Lv(inpic, shape='same'):
    lx = Lx(inpic, shape)
    ly = Ly(inpic, shape)
    return np.sqrt(lx**2 + ly**2)

def Lvv(inpic, shape='same'):
    conv1 = inpic
    lx = convolve2d(conv1, Dxmask(), shape=shape)
    ly = convolve2d(conv1, Dymask(), shape=shape)
    lxy = convolve2d(lx, ly, shape=shape)
    lxx = convolve2d(conv1,Dxxmask(), shape=shape)
    lyy = convolve2d(conv1, Dyymask(), shape=shape)
    lvv = lx**2*lxx+2*lx*ly*lxy+ly**2*lyy
    return lvv
    # TODO

def Lvvv(inpic, shape='same'):
    conv1 = inpic
    lx = convolve2d(conv1, Deltax(), shape=shape)
    ly = convolve2d(conv1, Deltay(), shape=shape)
    lxy = convolve2d(lx, ly, shape=shape)
    lxx = convolve2d(conv1, Deltaxx(), shape=shape)
    lyy = convolve2d(conv1, Deltayy(), shape=shape)
    lxxx = convolve2d(lx, lxx, shape=shape)
    lyyy = convolve2d(ly, lyy, shape=shape)
    lxxy = convolve2d(lxx, ly, shape=shape)
    lxyy = convolve2d(lx, lyy, shape=shape)
    lvvv = lx**3*lxxx+3*lx**2*ly*lxxy+3*lx*ly**2*lxyy+ly**3*lyyy
    return lvvv

def Lvvtilde(inpic, shape='same'):
    return Lv(inpic, shape='same')**2*Lvv(inpic, s, shape='same')

def Lvvvtilde(inpic, shape='same'):
    return Lv(inpic, shape='same')**3*Lvvv(inpic, s, shape='same')

# %% Gauss
s = 3
tools = np.load("Images-npy/few256.npy")
showgrey(tools)
conv1 = discgaussfft(tools, s)
showgrey(conv1)
plt.show()
#%% derivativ with and without gauss
dxtools = convolve2d(tools, Deltax(), 'valid')
dytools = convolve2d(tools, Deltay(), 'valid')
showgrey(dxtools)
showgrey(dytools)
showgrey(Lx(tools, s))
showgrey(Ly(tools, s))
# %% Histogram
s=3
mag = Lv(tools, s)
plt.hist(mag.flatten(), bins=50, range=(0, 20))
showgrey((mag > 8).astype(int))
plt.show()


# %%
dxm = Dxmask()
dxxm = Dxxmask()
[x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
dxxxm = convolve2d(dxm, dxxm, 'same')
dxxymask = convolve2d(Dxxmask(), Dymask(), 'same')
print(convolve2d(x**3,dxxxm, 'valid'))
print(convolve2d(x**3, Dxxmask(), 'valid'))
print(convolve2d(x**2*y, dxxymask, 'valid'))
#%%
house = np.load("Images-npy/godthem256.npy")
scale = 3
print(Lvvtilde(discgaussfft(house, scale)))
showgrey(contour(Lvvtilde(discgaussfft(house, scale))))












#
# showgrey(conv1)
# dx = deltax()
# dxtools = convolve2d(conv1, dx)
# showgrey(dxtools)
#
# #%% deltaX
# dxtools = deltax(conv1, s)
# showgrey(dxtools)
# plt.show()
# #%% deltaY
# dytools = deltay(conv1, s)
# showgrey(dytools)
# plt.show()
# #%%
# # the edge strength in the image as magnitude.
# gradmagntools = np.sqrt(dxtools**2 + dytools**2)
graph = np.histogram(gradmagntools)
print(gradmagntools)
showgrey((gradmagntools > 3).astype(int))
#
# #%%
