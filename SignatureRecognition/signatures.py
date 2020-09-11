import numpy as np
from scipy.misc import imread,imsave
import matplotlib.cm as cm
#%matplotlib inline
import matplotlib.pyplot as plt



plt.rcParams['axes.labelsize'] = 14

andysign = imread('./signatures/andysign.png',1)

(w,h) = andysign.shape


sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobely = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
convolutedx = np.empty_like(andysign)
convolutedy = np.empty_like(andysign)
convolutedhat = np.empty_like(andysign)
convoluted = np.empty_like(andysign)


#slow unoptimzed code
for x in range(1,w-1):
    for y in range(2,h-1):
        convolutedx[x,y] = sobelx[0,0]* andysign[x-1,y-1] + sobelx[0,1]* andysign[x-1,y] + sobelx[0,2]* andysign[x-1,y+1] + sobelx[1,0]* andysign[x,y-1] + sobelx[1,1]* andysign[x,y] + sobelx[1,2]* andysign[x,y+1] + sobelx[2,0]* andysign[x+1,y-1] + sobelx[2,1]* andysign[x+1,y]+ sobelx[2,2]* andysign[x+1,y+1]
        convolutedy[x,y] = sobely[0,0]* andysign[x-1,y-1] + sobely[0,1]* andysign[x-1,y] + sobely[0,2]* andysign[x-1,y+1] + sobely[1,0]* andysign[x,y-1] + sobely[1,1]* andysign[x,y] + sobely[1,2]* andysign[x,y+1] + sobely[2,0]* andysign[x+1,y-1] + sobely[2,1]* andysign[x+1,y]+ sobely[2,2]* andysign[x+1,y+1]



#fix the overflows
convolutedx[convolutedx>255]=255
convolutedx[convolutedx<0]=0
convolutedy[convolutedy>255]=255
convolutedy[convolutedy<0]=0

#combine
convolutedhat = convolutedx + convolutedy
convoluted = np.sqrt(np.power(convolutedx,2)+np.power(convolutedy,2))


#fix the overflows
convolutedhat[convolutedhat>255]=255
convolutedhat[convolutedhat<0]=0

#fix the overflows
convoluted[convoluted>255]=255
convoluted[convoluted<0]=0

#show the image
plt.subplot(131).set_axis_off()
plt.imshow(andysign, cmap=plt.cm.gray)
plt.title('Input image')

#show the result
plt.subplot(132).set_axis_off()
plt.imshow(convolutedhat, cmap=plt.cm.gray)
plt.title('using ehat')
plt.show()

plt.rcParams['axes.labelsize'] = 14

andyssign1 = imread('./signatures/andyssign1.png', 1)


(w, h) = andyssign1.shape


sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobely = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
convolutedx = np.empty_like(andyssign1)
convolutedy = np.empty_like(andyssign1)
convolutedhat = np.empty_like(andyssign1)
convoluted = np.empty_like(andyssign1)


# slow unoptimzed code
for x in range(1, w - 1):
    for y in range(2, h - 1):

        convolutedx[x, y] = sobelx[0, 0] * andyssign1[x - 1, y - 1] + sobelx[0, 1] * andyssign1[x - 1, y] + sobelx[
            0, 2] * andyssign1[x - 1, y + 1] + sobelx[1, 0] * andyssign1[x, y - 1] + sobelx[1, 1] * andyssign1[x, y] + \
                            sobelx[1, 2] * andyssign1[x, y + 1] + sobelx[2, 0] * andyssign1[x + 1, y - 1] + sobelx[
                                2, 1] * andyssign1[x + 1, y] + sobelx[2, 2] * andyssign1[x + 1, y + 1]
        convolutedy[x, y] = sobely[0, 0] * andyssign1[x - 1, y - 1] + sobely[0, 1] * andyssign1[x - 1, y] + sobely[
            0, 2] * andyssign1[x - 1, y + 1] + sobely[1, 0] * andyssign1[x, y - 1] + sobely[1, 1] * andyssign1[x, y] + \
                            sobely[1, 2] * andyssign1[x, y + 1] + sobely[2, 0] * andyssign1[x + 1, y - 1] + sobely[
                                2, 1] * andyssign1[x + 1, y] + sobely[2, 2] * andyssign1[x + 1, y + 1]


# fix the overflows
convolutedx[convolutedx > 255] = 255
convolutedx[convolutedx < 0] = 0
convolutedy[convolutedy > 255] = 255
convolutedy[convolutedy < 0] = 0

# combine
convolutedhat = convolutedx + convolutedy
convoluted = np.sqrt(np.power(convolutedx, 2) + np.power(convolutedy, 2))

# fix the overflows
convolutedhat[convolutedhat > 255] = 255
convolutedhat[convolutedhat < 0] = 0

# fix the overflows
convoluted[convoluted > 255] = 255
convoluted[convoluted < 0] = 0

# show the image
plt.subplot(131).set_axis_off()
plt.imshow(andysign, cmap=plt.cm.gray)
plt.title('Input image')

# show the result
plt.subplot(132).set_axis_off()
plt.imshow(convolutedhat, cmap=plt.cm.gray)
plt.title('using ehat')
plt.show()


plt.rcParams['axes.labelsize'] = 14
asign = imread('./signatures/asign.png', 1)

(w, h) = asign.shape

sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobely = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
convolutedx = np.empty_like(asign)
convolutedy = np.empty_like(asign)
convolutedhat = np.empty_like(asign)
convoluted = np.empty_like(asign)


# slow unoptimzed code
for x in range(1, w - 1):
    for y in range(2, h - 1):

        convolutedx[x, y] = sobelx[0, 0] * asign[x - 1, y - 1] + sobelx[0, 1] * asign[x - 1, y] + sobelx[
            0, 2] * asign[x - 1, y + 1] + sobelx[1, 0] * asign[x, y - 1] + sobelx[1, 1] * asign[x, y] + \
                            sobelx[1, 2] * asign[x, y + 1] + sobelx[2, 0] * asign[x + 1, y - 1] + sobelx[
                                2, 1] * asign[x + 1, y] + sobelx[2, 2] * asign[x + 1, y + 1]
        convolutedy[x, y] = sobely[0, 0] * asign[x - 1, y - 1] + sobely[0, 1] * asign[x - 1, y] + sobely[
            0, 2] * asign[x - 1, y + 1] + sobely[1, 0] * asign[x, y - 1] + sobely[1, 1] * asign[x, y] + \
                            sobely[1, 2] * asign[x, y + 1] + sobely[2, 0] * asign[x + 1, y - 1] + sobely[
                                2, 1] * asign[x + 1, y] + sobely[2, 2] * asign[x + 1, y + 1]

# fix the overflows
convolutedx[convolutedx > 255] = 255
convolutedx[convolutedx < 0] = 0
convolutedy[convolutedy > 255] = 255
convolutedy[convolutedy < 0] = 0

# combine
convolutedhat = convolutedx + convolutedy
convoluted = np.sqrt(np.power(convolutedx, 2) + np.power(convolutedy, 2))

# fix the overflows
convolutedhat[convolutedhat > 255] = 255
convolutedhat[convolutedhat < 0] = 0

# fix the overflows
convoluted[convoluted > 255] = 255
convoluted[convoluted < 0] = 0

# show the image
plt.subplot(131).set_axis_off()
plt.imshow(andysign, cmap=plt.cm.gray)
plt.title('Input image')

# show the result
plt.subplot(132).set_axis_off()
plt.imshow(convolutedhat, cmap=plt.cm.gray)
plt.title('using ehat')
plt.show()

