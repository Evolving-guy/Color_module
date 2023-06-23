import numpy as np
import imageio.v2 as imageio
import pandas as pd
import matplotlib.pyplot as plt
import colour

global df
df = pd.DataFrame([])


def func(t):
    if t > 0.008856:
        return np.power(t, 1 / 3.0)
    else:
        return 7.787 * t + 16 / 116.0


path = input("Enter the path to the image")  
pic = imageio.imread(path)

plt.figure(figsize=(15, 15))
plt.imshow(pic)

print("Shape of the image : {}".format(pic.shape))
print("Image Hight {}".format(pic.shape[0]))
print("Image Width {}".format(pic.shape[1]))

print("Image size {}".format(pic.size))
print("Maximum RGB value in this image {}".format(pic.max()))
print("Minimum RGB value in this image {}".format(pic.min()))


column = input("\n Enter the x coordinate of the pixel ")
row = input("\n Enter the y coordinate of the pixel ")

# Showing the RGB intensity values at the pixel
# pic[300,250] # pixel at row 300 column 50
print(
    "\nValue of only R channel {}".format(pic[int(row), int(column), 0])
)  # (imageio uses rgb so 0,1,2)
print("Value of only G channel {}".format(pic[int(row), int(column), 1]))
print("Value of only B channel {}".format(pic[int(row), int(column), 2]))

# CONVERSIONS
R = pic[int(row), int(column), 0]
G = pic[int(row), int(column), 1]
B = pic[int(row), int(column), 2]


r = R / 255
g = G / 255
b = B / 255
rgb = [r, g, b]
print("\n The rgb values are given by {}".format(rgb))

matrix = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
]

if r <= 0.04045:
    sr = r / 12.92
else:
    sr = np.power(((r + 0.055) / 1.055), 2.4)

if g <= 0.04045:
    sg = g / 12.92
else:
    sg = np.power(((g + 0.055) / 1.055), 2.4)

if b <= 0.04045:
    sb = b / 12.92
else:
    sb = np.power(((b + 0.055) / 1.055), 2.4)

srgb = [sr, sg, sb]

cie = np.dot(matrix, srgb)
XYZ = 100 * cie
print("\n The cie values are {}".format(100 * cie))

x0 = cie[0] / (cie[0] + cie[1] + cie[2])
y0 = cie[1] / (cie[0] + cie[1] + cie[2])

xy = [x0, y0]

print("\n The chromaticity values x and y are {} and {}".format(x0, y0))


cie[0] = cie[0] / 0.950456
# D65 illuminant values
cie[2] = cie[2] / 1.088754
# Calculate the L
L = 116 * func(cie[1]) - 16.0

# Calculate the a
a = 500 * (func(cie[0]) - func(cie[1]))

# Calculate the b
b = 200 * (func(cie[1]) - func(cie[2]))

#  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100
Lab = [b, a, L]

print("\n The Lab values are {}".format(Lab))

# xy and chromaticity diagram using colour-science package
xy_n = [0.31270000, 0.32900000]  # D65 xy values
d_wl = colour.dominant_wavelength(xy, xy_n)

print(
    "\n The dominant wavelength of the pixel color is {} nm \(-+2nm) \n".format(d_wl[0])
)


# Making the spectra.
wv = np.array(
    [
        380,
        400,
        420,
        440,
        460,
        480,
        500,
        520,
        540,
        560,
        580,
        600,
        620,
        640,
        660,
        680,
        700,
        720,
        740,
        760,
        780,
    ]
)

D65 = np.array(
    [
        50.0,
        82.8,
        93.4,
        104.9,
        117.8,
        115.9,
        109.4,
        104.8,
        104.4,
        100.0,
        95.8,
        90.0,
        87.7,
        83.7,
        80.2,
        78.3,
        71.6,
        61.6,
        75.1,
        46.4,
        63.4,
    ]
)

match_func = np.array(
    [
        [0.0014, 0, 0.0065],
        [0.0143, 0.0004, 0.0679],
        [0.1344, 0.004, 0.6456],
        [0.3483, 0.023, 1.7471],
        [0.2908, 0.06, 1.6692],
        [0.0956, 0.139, 0.813],
        [0.0049, 0.323, 0.272],
        [0.0633, 0.71, 0.0782],
        [0.2904, 0.954, 0.0203],
        [0.5945, 0.995, 0.0039],
        [0.9163, 0.87, 0.0017],
        [1.0622, 0.631, 0.0008],
        [0.8544, 0.381, 0.0002],
        [0.4479, 0.175, 0],
        [0.1649, 0.061, 0],
        [0.0468, 0.017, 0],
        [0.0114, 0.0041, 0],
        [0.0029, 0.001, 0],
        [0.0007, 0.0003, 0],
        [0.0002, 0.0001, 0],
        [0, 0, 0],
    ]
)

A = np.matrix.transpose(match_func)

s = np.dot(A, D65)
print(" scaling factor {} \n".format(s[1]))

c1 = np.array(s[1] * XYZ)

e = np.linalg.pinv(match_func)

f1 = np.dot(c1, e)

print(f1)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
plt.plot(wv, f1)

# Plotting chromaticity point
colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

x, y = xy
plt.plot(x, y, "o-", color="white")

plt.annotate(
    text="color of pixel",
    xy=xy,
    xytext=(-50, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
)

colour.plotting.render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True,
)


'''"""#Masking to show images with only r, only b and only g
fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=(15,5))

for c, ax in zip(range(3), ax):
    
    # create zero matrix
    split_img = np.zeros(pic.shape, dtype="uint8") # 'dtype' by default: 'numpy.float64'
    
    # assing each channel 
    split_img[ :, :, c] = pic[ :, :, c]
    
    # display each channel
    ax.imshow(split_img)"""'''

plt.show()
