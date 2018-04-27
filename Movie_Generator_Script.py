import matplotlib
# #%matplotlib inline
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as colors
from matplotlib import animation
from IPython import display

import h5py

filename = 'BU_output_pick.h5'
f = h5py.File(filename, 'r')
data_BU = f['array'][:]

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

# TODO: initialize the 2D scene here
data = np.random.random((8,10))

im = ax.imshow(
    data,
    aspect='auto',
    interpolation="none",
    cmap = cm.coolwarm,
    vmin = 0,
    vmax = 1 # TODO: setup the range of your data accrodingly
)
ax.set_title('Noise Flicker')
ax.set_ylabel('y')
ax.set_xlabel('x')

cbar = fig.colorbar(im, ax = ax)
cbar.ax.set_ylabel('Amplitude')

plt.tight_layout()

def init():
    im.set_data(data)
    return im,

def animate(i):
    # TODO: update your data here, ex. p = mapRetinaOutputTo2D(...)
    p = np.zeros((8, 10))

    data = data_BU[i]

    for j in range(80):
        y_idx = j % 8
        x_idx = j / 8
        p[y_idx][x_idx] = data[j]

    im.set_data(p)
    return im,



num_frames = 2000
for i in xrange(num_frames):
    animate(i)
    display.clear_output(wait=True)
    display.display(fig)

anim = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=num_frames, blit=True
)

# TODO: save/encode the movie
anim.save('moving_bar_BU.mp4', dpi=300, fps=40, extra_args=['-vcodec', 'mpeg4'])

