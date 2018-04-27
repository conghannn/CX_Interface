#!/usr/bin/env python

"""
CX input generation script.
"""

import logging
import sys

import h5py
import numpy as np

import matplotlib as mpl
import matplotlib.animation as anim
import matplotlib.image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

try:
    from shutil import which
except:
    from shutilwhich import which

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

class PlaySavedSignal(anim.TimedAnimation):
    """
    Animate a sequence of frames saved in an HDF5 file.

    The stored signal must be stored in an order-3 tensor
    with dimensions `(number_of_frames, frame_rows, frame_cols)`.
    """

    def __init__(self, data, interval=100):
        fig = plt.figure()
        self.data = data
        self.ax = plt.imshow(self.data[0, :, :], cmap=cm.get_cmap('gray'))
        super(self.__class__, self).__init__(fig, interval=interval,
                                             blit=True, repeat=False)

    def _draw_frame(self, framedata):
        frame = self.data[framedata, :, :]
        self.ax.set_data(frame)

    def new_frame_seq(self):
        return iter(range(1, self.data.shape[0]))

    @classmethod
    def from_file(self, file_name, dataset='array', interval=100):
        f = h5py.File(file_name, 'r')
        data = f[dataset][:]
        return PlaySavedSignal(data, interval)

class PlayCXInputSignals(anim.TimedAnimation):
    """
    Animate CX input signals.
    """

    def __init__(self, input, bu_input_l, bu_input_r, pb_input, interval=100, step=1):
        # Frame data step:
        self.step = step

        assert bu_input_l.shape[0] == pb_input.shape[0]
        assert bu_input_r.shape[0] == pb_input.shape[0]
        assert bu_input_l.shape[1] == bu_input_r.shape[1]

        self.input = input

        self.bu_input_l = bu_input_l
        self.bu_input_r = bu_input_r
        self.pb_input = pb_input

        fig = plt.figure()
        self.ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        self.ax_im = self.ax.imshow(input[0],
                                    vmin=np.min(input), vmax=np.max(input),
                                    cmap=cm.get_cmap('gray'))
        self.ax.axis('tight')
        plt.title('Input')
 
        self.ax_bu_l = plt.subplot2grid((3, 2), (1, 0))
        self.ax_bu_l_im = self.ax_bu_l.imshow(bu_input_l[0],
                                              vmin=np.min(bu_input_l), vmax=np.max(bu_input_l),
                                              cmap=cm.get_cmap('gray'))
        self.ax_bu_l.axis('tight')
        plt.title('BU RF Response')

        self.ax_bu_r = plt.subplot2grid((3, 2), (1, 1))
        self.ax_bu_r_im = self.ax_bu_r.imshow(bu_input_r[0],
                                              vmin=np.min(bu_input_r), vmax=np.max(bu_input_r),
                                              cmap=cm.get_cmap('gray'))
        self.ax_bu_r.axis('tight')
        plt.title('bu RF Response')

        self.ax_pb = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.ax_pb_im = self.ax_pb.imshow(pb_input[0][np.newaxis],
                                          vmin=np.min(pb_input), vmax=np.max(pb_input),
                                          cmap=cm.get_cmap('gray'))
        self.ax_pb.axis('tight')
        plt.title('PB RF Response')

        plt.tight_layout()

        super(self.__class__, self).__init__(fig, interval=interval,
                                             blit=True, repeat=False)

    def _draw_frame(self, framedata):
        self.ax_im.set_data(self.input[framedata])

        self.ax_bu_l_im.set_data(self.bu_input_l[framedata])
        self.ax_bu_r_im.set_data(self.bu_input_r[framedata])
        self.ax_pb_im.set_data(self.pb_input[framedata][np.newaxis])

    def new_frame_seq(self):
        return iter(range(1, self.pb_input.shape[0], self.step))

    @classmethod
    def from_file(self, input_file, bu_input_l_file, bu_input_r_file, pb_input_file,
                  dataset='array', interval=100, step=1):
        f = h5py.File(input_file, 'r')
        input = f[dataset][:]
        f = h5py.File(bu_input_l_file, 'r')
        bu_input_l = f[dataset][:]
        f = h5py.File(bu_input_r_file, 'r')
        bu_input_r = f[dataset][:]
        f = h5py.File(pb_input_file, 'r')
        pb_input = f[dataset][:]
        return PlayCXInputSignals(input, bu_input_l, bu_input_r, pb_input,
                                  interval, step)

    def save(self, filename):
        if which('ffmpeg'):
            w = anim.FFMpegFileWriter()
        elif which('avconv'):
            w = anim.AVConvFileWriter()
        else:
            raise RuntimeError('avconv or ffmpeg required')
        super(self.__class__, self).save(filename, writer=w)

class CreateSignal(object):
    """
    Create a test video signal.
    """

    def __init__(self, shape, dt, dur):
        self.shape = shape
        self.dt = dt
        self.dur = dur
        self.N_t = int(self.dur/self.dt)

    def moving_bar_l2r(self, width):
        data = np.empty((self.N_t, self.shape[0], self.shape[1]), np.float64)
        for i in range(self.N_t):
            start = int(np.ceil(i*(self.shape[1]-width)/float(self.N_t)))
            frame = np.zeros(self.shape, dtype=np.float64)
            frame[:, start:start+width] = 1.0
            data[i, :, :] = frame
        return data

    def moving_bar_r2l(self, width):
        data = np.empty((self.N_t, self.shape[0], self.shape[1]), np.float64)
        for i in range(self.N_t):
            start = int(np.ceil(i*(self.shape[1]-width)/float(self.N_t)))
            frame = np.zeros(self.shape, dtype=np.float64)
            frame[:, start:start+width] = 1.0
            data[i, :, :] = frame
        return data[::-1]

    @classmethod
    def write(cls, data, file_name, dataset='array'):
        f = h5py.File(file_name, 'w')
        f.create_dataset(dataset, data.shape, data.dtype,
                         maxshape=(None,)+data.shape[1:])
        f[dataset][:] = data
        f.close()

class CircularGaussianFilterBank(object):
    """
    Create a bank of circular 2D Gaussian filters.

    Parameters
    ----------
    shape : tuple
        Image dimensions.
    sigma : float
        Parameter of Gaussian.
    n : int
        How many blocks should occupy the x-axis.
    """

    def __init__(self, shape, sigma, n):
        self.shape = shape
        self.sigma = sigma
        self.n = n

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))

        self.filters = self.create_filters(shape, sigma, n)

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_gaussian(cls, x, y, sigma):
        """
        2D Gaussian function.
        """

        return (1.0/(1*np.pi*(sigma**2)))*np.exp(-(1.0/(2*(sigma**2)))*(x**2+y**2))

    @classmethod
    def gaussian_mat(cls, shape, sigma, n_x_offset, n_y_offset, n):
        """
        Compute offset circular 2D Gaussian.
        """

        # Image dimensions in pixels:
        N_y, N_x = shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y)-(n_y_offset/float(n)))
        return cls.func_gaussian(X, Y, sigma)

    @classmethod
    def create_filters(cls, shape, sigma, n):
        """
        Create filter bank as order-4 tensor.
        """

        N_y, N_x = shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Compute how many blocks to use along the y-axis:
        m = n*N_y/N_x

        # Construct filters offset by the blocks:
        n_x_offsets = np.linspace(np.ceil(-n/2.0), np.floor(n/2.0), n)
        n_y_offsets = np.linspace(np.ceil(-m/2.0), np.floor(m/2.0), m)
        filters = np.empty((m, n, N_y, N_x), np.float64)
        for j, n_x_offset in enumerate(n_x_offsets):
            for i, n_y_offset in enumerate(n_y_offsets):
                filters[i, j] = cls.gaussian_mat(shape, sigma,
                                                 n_x_offset, n_y_offset, n)
        return filters

    def apply_filters(self, frame, normalize=True):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)
        else:
            return result

class RectangularFilterBank(object):
    """
    Create a bank of 2D rectangular filters that tile the x-axis.
    """

    def __init__(self, shape, n):
        self.shape = shape
        self.n = n

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))

        self.filters = self.create_filters(shape, n)

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_rect(cls, x, y, width):
        return np.logical_and(x > -width/2.0, x <= width/2.0).astype(np.float64)

    @classmethod
    def rect_mat(cls, shape, n_x_offset, n):
        N_y, N_x = shape

        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y))
        return cls.func_rect(X, Y, 1.0/n)
        
    @classmethod
    def create_filters(cls, shape, n):
        N_y, N_x = shape
                
        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Construct filters offset by the blocks:
        n_x_offsets = np.linspace(np.ceil(-n/2.0), np.floor(n/2.0), n)
        filters = np.empty((n, N_y, N_x), np.float64)

        for j, n_x_offset in enumerate(n_x_offsets):
            filters[j] = cls.rect_mat(shape, n_x_offset, n)                                                 
        return filters

    def apply_filters(self, frame, normalize=True):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)
        else:
            return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='l2r', type=str,
                        help='Direction [l2r, r2l]')
    args = parser.parse_args()

    logger.info('generating input video signal (%s)' % args.d)
    shape = (200, 500)
    dt = 1e-4
    dur = 0.2
    c = CreateSignal(shape, dt, dur)
    if args.d == 'l2r':
        data = c.moving_bar_l2r(50)
    elif args.d == 'r2l':
        data = c.moving_bar_r2l(50)
    else:
        raise RuntimeError('unsupported signal type')

    c.write(data, 'moving_bar.h5')

    logger.info('generating Gaussian RFs for BU')
    fc = CircularGaussianFilterBank((shape[0], shape[1]/2), 0.05, 10)

    logger.info('generating rectangular RFs for PB')
    fr = RectangularFilterBank(shape, 18)

    logger.info('filtering with Gaussian RFs')
    BU_input_pre = np.empty((len(data),)+fc.filters.shape[0:2])
    bu_input_pre = np.empty((len(data),)+fc.filters.shape[0:2])
    for i, frame in enumerate(data):
        BU_input_pre[i, :, :] = 3*fc.apply_filters(frame[:, :shape[1]/2])
        bu_input_pre[i, :, :] = 3*fc.apply_filters(frame[:, shape[1]/2:])

    logger.info('filtering with rectangular RFs')
    PB_input_pre = np.empty((len(data), fr.filters.shape[0]))
    for i, frame in enumerate(data):
        PB_input_pre[i, :] = 3*fr.apply_filters(frame)

    logger.info('saving RF responses')
    with h5py.File('BU_input_pre.h5', 'w') as f:
        f.create_dataset('/array', data=BU_input_pre)
    with h5py.File('bu_input_pre.h5', 'w') as f:
        f.create_dataset('/array', data=bu_input_pre)
    with h5py.File('PB_input_pre.h5', 'w') as f:
        f.create_dataset('/array', data=PB_input_pre)
