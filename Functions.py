# imports

import cPickle as pickle
from numpy import linspace, exp, uint16, uint32
from numpy import where, NaN
from numpy.random import randint
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import measurements as meas
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from matplotlib.pyplot import ion, ioff, close, imshow, cm
from IPython.display import display
import plotly as plotly


class SPring8_image(object):

    '''
    image class that reads from decetor, ccd, pilatus

    ---
    list of attributes
    file_name = name of the file read
    time = time stamp in second
    ROI = the region of interest selected
    ----
    '''

    # Gaussian function

    def __init__(self, file_name, ROI=None):
        self.file_name = file_name
        self.ROI = ROI

    def read_detector(self, name=1):
        '''
        Read the detector image and return a dictionary with the
        information contained
        in the header and the image as a numpy array.
        name can be 1 for ccd or 2 for pilatus
        TO - Do add angles from the diffractometers
        '''
        self.header_dict = {}
        if name == 1:
            byte = 2048
            res = 512
            user_dtype = np.uint16
        if name == 2:
            byte = 4096
            res = [487, 195]  # rect
            user_dtype = np.uint32
        # if you only wanted to read 512 bytes, do .read(512)
        in_file = open(self.file_name, 'rb')
        header_file = in_file.read(byte)
        image = in_file.read()  # read the rest of image
        in_file.close()  # save memory
        # convert the  binary image to integers
        if name == 1:
            self.img = np.fromstring(
                image, dtype=user_dtype, count=res * res).reshape(res, res)
        if name == 2:
            self.img = np.fromstring(
                image, dtype=user_dtype).reshape(res[1], res[0])
        self.img = median_filter(self.img, 3)  # perform median filter 3x3
        header = header_file.split('\n')
        try:
            for i in range(len(header)):
                if '#L' in header[i]:
                    for k, l in zip(filter(None, header[i][2:].split(' ')),  # not robust way to filter out the comment on the first line
                                    filter(None, header[i + 1].split(' '))):
                        self.header_dict[k] = l
                if '#V1' in header[i]:
                    # first presssure saved
                    self.NIG = float(header[i].split(' ')[3])
                    self.Tc = float(header[i].split(' ')[6])  # silly detail
                   # self.Tpyro = float(header[i].split(' ')[9])  # silly
                   # detail
                if '#C3' in header[i]:
                    self.c3 = header[i]
                if '#V2' in header[i]:
                    if header[i].split(' ')[2] == 'shutter':
                        # save iformation on the shutters
                        self.Ga = int(header[i].split(' ')[3])
                        self.In = int(header[i].split(' ')[4])
                        self.Au = int(header[i].split(' ')[5])
                        self.Sb = int(header[i].split(' ')[6])
                        self.As = int(header[i].split(' ')[7])
                    # sometimes the reading just fucks up
                    if header[i].split(' ')[2] == 'graphtech':
                        self.NIG = float(header[i].split(' ')[3])
                        # silly detail
                        self.Tpyro = float(header[i].split(' ')[4])
                if '#L' in header[i]:
                    self.time = float(header[i + 1].split(' ')[5])
                    self.monitor = float(header[i + 1].split(' ')[-2])
        except Exception as e:
            print e

    def show(self, log=False, size_fig=(10, 5)):
        'sowh detecotr image, optioanlly in log scale'
        fig, ax = plt.subplots(figsize=size_fig)
        ax.grid(color='white')
        if log:
            imshow(np.log(self.img), cmap=cm.Oranges)
        if not log:
            imshow(self.img, cmap=cm.Oranges)
        plt.colorbar()
        return fig, ax

    def select_ROI(self, img, display_plot=False, ROI=None, log=False):
        '''
        takes single image as input and ask to select rectangular roi
        '''
        self.ROI = ROI
        try:
            ioff()
        except Exception, e:
            print e
        fig, ax = plt.subplots()
        ax.grid(color='black')
        if type(self.ROI) == list:
            x_1, x_2, y_1, y_2 = self.ROI
            ax.grid(color='black')
            img_intrest = img[y_1:y_2, x_1:x_2]
            if display_plot:
                imshow(img_intrest)  # (yy,xx)
            if display_plot:
                display(fig)
        else:
            if display_plot:
                if log:
                    imshow(np.log(self.img), cmap=cm.Oranges)
                if not log:
                    imshow(self.img, cmap=cm.Oranges)
            if display_plot:
                try:
                    display(fig)
                except Exception, e:
                    print e
            cond = False
            while not cond:
                x_1 = float(raw_input('x_1:'))
                x_2 = float(raw_input('x_2:'))
                y_1 = float(raw_input('y_1:'))
                y_2 = float(raw_input('y_2:'))
                img_intrest = img[y_1:y_2, x_1:x_2]
                if display_plot:
                    if log:
                        imshow(np.log(img_intrest), cmap=cm.Oranges)
                    if not log:
                        imshow(img_intrest, cmap=cm.Oranges)
                if display_plot:
                    try:
                        display(fig)
                    except Exception, e:
                        print e
                check = str(raw_input('return if ok,space to redo'))
                if len(check) < 1:
                    cond = True
        try:
            close('all')
            ion()
        except Exception, e:
            print e

        self.ROI = [x_1, x_2, y_1, y_2]
        self.img_roi = img_intrest

    def analyze_img(self, std=15, plot_me=False, fit=True, details=False):
        '''
        analyze the image and gives the result black.
        Fitting with Gaussian and showing the cuts, and the fits
        is controlled via the
        optional boolean paramters fit, plot_mem and details
        '''
        try:
            ioff()
        except Exception, e:
            print e
        results = []
        if type(self.ROI) == list:
            x_1, x_2, y_1, y_2 = self.ROI
            max_int_pos = where(self.img_roi >= self.img_roi.max())
        else:
            print 'not a list, aborting'
        # extract x,y pos of maxima and sum the intensisties in the roi to make
        # an integreated intensity
        self.max_pos_x = max_int_pos[1][0]
        self.max_pos_y = max_int_pos[0][0]
        self.int_intensity = np.sum(self.img_roi)
        if fit:
            img_crop_x = self.img_roi[self.max_pos_x]
            img_crop_y = self.img_roi[:, self.max_pos_x].flatten()
            xx = linspace(y_1, y_2, len(img_crop_x))
            xy = linspace(x_1, x_2, len(img_crop_y))

        if plot_me:
            fig_update, ax_update = plt.subplots()
            imshow(self.img_roi, interpolation='nearest', cmap=cm.Oranges)
            ax_update.axhline(self.max_pos_y, color='red', ls='--')
            ax_update.axvline(self.max_pos_x, color='black', ls='--')
            # ax_update.set_title(str(image)[-9:-7] + 'minutes of growth')
            display(fig_update)
        # fit x
        y = img_crop_x
        amplitude = y.max()
        self.int_x = amplitude
        a0 = y.min()
        mean_x = xx[where(y == y.max())][0]
        if fit:
            poptx, pcov = curve_fit(
                gauss_function, xx, y, p0=[amplitude, mean_x, std, a0])
        # fit y
        y = img_crop_y
        amplitude = y.max()
        self.int_y = amplitude
        a0 = y.min()
        mean_y = xy[where(y == y.max())][0]
        if fit:
            popty, pcov = curve_fit(
                gauss_function, xy, y, p0=[amplitude, mean_y, std, a0])
            results.append([poptx, popty])
        if details:
            fig_det, [ax_det1, ax_det2] = plt.subplots(2, 1)
            ax_det1.plot(xx, img_crop_x, 'o')
            ax_det1.set_title('img_crop_x')
            ax_det1.plot(
                xx, gauss_function(xx, poptx[0], poptx[1], poptx[2], poptx[3]), 'r-')
            ax_det2.plot(xy, img_crop_y, 'o')
            ax_det2.set_title('img_crop_y')
            ax_det2.plot(
                xy, gauss_function(xy, popty[0], popty[1], popty[2], popty[3]), 'k-')
            display(fig_det)
        self.FWHM_y = poptx[2] * 2.355
        self.FWHM_x = popty[2] * 2.355
        try:
            close('all')
            ion()
        except Exception, e:
            print e


def gauss_function(x, a, x0, sigma, a0):
    return (a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))) + a0


def get_roi(i, kind, log=False):
    temp_image = SPring8_image(i)
    temp_image.read_detector(name=kind)
    temp_image.select_ROI(temp_image.img, True, log=log)
    return temp_image.ROI


def parse_sample(files, samples, roi, kind):
    '''
    list of filenames and retrun result of batch analyis
    '''

    try:
        ion()
    except Exception as e:
        print e + 'not in interactive notebook'
    sample = []
    for i in files:
        temp_image = SPring8_image(i)
        temp_image.read_detector(name=kind)
        temp_image.select_ROI(temp_image.img, ROI=roi)
        temp_image.analyze_img()
        sample.append(temp_image)
    samples[str(roi)] = sample
    return samples


def parse_sample_parallel(i, roi, kind, std=10):
    '''
    list of filenames and retrun result of batch analyis
    '''

    try:
        ion()
    except Exception as e:
        print str(e) + 'not in interactive notebook'
    sample = []

    temp_image = SPring8_image(i)
    temp_image.read_detector(name=kind)
    temp_image.select_ROI(temp_image.img, ROI=roi)
    try:
        temp_image.analyze_img(std=std)
    except Exception as e:
        print e
    sample.append(temp_image)

    return sample


def showrois(samples, roi, num):
    '''
    show the roi for a given number that rapresent the inxed of the detector pictures:
    rule of thumb: lower index means shorter time passed since X-ray on
    '''
    fig, ax = samples[samples.keys()[0]][num].show(log=True)
    for j in roi:
        i = roi[j]
        ax.add_patch(
            Rectangle(i[0::2], i[1] - i[0], i[3] - i[2], ec='black', fc=None, fill=False))
        ax.text(i[0] - 2, i[2] - 2, j)


def norm(df, scale=1):
    '''
    normalize values of imput dataframe in range 0-1
    '''
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm * scale


def define_roi(files, name, cond=None, kind=2):
    '''
    cond = yes to delete roi
    '''
    # Check if dumped ROIs exist already:
    try:
        roi = pickle.load(open(str(name) + 'ROI' + ".p", "rb"))
        print '- ROI dump found, using it !'
        if cond == 'yes':
            name = str(name) + ".p"
        else:
            print '- using loaded ROIs'
        roi = pickle.load(open(str(name) + 'ROI' + ".p", "rb"))
    except:
        print '- no ROI found, define them'
    # Define New ROIs
        try:
            roi
            print '- ROI exists, finishing'
            if not roi:
                print '- empty ROI dictionary'

                raise NameError
        except NameError:
            print '- create new ROI'
            roi = {}
        try:
            ZB
            print '-  ZB exists'
        except NameError:
            print '- create ZB'
            # first file  should only have ZB/twin
            print files[0]
            ZB = get_roi(files[0], kind, log=True)
            roi['ZB'] = ZB

        try:
            TW
            print '-  TW exists'
        except NameError:
            print '- create TW'
            # random late file  should  have strong  ZB/twin
            print files[50]
            TW = get_roi(files[50], kind, log=True)
            roi['TW'] = TW

        try:
            WZ
            print '-  WZ exists'
        except NameError:
            print '- create wZ'
            # any file in between the fist and the last should be good
            name_file = files[randint(0, high=len(files))]
            print name_file
            WZ = get_roi(name_file, kind, log=True)
            roi['WZ'] = WZ

        # Dump roi to disk
    try:
        roi_T = pickle.load(open(str(name) + 'ROI' + ".p", "rb"))
        print '- ROI-dump-file found'
        print '---------------------'
        print 'done'
    except:
        print '- ROI-dump-file not found, creating it..'
        try:
            roi
            print 'ROI found, dump it to disk'
            pickle.dump(roi, open(str(name) + 'ROI' + ".p", "wb"))
        except:
            'no ROI at all , problem ! '
    return roi


def save_suhtters(name):
    try:
        on_off = pickle.load(open(str(name) + 'shutter' + ".p", "rb"))
        print 'on_off dump found, using it !'
    except:
        print 'no on_off found, define them'
        on_off = []
        cond = True
        while cond:
            value = raw_input('Suhtter on  time:')
            on_off.append(float(value))
            value = raw_input('Suhtter on  time:')
            on_off.append(float(value))
            test = raw_input('more (leave empty to continue)?')
            if test:
                cond = False
        try:
            roi_T = pickle.load(open(str(name) + 'shutter' + ".p", "rb"))
            print 'on_off-dump-file found, skipping'
        except:
            print 'on_off-dump-file not found, creating it..'
            try:
                on_off
                print 'on_off object found, dumping it to disk'
                pickle.dump(on_off, open(str(name) + 'shutter' + ".p", "wb"))
            except:
                print 'no on_off at all , problem ! '

    return on_off


def plot_dataframe(data, error=False, log=False):
    '''
    plot data frame columns in a long plot
    '''
    try:
        ioff()
        fig, ax = plt.subplots(
            (len(data.columns)), figsize=(10, 50), sharex=True)
        close('all')
        ion()
    except:
        print 'you may be in a non interactive environment'
    if not error:
        for i, j in zip(data.columns, ax):
            if i != 'name':
                j.plot(data.index, data[i].values)
                j.set_ylabel(str(i))
    if error:
        for i, j in zip(data.columns, ax):
            if i != 'name':
                j.errorbar(
                    data.index, data[i].values, yerr=sqrt(data[i].values))
                j.set_ylabel(str(i))
    if log:
        for i, j in zip(data.columns, ax):
            if i != 'name':
                j.set_yscale('log')
    return fig, ax


def plot_shutters(on_off, data, ax,
                  offset, alpha=.22, facecolor='#9EB57F', **kwargsplot):
    '''
    Examples:

    * draw a vertical green translucent rectangle from x=1.25 to 1.55 that
      spans the yrange of the axes::

        >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)
    some arguments:
    ---
    for plotting
    xmin=0, xmax=1,
    linewidth=4, color='r'
    ls = ''

   '''
    for value in on_off:
        for j in ax:
            # not sure about this !!!!! works in the notebook but not in shell
            j.axvline(value + data.index.values.min(), **kwargsplot)
    if offset:
        if float(len(on_off)) == float(len(offset)):
            print 'ok'
        else:
            print 'wrong list baby'
        for value, off in zip(on_off, offset):
            for j in ax:
                j.axvspan(
                    value + data.index.values.min() - off[
                        0], value + data.index.values.min(
                    ) + off[1],
                    alpha=alpha, facecolor=facecolor)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len / 2):-(window_len / 2)]


def spline_do(x, y, ax, k=3, s=7, n=1, diff=True, plot=True, error=False):
    '''
    Finds  spline with degree k and smoothing factor s for x and y.
    Returns ys,d_ys: the y values of the spline and the derivative.

    '''
    # xs = np.array(x).astype(float)
    s = UnivariateSpline(x, y, k=k, s=s)
    xs = linspace(x.values.min(), x.values.max(), len(x))
    ys = s(xs)
    if plot:
        if not error:
            ax.plot(x, norm(y), '.', label='Data')
        ax.plot(xs, norm(ys), label='Spline fit')
        if error:
            ax.errorbar(x, norm(y), yerr=sqrt(norm(y)))

    d_ys = np.gradient(ys)
    dd_ys = np.gradient(d_ys)
    if diff:
        if plot:
            ax.plot(xs, norm(d_ys), '-.', label='1st derivative')
            #ax.plot(xs, norm(dd_ys),'--',label='2nd derivative')
            ax.plot(xs, norm(smooth(dd_ys, 81)),
                    '--', label='2nd derivative (smothed)')
    if plot:
        ax.legend(loc=0)
    return xs, ys, d_ys, dd_ys


def load_data_frame(name):
    try:
        data = pd.read_csv(str(name) + 'data_frame.csv', index_col=0)
        print 'data frame loaded'
        return data
    except:
        print 'not found'
        return None


def pandify(samples, roi, name_file, save=True):
    '''
    takes the result of the calculation and turns them into a sorted
    dataframe!
    '''
    for a in samples.keys():
        time, NIG, = [], []
        for i in samples[a]:
            time.append(i.time)
            NIG.append(i.NIG)
    data = pd.DataFrame(
        index=time, columns=['NIG', 'Int_WZ', 'Int_ZB', 'Int_TW',
                             'FWHM_x_WZ', 'FWHM_x_ZB', 'FWHM_x_TW',
                             'FWHM_y_WZ', 'FWHM_y_ZB', 'FWHM_y_TW',
                             'max_pos_x_WZ', 'max_pos_x_ZB', 'max_pos_x_TW',
                             'max_pos_y_WZ', 'max_pos_y_ZB', 'max_pos_y_TW', 'name'
                             ])
    Int_WZ = []
    Int_ZB = []
    Int_TW = []
    FWHM_x_WZ = []
    FWHM_x_ZB = []
    FWHM_x_TW = []
    FWHM_y_WZ = []
    FWHM_y_ZB = []
    FWHM_y_TW = []
    max_pos_x_WZ = []
    max_pos_x_ZB = []
    max_pos_x_TW = []
    max_pos_y_WZ = []
    max_pos_y_ZB = []
    max_pos_y_TW = []
    name = []

    for i in roi.keys():
        if i == 'ZB':
            for i in samples[i]:
                Int_ZB.append(i.int_intensity / i.monitor)
                try:
                    FWHM_x_ZB.append(i.FWHM_x)
                    FWHM_y_ZB.append(i.FWHM_y)
                except:
                    FWHM_x_ZB.append(NaN)
                    FWHM_y_ZB.append(NaN)
                max_pos_x_ZB.append(i.max_pos_x)
                max_pos_y_ZB.append(i.max_pos_y)
        if i == 'TW':
            for i in samples[i]:
                Int_TW.append(i.int_intensity / i.monitor)
                try:
                    FWHM_x_TW.append(i.FWHM_x)
                    FWHM_y_TW.append(i.FWHM_y)
                except:
                    FWHM_x_TW.append(NaN)
                    FWHM_y_TW.append(NaN)
                max_pos_x_TW.append(i.max_pos_x)
                max_pos_y_TW.append(i.max_pos_y)
        if i == 'WZ':
            for i in samples[i]:
                Int_WZ.append(i.int_intensity / i.monitor)
                try:
                    FWHM_x_WZ.append(i.FWHM_x)
                    FWHM_y_WZ.append(i.FWHM_y)
                except:
                    FWHM_x_WZ.append(NaN)
                    FWHM_y_WZ.append(NaN)
                max_pos_x_WZ.append(i.max_pos_x)
                max_pos_y_WZ.append(i.max_pos_y)
                name.append(str(i.file_name))

    data.Int_ZB = Int_ZB
    data.FWHM_x_ZB = FWHM_x_ZB
    data.FWHM_y_ZB = FWHM_y_ZB
    data.max_pos_x_ZB = max_pos_x_ZB
    data.max_pos_y_ZB = max_pos_y_ZB

    data.Int_TW = Int_TW
    data.FWHM_x_TW = FWHM_x_TW
    data.FWHM_y_TW = FWHM_y_TW
    data.max_pos_x_TW = max_pos_x_TW
    data.max_pos_y_TW = max_pos_y_TW

    data.Int_WZ = Int_WZ
    data.FWHM_x_WZ = FWHM_x_WZ
    data.FWHM_y_WZ = FWHM_y_WZ
    data.max_pos_x_WZ = max_pos_x_WZ
    data.max_pos_y_WZ = max_pos_y_WZ

    data.NIG = NIG
    data.name = name

    data = data.sort_index()
    data.to_csv(str(name_file) + 'data_frame.csv')
    return data


def connect_to_ploty(usr='giulio', api='622ge3l1ww'):
    ply = plotly.plotly(usr, api)
    return ply


def create_stacked_layout(n=3):
    '''
    Creates a stacked layout with shared x axis.
    n controls the number of stacked rows.
    '''
    layoutlst = []
    layoutlst.append(0)
    for i in range(int(n)):
        layoutlst.append((i + 1) / n)

    layout = {}
    for i in range(int(n)):
        if i == 0:
            layout['yaxis'] = {'domain': layoutlst[i:i + 2]}
        else:
            layout['yaxis' + str(i + 1)] = {'domain': layoutlst[i:i + 2]}

    return layout


def save_spline_par(parameters_dict, name):
    '''
    Saves parameters to disk with suffix "spline_parameters" as a pickle
    object
    -----
    The required arguments are the dictionary of the data and the name
    of the folder
    '''
    try:
        print 'searching parameters_dict object'
        parameters_dict
        print 'parameters_dict object found, dumping it to disk'
        pickle.dump(parameters_dict,
                    open(str(name) + 'spline_parameters' + ".p", "wb"))
    except Exception as e:
        print e
        print 'no spline_parameters at all , problem ! '


def load_spline_par(name):
    '''
    Load parameters dictionary from disk with suffix "spline_parameters"
    -----
    The required argument is the name of the folder (sample).
    ----
    Returns the dictionary.
    '''
    try:
        parameters_dict = pickle.load(
            open(str(name) + 'spline_parameters' + ".p", "rb"))
        print 'spline_parameters dump found, using it !'
        return parameters_dict
    except:
        print 'no parameters_dict found, define them'
        return None


def shade(on_off, ax):
    '''
    shade the ax axsis object between on and off
    Takes the list of on-off values (!) and one
    axis object.
    '''
    for on, off in zip(on_off[:-1:2], on_off[1::2]):
            # on = on + data.index.values.min()
            # off = off + data.index.values.min()
        ax.add_patch(
            Rectangle((on, -1),
                      off - on, ax.get_ylim()[1],
                      facecolor="grey", alpha=0.5))


def update_legend(fig, loc=0):
    '''
    takes a figure as argument and updates
    the legend with the label of each line
    except ones without label.
    '''
    lines = []
    for ax in fig.get_axes():
        ax.legend_ = None
        plt.draw()
        for line in ax.lines:
            if '_line' in line.get_label():
                continue
            else:
                lines.append(line)
    ax.legend(lines, [l.get_label()
                      for l in lines], loc=loc)  # to create nice legend


def color_ticks(fig, color_labeled_lines=True):
    '''
    change color of y tick labels with the color
    of the labeled line, works even with multiple lines
    but will take the color of the last line, in order
    of plotting.
    Not really working perfectly with mislabeled plots)
    '''
    for ax in fig.get_axes():
        if len(ax.lines) > 2:
            print 'check the result'
        for line in ax.lines:
            print ax.get_ylim()
            print line.get_label()
            print '--'
            if '_line' not in line.get_label():
                    for i in ax.get_yticklabels():
                        i.set_color(line.get_color())
            else:
                continue
            # else:
            #     if '_line' in line.get_label():
            #         for i in ax.get_yticklabels():
            #             i.set_color(line.get_color())


def Version():
    return 'last update Date: Date: 7-1-2014'
