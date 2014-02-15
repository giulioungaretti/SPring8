# imports
import cPickle as pickle
from numpy import linspace, uint16, uint32
from numpy import where, NaN
from numpy.random import randint
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.mlab import find
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.ndimage.measurements import center_of_mass
from matplotlib.patches import Rectangle
from matplotlib.pyplot import ion, ioff, close, imshow, cm
from IPython.display import display
import plotly as plotly
from collections import deque
import agpy as agf

class SPring8_image(object):

    '''
    image class that reads from detector: ccd rikagu and  pilatus

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
        # self.img = median_filter(self.img, 11)  # perform median filter 5x5
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
        # tuple (x coordinate, y coordinate)
        self.COM = center_of_mass(self.img_roi)
        self.int_intensity = np.sum(self.img_roi)
        if fit:
            img_crop_x = self.img_roi[self.max_pos_x]
            img_crop_y = self.img_roi[:, self.max_pos_x].flatten()
            xx = np.linspace(y_1, y_2, len(img_crop_x))
            xy = np.linspace(x_1, x_2, len(img_crop_y))

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


def parse_headers(i, kind=2):
    '''
    optimized for speed!
    '''
    sample = deque()
    temp_image = SPring8_image(i)
    temp_image.read_detector(name=kind)
    sample.append(temp_image)

    return sample


def fit_parallel(i, data_df, range_pixel=[70,125, 90,370], kind=2):
    '''
    optimized for speed!
    range_pixel= [y1,y2,x1,x2] '''

    Monitor = data_df.Monitor[data_df.name == i].values[0]
    temp_image = SPring8_image(i)
    temp_image.read_detector(name=kind)
    image = temp_image.img
    image = image[range_pixel[0]:range_pixel[1],range_pixel[2]:range_pixel[3]] # take right 
    image = image/Monitor
    return agf.gaussfitter.gaussfit(image)



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
    files MUST be string to the iamge name with paht!
    cond = 'yes' to delete roi
    '''
    # Check if dumped ROIs exist already:
    try:
        roi = pickle.load(open(str(name) + 'ROI' + ".p", "rb"))
        print '- ROI dump found, using it !'
        if cond == 'yes':
            print 'creating new roi, overwriting old file, commit!!!'
            roi = {}
            raise Exception
        else:
            print '- using loaded ROIs'
    except:
        print '- no ROI found, define them'
    # Define New ROIs
        try:
            roi
            print '- ROI object exists '
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
            ZB = get_roi(files, kind, log=True)
            roi['ZB'] = ZB

        try:
            TW
            print '-  TW exists'
        except NameError:
            print '- create TW'
            # random late file  should  have strong  ZB/twin
            TW = get_roi(files, kind, log=True)
            roi['TW'] = TW

        try:
            WZ
            print '-  WZ exists'
        except NameError:
            print '- create wZ'
            WZ = get_roi(files, kind, log=True)
            roi['WZ'] = WZ

        # Dump roi to disk
    print 'saving to disk'
    try:
        roi
        print 'ROI found, dump it to disk'
        pickle.dump(roi, open(str(name) + 'ROI' + ".p", "wb"))
    except:
        'no ROI at all , problem ! '
    return roi


def save_suhtters(name, new=False):
    try:
        on_off = pickle.load(open(str(name) + 'shutter' + ".p", "rb"))
        print 'on_off dump found, using it !'
        if new == True:
            print 'skipping on_off dump, creating new one'
            raise Exception
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
            if new == True:
                print 'skipping on_off dump, creating new one'
                raise Exception
        except:
            print 'on_off-dump-file not found, creating it..'
            try:
                on_off
                print 'on_off object found, dumping it to disk'
                pickle.dump(on_off, open(str(name) + 'shutter' + ".p", "wb"))
            except:
                print 'no on_off at all , problem ! '
    filename = str(name) + 'shutter' + ".p"
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
    xs = np.linspace(x.values.min(), x.values.max(), len(x))
    ys = s(xs)
    if plot:
        if not error:
            ax.plot(x, norm(y), '.', label='Data')
        ax.plot(xs, norm(ys), label='Spline fit')
        if error:
            ax.errorbar(x, norm(y), yerr=np.sqrt(norm(y)))

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


def load_spec_log(name, lines=2):
    '''
    Read the spec log file and return the log store
    as a panda data frame.
    Last  n * 'lines' skipped.
    '''
    names = 'Time Epoch  Seconds  F.M.  bg  pico1  roi0  roi1  roi2  Monitor  Detector'
    names = names.replace('  ', ' ').split(' ')
    spec_name = 'spec/{0}'.format(name)
    spec_log = pd.read_csv(spec_name, names=names, skiprows=0,
                           sep=' ', comment='#', skip_footer=lines)  # skip last 3 lines
    return spec_log.dropna()


def pandify(samples, roi, name_file, save=True):
    '''
    takes the result of the calculation and turns them into a sorted
    dataframe!
    ----
    samples is a dict coming from the parallel calculation
    '''
    for a in samples.keys():
        time, NIG, monitor_value = [], [], []
        for i in samples[a]:
            time.append(i.time)
            NIG.append(i.NIG)
            monitor_value.append(i.monitor)
    data = pd.DataFrame(
        index=time, columns=['NIG', 'Int_WZ', 'Int_ZB', 'Int_TW',
                             'FWHM_x_WZ', 'FWHM_x_ZB', 'FWHM_x_TW',
                             'FWHM_y_WZ', 'FWHM_y_ZB', 'FWHM_y_TW',
                             'max_pos_x_WZ', 'max_pos_x_ZB', 'max_pos_x_TW',
                             'max_pos_y_WZ', 'max_pos_y_ZB', 'max_pos_y_TW',
                             'name',
                             'COM_x_WZ', 'COM_x_ZB', 'COM_x_TW',
                             'COM_y_WZ', 'COM_y_ZB', 'COM_y_TW', 'Monitor'])

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
    COM_x_WZ = []
    COM_x_ZB = []
    COM_x_TW = []
    COM_y_WZ = []
    COM_y_ZB = []
    COM_y_TW = []
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
                COM_x_ZB.append(i.COM[0])
                max_pos_y_ZB.append(i.max_pos_y)
                COM_y_ZB.append(i.COM[1])
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
                COM_x_TW.append(i.COM[0])
                max_pos_y_TW.append(i.max_pos_y)
                COM_y_TW.append(i.COM[1])

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
                COM_x_WZ.append(i.COM[0])
                max_pos_y_WZ.append(i.max_pos_y)
                COM_y_WZ.append(i.COM[1])
                # add name of file to the data
                name.append(str(i.file_name))

    data.Int_ZB = Int_ZB
    data.FWHM_x_ZB = FWHM_x_ZB
    data.COM_x_ZB = COM_x_ZB
    data.FWHM_y_ZB = FWHM_y_ZB
    data.COM_y_ZB = COM_y_ZB
    data.max_pos_x_ZB = max_pos_x_ZB
    data.max_pos_y_ZB = max_pos_y_ZB
    data.Int_TW = Int_TW
    data.FWHM_x_TW = FWHM_x_TW
    data.COM_x_TW = COM_x_TW
    data.FWHM_y_TW = FWHM_y_TW
    data.COM_y_TW = COM_y_TW
    data.max_pos_x_TW = max_pos_x_TW
    data.max_pos_y_TW = max_pos_y_TW
    data.Int_WZ = Int_WZ
    data.FWHM_x_WZ = FWHM_x_WZ
    data.COM_x_WZ = COM_x_WZ
    data.FWHM_y_WZ = FWHM_y_WZ
    data.COM_y_WZ = COM_y_WZ
    data.max_pos_x_WZ = max_pos_x_WZ
    data.max_pos_y_WZ = max_pos_y_WZ
    data.NIG = NIG
    data.name = name
    data.Monitor = monitor_value
    data = data.sort_index()
    data.to_csv(str(name_file) + 'data_frame.csv')
    return data


def pandify_header(samples, save=True, name_file=''):
    '''
    takes the result of the calculation and turns them into a sorted
    dataframe!
    ----
    samples is a dict coming from the parallel calculation
    '''
    timestamp, NIG, monitor_value, name = [], [], [], []
    for i in samples:
        timestamp.append(i.time)
        NIG.append(i.NIG)
        monitor_value.append(i.monitor)
        name.append(str(i.file_name))
    data = pd.DataFrame(
        index=timestamp, columns=['NIG', 'name', 'Monitor', 'Timestamp'])
    data.NIG = NIG
    data.name = name
    data.Monitor = monitor_value
    data = data.sort_index()
    if save:
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


def shade(on_off, ax, data=None, lim=None):
    '''
    shade the ax axsis object between on and off
    Takes the list of on-off values (!) and one
    axis object.
    '''
    if lim != None:
        limit = lim
    if lim == None:
        limit = ax.get_ylim()[1]
    for on, off in zip(on_off[:-1:2], on_off[1::2]):
        if data:
            on = on + data.index.values.min()
            off = off + data.index.values.min()
        ax.add_patch(
            Rectangle((on, -1),
                      off - on, limit,
                      facecolor="grey", alpha=0.5))


def update_legend(fig, loc=0, frameon=True):
    '''
    takes a figure as argument and updates
    the legend with the label of each line
    except ones without label.
    == == == == == == == = == == == == == == =
    Location String   Location Code
    == == == == == == == = == == == == == == =
    'best'            0
    'upper right'     1
    'upper left'      2
    'lower left'      3
    'lower right'     4
    'right'           5
    'center left'     6
    'center right'    7
    'lower center'    8
    'upper center'    9
    'center'         1 0
    == == == == == == == = == == == == == == =
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
                      for l in lines], loc=loc, frameon=frameon)  # to create nice legend


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
            print '''
                coloring ticks...
                too much lines in one axis check the result
                coloring with last line drawn
                 '''
        for line in ax.lines:
            if '_line' not in line.get_label():
                for i in ax.get_yticklabels():
                    i.set_color(line.get_color())
            else:
                continue
            # else:
            #     if '_line' in line.get_label():
            #         for i in ax.get_yticklabels():
            #             i.set_color(line.get_color())


def make_spine_invisible(ax, direction):
    if direction in ["right", "left"]:
        ax.yaxis.set_ticks_position(direction)
        ax.yaxis.set_label_position(direction)
    elif direction in ["top", "bottom"]:
        ax.xaxis.set_ticks_position(direction)
        ax.xaxis.set_label_position(direction)
    else:
        raise ValueError("Unknown Direction : %s" % (direction,))


def make_patch_spines_invisible(ax):
    '''
    makes the patch spine invisible for given ax
    '''
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)


def offset_axis(ax, value=1.2, orientation="right"):
    '''
    offset secondary y axis spine by value
    ---
    value is in axes coordinate 1 = max x, >1 outisde the graph
    '''
    ax.spines[orientation].set_position(("axes", value))
    make_patch_spines_invisible(ax)
    ax.spines[orientation].set_visible(True)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number. Symmetric window.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized symmetric window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2  # symmetric window

    # precompute coefficients
    b = np.mat([[k ** i for i in order_range]
               for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def savitzky_golay_piecewise(xvals, data, kernel=11, order=4):
    turnpoint = 0
    last = len(xvals)
    if xvals[1] > xvals[0]:  # x is increasing?
        for i in range(1, last):  # yes
            if xvals[i] < xvals[i - 1]:  # search where x starts to fall
                turnpoint = i
                break
    else:  # no, x is decreasing
        for i in range(1, last):  # search where it starts to rise
            if xvals[i] > xvals[i - 1]:
                turnpoint = i
                break
    if turnpoint == 0:  # no change in direction of x
        return savitzky_golay(data, kernel, order)
    else:
        # smooth the first piece
        firstpart = savitzky_golay(data[0:turnpoint], kernel, order)
        # recursively smooth the rest
        rest = savitzky_golay_piecewise(
            xvals[turnpoint:], data[turnpoint:], kernel, order)
        return numpy.concatenate((firstpart, rest))


def loadme(name):
    '''
    arguments:
    name = the name of the folder
    -----
    returns:
    data, parameters_dict, on_off, offset_list
    '''
    try:
        data = pd.read_csv(str(name) + 'data_frame.csv', index_col=0)
        print 'data frame loaded'
        parameters_dict = pickle.load(
            open(str(name) + 'spline_parameters' + ".p", "rb"))
        print 'parameters_dict loaded'
        on_off = save_suhtters(name)
        print 'on off loaded'
        offset_list = [[5, 5] for i in range(len(on_off))]  # [[5,56],[10,56]]
        print 'offset_list done'
    except:
        Exception
        print'not found'
        data = None
        parameters_dict = None
    return data, parameters_dict, on_off, offset_list


def do(data, structure, parameters_dict, name, res=False):
    '''
    load data frame and plot shit.
    -------
    input:
    data : the dataframe to be plotted
    structure : string to select WZ,ZB,TW
    parameters_dict: the parameters of the spline
    '''
    fig, ax = plt.subplots(3, 1, figsize=(30, 20), sharex=True)

    on_off = save_suhtters(name)

    data_temp = data

    offset_list = [[5, 5] for i in range(len(on_off))]  # [[5,56],[10,56]]

    x = data.index
    y = data_temp['Int_' + str(structure)]
    xs, ys, d_ys, dd_ys = spline_do(
        x, y, ax, n=3, s=parameters_dict[str(structure)], k=3, plot=False)

    ax[0].plot(x, y, '.', label=y.name)
    ax[0].plot(xs, ys, '-', label=y.name + 'Spline')
    ax2 = ax[0].twinx()

    ax2.plot(xs, d_ys, '--', label=y.name + '1st d erivative')
    ax2.set_ylabel('Normalized intensity (a.u)')
    ax2.legend()
    plot_shutters(on_off, data_temp, ax,
                  offset=offset_list, ls='--', color='k')

    y = data_temp.NIG

    ax[1].plot(x[data.NIG != 1], y[data.NIG != 1], label=y.name)

    ax[1].set_ylabel('Ion Gauge Pressure (Pa)')
    y = data_temp['FWHM_x_' + str(structure)]
    ax[2].plot(x, y[data_temp.FWHM_x_WZ != 0], label=y.name)
    y = data_temp['FWHM_y_' + str(structure)]
    ax[2].plot(x, y[data_temp.FWHM_y_WZ != 0], label=y.name)
    ax[2].set_ylabel('Normalized intensity (a.u)')
    [j.legend(loc=0) for j in ax]
    ax[2].set_xlabel('Time (s)')
    return fig, ax


def do_derivatives(data, parameters_dict, name, shutter=False):
    fig, ax = subplots()
    on_off = save_suhtters(name)
    data_temp = data
    offset_list = [[5, 5] for i in range(len(on_off))]  # [[5,56],[10,56]]

    x = data.index
    # WZ
    y = data_temp.Int_WZ
    xs, ys, d_ys, dd_ys = spline_do(
        x, y, [ax], n=3, s=parameters_dict['WZ'], k=3, plot=False)

    ax.plot(xs, norm(d_ys), '--', label=y.name + '1st d erivative')
    ax.set_ylabel('Normalized intensity (a.u)')
    ax.legend()
    if shutter:
        plot_shutters(on_off, data_temp, [ax],
                      offset=offset_list, ls='--', color='k')
    # now to ZV
    y = data_temp.Int_TW
    xs, ys, d_ys, dd_ys = spline_do(x, y,
                                    [ax], n=3,
                                    s=parameters_dict['TW'],
                                    k=3, plot=False)
    ax.plot(xs, norm(d_ys), '--', label=y.name + '1st d erivative')
    ax.set_ylabel('Normalized intensity (a.u)')
    ax.legend()

    return fig, ax


def create_par_dict():
    parameters_dict = {}
    structures = ['WZ', 'TW', 'ZB']
    for i in structures:
        value = raw_input('Insert spline value for %s: ' % (i))
        parameters_dict[i] = float(value)
    print 'done'
    return parameters_dict


def do_diff_derivatives(data, parameters_dict,
                        structure, name,
                        shutter=False, nig=True,
                        raw_int_wz=True, shade=True,
                        numeric=False, smooth_int_wz=False,
                        **kwargs):
    '''
    Perform the difference of the derivatives I_WZ  and I[structure]
    of the spline fitted data with  the parameters specified
    in the parameters_dict.
    '''
    fig, ax = subplots(**kwargs)
    on_off = save_suhtters(name)
    data_temp = data
    offset_list = [[5, 5] for i in range(len(on_off))]  # [[5,56],[10,56]]

    x = data.index
    y = data_temp.Int_WZ
    xs_WZ, ys_WZ, d_ys_WZ, dd_ys_WZ = spline_do(
        x, y, [ax], n=3, s=parameters_dict['WZ'], k=3, plot=False)
    y_wz = array(y)

    # start plotting
    if shutter:
        plot_shutters(on_off, data_temp, [ax],
                      offset=offset_list, ls='--', color='k')

    # spline derivative
    if not numeric:
        line1, = ax.plot(xs_WZ, diff, c='#00A0B0',
                         label=r'$I{\prime}_{WZ} ' +
                         '- I{\prime}_{%s}$' % (structure))
    # numeric derivative
    if numeric:
        mean = 15
        median = 1
        wz = pd.rolling_mean(
            pd.rolling_median(data.Int_WZ, median), mean)
        d_wz = pd.rolling_mean(
            pd.rolling_median(data.Int_WZ, median).diff(), mean)
        d_tw = pd.rolling_mean(
            pd.rolling_median(data.Int_TW, median).diff(), mean)
        line1, = ax.plot(xs_WZ, d_wz, 'o-',   c='#D95B43',
                         label=r'$I{\prime}_{WZ}, smooth 15 $')
        line1, = ax.plot(xs_WZ, smooth(d_wz, 11), 'o--',
                         c='#C02942', label=r'$I{\prime}_{WZ}$')
    if nig:
        ax2 = ax.twinx()
        ax2.set_ylabel('Pressure (Pa)')
        # to create nice legend
        line2, = ax2.plot(x[data_temp.NIG != 1],
                          data_temp.NIG[data_temp.NIG != 1],
                          c='#72C086', label='NIG')
        ax2.yaxis.label.set_color(line2.get_color())  # color the label
        ax2.tick_params(axis='y', colors=line2.get_color())
        lines = [line1, line2]  # to create nice legend
        ax.legend(lines, [l.get_label()
                  for l in lines], loc=0)  # to create nice legend

    ax3 = ax.twinx()
    ax3.set_ylabel(' intensity (a.u)')
    if smooth_int_wz:
        line2, = ax3.plot(xs_WZ, smooth(data.Int_WZ, 15),
                          'o-',
                          c='#9FE5F2', label=r'$I_{WZ} $')
    if raw_int_wz:
        line2, = ax3.plot(data.index, data.Int_WZ,
                          'o-',
                          c='#0095B1', label=r'$I_{WZ} $')
    lines = [line1, line2]  # to create nice legend
    ax.legend(lines, [l.get_label()
                      for l in lines], loc=3)  # to create nice legend
    if shade:
        for on, off in zip(on_off[:-1:2], on_off[1::2]):
            on = on + data.index.values.min()
            off = off + data.index.values.min()
            ax.add_patch(
                Rectangle((on, -1), off - on, 2, facecolor="grey", alpha=.5))
    # uncomment below to simulate a 10 sec delay, which is unrasonable
        # for on, off in zip(on_off[:-1:2], on_off[1::2]):
        #     on = on  - 10 + data.index.values.min()
        #     off = off -10 + data.index.values.min()
        #     ax.add_patch(Rectangle((on, -1), off - on, 2, facecolor="red",alpha=.5))
    ax.set_ylim(d_wz.min(), d_wz.max())
    ax3.set_ylim(wz.min(), wz.max())
    ax.set_xlabel('Time(s)')
    value_reference_for_label = d_wz
    print value_reference_for_label.max()
    ax.annotate('Open',
                xy=(203, value_reference_for_label.max() * .55),
                xytext = (0, 0),  fontsize=20,
                color='white',
                horizontalalignment='center', rotation=45,
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.annotate('''In
                    shutter:''',
                xy=(135, value_reference_for_label.max() * .54),
                xytext = (0, 0),   fontsize=20,
                color='black',
                horizontalalignment='center',
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.annotate('Close',
                xy=(175, value_reference_for_label.max() * .55),
                xytext = (0, 0),   fontsize=20,
                color= '#353535',
                horizontalalignment='center', rotation=45,
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.set_ylabel('Normalized intensity (a.u)')
    return fig, ax


def ddderive(values):
    temp_array = np.diff(values)
    y = np.insert(temp_array, 0, 0)
    return y


def smooth_deriv(values, window=15, poly_degree=3):
    '''
    calculate the first derivative of the value array
    and smooth it with SG alghoritm with default windows
    of 15 points and poly degree of 3
    '''
    temp_array = np.diff(values)
    d_sav = savitzky_golay(temp_array, window, poly_degree)
    y = np.insert(d_sav, 0, 0)
    return y


def FWHM(X, Y):
    half_max = (max(Y) + min(Y)) / 2
    d = np.sign(half_max - np.array(Y[0:-1])) - \
        np.sign(half_max - np.array(Y[1:]))
    # find the left and right most indexes
    left_idx = find(d > 0)[0]
    right_idx = find(d < 0)[-1]
    # return the difference (full width)
    return X[right_idx], X[left_idx], half_max


def css_styling():
    '''
    stylish notebook
    '''
    try:
        from IPython.core.display import HTML
        styles = open("styles/custom.css", "r").read()
        return HTML(styles)
    except:
        print 'what are you tring to do ???'


def Version():
    return 'date 2014-01-17 tale IIII no filter '
