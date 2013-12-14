from pylab import *
from Functions import *
import pandas as pd

name = 'time2a'


def loadme(name):
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


def do(data, structure, parameters_dict):
    '''
    load data frame and plot shit.
    -------
    input:
    data : the dataframe to be plotted
    structure : string to select WZ,ZB,TW
    parameters_dict: the parameters of the spline
    '''
    fig, ax = subplots(3, 1, figsize=(30, 20), sharex=True)

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
    fig.show()


def do_derivatives(data, parameters_dict, shutter=False):
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

    return fig


def do_diff_derivatives(data, parameters_dict, shutter=False):
    fig, ax = subplots()

    on_off = save_suhtters(name)

    data_temp = data

    offset_list = [[5, 5] for i in range(len(on_off))]  # [[5,56],[10,56]]

    x = data.index
    y = data_temp.Int_WZ
    xs_WZ, ys_WZ, d_ys_WZ, dd_ys_WZ = spline_do(
        x, y, [ax], n=3, s=parameters_dict['WZ'], k=3, plot=False)
    ax.legend()
    if shutter:
        plot_shutters(on_off, data_temp, [ax],
                      offset=offset_list, ls='--', color='k')
    # now to ZV
    y = data_temp.Int_TW
    xs_TW, ys_TW, d_ys_TW, dd_ys_TW = spline_do(
        x, y, [ax], n=3,
        s=parameters_dict['TW'], k=3, plot=False)
    diff = (norm(d_ys_WZ) - norm(d_ys_TW))  # / (d_ys_WZ + d_ys_TW )

    ax.plot(xs_TW, diff)
    ax2 = ax.twinx()
    ax2.set_ylabel('NIG (Pa)')
    ax2.plot(x[data_temp.NIG != 1], data_temp.NIG[data_temp.NIG != 1])
    ax.set_ylabel('Normalized intensity (a.u)')
    ax.legend()
    fig.show()

    return fig
