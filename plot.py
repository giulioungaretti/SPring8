from quickplot import loadme, update_legend
from Functions import shade, color_ticks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_fig():
    name = 'InAs0799time3'
    data, parameters_dict, on_off, offset_list = loadme(name)
    fig, ax = plt.subplots(figsize=(15, 10))
    WZ_Color = '#C02942'
    TW_Color = '#69D2E7'
    WZ_color_Data = "#00A0B0"
    TW_Color_Data = "#F38630"
    # WZ
    ax.plot(data.index, data.Int_WZ, 'o', c=WZ_color_Data, label=r'$I_{WZ} $')
    ax.set_xlim(100, 600)
    ax.set_ylim(13, 16.5)
    shade(on_off, ax, lim=35)
    data['time'] = data.index
    high = on_off[0]
    low = 100
    means = []
    x1 = data.time[(data.time < high) & (data.time > low)]
    y = data.Int_WZ[(data.time < high) & (data.time > low)]
    means.append(y.mean())
    avg = y.mean() * np.ones(len(x1))
    ax.plot(x1, avg, 'r--')
    cond = True
    if cond:
        for i, j in zip(on_off[1::2], on_off[2::2]):
            low = i
            high = j
            x = data.time[(data.time < high) & (data.time > low)]
            y = data.Int_WZ[(data.time < high) & (data.time > low)]
            avg = y.mean() * np.ones(len(x))
            ax.plot(x, avg, '--', c=WZ_Color)
            means.append(y.mean())
    means = np.array(means)
    value_reference_for_label = 16.20
    for i, j, k in zip(on_off[0::2], on_off[1::2], means):
        low = i
        high = j
        x = data.time[(data.time < high) & (data.time > low)]
        y = data.Int_WZ[(data.time < high) & (data.time > low)]
        x = x.values.astype(float)[y > k]
        y = y.values.astype(float)[y > k]
        p = np.polyfit(x, y, 1)
        ax.plot(x, p[0] * x + p[1], '--', c=WZ_Color)  # A red solid
        # initual value
        base_value_WZ = 0.47
        ax.annotate('{0:.2f} '.format((k / (j - i)) / base_value_WZ),
                    xy=(j, value_reference_for_label),
                    xytext = (0, 0),   fontsize=18,
                    color= WZ_color_Data,
                    horizontalalignment='right',  # rotation=45,
                    bbox = dict(fc='k', alpha=.0),
                    textcoords = 'offset points')

    ax.annotate('''Growth rate:
(a.u)''',
                xy=(178, 16.2),
                xytext = (0, 0),   fontsize=20,
                color= '#353535',
                horizontalalignment='right',  # rotation=45,
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.annotate('Open:',
                xy=(140, 15.05),
                xytext = (0, 0),  fontsize=20,
                color='#353535',
                horizontalalignment='right',  # rotation=45,
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.annotate('In shutter:',
                xy=(165.2, 15.2),
                xytext = (0, 0),   fontsize=20,
                color='black',
                horizontalalignment='right',
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.annotate('Close:',
                xy=(140, 14.9),
                xytext = (0, 0),   fontsize=20,
                color= '#353535',
                horizontalalignment='right',  # rotation=45,
                bbox = dict(fc='k', alpha=.0),
                textcoords = 'offset points')
    ax.add_patch(Rectangle((145, 15.05), 10, height=.1,
                 fill=False, alpha=.5, color='k'))
    ax.add_patch(
        Rectangle((145, 14.9), 10, height=.1, facecolor="grey", alpha=.5))
    ax.set_ylabel(' Intensity (a.u)')
    ax.legend(loc=5)
    ax.set_xlabel('Time(s)')

    ax.set_xticklabels(range(0, 600, 100))

    fig.get_axes()[0].tick_params('both', length=10, width=2, which='major')
    ###########
    # TWIN
    ###########

    ax2 = ax.twinx()
    ax2.plot(data.index, data.Int_TW, 'o', c=TW_Color_Data, label=r'$I_{TW} $')
    ax2.set_xlim(100, 600)
    ax2.set_ylim(225, 320)

    data['time'] = data.index
    high = on_off[0]
    low = 100
    means = []
    x1 = data.time[(data.time < high) & (data.time > low)]
    y = data.Int_TW[(data.time < high) & (data.time > low)]
    means.append(y.mean())
    avg = y.mean() * np.ones(len(x1))
    ax2.plot(x1, avg, '--', c=TW_Color)

    cond = True
    if cond:
        for i, j in zip(on_off[1::2], on_off[2::2]):
            low = i
            high = j
            x = data.time[(data.time < high) & (data.time > low)]
            y = data.Int_TW[(data.time < high) & (data.time > low)]
            avg = y.mean() * np.ones(len(x))
            ax2.plot(x, avg, '--', c=TW_Color)
            means.append(y.mean())
    means = np.array(means)

    value_reference_for_label = 315
    base_value_TW = 9.11
    for i, j, k in zip(on_off[0::2], on_off[1::2], means):
        low = i
        high = j
        x = data.time[(data.time < high) & (data.time > low)]
        y = data.Int_TW[(data.time < high) & (data.time > low)]
        x = x.values.astype(float)[y > k]
        y = y.values.astype(float)[y > k]
        p = np.polyfit(x, y, 1)
        ax2.plot(x, p[0] * x + p[1], '--', c=TW_Color)  # hipstery blue
        ax2.annotate('{0:.2f} '.format((k / (j - i)) / base_value_TW),
                     xy=(j, value_reference_for_label),
                     xytext = (0, 0),   fontsize=18,
                     color= TW_Color_Data,
                     horizontalalignment='right',  # rotation=45,
                     bbox = dict(fc='k', alpha=.0),
                     textcoords = 'offset points')

    ax2.axhline(310, ls='--', c='gray')
    ax2.add_patch(
        Rectangle((145, 15.05), 10, height=.1, fill=False, alpha=.5, color='k'))
    ax2.add_patch(
        Rectangle((145, 14.9), 10, height=.1, facecolor="grey", alpha=.5))
    ax2.set_ylabel('Normalized intensity (a.u)')
    ax2.legend(loc=5)
    ax2.set_xlabel('Time(s)')
    ax2.set_xticklabels(range(0, 600, 100))
    ax2.grid('off')
    # fig.get_ax2es()[0].tick_params('both', length=10, width=2, which='major')
    color_ticks(fig)
    update_legend(fig, loc=4)
    fig.savefig('2.pdf')
    return fig
