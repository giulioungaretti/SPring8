from quickplot import loadme, do_diff_derivatives, shade
import matplotlib.pyplot as plt
name = 'InAs0799time3'
data, parameters_dict, on_off, offset_list = loadme(name)
fig, ax = do_diff_derivatives(
    data, parameters_dict, 'TW',
    name, False, False, False, True, True,True)

fig.show()
def aaa():
    fig, ax = plt.subplots()
    ax.plot(data.index, data.Int_WZ, 'o')
    ax2 = ax.twinx()
    ax2.plot(data.index[data.NIG != 1], data.NIG[data.NIG != 1])
    shade(on_off, ax)
    return fig
