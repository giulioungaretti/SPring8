from quickplot import *
import pandas as pd
names = ['InAs0799time3' , 'InAs0799time2', 'InAs0799time1a']
for name in names:
    fig, ax = plt.subplots()
    data, parameters_dict, on_off, offset_list = loadme(name)
    ax.plot(data.index[data.NIG != 1], data.NIG[data.NIG != 1])
    ax.set_title(name)
    fig.show()
