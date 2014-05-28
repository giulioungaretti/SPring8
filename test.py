from scipy import *
from Functions import *
import matplotlib.pyplot as plt
name = 'InAs0799time1a'
data, parameters_dict, on_off, offset_list = loadme(name)
fig, ax = plt.subplots()
x = data.index
y = data.Int_TW
y_sav = savitzky_golay(y,25,3)
ax.plot(x,y,'.')
ax.plot(x,y_sav,"-") 
ax2 = ax.twinx()
ax2.plot(x, y-y_sav)
fig.show()
raw_input('asd')
