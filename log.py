class mbelog(object):

    def __init__(self, name):
        self.name = name

    def readlog(self):
        '''
        read the log
        '''
        import time
        format = "%Y/%m/%d %H:%M:%S"
        data_time = []
        data_NIG = []
        temp = []
        # cell_temp_As = []
        cell_temp_In = []
        cell_temp_Sb = []
        shutter_In = []
        shutter_As = []
        shutter_Sb = []
        shut_time = []
        with open(self.name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                    if 'NIG' in line:
                        NIG = float(line[-8:])
                        data_NIG.append(NIG)
                    if '#D' in line:
                        time_val = time.strptime(line[3:-1], format)
                        data_time.append(time.mktime(time_val))
                    if 'IR' in line:
                        temp.appned(float(line[5:]))
                    if '#P Kcells' in line:
                        temperatures =  line[11:-1].split(' ')
                        # cell_temp_As.append(temperatures[4])
                        cell_temp_In.append(temperatures[2])
                        cell_temp_Sb.append(temperatures[4])
                    if '#C Shutter(s) opened or closed:' in line:
                        shutt =  line[32:-1].split(' ')
                        shut_time.append(time.mktime(time_val))
                        shutter_In.append(shutt[1])
                        shutter_As.append(shutt[-1])
                        shutter_Sb.append(shutt[-2])
        self.sub_temp = temp
        self.NIG = data_NIG
        self.time = data_time
        # self.cell_temp_As = cell_temp_As
        self.cell_temp_In = cell_temp_In
        self.cell_temp_Sb = cell_temp_Sb
        self.shutter_In = shutter_In 
        self.shutter_As = shutter_As 
        self.shutter_Sb = shutter_Sb 
        self.shut_time = shut_time
#P Kcells  
