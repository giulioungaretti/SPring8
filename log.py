

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
                        temp.append(float(line[5:]))
        self.sub_temp = temp
        self.NIG = data_NIG
        self.time = data_time
        # return data_time, data_NIG, temp
