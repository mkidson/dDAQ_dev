import numpy as np

# import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Create event object with attributes : 
#----------------------------------------------------------------

class event(object):

    def __init__(self, event_id, active_channels, t0, trace, baseline, integrals, alignment_method, align_args, calculate_integrals=True, compute_fails=True):
        #init attributes
        self.eventID = event_id
        self.active_channels = active_channels
        self.triggerTime = t0

        self.shortIntegral = np.zeros(len(integrals[0]))
        self.longIntegral = np.zeros(len(integrals[0]))
        self.fails = [0,0,0,0,0] #start, long, short, integral, cfd zero


        # self.integralBounds = integrals #time ns
        self.baseline = np.mean(trace[:baseline])

        self.trace = self.baseline-trace
        self.__check_polarity()


        if alignment_method == 'CFD':
            self.CFD_arr, self.align_pos = self.__cfd(*align_args)

        elif alignment_method == 'max':
            self.align_pos = np.where(self.trace == np.max(self.trace))[0][0]
        
        elif alignment_method == 'fast_CFD':
            self.CFD_arr, self.align_pos = self.__fast_cfd(*align_args)

        if calculate_integrals == True:
            self.istart = self.align_pos + integrals[0]
            self.ishort = self.align_pos + integrals[1]
            self.ilong = self.align_pos + integrals[2]

            self.longIntegral = np.array([self.__sum_integral(i) for i in self.ilong])
            self.shortIntegral = np.array([self.__sum_integral(i) for i in self.ishort])

            if len(self.ilong) == 1:
                self.longIntegral = self.longIntegral[0]
            if len(self.ishort) == 1:
                self.shortIntegral = self.shortIntegral[0]


        #fails
        if compute_fails == True:
            for i in self.ilong:
                for j in self.ishort:
                    for k in self.istart:
                        if i > len(self.trace) or i < 0: 
                            self.fails[1] = 1
                            # plt.plot(self.trace)
                            # plt.show()
                            # print(self.longIntegral,self.shortIntegral)
                        if k < 0 or k > len(self.trace): 
                            self.fails[0] = 1
                            # print(self.longIntegral,self.shortIntegral)
                        if j < 0 or j > len(self.trace):
                            self.fails[2] = 1
                            # print(self.longIntegral,self.shortIntegral)
                        if j < 0 or j > i:
                            self.fails[3] = 1
                            # print(self.ch,self.longIntegral,self.shortIntegral)
                            # plt.plot(self.trace)
                            # plt.show()




#------------------------------------------------------------------
#Getter functions
#------------------------------------------------------------------
    def get_event_id(self):
        return self.eventID
    def get_active_channels(self):
        return self.active_channels
    def get_t0(self):
        return self.triggerTime
    def get_trace(self):
        return self.trace 
    def get_CFD(self):
        return self.CFD
    def get_baseline(self):
        return self.baseline 
    def get_long_integral(self):
        return self.longIntegral
    def get_short_integral(self):
        return self.shortIntegral
    def get_pulse_shape(self):

        if  len(self.ilong) > 1 and len(self.ishort) > 1:
            print('Multi integrals used, pulse shape cannot be calculated')
            return -1
        else:
            if self.fails[3] == 1:
                return -1
            else:
                return self.shortIntegral / self.longIntegral

    def get_times(self):
        return self.istart, self.ishort, self.ilong, self.align_pos

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Pulse height fitting RETURNS MAX at the moment and the index of max as representation for t
#        if gaussian fitting fails returns max
#----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_pulse_height(self):
        return max(self.trace), np.argmax(self.trace)
    def get_fails(self, display=False):
        if display:
            if np.sum(self.fails) == 0:
                print(f'Event {self.eventID} Fails: {np.sum(self.fails)} fails')
            else:
                print(f'Event {self.eventID} Fails: {np.sum(self.fails)} fails\ntstart: {self.fails[0]}\ttlong: {self.fails[1]}\ttshort: {self.fails[2]}\tintegral: {self.fails[3]}\tt0: {self.fails[4]}')
        return self.fails

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Constant Fraction Discriminator, requires parameters, y (the trace) F (scaling fraction) L (filter window) O (filter offset) 
#------------------------------------------------------------------------------------------------------------------------------------------------- 

    def __cfd(self, F, L, O):
        # F = self.CFD[0]
        # L = self.CFD[1]
        # O = self.CFD[2]
        cfdArr = np.zeros(len(self.trace))
        zero_cross = 0
        #calculate CFD 
        for i in range(O + L + 1, len(self.trace) - O - L - 1, 1):
            cfdArr[i] = np.sum(F * self.trace[i - L : i - 1] - self.trace[i - L - O : i - 1 - O])
        #find the inflection point
        closest = np.argmin(np.subtract(cfdArr, np.roll(cfdArr, 1)))
        lower = 0
        upper = 0
        #Locate the zero crossing near to the inflection point
        if cfdArr[closest] < 0:
            while cfdArr[closest] < 0:
                closest -= 1
            lower = closest
            upper = closest + 1
        else:
            while cfdArr[closest] > 0:
                closest += 1
            upper = closest
            lower = closest - 1

        #return error -1 result no crossover found
        if lower == upper:
            print('No CFD zero crossing found')
            self.fails[4] = 1
            return cfdArr, -1

        #calculate the crossover time through a weighted average of the two nearest samples to the zero 
        sum_y = np.sum(np.abs(cfdArr[lower:upper + 1]))
        zero_cross = lower * (1 - np.abs(cfdArr[lower]) / sum_y) + (upper) * (1 - np.abs(cfdArr[upper]) / sum_y)
        
        #return error -1 result if calculation fails
        if np.isnan(zero_cross):
            print('CFD calculation fail')
            self.fails[4] = 1
            return cfdArr, -1


        return cfdArr, zero_cross


# Hopefully a faster method of finding the crossover time

    def __fast_cfd(self, frac, offset):

        # We have one trace scaled down and the other inverted and delayed
        frac_trace = self.trace * frac
        delay_trace = np.roll(self.trace, offset)

        # Then subtract one from the other
        cfd_array = frac_trace - delay_trace

        # If there is only one pulse in the window, this will find the index positions of
        # the min and max, between which should be the zero crossing event that we care about
        cfd_array_max_index = np.where(cfd_array == np.max(cfd_array))[0][0]
        cfd_array_min_index = cfd_array_max_index + np.where(cfd_array[cfd_array_max_index:] == np.min(cfd_array[cfd_array_max_index:]))[0][0]

        zero_cross_index = -1

        try:
            # We use np.diff to find where the sign of two adjacent points is different and that 
            # should be the crossing event. We then get the index of that point
            zero_cross_index = cfd_array_max_index + np.where( np.diff( np.sign( cfd_array[cfd_array_max_index:cfd_array_min_index] ) ) != 0 )[0][0]
        except:   # This used to only except IndexError but I think this is more general
            self.fails[4] = 1
            return cfd_array, -1


        return cfd_array, zero_cross_index



#-------------------------------------------------------------------------------------------------------------------------------------------------
# Integral calculation: just a sum like QtDaq B) 
#------------------------------------------------------------------------------------------------------------------------------------------------- 
        
    def __sum_integral(self, end):

        sum_int = np.sum(self.trace[int(self.istart):int(end)])
        return sum_int

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Determine if the pulse is negative going or positive going
#            flips polarity so they are positive going
#----------------------------------------------------------------------------------------------------------------------------------------------------------
    def __check_polarity(self):
        peak = max(self.trace) + min(self.trace)
        if peak < 0:
            self.trace = -1 * self.trace


