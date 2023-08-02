import numpy as np

# import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Create event object with attributes : 
#----------------------------------------------------------------

class event(object):

    def __init__(self, event_id, active_channels, t0, trace, baseline, integrals, alignment_method, align_args, calculate_integrals=True, compute_fails=True, calibration_m=1, calibration_c=0):
        #init attributes
        self.eventID = event_id
        self.active_channels = active_channels
        self.triggerTime = t0
        self.calibration_m = calibration_m
        self.calibration_c = calibration_c

        self.shortIntegral = 0
        self.longIntegral = 0
        self.fails = [0,0,0,0,0] #start, long, short, integral, align pos


        # self.integralBounds = integrals #time ns
        self.baseline = np.mean(trace[:baseline])
        self.baseline_bits = baseline

        self.trace = self.baseline-trace
        self.__check_polarity()

        self.align_args = align_args


        if alignment_method == 'CFD_old':
            self.CFD_arr, self.align_pos = self.__cfd_old(*align_args)

        elif alignment_method == 'max':
            self.align_pos = np.where(self.trace == np.max(self.trace))[0][0]
        
        elif alignment_method == 'CFD':
            self.CFD_arr, self.align_pos = self.__cfd(*align_args)

        if calculate_integrals == True:
            self.istart = self.align_pos + integrals[0]
            self.ishort = self.align_pos + integrals[1]
            self.ilong = self.align_pos + integrals[2]

            # self.longIntegral = np.array([self.__sum_integral(i) for i in self.ilong])
            # self.shortIntegral = np.array([self.__sum_integral(i) for i in self.ishort])

            self.longIntegral = self.__sum_integral(self.ilong)
            self.shortIntegral = self.__sum_integral(self.ishort)

            # if len(self.ilong) == 1:
            #     self.longIntegral = self.longIntegral[0]
            # if len(self.ishort) == 1:
            #     self.shortIntegral = self.shortIntegral[0]


        #fails
        if compute_fails == True:
            # for i in self.istart:
            if self.istart < 0 or self.istart > len(self.trace): 
                # If the start of the integration window is outside the event window, fail
                self.fails[0] = 1

            # for j in self.ilong:
            if self.ilong > len(self.trace) or self.ilong < 0: 
                # If the end of the long integral window is outside the event window, fail
                self.fails[1] = 1

            # for k in self.ishort:
            if self.ishort < 0 or self.ishort > len(self.trace):
                # If the end of the short integral window is outside the event window, fail
                self.fails[2] = 1
                    
            # if len(self.ilong) == 1:
            if self.longIntegral < 0:# or self.longIntegral < self.shortIntegral:
                # If the long integral is less than 0, then fail
                # also possible to include the long integral being smaller than the short but that sometimes has issues
                self.fails[3] = 1
            # else:
            #     for l in range(len(self.longIntegral)):
            #         if self.longIntegral[l] < 0 or self.longIntegral[l] < self.shortIntegral[l]:
            #             # Same thing but for mulitple long integrals per event
            #             self.fails[3] = 1




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
        return self.CFD_arr
    def get_baseline(self):
        return self.baseline 
    def get_long_integral(self):
        return self.calibration_m * self.longIntegral + self.calibration_c
    def get_short_integral(self):
        return self.calibration_m * self.shortIntegral + self.calibration_c
    def get_pulse_shape(self):

        # if  len(self.ilong) > 1 and len(self.ishort) > 1:
        #     print('Multi integrals used, pulse shape cannot be calculated')
        #     return -1
        # else:
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

    def get_geometric_mean_trace(self, trace_list):
        cfd_list = []
        for tr in trace_list:
            cfd_list.append(self.__cfd_with_trace_input(self.align_args[0], self.align_args[1], tr)[1])
        
        for i in range(len(trace_list[1:])):
            trace_list[i+1] = np.roll(trace_list[i+1], cfd_list[0] - cfd_list[i+1])
        
        geometric_mean_trace = np.nan_to_num(np.power(np.prod(trace_list, axis=0), 1/len(trace_list)))

        baseline_geo = np.mean(geometric_mean_trace[:self.baseline_bits])
        geometric_mean_trace = geometric_mean_trace - baseline_geo

        L_geo = np.sum(geometric_mean_trace[int(self.istart):int(self.ilong)])
        S_geo = np.sum(geometric_mean_trace[int(self.istart):int(self.ishort)]) / L_geo

        T_trigger_geo = self.triggerTime

        pulse_height_geo = np.max(geometric_mean_trace)

        return geometric_mean_trace, L_geo, S_geo, T_trigger_geo, baseline_geo, pulse_height_geo    

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Constant Fraction Discriminator, requires parameters, y (the trace) F (scaling fraction) L (filter window) O (filter offset) 
#------------------------------------------------------------------------------------------------------------------------------------------------- 

    def __cfd_old(self, F, L, O):
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

    def __cfd(self, frac, offset):

        # We have one trace scaled down and the other inverted and delayed
        frac_trace = self.trace * frac
        delay_trace = np.roll(self.trace, offset)

        # Then subtract one from the other
        cfd_array = frac_trace - delay_trace

        # If there is only one pulse in the window, this will find the index positions of
        # the min and max, between which should be the zero crossing event that we care about
        # cfd_array_max_index = np.where(cfd_array == np.max(cfd_array))[0][0]
        cfd_array_max_index = np.argmax(cfd_array)
        # cfd_array_min_index = cfd_array_max_index + np.where(cfd_array[cfd_array_max_index:] == np.min(cfd_array[cfd_array_max_index:]))[0][0]
        cfd_array_min_index = cfd_array_max_index + np.argmin(cfd_array[cfd_array_max_index:])

        zero_cross_index = -1

        try:
            # We use np.diff to find where the sign of two adjacent points is different and that 
            # should be the crossing event. We then get the index of that point
            zero_cross_index = cfd_array_max_index + np.where( np.diff( np.sign( cfd_array[cfd_array_max_index:cfd_array_min_index] ) ) != 0 )[0][0]
        except:   # This used to only except IndexError but I think this is more general
            self.fails[4] = 1
            return cfd_array, -1


        return cfd_array, zero_cross_index

    def __cfd_with_trace_input(self, frac, offset, trace):

        # We have one trace scaled down and the other inverted and delayed
        frac_trace = trace * frac
        delay_trace = np.roll(trace, offset)

        # Then subtract one from the other
        cfd_array = frac_trace - delay_trace

        # If there is only one pulse in the window, this will find the index positions of
        # the min and max, between which should be the zero crossing event that we care about
        # cfd_array_max_index = np.where(cfd_array == np.max(cfd_array))[0][0]
        cfd_array_max_index = np.argmax(cfd_array)
        # cfd_array_min_index = cfd_array_max_index + np.where(cfd_array[cfd_array_max_index:] == np.min(cfd_array[cfd_array_max_index:]))[0][0]
        cfd_array_min_index = cfd_array_max_index + np.argmin(cfd_array[cfd_array_max_index:])

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


