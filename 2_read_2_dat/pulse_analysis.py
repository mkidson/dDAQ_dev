from scipy import signal
from scipy.signal import find_peaks
from event import *
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt 
import matplotlib.patches as pat
from scipy.interpolate import interp1d
import csv



class read_dat(object):

    header_size = 72
    max_channels = 64
    preamble_size = 4+20+4*max_channels

    def __init__(self, filename):
        self.filename = filename
        self.input_file = open(self.filename, 'rb')
        self.header = self.input_file.read(self.header_size)
        self.end_file = False
        self.event_counter = 0
        self.active_channels = []

        # These two exist 
        self.selection = [[], []]
        self.cuts = []

        print('init complete')

    
    def reinitialise_file(self):
        """Reinitialises the dat_file so that the next event read out will be the first in the file
        """
        self.input_file.close()

        self.input_file = open(self.filename, 'rb')
        self.header = self.input_file.read(self.header_size)
        self.end_file = False


    def read_event(self, baseline_samples, raw_traces=False):
        """Reads every active channel and returns the traces after the baseline has been subtracted and the polarity checked to make it positive-going. Optionally returns the traces before any processing. Takes the number of samples used to determine the baseline as an input.

            Args
            ----
            baseline_samples : (int)
                Number of samples used to calculate the baseline of a pulse at the start of the acquisition window. 

            raw_traces : (bool, optional)
                Flag that determines if the traces before any processing should be returned to the user instead of the processed ones. Defaults to False.

            Returns
            -------
            traces : (array)
                Array of traces from each channel, in order. If any channels in between are not used (e.g. we use 0, 1, and 3), then the array will get flattened so there are only `n` elements, where `n` is the number of active channels.
        """
        # Reads the preamble that sits at the front of each event
        preamble = np.frombuffer(self.input_file.read(self.preamble_size), dtype=np.uint32)
        if not preamble.any(): # Checks end of file
            self.end_file = True
            return self.end_file
        
        # Array of all channels. 0 if that channel isn't active, int value being equal to the number of samples in that active channel
        self.channel_sizes = preamble[6:]
        # Array of active channel numbers, starting from 0
        self.active_channels = np.argwhere(self.channel_sizes > 0).flatten()

        traces_raw = np.empty((len(self.active_channels), self.channel_sizes[self.active_channels[0]]))
        # Gets the raw trace as it comes out of the detector. y-axis is in bits and x-axis is in samples
        for i in range(len(self.active_channels)):
            traces_raw[i] = np.array(np.frombuffer(self.input_file.read(self.channel_sizes[self.active_channels[i]]*2), dtype=np.uint16), dtype=int)
        
        traces = np.empty((len(self.active_channels), self.channel_sizes[self.active_channels[0]]))
        # Iterates through the channels and outputs pulses that have a baseline at 0
        for i in range(len(traces_raw)):
            baseline = np.mean(traces_raw[i][:baseline_samples])
            trace_to_append = baseline - traces_raw[i]

            peak = max(trace_to_append) + min(trace_to_append)
            if peak < 0:
                traces[i] = -1 * trace_to_append
            else:
                traces[i] = trace_to_append
        
        if raw_traces:
            return traces_raw
        else:
            return traces


    def calculate_integrals(self, trace, align_point, t_start, t_short, t_long):
        """Calculates the short and long integrals of a user-supplied trace. Also supplied is the align point (usually determined with CFD), the start time (relative to the alignment point), and the two end times (also relative to the alignment point)

            Args
            ----
            trace : (array, float)
                Signal trace that will be integrated.

            align_point : (float)
                Sample number that the integration window will be determined relative to.

            t_start : (int)
                Number of samples before the `align_point` where the integration window will start.

            t_short : (int)
                Number of samples after the `align_point` that the short integration window will end.

            t_long : (int)
                Number of samples after the `align_point` that the long integration window will end.

            Returns
            -------
            S : (float)
                Short integral
            L : (float)
                Long integral
        """
        short_int = np.sum(trace[int(align_point+t_start):int(align_point+t_short)])
        long_int = np.sum(trace[int(align_point+t_start):int(align_point+t_long)])

        return short_int, long_int


    def get_geometric_mean_trace(self, traces, align_points):
        """Computes the trace that results from taking the geometric mean of two or more traces. The traces all get aligned by their respective entry in `align_points` and shifted up by 100 in order to properly compute the geometric mean of points that straddle the baseline, as otherwise these are not well defined. The result gets shifted back down by 100 at the end.

            Args
            ----
            traces : (array)
                Array of traces that will be included in the geometric mean. Minimum number of traces is 2.

            align_points : (array, int)
                Array of alignment points that will be used to align the traces to each other so the geometric mean makes sense.

            Returns
            -------
            geometric_mean_trace : (array, float)
                Array containing the trace that is the result of taking the geometric mean of two or more traces at each sample point.
        """
        # Always aligns everything to the first trace in the list
        for i in range(len(traces[1:])):
            traces[i+1] = np.roll(traces[i+1], align_points[0] - align_points[i+1])
        
        geometric_mean_trace = np.nan_to_num(np.power(np.prod(traces+100, axis=0), 1/len(traces)))

        return geometric_mean_trace-100


    def time_of_flight(self, trace, tof_stop_trace, stop_distance, stop_height, cfd_params=(0.75, 6)):
        """Computes the difference in samples between the CFD zero crossing point of `trace` and the CFD zero crossing points of `tof_stop_trace`. Since multiple stop pulses can appear in each acquisition window, depending on the frequency, it finds the difference between the very last pulse in the window. This way we avoid seeing a ghost spectrum.

        The interpolated positions are used to determine the time difference in order to avoid an artefact of subtracting integers from integers. 

        The frequency (in MHz) and height of the pulses (in bits) need to be supplied in order to best select the peaks in the stop pulse trace.

        CFD parameters can also be supplied but will default to whatever I've found to be optimal in my work :)

            Args
            ----
            trace : (array, float)
                Array of the signal trace.

            tof_stop_trace : (array, float)
                Array of the stop pulse trace.

            stop_distance : (int)
                Approximate distance, in samples, between the stop pulses. 

            stop_height : (int)
                Approximate height of the stop pulses, in bits.

            cfd_params : (tuple, float, optional)
                Tuple containing the CFD parameters to be used. Defaults to (0.75, 6).

            Returns
            -------
            t_diff_samples : (float)
                Difference in time (in samples) between the CFD zero-crossing points of the stop pulse and the signal trace.
        """
        trace_cfd_interp = self.cfd(trace, *cfd_params)[2]

        # Frequency and height are modified for more optimal peak finding. If they're exact, it doesn't work as well.
        peaks = find_peaks(tof_stop_trace, distance=stop_distance-50, height=stop_height-500)

        tof_stop_trace_for_cfd = tof_stop_trace[peaks[0][-1]-50:peaks[0][-1]+50]

        tof_stop_cfd_interp = self.cfd(tof_stop_trace_for_cfd, *cfd_params)[2] + peaks[0][-1] - 50

        t_diff_samples = tof_stop_cfd_interp - trace_cfd_interp

        return t_diff_samples


    def draw_polygons(self, L, S, lims=None):
        """Plots all supplied L and S points and allows for creation of polygons that can be used to cut out (or in) certain points. Returns nothing but "o" should be pressed after creation of the polygons to get a file out the other end.

            Args
            ----
            L : (array, float)
                Array of L values to be plotted

            S : (array, float)
                Array of S values to be plotted

            lims : (tuple (2x2), optional)
                Tuple containing the x- and y-lims for the plot. If None, uses the automatic limits determined by pyplot. Defaults to None.
        """

        fig = plt.figure()
        if lims == None:
            plt.hist2d(L, S, [512,512], norm=colors.LogNorm(vmin=1), cmap='inferno')
        else:
            plt.hist2d(L, S, [512,512], lims, norm=colors.LogNorm(vmin=1), cmap='inferno')

        plt.colorbar()
        
        plt.legend(title='a: Start new selection\nu: Undo previous point\nx: End current selection\nd: Delete previous selection\no: Output selections to file\nq: Quit', loc='lower right', frameon=False, framealpha=0)


        cip = fig.canvas.mpl_connect('key_press_event', self.__press)
        plt.show(block=True)


    def read_polygons(self, polygon_filename):
        """Reads a file containing polygons created by `draw_polygons` and outputs an array of matplotlib Polygon objects.

            Args
            ----
            polygon_filename : (str)
                Name of the file containing the polygons created by `draw_polygons`.

            Returns
            -------
            polygons : (array, mpl.pat.Polygon)
                Array of matplotlib Polygon objects.
        """
        polygon_file = open(polygon_filename, 'r')
        polygon_csv_reader = csv.reader(polygon_file)

        polygons_x = []
        polygons_y = []

        i = 0
        for row in polygon_csv_reader:
            if i % 2 == 0:
                polygons_x.append([float(x) for x in row])
            elif i % 2 == 1:
                polygons_y.append([float(x) for x in row])

            i += 1

        polygons = []

        for c in range(len(polygons_x)):
            polygons.append(pat.Polygon(np.transpose([polygons_x[c], polygons_y[c]])))

        return polygons


##################################################################################################################################################
#   HELPER FUNCTIONS FOR: add_selections
#               while adding cuts:
#                    key press a or A: starting a new selection
#                    key press x or X: terminate current cut
#                    key press q or Q: terminate 
#                    key press o or O: Output the cuts created 
##################################################################################################################################################

    def __onclick(self, event):
        if event.xdata != None and event.ydata != None:
            print(event.xdata, event.ydata)
            self.selection[0].append(event.xdata)
            self.selection[1].append(event.ydata)

            if len(self.selection[0]) - sum(self.cuts) == 1:
                plt.plot(self.selection[0][-1], self.selection[1][-1], '.', color=f'C0{len(self.cuts)}')
            else:
                plt.plot(self.selection[0][-2:], self.selection[1][-2:], '-', color=f'C0{len(self.cuts)}')
            plt.draw()
            return

    def __press(self, event):
        fig = plt.gcf()
        ax = plt.gca()
                
        if event.key == 'a' or event.key == 'A':
            if len(self.cuts) != 0 and self.cuts[-1] == 0:
                print('Press X to end the current selection and A to start a new selection!')
            else:
                self.cuts.append(0)
                print(f'Begin selection for cut {len(self.cuts)}')
                
                cid = fig.canvas.mpl_connect('button_press_event', self.__onclick)

        elif event.key == 'x' or event.key == 'X':
            if len(self.cuts) == 1:
                if len(self.selection[0]) < 3:
                    print('A selection is required to have at least three points to enclose an area.')
                else:
                    self.cuts[0] = len(self.selection[0])
            else:
                self.cuts[-1] = len(self.selection[0]) - sum(self.cuts)
            plt.plot([self.selection[0][-1], self.selection[0][-self.cuts[-1]]], [self.selection[1][-1], self.selection[1][-self.cuts[-1]]], '-', color=f'C0{len(self.cuts)}')
            plt.draw()
            print('End selection')
            cid = fig.canvas.mpl_connect('button_press_event', self.__onclick)
            fig.canvas.mpl_disconnect(cid)

        elif event.key == 'q' or event.key == 'Q':
            cip = fig.canvas.mpl_connect('key_press_event', self.__press)
            fig.canvas.mpl_disconnect(cip)
            plt.close()

        elif event.key == 'u' or event.key == 'U':
            if len(self.cuts) != 0 and self.cuts[-1] == 0:
                l = ax.lines
                l[-1].remove()
                self.selection[0].pop()
                self.selection[1].pop()
                plt.draw()
            else:
                print('You can\'t use "u" functionality on a completed selection. Please delete the selection using "d" and redo it.')

        elif event.key == 'd' or event.key == 'D':
            if len(self.cuts) != 0 and self.cuts[-1] == 0:
                print('Press X to end the current selection and then D to delete the whole selection!')
            else:
                for i in range(self.cuts[-1]):
                    l = ax.lines
                    l[-1].remove()
                    self.selection[0].pop()
                    self.selection[1].pop()

                l[-1].remove()
                self.cuts.pop()
                print(len(self.cuts))
                plt.draw()

        elif event.key == 'o' or event.key == 'O': 
            ax = plt.gca()
            x = ax.get_xlabel()
            y = ax.get_ylabel()
            header = [f'{self.filename}, x:{x} vs y:{y}\n ']
            indices = np.cumsum(self.cuts)
            split = np.split(self.selection, indices, axis=1)[:-1]
            
            with open(f'{self.filename[:-4]}_cuts.csv', 'w', newline='') as f:
            # create the csv writer
                writer = csv.writer(f)

                # write a header to the csv file
                writer.writerow(header)
                for i in range(len(split)):
                    for j in range(len(split[i])):
                        writer.writerow(split[i][j])
            print(f'Selections outputted to file: {self.filename[:-4]}_cuts.csv')

        return


    def cfd(self, trace, frac, offset):
        """Determines the zero-crossing point of the CFD of a trace. Returns the trace after CFD is done to it, the sample point just before the crossing, and then a point determined with linear interpolation to get as close as possible to the point.

            Args
            ----
            trace : (array, float)
                Array containing the trace that CFD will be done to.

            frac : (float)
                Fraction to multiply the secondary trace before it gets shifted and subtracted.

            offset : (int)
                Number of samples to shift the secondary trace by before subtracting it.

            Returns
            -------
            cfd_array : (array, float)
                Array containing the trace after CFD has been done to it.
            
            zero_cross_index : (int)
                Sample point just before the zero-crossing point.
            
            zero_cross_interp : (float)
                Point determined to be the zero-crossing point using a linear interpolation from either side of the point.
        """

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
        zero_cross_interp = -1

        try:
            # We use np.diff to find where the sign of two adjacent points is different and that 
            # should be the crossing event. We then get the index of that point
            zero_cross_index = cfd_array_max_index + np.where( np.diff( np.sign( cfd_array[cfd_array_max_index:cfd_array_min_index] ) ) != 0 )[0][0]

            zero_cross_interp = (0 - cfd_array[zero_cross_index]) * ( 1 / (cfd_array[zero_cross_index+1] - cfd_array[zero_cross_index]) ) + zero_cross_index

        except Exception as err:   # This used to only except IndexError but I think this is more general
            # print(err)
            # self.fails[4] = 1
            return cfd_array, -1, -1


        return cfd_array, zero_cross_index, zero_cross_interp

