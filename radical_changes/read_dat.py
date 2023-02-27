# 25/03/2020
# read in raw waveform from dat files with some analysis modules
# Chloé Sole 
# Edited (and hopefully improved) 27/01/2023
# Miles Kidson


from scipy.signal import peak_widths
from scipy import signal
from scipy.signal import find_peaks
from event import *
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt 
import matplotlib.patches as pat
from scipy.interpolate import interp1d
import csv


# Open file and skip over header preamble
class read_dat(object):

    headerSize = 72
    maxChannels = 64
    preambleSize = 4+20+4*maxChannels
    
    def __init__(self, file_name, sample_rate=2, CFD=[0.75, 6, 6], t_start=[-80], t_long=[400], t_short=[10], baseline_samples=200, output=[0,0,0,0,0]): #time parameters in ns
        self.fileName = file_name
        self.inputFile = open(file_name, 'rb')
        self.header = self.inputFile.read(self.headerSize)
        self.eventCounter = 0
        self.eventTimeStamp = 0
        self.endFile = False
        self.nsPerSample = sample_rate
        self.CFD = CFD
        self.tStart = np.array(t_start)
        self.tShort = np.array(t_short)
        self.tLong = np.array(t_long)
        self.chActive = np.zeros(8)
        self.fails = np.zeros((len(self.chActive),5))#start, long, short, integral, zero
        self.totFails = np.zeros(len(self.chActive))
        self.baselineSamples = baseline_samples
        self.selection = [[],[]]
        self.cuts = []
        self.polygon_cuts = []
        print('init complete')
       

    def read_event(self):
        """Reads the next event in the file, starting from the beginning, and returns an array of `event` objects, one for each active channel. If the end of the file is reached, it returns `True`.

            Returns
            -------
            event array
                Array of `event` objects, one for each active channel.
        """
        preamble = np.frombuffer(self.inputFile.read(self.preambleSize), dtype=np.uint32)
        if not preamble.any(): #check end of file
            self.end_file = True
            return self.end_file
        #convert timestamp to microseconds
        self.eventTimeStamp = preamble[5]*8e-3 #us 
        self.channelSizes = preamble[6:]
        #array of all channels 0 if that channel isn't active, int value being equal to the 
        #number of samples in that active channel
        self.chActive = np.argwhere(self.channelSizes > 0).flatten()
        # print(self.ch_active)
        

        #init trace array
        traces = np.empty((len(self.chActive), self.channelSizes[self.chActive[0]]))
        self.eventCounter+=1
        #read traces for only active channels
        ev=[]
        for i in range(len(self.chActive)):
            y = np.array(np.frombuffer(self.inputFile.read(self.channelSizes[self.chActive[i]]*2), dtype=np.uint16), dtype=int)
            traces[i]=y
            ev.append(event(self.eventCounter, self.chActive[i], self.eventTimeStamp, traces[i], self.CFD, [self.tStart/self.nsPerSample, self.tShort/self.nsPerSample, self.tLong/self.nsPerSample], self.baselineSamples))
            if ev[i].get_fails() != [0,0,0,0,0]:  # Don't understand this
                self.totFails[i] += 1
            self.fails[i] = np.add(self.fails[i], ev[i].get_fails())

        return ev


    def lst_out(self, events=False, ch=True, output=True, traces=False, cuts=False, inc=None, filename=""):
        """Reads a number of events from the file buffer for the channels specified, applying cuts if given. These cuts can be made using `read_dat.add_selections()`. Outputs a csv file for each active channel, containing

            Args
            ----
            events : (False or int, optional) 
                If int, specifies the number of events to read from the file. If False, it reads all events left in the file. Defaults to False.

            ch : (True or int array, optional)
                If True, reads out from all channels. If int array, reads out from channels specified, with numbering from 0. Defaults to True.

            output : (bool or 5x1 int array, optional)
                If True L [ch], S [ch], T_trigger [us], baseline [bits], pulse height [bits] are read out into a file per channel. If int array, it specifies the parameters to output with binary, where 1 means output and 0 means ignore. Defaults to True.

            traces : (bool, optional)
                If True, traces will be output in list mode in a csv file per channel. Otherwise traces will not be output. Defaults to False.

            cuts : (bool or int array, optional)
                If False, no cuts are applied to output. If an int array, it acts just like the cuts argument for `read_dat.select_events()`. If True it defaults to no cuts. Defaults to False.

            inc : (None or int array, optional)
                If int array, acts precisely like the inc argument for `read_dat.select_events()`, where a 1 includes the cut and -1 excludes it. If left as None, all cuts are considered to be included. Defaults to None.

            filename : (str, optional)
                Desired output file name. If left empty, it uses the `file_name` of the original file. Defaults to "".

            Returns
            -------
            Should return nothing. If the arguments are supplied incorrectly, returns None.
        """
        ev = self.read_event()
        out = []
        writer_trace = []
        writer_params = []
        if ch == True:
            ch = np.arange(len(ev))
        elif ch == False:
            print('ERROR: ch must either be True or an array of ints. Returning None')
            return None
        if output != False:
            if output == True:
                for i in range(len(ch)):
                    out.append([1,1,1,1,1])
            else:
                out = output
        else:
            for i in range(len(ch)):
                out.append([0,0,0,0,0])

        out = np.array(out)
        if cuts != False:
            if cuts == True:
                print('Please input an array if you want to apply cuts. Defaulting to no cuts')
            else:
                cuts = np.array(cuts)
                if inc == None: # Setting inc to all 1, so all included
                    inc = np.ones(len(cuts))
                elif len(inc) != len(cuts):
                    print('ERROR: len(inc) must be the same as len(cuts). Returning None')
                    return None
                else:
                    inc = np.array(inc)


        # initiate the output files for the traces and other parameters, one per channel
        # only doing the headers now, no data yet
        for i in range(len(ch)):
            if output != False:
                header = [f'{self.fileName[:-4]} channel {ch[i]}, {events} events, cuts {cuts}']
                if len(filename)==0:
                    f = open(f'{self.fileName[:-4]}_params_{ch[i]}.csv', 'w', newline='')
                else:
                    f = open(filename, 'w', newline='')
                writer_params.append(csv.writer(f))
                writer_params[i].writerow(header)
                if len(self.tLong) > 1 or len(self.tShort) > 1: # idk what this is doing
                    writer_params[i].writerow([f'long integral (ns): {self.tLong}, short integral (ns): {self.tShort}'])
                labels = np.array(['L [ch]', 'S[ch]', 'T (trigger) [us]', 'baseline', 'pulse height [bits]'])
                writer_params[i].writerow(labels[out[i] == 1])

            if traces == True:
                header = [f'{self.fileName[:-4]} channel {ch[i]}, {events} events, cuts {cuts}']
                f = open(f'{self.fileName[:-4]}_trace_{ch[i]}.csv', 'w', newline='')
                # create the csv writer
                writer_trace.append(csv.writer(f))

                # write a header to the csv file
                writer_trace[i].writerow(header)

        counter = 1
        #iterate over the desired number of events and write out the traces and other parameters      
        while True:
            for i in range(len(ch)):
                if output != False:
                    calc_params = np.array([np.array(ev[ch[i]].get_long_integral()), np.array(ev[ch[i]].get_pulse_shape()), ev[ch[i]].get_t0(), ev[ch[i]].get_baseline(), ev[ch[i]].get_pulse_height()[0]])
                    
                    if type(cuts) != bool and i == 0:   # checks if there are cuts that need to be made and does them one at a time
                        # Needs to be this way so we can get a specific number of events
                        L, S = self.select_events(calc_params[0], calc_params[1], 'L', 'S', cuts, inc, visual=False)
                        if len(L) == 0:
                            counter -= 1
                            break
                    writer_params[i].writerow(calc_params[out[i] == 1])

                if traces == True:
                    writer_trace[i].writerow(ev[ch[i]].get_trace())

            if events > counter or events == False:
                ev = self.read_event()
                counter += 1
                if counter % 1000 == 0:
                    print(f'{counter} events')

                if ev == True: #if end of file was reached break read loop
                    break 
            else:
                break
        print('End reading')


    def get_fails(self, display=False):
        """Returns an nx5 array of ints, for n channels, in the format of [start, long, short, integral, zero]. The value at the associated index indicates the number of fails out of the processed events that have failed that check. Runs this check for all events processed.

        ## Fail Details

        | Index | Fail Name | Fail Condition    |
        |---    |---    |---    |
        | 0 | start | The start time is set outside of the acquisition window   |
        | 1 | long  | The long integral end gate is outside of the acquisition window   |
        | 2 | short | The short integral end gate is outside of the acquisition window  |
        | 3 | integral  | The calculated short integral is negative or the calculated long integral has a smaller value than the calculated short integral  |
        | 4 | zero  | The CFD calculation failed to return a reasonable t_0  |

            Args
            ----
            display : (bool, optional)
                If True, prints out the breakdown of fails to terminal. Defaults to False.

            Returns
            -------
            fails : nx5 int array
                Fail information in the format of [start, long, short, integral, zero] for n channels. The value at the associated index indicates the number of fails out of the processed events that have failed that check. See description for details on what each fail means.

            totFails : int array
                Number of events failed per channel.

            eventCounter : int
                Total number of events processed
        """
        if display:
            for i in range(len(self.chActive)):
                if np.sum(self.fails) == 0:
                    print(f'Channel: {self.chActive[i]}\tEvents: {self.eventCounter}\tFails: {self.totFails[i]}')
                else:
                    print(f'Channel: {self.chActive[i]}\tEvents: {self.eventCounter}\tFails: {self.totFails[i]}\ntstart: {self.fails[i][0]}\ttlong: {self.fails[i][1]}\ttshort: {self.fails[i][2]}\tintegral: {self.fails[i][3]}\tt0: {self.fails[i][4]}')
        return self.fails, self.totFails, self.eventCounter


    def add_selections(self, x_param=[], y_param=[], x_param_name='L', y_param_name='S', mode='m', lims=[[0, 50000], [0, 1]], file=False):
        """Method to add multiple cuts to the events. Can be run in manual `m` or predetermined `p` modes. Manual mode allows the user to input arrays of `x_param` and `y_param` values so that cuts can be made to separate, for example, the neutron and gamma events. Predetermined mode allows the selections to be input from a file provided and no input is needed.

            Args
            ----
            x_param : (float array, optional)
                Array of x_param values for the processed events. Defaults to [].

            y_param : (float array, optional)
                Array of y_param values for the processed events. Defaults to [].

            x_param_name : (str, optional)
                Name for the value provided to x_param, used for display only. Defaults to 'L'.

            y_param_name : (str, optional)
                Name for the value provided to y_param, used for display only. Defaults to 'S'.

            mode : (str, optional)
                Character specifying mode of operation. 'm' is manual mode, where x_param and y_param arrays are required so selections can be made visually on a 2d histogram. 'p' is predetermined mode, where cuts are created from selections obtained from `file`. Defaults to 'm'.

            lims : (2x2 float array, optional)
                Array of x and y limits for the visual aid 2d histogram. Defaults to [[0, 50000], [0, 1]].

            file : (bool or str, optional)
                Specifies the input file to be used when in predetermined mode. If False when mode='p', it will request the file as input. Defaults to False.
        
        ---
        
        Selections are made by clicking on points on a 2D histogram, defining a polygon that encloses the events of interest. In manual mode, these arrays must be supplied. Once the cuts have been made, they get saved to the `read_dat` object as polygons and are used by the `lst_out` and `select_events` methods. 

        By default, no file containing the coordinates of the selections is output when in manual mode. A key needs to be pressed and they will be output to a file named "`file_name`\_cuts.csv", where `file_name` is the file used to instantiate the `read_dat` class.

        ### Making selections in manual mode

        When in manual mode, we can make selections on the 2D histogram. There are a number of keys that can be pressed to activate certain commands.

        | Key | Action    |
        |---    |---    |
        | a, A:  | Start a new selection   |
        | u, U:  | Undo previous point, only usable while in a selection |
        | x, X:  | End current selection. Can only end a selection if there are more than 2 co-ordinates in the selection    |
        | d, D:  | Delete previous completed selection   |
        | o, O:  | Output the selections added to the file "`file_name`\_cuts.csv"  |
        | q, Q:  | Quit, ends visual guide and re-enters the main code segment   |

        A typical selection would proceed as follows:
        - Press "a" to start the selection.
        - Choose at least 3 coordinates to enclose an area. If less than 3 are chosen before the next step, it will throw an error.
        - Press "x" to end the selection. 
        - If you want to export the selection so it can be used again later, press "o".
        - Finally press "q" to quit.
        """
        if mode == 'm':
            if len(x_param) == 0 or len(y_param) == 0:  # checking if there were arrays input, otherwise it doesn't work
                print('ERROR: No x_param and y_param values recieved. x_param and y_param values required for manual cut mode. Returning None.')
                return None
            else:
                if len(x_param) != len(y_param):    # checking length of input arrays
                    print('ERROR: x_param and y_param need to be the same length. Returning None')
                    return None
                else:
                    fig = plt.figure(1)
                    cmap_r = cm.get_cmap('Blues_r')
                    plt.hist2d(x_param, y_param, [256,256], lims, norm=colors.LogNorm(vmin=1), cmap=cmap_r)
                    plt.colorbar()
                    plt.xlabel(f'{x_param_name} [ch]')
                    plt.ylabel(f'{y_param_name} [ch]')
                    cip = fig.canvas.mpl_connect('key_press_event', self.__press)
                    plt.show(block=True)

        elif mode == 'p':
            if file == False:
                file = input('An input file is required to run in mode "p"\nPlease enter your filename.csv:')
            else:
                if file[-3:] != 'csv':  # checking if the filename has .csv, if not it adds it
                    file = file + '.csv'
                with open(file, 'r') as read_file:
                    csv_reader = csv.reader(read_file)
                    next(csv_reader)
                    i = 0
                    for row in csv_reader:
                        if i % 2 == 0:
                            self.cuts.append(len(row))
                        A = [float(x) for x in row]
                        self.selection[i % 2].extend(A)
                        i += 1
                    print('Selections Imported')
        
        # Turning points into polygons
        indices = np.cumsum(self.cuts)
        split_cuts = np.split(self.selection, indices, axis=1)[:-1]     # splitting the selections into their separate cuts. It's inefficient but a mess to redo so I'm leaving it
        for split in split_cuts:
            self.polygon_cuts.append(pat.Polygon(np.transpose(split)))
        
        print('Polygons Created')



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
            header = [f'{self.fileName}, x:{x} vs y:{y}\n ']
            indices = np.cumsum(self.cuts)
            split = np.split(self.selection, indices, axis=1)[:-1]
            
            with open(f'{self.fileName[:-4]}_cuts.csv', 'w', newline='') as f:
            # create the csv writer
                writer = csv.writer(f)

                # write a header to the csv file
                writer.writerow(header)
                for i in range(len(split)):
                    for j in range(len(split[i])):
                        writer.writerow(split[i][j])
            print(f'Selections outputted to file: {self.fileName[:-4]}_cuts.csv')

        return


    def select_events(self, x_param, y_param, x_param_name='L', y_param_name='S', cut_id=[0], inc=[1], visual=False, lims = [[0, 50000], [0, 1]]):
        """Method to pull the events which fall within the desired area. The area is defined by the inclusion or exclusion of cuts made using `add_selections`. If no selections have been made, then all events will be returned. A visual aid can be shown to aid in understanding which events are included or excluded.

            Args
            ----
            x_param : (float array, optional)
                Array of x_param values for the processed events. Defaults to [].

            y_param : (float array, optional)
                Array of y_param values for the processed events. Defaults to [].

            x_param_name : (str, optional)
                Name for the value provided to x_param, used for display only. Defaults to 'L'.

            y_param_name : (str, optional)
                Name for the value provided to y_param, used for display only. Defaults to 'S'.

            cut_id : (int array, optional)
                Array of cut ids which are either included or excluded. Cuts are in the order they were created. Defaults to [0].

            inc : (int array, optional)
                Array of 1 or -1 to indicate which cuts include or exclude events. Requires `len(inc) = len(cut_id)`. Defaults to [1].

            visual : (bool, optional)
                If True a 2D histogram with the included and excluded events is displayed with the cut boundaries. If False nothing is displayed. Defaults to False.

            lims : (2x2 float array, optional)
                Array of x and y limits for the visual aid 2D histogram. Defaults to [[0, 50000], [0, 1]].

            Returns
            -------
            x_param_cut : ndarray
                Array of x_param values for events that passed the cuts applied.
            
            y_param_cut : ndarray
                Array of y_param values for events that passed the cuts applied.
        """        
        if visual == True:  # creating the subplots now with a check or else there were millions produced when doing lst_out
            fig, ax = plt.subplots()

        x_param = np.array(x_param) # numpy arrays are easier to work with but I wanted the input to not have to be.
        y_param = np.array(y_param)
        check_points = np.transpose([x_param, y_param]) # array of x-y pairs that will be checked whether they are in or out of the polygons
        mask = []

        for i in range(len(cut_id)):
            if len(np.shape(check_points)) < 2: # checks if there is more than one point, otherwise it needs to use a different method
                temp_mask = self.polygon_cuts[cut_id[i]].contains_point(check_points)
            else:
                temp_mask = self.polygon_cuts[cut_id[i]].contains_points(check_points)


            if inc[i] == -1:    # a bit of chloe magic to determine if the point should be included or excluded based on the cuts and inc args
                temp_mask = np.invert(temp_mask)
            if i != 0:
                if inc[i] == 1:
                    mask = (mask) | (temp_mask)
                elif inc[i] == -1:
                    mask = (mask) & (temp_mask)
            else:
                mask = temp_mask
        
            if visual == True:  # plotting outline of the polygon used
                self.polygon_cuts[cut_id[i]].set_fill(False)
                self.polygon_cuts[cut_id[i]].set_ec('r')
                self.polygon_cuts[cut_id[i]].set_ls('--')
                ax.add_patch(self.polygon_cuts[cut_id[i]])

        if visual == True:  # plotting red if it's included, blue if it's excluded
            cmap_b, cmap_r = cm.get_cmap('Blues_r'), cm.get_cmap('Reds_r')
            plt.title('Cut Check')
            if x_param[mask] != []:
                plt.hist2d(x_param[mask], y_param[mask], [256,256], lims, norm=colors.LogNorm(vmin=1), cmap=cmap_r)
                plt.colorbar(label='Included Events [Counts]', pad=0.1, shrink=0.5, anchor=(0.0, 0.5))
            if x_param[~mask] != []:
                plt.hist2d(x_param[~mask], y_param[~mask], [256,256], lims, norm=colors.LogNorm(vmin=1), cmap=cmap_b)
                plt.colorbar(label='Excluded Events [Counts]', shrink=0.5)

            plt.xlabel(f'{x_param_name} [ch]')
            plt.ylabel(f'{y_param_name} [ch]')
            plt.show(block=True)
        return x_param[mask], y_param[mask]
