#25/03/2020
#read in raw waveform from dat files with some analysis modules
#Chloé Sole


from scipy.signal import peak_widths
from scipy import signal
from scipy.signal import find_peaks
from event import *
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import csv


# Open file and skip over header preamble
class read_dat(object):
    headerSize = 72
    maxChannels = 64
    preambleSize = 4+20+4*maxChannels
    
    def __init__(self,file_name,sample_rate = 2,CFD = [0.75,6,6],t_start = [-80],t_long = [400], t_short = [10],baseline_samples = 200,output=[0,0,0,0,0]): #time parameters in ns
        self.fileName = file_name
        self.inputFile = open(file_name, "rb")
        self.header = self.inputFile.read(self.headerSize)
        self.eventCounter=0
        self.eventTimeStamp=0
        self.endFile = False
        self.nsPerSample = sample_rate
        self.CFD = CFD
        self.tStart = np.array(t_start)
        self.tShort = np.array(t_short)
        self.tLong = np.array(t_long)
        self.chActive =np.zeros(8)
        self.fails = np.zeros((len(self.chActive),5))#start, long, short, integral, zero
        self.totFails=np.zeros(len(self.chActive))
        self.baselineSamples=baseline_samples
        self.selection=[[],[]]
        self.cuts=[]
        print('init complete')
       
#--------------------------------------------------------------------------------------------------------------------------------------------------
# Function to read a single event
#--------------------------------------------------------------------------------------------------------------------------------------------------
    def read_event(self):
        preamble = np.frombuffer(self.inputFile.read(self.preambleSize),dtype=np.uint32)
        if not preamble.any(): #check end of file
            self.end_file = True
            return self.end_file
        #convert timestamp to microseconds
        self.eventTimeStamp = preamble[5]*8e-3 #us 
        self.channelSizes = preamble[6:]
        #array of all channels 0 if that channel isn't active, int value being equal to the 
        #number of samples in that active channel
        self.chActive = np.argwhere(self.channelSizes>0).flatten()
        # print(self.ch_active)
        

        #init trace array
        traces = np.empty((len(self.chActive),self.channelSizes[self.chActive[0]]))
        self.eventCounter+=1
        #read traces for only active channels
        ev=[]
        for i in range(len(self.chActive)):
            y = np.array(np.frombuffer(self.inputFile.read(self.channelSizes[self.chActive[i]]*2), dtype=np.uint16),dtype=int)
            traces[i]=y
            ev.append(event(self.eventCounter,self.chActive[i],self.eventTimeStamp,traces[i],self.CFD,[self.tStart/self.nsPerSample,self.tShort/self.nsPerSample,self.tLong/self.nsPerSample],self.baselineSamples))
            if ev[i].get_fails()!=[0,0,0,0,0]:  # Don't understand this
                self.totFails[i]+=1
            self.fails[i] = np.add(self.fails[i],ev[i].get_fails())

        return ev

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#  Function to read multiple events in the file or whole file 
#       events: number of events to be read, must be integer, if False, the full file will be read. 
#       ch:     which channels to read, True reads all active channels, else value must be an array of channel indices (starting from zero) of the ACTIVE ch
#       output: lst mode output of all parameters if true. if only certain parameters to be read [1,0,0,1,0] where each index represents a different 
#               parameter. 
#                   [L, S, (trigger)T us, baseline, pulse height]    
#       traces: if True traces will be output in lst mode one file per channel. 
#       cuts: if False no cuts will be selected. else an array of len cuts is expected 1 for include, 0 for ignore, -1 for exclude.
#                   example: 3 cuts have been added, I want to include cut 1, and remove cut 2 and ignore cut 3 --> [1,-1,0]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
    def lst_out(self,events=False, ch=True,output=True, traces=False, cuts=False,filename = ""):
        ev = self.read_event()
        out=[]    
        writer_trace =[]
        writer_params =[]
        if ch==True:
            ch = np.arange(len(ev))
        if output !=False:
            if output == True:
                for i in range(len(ch)):
                    out.append([1,1,1,1,1])
            else:
                out=output
        else:
            for i in range(len(ch)):
                out.append([0,0,0,0,0])

        out = np.array(out)
        if cuts!=False:
            if cuts == True:
                print('Please input an array if you want to apply cuts. Defaulting to no cuts')
            else:
                cuts = np.array(cuts)
                inc = cuts[cuts!=0]
                cuts = np.arange(len(cuts))[cuts!=0]


        #initiate the output files for the traces and other parameters, one per channel
        for i in range(len(ch)):
            if output!=False:
                header = ["{} channel {}, {} events, cuts {}".format(self.fileName[:-4],ch[i],events,cuts)]
                if len(filename)==0:
                    f=open("{}_params_{}.csv".format(self.fileName[:-4],ch[i]), 'w',newline='')
                else:
                    f=open(filename, 'w',newline='')
                writer_params.append(csv.writer(f))
                writer_params[i].writerow(header)
                if len(self.tLong)>1 or len(self.tShort)>1:
                    writer_params[i].writerow(["long integral (ns): {}, short integral (ns): {}".format(self.tLong, self.tShort)])
                labels = np.array(["L [ch]", "S[ch]", "T (trigger) [us]", "baseline", "pulse height [bits]"])
                writer_params[i].writerow(labels[out[i]==1])

            if traces==True:
                header=["{} channel {}, {} events, cuts {}".format(self.fileName[:-4],ch[i],events,cuts)]
                f=open("{}_trace_{}.csv".format(self.fileName[:-4],ch[i]), 'w',newline='')
                # create the csv writer
                writer_trace.append(csv.writer(f))

                # write a header to the csv file
                writer_trace[i].writerow(header)

        counter=1
        #iterate over the desired number of events and write out the traces and other parameters      
        while True:
            for i in range(len(ch)):
                if output!=False:
                    calc_params = np.array([np.array(ev[ch[i]].get_long_integral()),np.array(ev[ch[i]].get_pulse_shape()),ev[ch[i]].get_t0(),ev[ch[i]].get_baseline(),ev[ch[i]].get_pulse_height()[0]])
                    
                    if type(cuts)!=bool and i==0:
                        L,S=self.select_events(calc_params[0],calc_params[1],cuts,inc)
                        if len(L)==0:
                            counter=counter-1
                            break
                    writer_params[i].writerow(calc_params[out[i]==1])

                if traces == True:
                    writer_trace[i].writerow(ev[ch[i]].get_trace())

            if events>counter or events==False:
                ev=self.read_event()
                counter+=1
                if counter%1000 ==0 :
                    print("{} events".format(counter))

                if ev == True: #if end of file was reached break read loop
                    break 
            else:
                break
        print("End reading")

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Function to pull fail information from number of events processed
#--------------------------------------------------------------------------------------------------------------------------------------------------
    def get_fails(self,display=False):
        if display:
            for i in range(len(self.chActive)):
                if np.sum(self.fails)==0:
                    print("Channel: {}\tEvents: {}\tFails: {}".format(self.chActive[i],self.eventCounter,self.totFails[i]))
                else:
                    print("Channel: {}\tEvents: {}\tFails: {}\ntstart: {}\ttlong: {}\ttshort: {}\tintegral: {}\tt0: {}".format(self.chActive[i],self.eventCounter,self.totFails[i],*self.fails[i]))
        return self.fails, self.totFails, self.eventCounter

#--------------------------------------------------------------------------------------------------------------------------------------------------
# enclosed area selections
#   a: auto cut 
#   m: manual with visual aid
#   p: manual user provided cut co-ordinates
#   file: input file for previously determined cuts 
#   L and S: required if mode is m
#--------------------------------------------------------------------------------------------------------------------------------------------------
    def add_selections(self,L=[],S=[],mode="m",lims = [[0,50000],[0,1]],file=False):
        if mode == "m":
            if len(L)==0 or len(S)==0:
                print("Error! No S and L values recieved. L and S values required for manual cut mode. ")
            else:
                fig = plt.figure(1)
                # plt = fig.add_subplot(111)
                cmap_r = cm.get_cmap("Blues_r")
                plt.hist2d(L,S,[256,256],lims,norm=colors.LogNorm(vmin=1),cmap=cmap_r)
                plt.colorbar()
                # plt.title(r"Press $a$ to add a cut, $x$ to end the cut, $u$ to undo the last point added to the current cut, $d$ ")
                plt.xlabel("L [ch]")
                plt.ylabel("S [ch]")
                cip=fig.canvas.mpl_connect('key_press_event', self.__press)
                plt.show()


        # elif mode == "a":
        #     print("Auto mode not integrated yet")
        elif mode == "p":
            if file == False:
                file = input("An input file is required to run in mode \"p\"\nPlease enter your filename.csv:")
            else:
                if file[-3:]!="csv":
                    file = file+".csv"
                with open (file,"r") as read_file:
                    csv_reader = csv.reader(read_file)
                    next(csv_reader)
                    i = 0
                    for row in csv_reader:
                        if i%2==0:
                            self.cuts.append(len(row))
                        A = [float(x) for x in row]
                        self.selection[i%2].extend(A)
                        i+=1
                    print("Selections Imported")


##################################################################################################################################################
#   HELPER FUNCTIONS FOR: add_selections
#               while adding cuts:
#                    key press a or A: starting a new selection
#                    key press x or X: terminate current cut
#                    key press q or Q: terminate 
#                    key press o or O: Output the cuts created 
##################################################################################################################################################
    def __onclick(self,event):
        if event.xdata != None and event.ydata !=None:
            print(event.xdata, event.ydata)
            self.selection[0].append(event.xdata)
            self.selection[1].append(event.ydata)

            if len(self.selection[0])-sum(self.cuts)==1:
                plt.plot(self.selection[0][-1],self.selection[1][-1],".",color="C0{}".format(len(self.cuts)))
            else:
                plt.plot(self.selection[0][-2:],self.selection[1][-2:], "-",color="C0{}".format(len(self.cuts)))
            plt.draw()
            return

    def __press(self,event):
        fig = plt.gcf()
        ax=plt.gca()
                
        if event.key =="a" or event.key =="A":
            if len(self.cuts)!=0 and self.cuts[-1]==0:
                print("Press X to end the current selection and A to start a new selection!")
            else:
                self.cuts.append(0)
                print("Begin selection for cut {}".format(len(self.cuts)))
                
                cid = fig.canvas.mpl_connect('button_press_event', self.__onclick)
        elif event.key =="x" or event.key =="X":
            if len(self.cuts)==1:
                if len(self.selection[0])<3:
                    print("A selection is required to have at least three points to enclose an area.")
                else:
                    self.cuts[0]=len(self.selection[0])
            else:
                self.cuts[-1]=len(self.selection[0])-sum(self.cuts)
            plt.plot([self.selection[0][-1],self.selection[0][-self.cuts[-1]]],[self.selection[1][-1],self.selection[1][-self.cuts[-1]]], "-",color="C0{}".format(len(self.cuts)))
            plt.draw()
            print("End selection")
            cid = fig.canvas.mpl_connect('button_press_event', self.__onclick)
            fig.canvas.mpl_disconnect(cid)
        elif event.key =="q" or event.key =="Q":
            cip=fig.canvas.mpl_connect('key_press_event', self.__press)
            fig.canvas.mpl_disconnect(cip)
            plt.close()
        elif event.key =="u" or event.key =="U":
            if len(self.cuts)!=0 and self.cuts[-1]==0:
                l=ax.lines
                l[-1].remove()
                self.selection[0].pop()
                self.selection[1].pop()
                plt.draw()
            else:
                print("You can't use \"u\" functionality on a completed selection. Please delete the selection using \"d\" and redo it.")
        elif event.key =="d" or event.key== "D":
            if len(self.cuts)!=0 and self.cuts[-1]==0:
                print("Press X to end the current selection and then D to delete the whole selection!")
            else:
                for i in range(self.cuts[-1]):
                    l=ax.lines
                    l[-1].remove()
                    self.selection[0].pop()
                    self.selection[1].pop()

                l[-1].remove()
                self.cuts.pop()
                print(len(self.cuts))
                plt.draw()
        elif event.key == "o" or event.key == "O": 
            # f = open("cuts_SL.txt","w")
            ax = plt.gca()
            x = ax.get_xlabel()
            y = ax.get_ylabel()
            header = ["{}, x:{} vs y:{}\n ".format(self.fileName,x,y)]
            indices = np.cumsum(self.cuts)
            split = np.split(self.selection,indices,axis=1)[:-1]
            
            with open("{}_cuts_SL.csv".format(self.fileName[:-4]), 'w',newline='') as f:
            # create the csv writer
                writer = csv.writer(f)

                # write a header to the csv file
                writer.writerow(header)
                for i in range(len(split)):
                    for j in range(len(split[i])):
                        writer.writerow(split[i][j])
            print("Selections outputted to file: {}_cuts_SL.csv".format(self.fileName[:-4]))

        return
###########################################################################################################################################
# method to return the events which fall within the desired cuts. 
# cuts need to have been added already to the read_dat object
#
#
###########################################################################################################################################
    def select_events(self,L,S, cut_id=[0],inc=[1],visual=False,lims = [[0,50000],[0,1]]):
        # cut_id=cut_id[inc!=0]
        L= np.array(L)
        S = np.array(S)
        mask = []
        for i in range(len(cut_id)):
            indices = np.cumsum(self.cuts)
            split = np.split(self.selection,indices,axis=1)[:-1]
            #select the cut that we are looking at
            # if len self.cuts==1:
                # cut = self.selection
            print(cut_id[i])
            cut = np.array(split[cut_id[i]])

            #split the cut into a top slice and a bottom slice
            split_1 = np.split(cut,[np.argmax(cut[0])+1],axis=1)[0]
            split_2 = np.flip(np.split(np.roll(cut,-1,axis=1),[np.argmax(cut[0])-1],axis=1)[1],axis=1)
            

            #Check which side is on top
            if (split_1[1][1]-split_1[1][0])/(split_1[0][1]-split_1[0][0])>(split_2[1][1]-split_2[1][0])/(split_2[0][1]-split_2[0][0]):
                upper_cut = split_1
                lower_cut = split_2
            else:
                lower_cut=split_1
                upper_cut=split_2

            #splines to represent the upper and lower lines of the cuts
            upp_spline = interp1d(upper_cut[0],upper_cut[1],bounds_error=False,fill_value=0)
            low_spline = interp1d(lower_cut[0],lower_cut[1],bounds_error=False,fill_value=1)
            
            temp_mask = (S-low_spline(L)>0)&(S-upp_spline(L)<0)

            if inc[i]==-1:
                temp_mask=~temp_mask
            if i !=0:
                if inc[i]==1:
                    mask = (mask)|(temp_mask)
                elif inc[i]==-1:
                    mask= (mask)&(temp_mask)
            else:
                mask = temp_mask
            if visual ==True:
                plt.plot(upper_cut[0],upp_spline(upper_cut[0]),"r--")
                plt.plot(lower_cut[0],low_spline(lower_cut[0]),"r--")

        if visual==True:


            cmap_b,cmap_r = cm.get_cmap("Blues_r"),cm.get_cmap("Reds_r")
            plt.title("Cut Check")
            if L[mask]!=[]:
                plt.hist2d(L[mask],S[mask],[256,256],lims,norm=colors.LogNorm(vmin=1),cmap=cmap_r)
                plt.colorbar(label="Included Events [Counts]", pad=0.1,shrink=0.5,anchor=(0.0, 0.5))
            if L[~mask]!=[]:
                plt.hist2d(L[~mask],S[~mask],[256,256],lims,norm=colors.LogNorm(vmin=1),cmap=cmap_b)
                plt.colorbar(label="Excluded Events [Counts]",shrink=0.5)

            plt.xlabel("L [ch]")
            plt.ylabel("S [ch]")
            plt.show()
        return L[mask],S[mask]

###########################################################################################################################################



