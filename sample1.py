
from read_dat import *


#create a read_dat object
file = read_dat("AmBe.dat")

#we will be pulling out the L and S values for both channels, or if you have 
#run the lst mode output already you can just import the L and S values 
L=[[],[]]
S=[[],[]]
for i in range(1000):#read 1000 events
	event = file.read_event()
	for j in range(len(event)): #per channel
		S[j].append(event[j].get_pulse_shape())
		L[j].append(event[j].get_long_integral())
		# print(event[j].ch,L[j][i])
#now we add a selection in p mode. if you want to add in a new cut you 
#need to run in mode m and send it the L and S values of the channel you want
file.add_selections(mode="p",file="AmBe_cuts_SL.csv")

#using the cuts we have added to our read_dat object we select for the neutron
#and gamma events, run with visual = True if you want to check visually what 
#cuts are active and which events you are selecting for
L_neutrons, S_neutrons = file.select_events(L[0],S[0],cut_id = [0],inc=[-1])
L_gammas, S_gammas = file.select_events(L[0],S[0],cut_id = [0],inc=[1])

#example with multiple cuts and visual
file.select_events(L[0],S[0],cut_id=[0,1,2],inc=[1,1,-1],visual=True)

#save a 2d hist forchannel 0
cmap_r = cm.get_cmap("Blues_r")

plt.figure()     
h, xedges, yedges,images= plt.hist2d(L[0],S[0],[256,256],[[0,40000],[0,1]],norm=colors.LogNorm(vmin=1),cmap=cmap_r)
plt.xlabel("L[ch]")
plt.ylabel("S[ch]")
plt.show()
#or without visual:
# h, xedges, yedges = np.histogram2d(L,S,[256,256],[[0,40000],[0,1]])
header = "AmBe.dat SL [256,256] [[0,40000],[0,1]]"
np.savetxt("SL_AmBe_2dhist.txt",h,header=header)

#save the 1d hists
plt.figure()
n, bins, patches = plt.hist(L_neutrons,256,(0,40000))
plt.xlabel("L[ch]")
plt.ylabel("Counts")
plt.show()
#for without visual:
n_g, bins_g = np.histogram(L_gammas,256,(0,40000))

header = "AmBe.dat L neutrons cut 0, bins 256, [0,40000]\n L[ch]\tCounts"
np.savetxt("SL_AmBe_1dL_neutrons.txt",np.array([bins[:-1],n]).T,header=header)

header = "AmBe.dat L gammas cut 0, bins 256, [0,40000]\n L[ch]\tCounts"
np.savetxt("SL_AmBe_1dL_gammas.txt",np.array([bins_g[:-1],n_g]).T,header=header)

file.get_fails(display=True)