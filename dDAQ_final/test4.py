from read_dat import *
from event import *

file = read_dat(r'STNG.dat')

L=[[],[]]
S=[[],[]]
for i in range(100):#read 1000 events
	event = file.read_event()
	for j in range(len(event)): #per channel
		S[j].append(event[j].get_pulse_shape())
		L[j].append(event[j].get_long_integral())

file.add_selections(L=L[0],S=S[0],mode="m")#, file='/home/mkidson/gitRepos/dDAQ_dev/dDAQ_final/STNG_cuts_SL.csv')

L_neutrons, S_neutrons = file.select_events(L[0], S[0], cut_id=[0], inc=[1], visual=True)

plt.figure()
plt.hist2d(L_neutrons, S_neutrons, [256, 256], norm=colors.LogNorm(vmin=1))
plt.show()