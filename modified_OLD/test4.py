from read_dat import *
from event import *

file = read_dat(r'../STNG.dat')

L=[[],[]]
S=[[],[]]
for i in range(1000):#read 1000 events
	event = file.read_event()
	for j in range(len(event)): #per channel
		S[j].append(event[j].get_pulse_shape())
		L[j].append(event[j].get_long_integral())

file.add_selections(L=L[0],S=S[0],mode="m",file="STNG_cuts_SL.csv")