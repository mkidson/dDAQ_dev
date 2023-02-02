from read_dat import *
from event import *
import matplotlib.patches as pat


file = read_dat(r'STNG.dat')

L=[[],[]]
S=[[],[]]
for i in range(500):#read 1000 events
	event = file.read_event()
	for j in range(len(event)): #per channel
		S[j].append(event[j].get_pulse_shape())
		L[j].append(event[j].get_long_integral())

file.add_selections(x_param=L[0],y_param=S[0],mode="m")#, file='/home/mkidson/gitRepos/dDAQ_dev/dDAQ_final/STNG_cuts_SL.csv')

L_neutrons, S_neutrons = file.select_events(L[0], S[0], cut_id=[0,1], inc=[1,-1], visual=True)
# print(L_neutrons, S_neutrons)

# poly = pat.Polygon(np.transpose([[504.032258064517,33770.16129032257,33770.16129032257,10710.685483870962,1260.0806451612898],[0.6066017316017317,0.7310606060606061,0.7851731601731602,0.7797619047619049,0.6823593073593074]]))
# mask = poly.contains_points(np.transpose([L[0], S[0]]))
# print(mask)
# print(np.array(L[0])[mask])

# plt.figure()
plt.hist2d(L_neutrons, S_neutrons, [256, 256], norm=colors.LogNorm(vmin=1))
plt.show()