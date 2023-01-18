#Test for multi short and long integrals. 

from read_dat import *


#create a read_dat object
file = read_dat("AmBe.dat",t_short=np.array([20,25,30,40,50,200]))

#we will be pulling out the L and S values for both channels, or if you have 
#run the lst mode output already you can just import the L and S values 
L=[]
S=[]
for i in range(1000):
	event = file.read_event()
	S.append(event[0].get_pulse_shape())
	L.append(event[0].get_long_integral())
	# print(event[0].get_short_integral())
L=np.array(L)
S= np.array(S).T

cmap_r = cm.get_cmap("Blues_r")
for i in range(len(S)):
	plt.figure()
	# print(L[i],S[i])
	h, xedges, yedges,images= plt.hist2d(L,S[i],[256,256],[[0,40000],[0,1]],norm=colors.LogNorm(vmin=1),cmap=cmap_r)
plt.show()


