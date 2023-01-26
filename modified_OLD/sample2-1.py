from read_dat import *

#-----------------------------------------------------------------------------
# lst mode read 100 events from ch 0 that fall outside of the area defined
# by cut 1
#-----------------------------------------------------------------------------
file = read_dat("../AmBe.dat")
file.add_selections(mode="p",file="AmBe_cuts_SL.csv")
file.lst_out(events=100,ch=[0],cuts=[0,-1])

#-----------------------------------------------------------------------------
# read the full file for ch 0 
#-----------------------------------------------------------------------------

#file.lst_out(events=False, ch=[0])

#----------------------------------------------------------------------------
# read 5 neutron traces (defined by excluding cut 0)
#----------------------------------------------------------------------------
#file.lst_out(events=5,output=False,traces=True,cuts=[-1])

#----------------------------------------------------------------------------
# read 5 gamma traces and output their L values 
#----------------------------------------------------------------------------
# file.lst_out(events=5,ch=[0],output=[[1,0,0,0,0]],traces=True,cuts=[1])

#---------------------------------------------------------------------------
# output L, S and T for 10 events for long integrals 400, 500 and 600 ns 
#---------------------------------------------------------------------------
# file = read_dat("AmBe.dat",t_long=np.array([400,500,600]))
# file.lst_out(events=10,ch=[0],output=[[1,1,1,0,0]],filename="multi_test.csv")