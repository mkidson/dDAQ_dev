from read_dat import *

file = read_dat('Cs137.dat')

file.lst_out(10000, [0], True, False, filename='Cs137_lst_out.csv')

# This should output a file that looks similar to the cut_lst_out.csv file 