import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x_y = [[3024.1935483870966,378.0241935483864,10332.661290322576,29233.870967741925,32132.056451612894,5418.346774193546], [0.5768398268398269,0.6525974025974026,0.7716450216450217,0.768939393939394,0.7067099567099567,0.5876623376623377]]
# x_y = [[0, 0.5, 0.7], [0, 0.7, 0.5]]

print(np.transpose(x_y))

# points = [[10000, 0.7], [5000, 0.1]]

# poly = pat.Polygon(np.transpose(x_y))

# print(poly.contains_points(points))

# ax.add_patch(poly)
# plt.xlim(0, 40000)
# plt.ylim(0, 1)
# plt.show()
