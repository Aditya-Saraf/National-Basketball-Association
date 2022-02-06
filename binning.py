import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

path = './data'
player_df = pd.read_csv(path + '/player_data.csv')
player_df['po_pts'] = player_df['po_pts'].fillna(0)
player_df['po_gp'] = player_df['po_gp'].fillna(0)
player_df['rs_gp'] = player_df['rs_gp'].fillna(0)
player_df['rs_pts'] = player_df['rs_pts'].fillna(0)
player_df['pts_avg'] = (player_df['po_pts'] + player_df['rs_pts']) / (player_df['po_gp'] + player_df['rs_gp'])
player_df.to_csv(path + "/temp_player_all_data.csv")

boxplot = player_df.boxplot(column=['pts_avg'])
plt.show()
"""Weak: < 3
Average: > 3 and < 9
Strong: > 9 
"""
weak_player_count = player_df.query('pts_avg < 3')['pts_avg'].count()
average_player_count = player_df[player_df['pts_avg'].between(3, 9)]['pts_avg'].count()
strong_player_count = player_df.query('pts_avg > 9')['pts_avg'].count()

barlist = plt.bar([1,2,3], [weak_player_count,average_player_count,strong_player_count], width = 0.4)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
plt.xticks(range(1,4), ['Weak', 'Average', 'Strong'])
plt.show()


