import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
from networkx.algorithms import community
import community as community_louvain
import collections

df = pd.read_excel('Data.xlsx')

#Nation renaming
df.rename(columns={'Points      ': 'Points'},inplace=True)
df['From country'] = df['From country'].str.replace('The Netherands','Netherlands')
df['To country'] = df['To country'].str.replace('The Netherands','Netherlands')
df['From country'] = df['From country'].str.replace('The Netherlands','Netherlands')
df['To country'] = df['To country'].str.replace('The Netherlands','Netherlands')
df['From country'] = df['From country'].str.replace('F.Y.R. Macedonia','Macedonia')
df['To country'] = df['To country'].str.replace('F.Y.R. Macedonia','Macedonia')
df['From country'] = df['From country'].str.replace('North Macedonia','Macedonia')
df['To country'] = df['To country'].str.replace('North Macedonia','Macedonia')
df['From country'] = df['From country'].str.replace('Bosnia & Herzegovina','Boznia&Herz')
df['To country'] = df['To country'].str.replace('Bosnia & Herzegovina','Boznia&Herz')  
df['From country'] = df['From country'].str.replace('Czech Republic','Czechia')
df['To country'] = df['To country'].str.replace('Czech Republic','Czechia')
df['From country'] = df['From country'].str.replace('United Kingdom','UK')
df['To country'] = df['To country'].str.replace('United Kingdom','UK')

#Drop Bosnia & Herzegovina for not earning points in a final
temp = df[df['From country']=='Boznia&Herz'].index 
df = df.drop(temp)
temp = df[df['To country']=='Boznia&Herz'].index 
df = df.drop(temp)

#Select only, finals, televotes
temp = df[df['(semi-) final']!='f'].index
df = df.drop(temp)
temp = df[df['Jury or Televoting']=='J'].index
df = df.drop(temp)

#Scale points to remove weighting to 1st and 2nd
df.replace(10, 9, inplace=True)
df.replace(12, 10, inplace=True)

#Calculate the points scored by each nation each year
pointsScored = np.zeros((len(df['To country'].unique()), len(df['Year'].unique())))
for i, country in enumerate(df['To country'].unique()):
    for k, year in enumerate(df['Year'].unique()):
        pointsScored[i,k] = df[df['Year']==year][df['To country']==country].Points.sum()

#Calculate point proportions and average them
for i, country in enumerate(df['To country'].unique()):
    for k, year in enumerate(df['Year'].unique()):
        indx = df[df['Year']==year][df['To country']==country]['Points'].index
        for ind in indx:
            if pointsScored[i, k] == 0:
                df.loc[ind, 'Points'] = np.NaN
            else:
                df.loc[ind, 'Points'] = (df.loc[ind, 'Points']) / (pointsScored[i, k])        

dfAverages = df.groupby(['From country','To country']).Points.mean()
dfAverages = dfAverages.unstack().fillna(0)

#Find louvian groups
np_matrix = dfAverages.values
names = dfAverages.columns
print(names)
G = nx.from_numpy_matrix(np_matrix)
G = nx.relabel_nodes(G, dict(enumerate(names)))
partition = community_louvain.best_partition(G)

#Asign groups on pandaframe
partition = collections.OrderedDict(sorted(partition.items())) 
dfAverages['group'] = partition.values()
dfAverages = dfAverages.sort_values(by ='group')

#Calculate stats of clusters
clusters = dfAverages['group'].unique()
for c in clusters:
    names = dfAverages[dfAverages['group']==c].index
    temp = dfAverages[names]
    m = np.zeros((len(names),len(names)))
    for i, n in enumerate(names):
        for i2, n2 in enumerate(names):
            t = temp[n]
            if n != n2:
                m[i,i2] = t[n2]
            else:
                m[i,i2] = np.NaN

    print(str(c) + ' mean: ' + str(np.nanmean(m)))
    print(str(c) + ' std: ' + str(np.nanstd(m)))
    print(str(c) + ' Sym: ' + str(np.nanmean(np.abs(m-np.transpose(m)))/2))

print(dfAverages)

dfAverages.drop(columns=['group'], inplace=True)
dfAverages = dfAverages[dfAverages.axes[0]]

#Calculate stats of all nations
np_matrix = dfAverages.values
m = np_matrix.copy()
for i, n in enumerate(names):
    for i2, n2 in enumerate(names):
        t = temp[n]
        if n == n2:
            m[i,i2] = np.NaN
print('All mean: ' + str(np.nanmean(m)))
print('All std: ' + str(np.nanstd(m)))
print('All Sym: ' + str(np.nanmean(np.abs(m-np.transpose(m)))/2))

#Calculate Asymmetry of each nation
df = pd.DataFrame(columns=list(['Name', 'Asymmetry']))
for i in range(0, np_matrix.shape[0]):
    total = 0
    for k in range(0, np_matrix.shape[0]):
        total += np.abs(np_matrix[i,k] - np_matrix[k,i])
    df = df.append({'Name' : dfAverages.axes[0][i], 'Asymmetry' : total/np_matrix.shape[0]}, ignore_index=True)
df = df.sort_values(by=['Asymmetry'])
print(df)

#Create heatmap
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
x_axis_labels = dfAverages.axes[1] # labels for x-axis
y_axis_labels = dfAverages.axes[0] # labels for y-axis
sns.heatmap(np_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels,cbar=False)
plt.xlabel("Average Proportion Of Points Received")
plt.ylabel("Average Proportion Of Points Given")
plt.show()
