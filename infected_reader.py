import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

df_total=pd.DataFrame
df_nodes_avg=[]
df_weight_avg=[]

def create_df(csv, num_steps):
    num_steps=num_steps+1
    with open(csv) as f:
        rows_to_keep_weight=[1]
        for i in range(10):
            rows_to_keep_weight.append(rows_to_keep_weight[i]+num_steps)
        df_weight=pd.read_csv(f, header=None, skiprows=lambda x: x not in rows_to_keep_weight)
        avg=str(df_weight.mean()).split(' ')[4]
        avg=avg.split('d')[0]
        avg=avg.strip()
        avg=float(avg)
        df_weight_avg.append(avg)
        print(avg)

    with open(csv) as f:
        rows_to_keep_nodes=[0]
        for i in range(10):
            rows_to_keep_nodes.append(rows_to_keep_nodes[i]+num_steps)
        df_nodes=pd.read_csv(f, delimiter=',', engine='python', skiprows=lambda y: y not in rows_to_keep_nodes)
        df_nodes=df_nodes.count(axis=1)
        df_nodes_avg.append(df_nodes.median())
        #print(df_nodes.mean())

    df_combine=pd.concat([df_nodes, df_weight], axis=1)
    #print(df_combine)

    return df_combine


df_total=create_df('infected.csv',50)
df_total=pd.concat([df_total, create_df('infected_20.csv',50)], axis=0)
df_total=pd.concat([df_total, create_df('infected_40.csv',50)], axis=0)
df_total=pd.concat([df_total, create_df('infected_60.csv',50)], axis=0)
df_total=pd.concat([df_total, create_df('infected_80.csv',50)], axis=0)
df_total=df_total.reset_index(drop=True)
df_total.set_axis(['y', 'x'], axis='columns', inplace=True)
#print(df_total)
print(df_nodes_avg)
print(df_weight_avg)

#df_total.plot('x','y',kind='scatter')
plt.xlabel('Average edge weight', fontsize=12)
plt.ylabel('Final number of infected nodes per run', fontsize=12)
plt.plot(df_weight_avg,df_nodes_avg)
plt.show()