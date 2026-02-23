import data_formater69 as df
import numpy as np
# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output
ii= 'BRANCHE' #Aggregeret branche på 69 opdeling


# Beregn afskrivningsraten
K_prev = df.K.groupby('BRANCHE')['Xt'].shift(1)
delta = (K_prev - df.K['Xt'] + df.I['Xt']) / K_prev
delta.index.names = [i, t]

# Beregn usercost of capital
df.P_I.index.names = [i, t]
P_K=(0.07+delta)*df.P_I['Pt']


# Beregn PM*M afgifter og told er ikke specificeret endnu 
P_MxM_F=(1+df.tau['tau'])*df.P_F['Pt']*df.M_F['Xt']

P_MxM_D=df.P_D['Pt']*df.M_D['Xt']

P_MxM=P_MxM_D+P_MxM_F

P_MxM_tot = P_MxM.groupby([i,t]).sum()
P_MxM_tot.index.names = [i, t]

# Beregn P_KLxKL
df.K.index.names = [i, t]
P_KLxKL=df.w['TIMELOEN_KR']*df.L['TIMER']/1000+P_K*df.K['Xt']

#Beregn P_O
P_O=(P_MxM_tot+P_KLxKL)/df.Y['Xt']

#Beregn markup
markup=df.P['Pt']/P_O-1
