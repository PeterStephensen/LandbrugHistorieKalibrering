import data_formater69 as df
import numpy as np
import pandas as pd
# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output
ii= 'BRANCHE' #Aggregeret branche på 69 opdeling

# Rente
r_map = {
    '01000': 0.05,  # Landbrug
    '10120': 0.07,
    'REST': 0.07
}
# Konverter r_map direkte til en Pandas Series
r = pd.Series(r_map)
r.index.name = i

EY=1.3     # substitution Y-niveau
EMtot=0  # substitution Mtot-niveau
EM_map = {
    '01000': 3.5,  # Landbrug
    '10120': 4,
    'REST': 2
}
EM = pd.Series(EM_map)     # substitution import/domestic
EM.index.name = j
EKL=0.4    # substitution K-L
# EJKL   # substitution J-KL (kun landbrug)

# Beregn afskrivningsraten
K_prev = df.K.groupby('ANVENDELSE')['Xt'].shift(1)
delta = (K_prev - df.K['Xt'] + df.I['Xt']) / K_prev
delta.index.names = [i, t]

# Beregn usercost of capital
df.P_I.index.names = [i, t]
P_K=(r+delta)*df.P_I['Pt']
P_K_final = P_K.reset_index(name='Pt')
P_K_final.to_csv('../Nationalregnskab/Data69/P_K.csv', index=False)

# Bestem først samlet dansk produktion ved at summe over tilgang i M_D
M_D_tot = df.M_D_loebende.groupby([i,t]).sum()
M_D_tot.index.names = [i, t]

# Beregn afgiftsats
tau_MD = df.afgift['afgift'] / M_D_tot['INDHOLD']
#slet 1966
tau_MD = tau_MD.loc[tau_MD.index.get_level_values('TID') != 1966]
tau_MD.index.names = [i, t]
tau_MD_final = tau_MD.reset_index(name='tau')
tau_MD_final.to_csv('../Nationalregnskab/Data69/tau_MD.csv', index=False) 

# Beregn PM*M 
P_MxM_F=(1+df.tau_MF['tau'])*df.P_F['Pt']*df.M_F['Xt']

P_MxM_D=(1+tau_MD)*df.P_D['Pt']*df.M_D['Xt']

P_MxM=P_MxM_D+P_MxM_F

P_MxM_tot = P_MxM.groupby([i,t]).sum()
P_MxM_tot.index.names = [i, t]

# Beregn P_KLxKL
df.K.index.names = [i, t]
P_KLxKL=df.w['Pt']*df.L['Xt']+P_K*K_prev

#Beregn først residual
landbrug_index = df.P['Pt'].loc['01000', :].index
# Reindex hektarstøtte til at matche landbrug index
hektarstotte_reindexed = df.hektarstotte['INDHOLD'].reindex(landbrug_index, fill_value=0)
# Sæt hektarstøtte til 0 for år >= 2005 (kun træk fra før 2005)
hektarstotte_adjusted = hektarstotte_reindexed.copy()
hektarstotte_adjusted.loc[hektarstotte_adjusted.index >= 2005] = 0

Res = (df.P['Pt'].loc['01000', :] * df.Y['Xt'].loc['01000', :] 
       - hektarstotte_adjusted
       - P_MxM_tot.loc['01000', :]
       - df.w['Pt'].loc['01000', :] * df.L['Xt'].loc['01000', :]
       - P_K.loc['01000', :] * K_prev.loc['01000', :])

#For landbrug specifikt
P_J=r.loc['01000']*df.P_Jord['Pt']
P_J.index.names = [t]
J_prev=df.J.groupby('ANVENDELSE')['Xt'].shift(1)
PxJ=P_J*J_prev.loc['01000']  

def P_O_landbrug(P_MxM_tot, P_KLxKL, PxJ, df):
    # 1. Hent mål-indekset (MultiIndex med ANVENDELSE og TID)
    target_index = df.Y['Xt'].index
    
    # 2. Sørg for at de andre variable matcher index-strukturen
    P_MxM_tot_adj = P_MxM_tot.reindex(target_index, fill_value=0)
    P_KL_adj = P_KLxKL.reindex(target_index, fill_value=0)
    
    # 3. Opret en tom serie til jordomkostninger fyldt med 0
    PxJ_final = pd.Series(0.0, index=target_index)
    
    # 4. Lav en maske for landbrug (01000)
    is_01000 = target_index.get_level_values('ANVENDELSE') == '01000'
    
    # 5. Map PxJKL værdierne ind kun for landbrug ved at matche på TID (år)
    # Vi tager årstallene fra target_index for landbrugs-rækkerne
    years_for_01000 = target_index.get_level_values('TID')[is_01000]
    
    # Vi henter værdierne fra din tidsserie (PxJKL) for de pågældende år
    # .values sikrer, at vi ikke får index-konflikter under indsættelsen
    PxJ_final.loc[is_01000] = PxJ.reindex(years_for_01000).values
    
    # 6. Beregn P_O (Nu er jord-leddet kun lagt til landbrug)
    P_O_values = (P_MxM_tot_adj + P_KL_adj + PxJ_final) / df.Y['Xt']
    
    return P_O_values

P_O = P_O_landbrug(P_MxM_tot, P_KLxKL, PxJ, df)

# Subsidier og hektarstøtte
subs_adj = df.subsidier['INDHOLD'].copy()
landbrug_mask = subs_adj.index.get_level_values('ANVENDELSE') == '01000'
tid_mask = subs_adj.index.get_level_values('TID') >= 2005
mask = landbrug_mask & tid_mask
tid_values = subs_adj.index.get_level_values('TID')[mask]
hektarstotte_reindexed = df.hektarstotte['INDHOLD'].reindex(tid_values, fill_value=0)
# Træk hektarstøtte fra
subs_adj.loc[mask] = subs_adj.loc[mask] + hektarstotte_reindexed.values
tau_Y = subs_adj / (df.Y_lob['INDHOLD'] - subs_adj)

#Beregn markup
markup=df.P['Pt']/((1+tau_Y)*P_O)-1

#Øverste CES
mu_Y_Mtot = (df.Mtot['Xt'] / df.Y['Xt']) * (df.P_Mtot['Pt'] / P_O)**EY
mu_Y_KL   = (df.KL['Xt']       / df.Y['Xt']) * (df.P_KL['Pt']     / P_O)**EY
mu_Y_KL.index.names = [i, t]
mu_Y_Mtot.index.names = [i, t]

# Materiale niveau
# broadcast Mtot og P_Mtot ned på (i,j,t)
M_index_df = df.M.index.to_frame(index=False)

# Merge Mtot
Mtot_df = df.Mtot['Xt'].reset_index(name='Xt')
Mtot_expanded = M_index_df.merge(Mtot_df, on=[i, t], how='left').fillna(0)
Mtot_exp = pd.Series(
    Mtot_expanded['Xt'].values,
    index=pd.MultiIndex.from_frame(Mtot_expanded[[i, j, t]])
)

# Merge P_Mtot
P_Mtot_df = df.P_Mtot['Pt'].reset_index(name='Pt')
P_Mtot_expanded = M_index_df.merge(P_Mtot_df, on=[i, t], how='left').fillna(0)
P_Mtot_exp = pd.Series(
    P_Mtot_expanded['Pt'].values,
    index=pd.MultiIndex.from_frame(P_Mtot_expanded[[i, j, t]])
)

mu_Mtot_M = (df.M['Xt'] / Mtot_exp) * (df.P_M['P_M'] / P_Mtot_exp)**EMtot
mu_Mtot_M.index.names = [i, j, t]

# Import vs dansk
mu_MD = (df.M_D['Xt'] / df.M['Xt']) * (((1+tau_MD)*df.P_D['Pt']) / df.P_M['P_M'])**EM
mu_MF = (df.M_F['Xt'] / df.M['Xt']) * (((1 + df.tau_MF['tau']) * df.P_F['Pt']) / df.P_M['P_M'])**EM

mu_MD.index.names = [i, j, t]
mu_MF.index.names = [i, j, t]

# KL niveau
mu_KL_K = (K_prev/ df.KL['Xt']) * (P_K/ df.P_KL['Pt'])**EKL
mu_KL_L = ((df.L['Xt']) / df.KL['Xt']) * (df.w['Pt'] / df.P_KL['Pt'])**EKL

# Beregn thetaer
theta_Y_KL=mu_Y_KL**(1/(EY-1))
theta_Y_Mtot=mu_Y_Mtot**(1/(EY-1))
theta_Mtot_M=mu_Mtot_M**(1/(EMtot-1))
theta_MD=mu_MD**(1/(EM-1))
theta_MF=mu_MF**(1/(EM-1))
theta_KL_K=mu_KL_K**(1/(EKL-1))
theta_KL_L=mu_KL_L**(1/(EKL-1))

# thetaer indekseret til 1994=1
theta_Y_KL_1994= theta_Y_KL.xs(1994, level='TID')
theta_Y_KL_indeks=theta_Y_KL/theta_Y_KL_1994
theta_Y_Mtot_1994=theta_Y_Mtot.xs(1994, level='TID')
theta_Y_Mtot_indeks=theta_Y_Mtot/theta_Y_Mtot_1994
theta_Mtot_M_1994=theta_Mtot_M.xs(1994, level='TID')
theta_Mtot_M_indeks=theta_Mtot_M/theta_Mtot_M_1994
theta_MD_1994=theta_MD.xs(1994, level='TID')
theta_MD_indeks=theta_MD/theta_MD_1994
theta_MF_1994=theta_MF.xs(1994, level='TID')
theta_MF_indeks=theta_MF/theta_MF_1994
theta_KL_K_1994=theta_KL_K.xs(1994, level='TID')
theta_KL_K_indeks=theta_KL_K/theta_KL_K_1994
theta_KL_L_1994=theta_KL_L.xs(1994, level='TID')
theta_KL_L_indeks=theta_KL_L/theta_KL_L_1994

