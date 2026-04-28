
import data_formater as df
import numpy as np
import pandas as pd
import sys
sys.path.append('../Indlæsning af data')
import data_formater69 as df69
import calib69 as calib69

# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output
ii= 'BRANCHE' #Aggregeret branche på 69 opdeling

def map_69_to_117(series_69, brancher_117, t):
    """
    Mapper en serie fra 69-opdeling til 117-opdeling.
    01000 -> 010000 (1-til-1)
    10120 -> 100010, 100030, 100040x100050 (1-til-mange, samme værdi)
    REST  -> REST (1-til-1)
    """
    mapping = {
        '01000':  ['010000'],
        '10120':  ['100010', '100030', '100040x100050'],
        'REST':   ['REST']
    }
    
    rows = []
    for branche_69, brancher_117_list in mapping.items():
        if branche_69 not in series_69.index.get_level_values(i):
            continue
        værdier = series_69.xs(branche_69, level=i)
        for branche_117 in brancher_117_list:
            temp = værdier.copy()
            temp.index = pd.MultiIndex.from_arrays(
                [[branche_117] * len(temp), temp.index],
                names=[i, t]
            )
            rows.append(temp)
    
    return pd.concat(rows).sort_index()


# Rente
r_map = {
    '010000': 0.05,  # Landbrug
    '100010': 0.07,
    '100030': 0.07,
    '100040x100050': 0.07,
    'REST': 0.07
}
# Konverter r_map direkte til en Pandas Series
r = pd.Series(r_map)
r.index.name = i

EY=0     # substitution Y-niveau
EMtot=0  # substitution Mtot-niveau
EM_map = {
    '010000': 3.5,  # Landbrug
    '100010': 4,
    '100030': 4,
    '100040x100050': 4,
    'REST': 2
}
EM = pd.Series(EM_map)     # substitution import/domestic
EM.index.name = j
EKL=0    # substitution K-L
EJKL=0   # substitution J-KL (kun landbrug)

##################################################################
# Beregn afskrivningsraten
P_K = map_69_to_117(calib69.P_K, i, t)
P_K.index.names = [i, t]
P_K_final = P_K.reset_index(name='Pt')
P_K_final.to_csv('../Nationalregnskab_117/Data117/P_K.csv', index=False)

delta = map_69_to_117(calib69.delta, i, t)
delta.index.names = [i, t]

P_I = map_69_to_117(df69.P_I, i, t)
P_I.index.names = [i, t]

# Bruttoinvesteringer
food_branches = ["100010", "100030", "100040x100050"]
non_food_branches = ['010000', 'REST']

I_non_food = map_69_to_117(df69.I, non_food_branches, t)
I_non_food = I_non_food[I_non_food.index.get_level_values(i).isin(non_food_branches)]
I_non_food.index.names = [i, t]

I_lob = df69.I_lob
I_10120 = I_lob.xs("10120", level=0)["Xt"]
gamma = df.s["gamma"] 

I_parts = []

for br in food_branches:
    g_br = gamma.xs(br, level=0)
    prod = g_br * I_10120.reindex(g_br.index).fillna(0)
    idx = pd.MultiIndex.from_arrays([[br] * len(prod), prod.index], names=[i, t])
    I_parts.append(pd.Series(prod.values, index=idx, name="Xt"))

I_food = pd.concat(I_parts).sort_index().to_frame("Xt")
I_real = I_food.copy()
I_real['Xt'] = I_food['Xt'] / P_I.reindex(I_food.index).fillna(1)['Pt']

I = pd.concat([I_real, I_non_food]).sort_index()
I.index.names = [i, t]

# Initalt kapital
K_non_food = map_69_to_117(df69.K, non_food_branches, t)
K_non_food = K_non_food[K_non_food.index.get_level_values(i).isin(non_food_branches)]
K_non_food.index.names = [i, t]

g = 0.02
I_first = I_real.xs(1993, level=t)['Xt']
delta_first = delta.xs(1993, level=t)
K0 = I_first / (g + delta_first)

#Kapitalapparat
år = sorted(I_real.index.get_level_values(t).unique())
brancher_K = I_real.index.get_level_values(i).unique()

K_food = pd.DataFrame(index=I_real.index, columns=['Xt'], dtype=float)

for b in brancher_K:
    if (b, 1992) not in I_real.index:
        continue
    K_food.loc[(b, 1992), 'Xt'] = K0[b]
    years = [yr for yr in år if yr >= 1993]
    for yr in years:
        d = delta.loc[(b, yr)]
        inv = I_real.loc[(b, yr), 'Xt']
        K_food.loc[(b, yr), 'Xt'] = (1 - d) * K_food.loc[(b, yr - 1), 'Xt'] + inv

K = pd.concat([K_food, K_non_food]).sort_index()
K.index.names = [i, t]
K_prev = K.groupby('ANVENDELSE')['Xt'].shift(1)
K.to_csv('../Nationalregnskab_117/Data117/K.csv', index=True)

####################################################################

# Bestem først samlet dansk produktion ved at summe over tilgang i M_D
M_D_tot = df.M_D_loebende.groupby([i,t]).sum()
M_D_tot.index.names = [i, t]

# Beregn afgiftsats
tau_MD = df.afgift['afgift'] / M_D_tot['INDHOLD']
#slet 1966
tau_MD = tau_MD.loc[tau_MD.index.get_level_values('TID') != 1966]
tau_MD.index.names = [i, t]
tau_MD_final = tau_MD.reset_index(name='tau')
tau_MD_final.to_csv('../Nationalregnskab_117/Data117/tau_MD.csv', index=True) 

# Beregn PM*M 
P_MxM_F=(1+df.tau_MF['tau'])*df.P_F['Pt']*df.M_F['Xt']

P_MxM_D=(1+tau_MD)*df.P_D['Pt']*df.M_D['Xt']

P_MxM=P_MxM_D+P_MxM_F

P_MxM_tot = P_MxM.groupby([i,t]).sum()
P_MxM_tot.index.names = [i, t]

# Beregn P_KLxKL
P_KLxKL=df.w['Pt']*df.L['Xt']+P_K*K_prev

################################################################################
# Beregn jordomkostninger
################################################################################
J = map_69_to_117(df69.J, i, t)
J.index.names = [i, t]
J.to_csv('../Nationalregnskab_117/Data117/Jordareal.csv', index=True)
J_prev=J.groupby('ANVENDELSE')['Xt'].shift(1)
P_J=calib69.P_J
PxJ=P_J*J_prev.loc['010000']  

def P_O_landbrug(P_MxM_tot, P_KLxKL, PxJ, df):
    # 1. Hent mål-indekset (MultiIndex med ANVENDELSE og TID)
    target_index = df.Y['Xt'].index
    
    # 2. Sørg for at de andre variable matcher index-strukturen
    P_MxM_tot_adj = P_MxM_tot.reindex(target_index, fill_value=0)
    P_KL_adj = P_KLxKL.reindex(target_index, fill_value=0)
    
    # 3. Opret en tom serie til jordomkostninger fyldt med 0
    PxJ_final = pd.Series(0.0, index=target_index)
    
    # 4. Lav en maske for landbrug (01000)
    is_01000 = target_index.get_level_values('ANVENDELSE') == '010000'
    
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
landbrug_mask = subs_adj.index.get_level_values('ANVENDELSE') == '010000'
tid_mask = subs_adj.index.get_level_values('TID') >= 2005
tid_grundskyld = subs_adj.index.get_level_values('TID') >= 2009
mask = landbrug_mask & tid_mask
mask_grundskyld = landbrug_mask & tid_grundskyld
tid_values = subs_adj.index.get_level_values('TID')[mask]
tid_values_grundskyld = subs_adj.index.get_level_values('TID')[mask_grundskyld]
hektarstotte_reindexed = df.hektarstotte['INDHOLD'].reindex(tid_values, fill_value=0)
grundskyld_reindexed = df.grundskyld['INDHOLD'].reindex(tid_values_grundskyld, fill_value=0)
# Træk hektarstøtte fra
subs_adj.loc[mask] = subs_adj.loc[mask] + hektarstotte_reindexed.values
subs_adj.loc[mask_grundskyld] = subs_adj.loc[mask_grundskyld] - 0.349*grundskyld_reindexed.values

tau_Y = subs_adj / (df.Y_lob['INDHOLD'] - subs_adj)
#Beregn markup
markup=df.P['Pt']/((1+tau_Y)*P_O)-1

#Øverste CES
mu_Y_Mtot = (df.Mtot['Xt'] / df.Y['Xt']) * (df.P_Mtot['Pt'] / P_O)**EY
mu_Y_KL   = (df.KL['Xt'] / df.Y['Xt']) * (df.P_KL['Pt'] / P_O)**EY
mu_Y_KL.index.names = [i, t]
mu_Y_Mtot.index.names = [i, t]

# JKL erstatter KL for landbrug i øverste CES
mu_Y_JKL = mu_Y_KL.copy()
mask_01000 = mu_Y_JKL.index.get_level_values('ANVENDELSE') == '010000'
tid_01000 = mu_Y_JKL.loc[mask_01000].index.get_level_values('TID')

jkl_reindexed   = df.JKL['Xt'].loc['010000'].reindex(tid_01000)
y_reindexed     = df.Y['Xt'].loc['010000'].reindex(tid_01000)
p_jkl_reindexed = df.P_JKL['Pt'].loc['010000'].reindex(tid_01000)
p_o_reindexed   = P_O.loc['010000'].reindex(tid_01000)

mu_Y_JKL.loc[mask_01000] = (jkl_reindexed.values / y_reindexed.values) * (p_jkl_reindexed.values / p_o_reindexed.values)**EY
mu_Y_JKL.index.names = [i, t]
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

mu_JKL_J  = (J_prev.loc['010000'] / df.JKL['Xt'].loc['010000']) * (P_J / df.P_JKL['Pt'].loc['010000'])**EJKL
mu_JKL_KL = (df.KL['Xt'].loc['010000'] / df.JKL['Xt'].loc['010000']) * (df.P_KL['Pt'].loc['010000'] / df.P_JKL['Pt'].loc['010000'])**EJKL
mu_JKL_J.index  = pd.MultiIndex.from_product([['010000'], mu_JKL_J.index],  names=[i, t])
mu_JKL_KL.index = pd.MultiIndex.from_product([['010000'], mu_JKL_KL.index], names=[i, t])
# KL niveau
mu_KL_K = (K_prev/ df.KL['Xt']) * (P_K/ df.P_KL['Pt'])**EKL
mu_KL_L = ((df.L['Xt']) / df.KL['Xt']) * (df.w['Pt'] / df.P_KL['Pt'])**EKL

# Beregn thetaer
theta_Y_KL=mu_Y_KL**(1/(EY-1))
theta_Y_JKL=mu_Y_JKL**(1/(EY-1))
theta_Y_Mtot=mu_Y_Mtot**(1/(EY-1))
theta_Mtot_M=mu_Mtot_M**(1/(EMtot-1))
theta_MD=mu_MD**(1/(EM-1))
theta_MF=mu_MF**(1/(EM-1))
theta_KL_K=mu_KL_K**(1/(EKL-1))
theta_KL_L=mu_KL_L**(1/(EKL-1))
theta_JKL_J  = mu_JKL_J**(1/(EJKL-1))
theta_JKL_KL = mu_JKL_KL**(1/(EJKL-1))

# thetaer indekseret til 1994=1
theta_Y_KL_1994= theta_Y_KL.xs(1994, level='TID')
theta_Y_KL_indeks=theta_Y_KL/theta_Y_KL_1994
theta_Y_JKL_1994= theta_Y_JKL.xs(1994, level='TID')
theta_Y_JKL_indeks=theta_Y_JKL/theta_Y_JKL_1994
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
theta_JKL_J_indeks  = theta_JKL_J  / theta_JKL_J.xs(1994, level='TID')
theta_JKL_KL_indeks = theta_JKL_KL / theta_JKL_KL.xs(1994, level='TID')
