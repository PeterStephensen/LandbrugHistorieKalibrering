import pandas as pd
import numpy as np

# Indlæs filen
df_materialer = pd.read_csv('../Nationalregnskab/Data69/landbrugsdata.csv')
df_tau_MD = pd.read_csv('../Nationalregnskab/Data69/tau_MD.csv')
df_tau_MF = pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_toldssats.csv')
df_kapital = pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_mængdeindeks_kapital.csv')
df_kapital_pris = pd.read_csv('../Nationalregnskab/Data69/P_K.csv')
df_lonsum=pd.read_csv('../Nationalregnskab/Data69/input_landbrugsdata.csv')
df_timer=pd.read_csv('../Nationalregnskab/Data69/Timer_landbrugsdata.csv')
df_timeLon=pd.read_csv('../Nationalregnskab/Data69/TimeLon_landbrugsdata.csv')
df_jord=pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_mængdeindeks_jord.csv')
df_jord_pris=pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_prisindeks_jord.csv')

mapping = {
    '01000 Landbrug og gartneri-(Tilgang)': '01000',
    '01000 Landbrug og gartneri- (Anvendelse)': '01000',
    '01000 Landbrug og gartneri': '01000',
    '10120 Føde-, drikke- og tobaksvareindustri-(Tilgang)': '10120',
    '10120 Føde-, drikke- og tobaksvareindustri- (Anvendelse)': '10120',
    '10120 Føde-, drikke- og tobaksvareindustri': '10120',
    'REST_TILGANG Øvrige brancher': 'REST',
    'REST_ANVENDELSE Øvrige brancher': 'REST'

}

brancher = ['01000', '10120', 'REST']

df_materialer['ANVENDELSE'] = df_materialer['ANVENDELSE'].replace(mapping)
df_materialer['TILGANG2'] = df_materialer['TILGANG2'].replace(mapping)
df_tau_MF['ANVENDELSE'] = df_tau_MF['ANVENDELSE'].replace(mapping)
df_tau_MF['TILGANG2'] = df_tau_MF['TILGANG2'].replace(mapping)
df_kapital['BRANCHE'] = df_kapital['BRANCHE'].replace(mapping)
df_lonsum['ANVENDELSE'] = df_lonsum['ANVENDELSE'].replace(mapping)

df_materialer=df_materialer.loc[df_materialer['ANVENDELSE'].isin(brancher)]
df_tau_MF=df_tau_MF.loc[df_tau_MF['ANVENDELSE'].isin(brancher)]

#filter for løbende priser og foregående års priser
df_materialer_MD=df_materialer.loc[(df_materialer['TILGANG1']=='Dansk produktion')]
df_materialer_MD=df_materialer_MD.loc[df_materialer_MD['TID']!=1966].copy()
df_materialer_MF=df_materialer.loc[(df_materialer['TILGANG1']=='Import eksklusiv told')]
df_materialer_MF=df_materialer_MF.loc[df_materialer_MF['TID']!=1966].copy()

df_kapital=df_kapital.loc[(df_kapital['BEHOLD']=='AN.11 Faste aktiver, nettobeholdning ultimo året')]
df_lonsom=df_lonsum.loc[(df_lonsum['TILGANG1']=='Aflønning af ansatte')].copy()

#fjern prishenhed
df_materialer_MD.drop(columns=['TILGANG1'],inplace=True)
df_materialer_MF.drop(columns=['TILGANG1'],inplace=True)

df_kapital.drop(columns=['BEHOLD'],inplace=True)
df_lonsom.drop(columns=['TILGANG1'],inplace=True)

#set index lidt besværligt men der skal lige ryddes op
df_materialer_MD_lob=df_materialer_MD.loc[(df_materialer_MD['PRISENHED']=='Løbende priser')].copy()
df_materialer_MF_lob=df_materialer_MF.loc[(df_materialer_MF['PRISENHED']=='Løbende priser')].copy()
df_materialer_MD_for=df_materialer_MD.loc[(df_materialer_MD['PRISENHED']!='Løbende priser')].copy()
df_materialer_MF_for=df_materialer_MF.loc[(df_materialer_MF['PRISENHED']!='Løbende priser')].copy()
df_materialer_MD_lob.drop(columns=['PRISENHED'],inplace=True)
df_materialer_MF_lob.drop(columns=['PRISENHED'],inplace=True)
df_materialer_MD_for.drop(columns=['PRISENHED'],inplace=True)
df_materialer_MF_for.drop(columns=['PRISENHED'],inplace=True)
df_materialer_MD_lob.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)
df_materialer_MF_lob.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)
df_materialer_MD_for.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)
df_materialer_MF_for.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)

# df_kapital_lob=df_kapital.loc[(df_kapital['PRISENHED']=='Løbende priser')].copy()
df_lonsom_lob=df_lonsom.loc[(df_lonsom['PRISENHED']=='Løbende priser')].copy()
# df_kapital_for=df_kapital.loc[(df_kapital['PRISENHED']=='Forige års priser')].copy()
# df_kapital_lob.drop(columns=['PRISENHED'],inplace=True)
df_lonsom_lob.drop(columns=['PRISENHED'],inplace=True)
# df_kapital_for.drop(columns=['PRISENHED'],inplace=True)
# df_kapital_lob.set_index(['BRANCHE', 'TID'],inplace=True)
# df_kapital_lob.rename_axis(index={'BRANCHE': 'ANVENDELSE'}, inplace=True)
df_lonsom_lob.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_kapital.set_index(['BRANCHE', 'TID'],inplace=True)
df_kapital.rename_axis(index={'BRANCHE': 'ANVENDELSE'}, inplace=True)
df_timer.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_timeLon.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_kapital_pris.set_index(['ANVENDELSE', 'TID'],inplace=True)

df_tau_MD.set_index([ 'ANVENDELSE', 'TID'],inplace=True)
df_tau_MF.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)

df_jord.set_index(['TID'],inplace=True)
df_jord_pris.set_index(['TID'],inplace=True)

##################################################################################################
# MATERIALER################################################################################
##################################################################################################

#### expand tau_MD#################################################################################
# Først, find hvilke ANVENDELSE værdier der findes i df_tau_MD
tau_MD_anvendelse = df_tau_MD.index.get_level_values('ANVENDELSE').unique()

# Reset index
df_tau_MD_reset = df_tau_MD.reset_index()
df_materialer_MD_lob_index_reset = df_materialer_MD_lob.reset_index()

# Filtrer df_materialer_MD_lob til kun de ANVENDELSE værdier der findes i df_tau_MD
df_materialer_MD_lob_filtered = df_materialer_MD_lob_index_reset[
    df_materialer_MD_lob_index_reset['ANVENDELSE'].isin(tau_MD_anvendelse)
]

# Merge for at få alle kombinationer af TILGANG2
df_tau_MD_expanded = df_materialer_MD_lob_filtered[['TILGANG2', 'ANVENDELSE', 'TID']].merge(
    df_tau_MD_reset,
    on=['ANVENDELSE', 'TID'],
    how='left'
)

# Sæt index og extract Series
df_tau_MD_expanded.set_index(['TILGANG2', 'ANVENDELSE', 'TID'], inplace=True)
df_tau_MD = df_tau_MD_expanded.copy() # Nu har samme index som df_materialer_MD_lob (filtreret)
##################################################################################################

# Beregn aggregater
df_materialer_aggregat_l=(1+df_tau_MD['tau'])*df_materialer_MD_lob['INDHOLD']+(1+df_tau_MF['tau'])*df_materialer_MF_lob['INDHOLD']
df_materialer_aggregat_loebende = df_materialer_aggregat_l.reset_index(name='INDHOLD')

tau_MD_prev = df_tau_MD.groupby(level=['TILGANG2', 'ANVENDELSE'])['tau'].shift(1)
tau_MF_prev = df_tau_MF.groupby(level=['TILGANG2', 'ANVENDELSE'])['tau'].shift(1)

df_materialer_aggregat_f=(1+tau_MD_prev)*df_materialer_MD_for['INDHOLD']+(1+tau_MF_prev)*df_materialer_MF_for['INDHOLD']
df_materialer_aggregat_for = df_materialer_aggregat_f.reset_index(name='INDHOLD')

#set index
df_materialer_aggregat_loebende.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)
df_materialer_aggregat_for.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)

##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################

Pt_Pt=df_materialer_aggregat_loebende/df_materialer_aggregat_for

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_Mjit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_Mjit.columns = ['TILGANG2', 'ANVENDELSE', 'TID', 'Pt']

df_Pt_Mjit.set_index(['TILGANG2', 'ANVENDELSE', 'TID'],inplace=True)

df_Xt=df_materialer_aggregat_loebende['INDHOLD']/df_Pt_Mjit['Pt']
df_Xt.columns = ['TILGANG2', 'ANVENDELSE', 'TID', 'Xt']
df_Xt_Mjit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_Mjit = df_Xt_Mjit.reset_index()


##################################################################################################
# MATERIALE DANSK################################################################################
##################################################################################################
df_materialer_MD_l=df_materialer_MD_lob.groupby(['ANVENDELSE', 'TID'])['INDHOLD'].sum()
df_materialer_MD_loebende = df_materialer_MD_l.reset_index(name='INDHOLD')

df_materialer_MD_f=df_materialer_MD_for.groupby(['ANVENDELSE', 'TID'])['INDHOLD'].sum()
df_materialer_MD_fore = df_materialer_MD_f.reset_index(name='INDHOLD')

df_materialer_MD_loebende.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_materialer_MD_fore.set_index(['ANVENDELSE', 'TID'],inplace=True)


##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################
Pt_Pt=df_materialer_MD_loebende/df_materialer_MD_fore

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_MDit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_MDit.columns = ['ANVENDELSE', 'TID', 'Pt']

df_Pt_MDit.set_index(['ANVENDELSE', 'TID'],inplace=True)

df_Xt=df_materialer_MD_loebende['INDHOLD']/df_Pt_MDit['Pt']
df_Xt.columns = ['ANVENDELSE', 'TID', 'Xt']
df_Xt_MDit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_MDit = df_Xt_MDit.reset_index()

##################################################################################################
# MATERIALE IMPORT################################################################################
##################################################################################################
df_materialer_MF_l=df_materialer_MF_lob.groupby(['ANVENDELSE', 'TID'])['INDHOLD'].sum()
df_materialer_MF_loebende = df_materialer_MF_l.reset_index(name='INDHOLD')

df_materialer_MF_f=df_materialer_MF_for.groupby(['ANVENDELSE', 'TID'])['INDHOLD'].sum()
df_materialer_MF_fore = df_materialer_MF_f.reset_index(name='INDHOLD')

df_materialer_MF_loebende.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_materialer_MF_fore.set_index(['ANVENDELSE', 'TID'],inplace=True)

##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################
Pt_Pt=df_materialer_MF_loebende/df_materialer_MF_fore

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_MFit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_MFit.columns = ['ANVENDELSE', 'TID', 'Pt']

df_Pt_MFit.set_index(['ANVENDELSE', 'TID'],inplace=True)

df_Xt=df_materialer_MF_loebende['INDHOLD']/df_Pt_MFit['Pt']
df_Xt.columns = ['ANVENDELSE', 'TID', 'Xt']
df_Xt_MFit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_MFit = df_Xt_MFit.reset_index()


##################################################################################################
# MATERIALE AGGREGAT################################################################################
##################################################################################################

# Tæller: Summen af løbende værdier på tværs af alle TILGANG2 for hver branche (ANVENDELSE)
# Dette svarer til tælleren i ligning (57)
loebende_Mtot = df_materialer_aggregat_loebende.groupby(['ANVENDELSE', 'TID'])['INDHOLD'].sum()

# Nævner: Årets mængde (Xt) ganget med sidste års pris (Pt-1)
# Vi shifter prisen for at få Pt-1
df_Pt_prev = df_Pt_Mjit.groupby(['TILGANG2', 'ANVENDELSE'])['Pt'].shift(1)
for_Mtot = (df_Xt_Mjit.set_index(['TILGANG2', 'ANVENDELSE', 'TID'])['Xt'] * df_Pt_prev).groupby(['ANVENDELSE', 'TID']).sum()

loebende_Mtot = loebende_Mtot.reset_index(name='INDHOLD')
for_Mtot = for_Mtot.reset_index(name='INDHOLD')

loebende_Mtot.set_index(['ANVENDELSE', 'TID'],inplace=True)
for_Mtot.set_index(['ANVENDELSE', 'TID'],inplace=True)

##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################
Pt_Pt=loebende_Mtot/for_Mtot

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_Mit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_Mit.columns = ['ANVENDELSE', 'TID', 'Pt']

df_Pt_Mit.set_index(['ANVENDELSE', 'TID'],inplace=True)

df_Xt=loebende_Mtot['INDHOLD']/df_Pt_Mit['Pt']
df_Xt.columns = ['ANVENDELSE', 'TID', 'Xt']
df_Xt_Mit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_Mit = df_Xt_Mit.reset_index()

##################################################################################################
#KL AGGREGAT ################################################################################
##################################################################################################

# Beregn aggregater
k_prev=df_kapital.groupby(level=['ANVENDELSE'])['Xt'].shift(1)
df_KL_aggregat_l=df_lonsom_lob['INDHOLD']+k_prev*df_kapital_pris['P']
df_KL_aggregat_lobende = df_KL_aggregat_l.reset_index(name='INDHOLD')

lon_prev = df_timeLon.groupby(level=['ANVENDELSE'])['TIMELOEN_KR'].shift(1)
pk_prev=df_kapital_pris.groupby(level=['ANVENDELSE'])['P'].shift(1)

df_KL_aggregat_f=lon_prev*df_timer['TIMER']/1000+pk_prev*k_prev
df_KL_aggregat_for = df_KL_aggregat_f.reset_index(name='INDHOLD')
#set index
df_KL_aggregat_lobende.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_KL_aggregat_for.set_index(['ANVENDELSE', 'TID'],inplace=True)

##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################

Pt_Pt=df_KL_aggregat_lobende/df_KL_aggregat_for

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_KLit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_KLit.columns = ['ANVENDELSE', 'TID', 'Pt']

df_Pt_KLit.set_index([ 'ANVENDELSE', 'TID'],inplace=True)

df_Xt=df_KL_aggregat_lobende['INDHOLD']/df_Pt_KLit['Pt']
df_Xt.columns = ['ANVENDELSE', 'TID', 'Xt']
df_Xt_KLit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_KLit = df_Xt_KLit.reset_index()

##################################################################################################
#JKL AGGREGAT ################################################################################
##################################################################################################

# Beregn aggregater
J_prev=df_jord['Xt'].shift(1)
df_JKL_aggregat_l=df_KL_aggregat_l+J_prev*df_jord_pris['Pt']*0.05
df_JKL_aggregat_lobende = df_JKL_aggregat_l.reset_index(name='INDHOLD')

jord_pris_prev = df_jord_pris['Pt'].shift(1)

df_JKL_aggregat_f=df_KL_aggregat_f+J_prev*jord_pris_prev*0.05
df_JKL_aggregat_for = df_JKL_aggregat_f.reset_index(name='INDHOLD')
#set index
df_JKL_aggregat_lobende.set_index(['ANVENDELSE', 'TID'],inplace=True)
df_JKL_aggregat_for.set_index(['ANVENDELSE', 'TID'],inplace=True)

##################################################################################################
# OMREGN DET HELE################################################################################
##################################################################################################

Pt_Pt=df_JKL_aggregat_lobende/df_JKL_aggregat_for

# Vi bruger .unstack() til at flytte 'TID' fra index til kolonner
df_wide = Pt_Pt['INDHOLD'].unstack(level='TID')

# Lav en tom tabel i samme form til vores resultat (Pt)
Pt = df_wide.copy()
Pt.loc[:, :] = np.nan  # Tøm tabellen for tal
Pt[2020] = 1.0         # Sæt 2020 til 1.0
# Find alle år i dit datasæt og sortér dem
years = sorted(df_wide.columns)

# FREMAD: Fra 2021 og op (Pt = Pt-1 * Pt_Pt_nu)
for y in years:
    if y > 2020:
        Pt[y] = Pt[y-1] * df_wide[y]

# BAGUD: Fra 2019 og ned (Pt = Pt+1 / Pt_Pt_næste_år)
for y in reversed(years):
    if y < 2020:
        Pt[y] = Pt[y+1] / df_wide[y+1]

df_Pt_JKLit = Pt.stack(future_stack=True).reset_index()

# Navngiv kolonnerne som før
df_Pt_JKLit.columns = ['ANVENDELSE', 'TID', 'Pt']

df_Pt_JKLit.set_index(['ANVENDELSE', 'TID'],inplace=True)

df_Xt=df_JKL_aggregat_lobende['INDHOLD']/df_Pt_JKLit['Pt']
df_Xt.columns = ['ANVENDELSE', 'TID', 'Xt']
df_Xt_JKLit = df_Xt.to_frame(name='Xt')

# 2. Fjern MultiIndex så den bliver "flad"
df_Xt_JKLit = df_Xt_JKLit.reset_index()

##################################################################################################
# GEM DET HELE ###################################################################################
##################################################################################################

# --- MASTER-FIL 1: AGGREGATER (Niveau 2) ---
# Bruges til at kalibrere Y, KL, Mtot, PO og Markup
df_master_aggregater = df_Pt_Mit.rename(columns={'Pt': 'P_Mtot'}).join([
    df_Xt_Mit.set_index(['ANVENDELSE', 'TID']).rename(columns={'Xt': 'M_tot'}),
    df_Pt_KLit.rename(columns={'Pt': 'P_KL'}),
    df_Xt_KLit.set_index(['ANVENDELSE', 'TID']).rename(columns={'Xt': 'KL'}),
    df_Pt_JKLit.rename(columns={'Pt': 'P_JKL'}),
    df_Xt_JKLit.set_index(['ANVENDELSE', 'TID']).rename(columns={'Xt': 'JKL'}),
    df_Pt_MDit.rename(columns={'Pt': 'P_MDtot'}),
    df_Xt_MDit.set_index(['ANVENDELSE', 'TID']).rename(columns={'Xt': 'MD_tot'}),
    df_Pt_MFit.rename(columns={'Pt': 'P_MFtot'}),
    df_Xt_MFit.set_index(['ANVENDELSE', 'TID']).rename(columns={'Xt': 'MF_tot'})

])

# Gem Niveau 2 filen
df_master_aggregater.to_csv('../Nationalregnskab/Data69/Aggregater_KL_M.csv')


# --- MASTER-FIL 2: VAREGRUPPER (Niveau 1) ---
# Bruges til at kalibrere fordelingen mellem de 5 brancher (Mjit)
df_master_varegrupper = df_Pt_Mjit.rename(columns={'Pt': 'P_M'}).join([
    df_Xt_Mjit.set_index(['TILGANG2', 'ANVENDELSE', 'TID']).rename(columns={'Xt': 'M'})
])

# Gem Niveau 1 filen
df_master_varegrupper.to_csv('../Nationalregnskab/Data69/Aggregater_Mjit.csv')
