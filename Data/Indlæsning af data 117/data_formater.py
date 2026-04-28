
import pandas as pd
import dreamtools as dt

# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output
ii= 'BRANCHE' #Aggregeret branche på 69 opdeling

# Indlæs data
df = pd.read_csv('../Nationalregnskab_117/Data117/landbrugsdata_mængdeindeks.csv')
df_priser = pd.read_csv('../Nationalregnskab_117/Data117/landbrugsdata_prisindeks.csv')
df_timer = pd.read_csv('../Nationalregnskab_117/Data117/Timer_landbrugsdata.csv')
df_timeLon = pd.read_csv('../Nationalregnskab_117/Data117/TimeLon_landbrugsdata.csv')
# df_kapital = pd.read_csv('../Nationalregnskab/Data117/Kapital_landbrugsdata.csv')
# df_kapital_mængder=pd.read_csv('../Nationalregnskab/Data117/landbrugsdata_mængdeindeks_kapital.csv')
# df_kapital_pris = pd.read_csv('../Nationalregnskab/Data117/landbrugsdata_prisindeks_kapital.csv')
df_toldssats = pd.read_csv('../Nationalregnskab_117/Data117/landbrugsdata_toldssats.csv')
df_afgift = pd.read_csv('../Nationalregnskab_117/Data117/landbrugsdata_afgift.csv')
df_produktion_løbende = pd.read_csv('../Nationalregnskab_117/Data117/landbrugsdata.csv')
df_aggregater = pd.read_csv('../Nationalregnskab_117/Data117/Aggregater_KL_M.csv')
df_aggregater_mjit = pd.read_csv('../Nationalregnskab_117/Data117/Aggregater_Mjit.csv')
df_input = pd.read_csv('../Nationalregnskab_117/Data117/input_landbrugsdata.csv')
df_jordareal = pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_jordareal.csv')
df_jordpris = pd.read_csv('../Nationalregnskab/Data69/jordpris.csv')
df_tilskud = pd.read_csv('../Nationalregnskab/Data69/landbrugsdata_tilskud.csv')
df_grundskyld = pd.read_csv('../Nationalregnskab/Data69/grundskyld.csv')
df_timer_lon = pd.read_csv('../Nationalregnskab_117/Data117/Timer_lon_landbrugsdata.csv')
df_realkredit=pd.read_csv('../Nationalregnskab/Data69/rente_landbrugsdata.csv')
df_gamma=pd.read_csv('../Nationalregnskab_117/Data117/gamma.csv')
df_s=pd.read_csv('../Nationalregnskab_117/Data117/s.csv')
makrodata= dt.Gdx('../Nationalregnskab/Data69/konjunktur_juni2025.gdx')

# Mapping for både Tilgang og Anvendelse
mapping = {
    '010000 Landbrug og gartneri-(Tilgang)': '010000',
    '010000 Landbrug og gartneri- (Anvendelse)': '010000',
    '100010 Slagterier-(Tilgang)': '100010',
    '100010 Slagterier- (Anvendelse)': '100010',
    '100030 Mejerier-(Tilgang)': '100030',
    '100030 Mejerier- (Anvendelse)': '100030',
    '100040x100050 Anden fødevareindustri (100040, 100050)': '100040x100050',
    'REST_TILGANG Øvrige brancher': 'REST',
    'REST_ANVENDELSE Øvrige brancher': 'REST'
}
###################################################################
###################   M Æ N G D E R    ############################
###################################################################

# Erstat navnene i de to kolonner
df['TILGANG2'] = df['TILGANG2'].replace(mapping)
df['ANVENDELSE'] = df['ANVENDELSE'].replace(mapping)
df_priser['TILGANG2'] = df_priser['TILGANG2'].replace(mapping)
df_priser['ANVENDELSE'] = df_priser['ANVENDELSE'].replace(mapping)
df_timer['ANVENDELSE'] = df_timer['ANVENDELSE'].replace(mapping)
df_timeLon['ANVENDELSE'] = df_timeLon['ANVENDELSE'].replace(mapping)
# df_kapital['BRANCHE'] = df_kapital['BRANCHE'].replace(mapping)
# df_kapital_mængder['BRANCHE'] = df_kapital_mængder['BRANCHE'].replace(mapping)
# df_kapital_pris['BRANCHE'] = df_kapital_pris['BRANCHE'].replace(mapping)
df_toldssats['ANVENDELSE'] = df_toldssats['ANVENDELSE'].replace(mapping)
df_toldssats['TILGANG2'] = df_toldssats['TILGANG2'].replace(mapping)
df_afgift['ANVENDELSE'] = df_afgift['ANVENDELSE'].replace(mapping)
df_produktion_løbende['ANVENDELSE'] = df_produktion_løbende['ANVENDELSE'].replace(mapping)
df_produktion_løbende['TILGANG2'] = df_produktion_løbende['TILGANG2'].replace(mapping)
df_input['ANVENDELSE'] = df_input['ANVENDELSE'].replace(mapping)

# Opdateret liste med de korte navne
brancher = ['010000', '100010', '100030', '100040x100050', 'REST']

# Materialer
import_data = df[(df['TILGANG1'] == 'Import eksklusiv told') & (df[i].isin(brancher))]
M_F = import_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

indenlandsk_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i].isin(brancher))]
M_D = indenlandsk_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

told_data = df[(df['TILGANG1'] == 'Told') & (df[i].isin(brancher))]
M_T = told_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

# Y-vektoren (Produktionsværdi i alt for de 5 brancher)
produktionsværdi_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Anvendelse, i alt-(Anvendelse)')]
Y = produktionsværdi_data.pivot_table(index=[j, t], values='Xt').fillna(0)
Y.index.names = [i, t]

# Produktionsværdi i løbende priser
produktionsværdi_data_lob = df_produktion_løbende[(df_produktion_løbende['TILGANG1'] == 'Dansk produktion') & (df_produktion_løbende[i] == 'Anvendelse, i alt-(Anvendelse)') & (df_produktion_løbende['PRISENHED'] == 'Løbende priser')]
Y_lob = produktionsværdi_data_lob.pivot_table(index=[j, t], values='INDHOLD').fillna(0)
Y_lob.index.names = [i, t]

#offentligt forbrug
offentligt_forbrug_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Offentligt forbrug, i alt-(Anvendelse)')]
G = offentligt_forbrug_data.pivot_table(index=[j, t], values='Xt').fillna(0)
G.index.names = [i, t]
target_index = Y.index
G = G.reindex(target_index).fillna(0)

#Privat forbrug
privat_forbrug_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Husholdningernes forbrugsudgifter + NPISH (Anvendelse)')]
C = privat_forbrug_data.pivot_table(index=[j, t], values='Xt').fillna(0)
C.index.names = [i, t]

#eksport
eksport_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Eksport - (Anvendelse)')]
X = eksport_data.pivot_table(index=[j, t], values='Xt').fillna(0)
X.index.names = [i, t]

#timer
timer_data = df_timer
L = timer_data.pivot_table(index=[i, t], values='TIMER').fillna(0)/1000
L.rename(columns={'TIMER': 'Xt'}, inplace=True)
L.index.names = [i, t]

timer_lon_data = df_timer_lon
L_lon = timer_lon_data.pivot_table(index=[i, t], values='TIMER').fillna(0)/1000
L_lon.rename(columns={'TIMER': 'Xt'}, inplace=True)
L_lon.index.names = [i, t]

# #kapitalapparat
# kapitalapparat_data = df_kapital[(df_kapital['BEHOLD'] == 'AN.11 Faste aktiver, nettobeholdning ultimo året') & (df_kapital['PRISENHED'] == '2020-priser, kædede værdier')]
# K = kapitalapparat_data.pivot_table(index=[ii, t], values='INDHOLD').fillna(0)
# K.rename(columns={'INDHOLD': 'Xt'}, inplace=True)
# K.index.names = [i, t]

# #Bruttoinvesteringer
# bruttoinvesterings_data = df_kapital_mængder[(df_kapital_mængder['BEHOLD'] == 'P.51g Faste bruttoinvesteringer')]
# I = bruttoinvesterings_data.pivot_table(index=[ii, t], values='Xt').fillna(0)
# I.index.names = [i, t]

M_D_loebende=df_produktion_løbende[(df_produktion_løbende['TILGANG1'] == 'Dansk produktion') & (df_produktion_løbende[i].isin(brancher)) & (df_produktion_løbende['PRISENHED'] == 'Løbende priser')]
M_D_loebende = M_D_loebende.pivot_table(index=[i, j, t], values='INDHOLD').fillna(0)
M_D_loebende.index.names = [i, j, t]

#KL aggregat
KL_aggregat_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
KL = KL_aggregat_data.pivot_table(index=[i, t], values='KL').fillna(0)
KL.rename(columns={'KL': 'Xt'}, inplace=True)
KL.index.names = [i, t]

#M aggregat
M_aggregat_data = df_aggregater_mjit[(df_aggregater_mjit['ANVENDELSE'].isin(brancher))]
M = M_aggregat_data.pivot_table(index=[i, j, t], values='M').fillna(0)
M.rename(columns={'M': 'Xt'}, inplace=True)
M.index.names = [i, j, t]

#Mtot aggregat
Mtot_aggregat_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
Mtot = Mtot_aggregat_data.pivot_table(index=[i, t], values='M_tot').fillna(0)
Mtot.rename(columns={'M_tot': 'Xt'}, inplace=True)
Mtot.index.names = [i, t]

# Subsider
subsidier_data = df_input[(df_input['TILGANG1'] == 'Andre produktionsskatter, netto') & (df_input['PRISENHED'] == 'Løbende priser')]
subsidier = subsidier_data.pivot_table(index=[i, t], values='INDHOLD').fillna(0)
subsidier.index.names = [i, t]

# Hektarstøtte
hektarstotte_data = df_tilskud
hektarstotte = hektarstotte_data.pivot_table(index=[t], values='INDHOLD').fillna(0)
hektarstotte.index.names = [t]

# Grundskyld
grundskyld_data = df_grundskyld
grundskyld = grundskyld_data.pivot_table(index=[t], values='INDHOLD').fillna(0)
grundskyld.index.names = [t]

# Jordareal
jordareal_data = df_jordareal
J_temp = jordareal_data.pivot_table(index=[t], values='INDHOLD').fillna(0).squeeze()

# Lav J til MultiIndex format [i, t] med kun værdier for branche 01000
J_index = pd.MultiIndex.from_product([['010000'], J_temp.index], names=[i, t])
J_series = pd.Series(J_temp.values, index=J_index, name='INDHOLD')

# Tilføj de andre brancher med værdier 0
other_branches = ['100010', '100030', '100040x100050', 'REST']
for branch in other_branches:
    branch_index = pd.MultiIndex.from_product([[branch], J_temp.index], names=[i, t])
    J_other = pd.Series(0, index=branch_index, name='INDHOLD')
    J_series = pd.concat([J_series, J_other])

J = J_series.to_frame(name='INDHOLD')
J = J.sort_index()
J.rename(columns={'INDHOLD': 'Xt'}, inplace=True)

#JKL aggregat
#JKL aggregat - kun for branche 01000
JKL_aggregat_data = df_aggregater[(df_aggregater['ANVENDELSE'] == '010000')]
JKL_temp = JKL_aggregat_data.pivot_table(index=[t], values='JKL').fillna(0).squeeze()

# Lav JKL til MultiIndex format [i, t] med kun værdier for branche 01000
JKL_index = pd.MultiIndex.from_product([['010000'], JKL_temp.index], names=[i, t])
JKL_series = pd.Series(JKL_temp.values, index=JKL_index, name='Xt')

# Tilføj de andre brancher med værdier 0
other_branches = ['100010', '100030', '100040x100050', 'REST']
for branch in other_branches:
    branch_index = pd.MultiIndex.from_product([[branch], JKL_temp.index], names=[i, t])
    JKL_other = pd.Series(0, index=branch_index, name='Xt')
    JKL_series = pd.concat([JKL_series, JKL_other])

JKL = JKL_series.to_frame(name='Xt')
JKL = JKL.sort_index()

# MD aggregat
MD_aggregat_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
MDtot = MD_aggregat_data.pivot_table(index=[i, t], values='MD_tot').fillna(0)
MDtot.rename(columns={'MD_tot': 'Xt'}, inplace=True)
MDtot.index.names = [i, t]

# MF aggregat
MF_aggregat_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
MFtot = MF_aggregat_data.pivot_table(index=[i, t], values='MF_tot').fillna(0)
MFtot.rename(columns={'MF_tot': 'Xt'}, inplace=True)
MFtot.index.names = [i, t]

# ###################################################################
# ###################   P R I S E R    ##############################
# ###################################################################
# output priser
produktionsværdi_data_priser = df_priser[(df_priser['TILGANG1'] == 'Dansk produktion') & (df_priser[i] == 'Anvendelse, i alt-(Anvendelse)')]
P = produktionsværdi_data_priser.pivot_table(index=[j, t], values='Pt').fillna(0)
P.index.names = [i, t]

#løn
lon_data = df_timeLon
w = df_timeLon.pivot_table(index=[i, t], values='TIMELOEN_KR').fillna(0)
w.rename(columns={'TIMELOEN_KR': 'Pt'}, inplace=True)
w.index.names = [i, t]


# #investeringspriser
# investeringspriser_data = df_kapital_pris[(df_kapital_pris['BEHOLD'] == 'P.51g Faste bruttoinvesteringer')]
# P_I = investeringspriser_data.pivot_table(index=[ii, t], values='Pt').fillna(0)
# P_I.index.names = [ii, t]

#pris import
pris_import_data = df_priser[(df_priser['TILGANG1'] == 'Import eksklusiv told')  & (df[i].isin(brancher))]
P_F = pris_import_data.pivot_table(index=[i, j, t], values='Pt').fillna(0)
P_F.index.names = [i, j, t]

#pris indenlandsk produktion
pris_indenlandsk_produktion_data = df_priser[(df_priser['TILGANG1'] == 'Dansk produktion') & (df[i].isin(brancher))]
P_D = pris_indenlandsk_produktion_data.pivot_table(index=[i, j, t], values='Pt').fillna(0)
P_D.index.names = [i, j, t]

#pris told
pris_told_data = df_priser[(df_priser['TILGANG1'] == 'Told') & (df[i].isin(brancher))]
P_T = pris_told_data.pivot_table(index=[i, j, t], values='Pt').fillna(0)
P_T.index.names = [i, j, t]

#toldssats
toldssats_data = df_toldssats[(df_toldssats[i].isin(brancher))]
tau_MF = toldssats_data.pivot_table(index=[i, j, t], values='tau').fillna(0)
tau_MF.index.names = [i, j, t]

#afgift
afgift_data = df_afgift[(df_afgift[i].isin(brancher))]
afgift = afgift_data.pivot_table(index=[i, t], values='afgift').fillna(0)
afgift.index.names = [i, t]

#KL aggregat pris
KL_aggregat_pris_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
P_KL = KL_aggregat_pris_data.pivot_table(index=[i, t], values='P_KL').fillna(0)
P_KL.rename(columns={'P_KL': 'Pt'}, inplace=True)
P_KL.index.names = [i, t]


#M aggregat pris
M_aggregat_pris_data = df_aggregater_mjit[(df_aggregater_mjit['ANVENDELSE'].isin(brancher))]
P_M = M_aggregat_pris_data.pivot_table(index=[i, j, t], values='P_M').fillna(0)
P_M.index.names = [i, j, t]

#Mtot aggregat pris
Mtot_aggregat_pris_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
P_Mtot = Mtot_aggregat_pris_data.pivot_table(index=[i, t], values='P_Mtot').fillna(0)
P_Mtot.rename(columns={'P_Mtot': 'Pt'}, inplace=True)
P_Mtot.index.names = [i, t]

#Jordpris
jordpris_data = df_jordpris
P_Jord = jordpris_data.pivot_table(index=[t], values='INDHOLD').fillna(0)
P_Jord.index.names = [t]
P_Jord.rename(columns={'INDHOLD': 'Pt'}, inplace=True)

#JKL aggregat pris
JKL_aggregat_pris_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
P_JKL = JKL_aggregat_pris_data.pivot_table(index=[i, t], values='P_JKL').fillna(0)
P_JKL.rename(columns={'P_JKL': 'Pt'}, inplace=True)
P_JKL.index.names = [i, t]

#MD aggregat pris
MD_aggregat_pris_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
P_MDtot = MD_aggregat_pris_data.pivot_table(index=[i, t], values='P_MDtot').fillna(0)
P_MDtot.index.names = [i, t]

#MF aggregat pris
MF_aggregat_pris_data = df_aggregater[(df_aggregater['ANVENDELSE'].isin(brancher))]
P_MFtot = MF_aggregat_pris_data.pivot_table(index=[i, t], values='P_MFtot').fillna(0)
P_MFtot.index.names = [i, t]

# Gamma (kun fødevarebrancher — opsplit mod 10120)
gamma_brancher = ['100010', '100030', '100040x100050']
gamma_df = df_gamma.copy()
gamma_df[i] = gamma_df[i].replace(mapping)
gamma_df = gamma_df[gamma_df[i].isin(gamma_brancher)]
gamma = gamma_df.pivot_table(index=[i, t], values='gamma').fillna(0)
gamma.index.names = [i, t]

#s
s_brancher = ['100010', '100030', '100040x100050']
s_df = df_s.copy()
s_df[i] = s_df[i].replace(mapping)
s_df = s_df[s_df[i].isin(s_brancher)]
s = s_df.pivot_table(index=[i, t], values='gamma').fillna(0)
s.index.names = [i, t]

#Rente
# Step 1: behold din tidsserie
R_t = df_realkredit.pivot_table(index=[t], values='INDHOLD').fillna(0)

# Step 2: lav MultiIndex for (i, t)
R_index = pd.MultiIndex.from_product([brancher, R_t.index], names=[i, t])

# Step 3: gentag værdier for hver branche
R_geld = pd.DataFrame(
    R_t.loc[R_index.get_level_values(t)].values,
    index=R_index,
    columns=['Rt']
)


# Makrodata
qK=makrodata.qK[:,:,:]
qK = qK[(qK.index.get_level_values('t') >= 1992) & (qK.index.get_level_values('t') <= 2022) & (qK.index.get_level_values('i_').isin(['iB', 'iM']))]
qK = qK.rename_axis(index={'i_': 'k'})
rAfskr = makrodata.rAfskr[:,:,:]
rAfskr = rAfskr[(rAfskr.index.get_level_values('t') >= 1992) & (rAfskr.index.get_level_values('t') <= 2022)]
qKxrAfskr = qK*rAfskr
qKxrAfskr_sum=qKxrAfskr.groupby(['s_','t']).sum()
mask = ~qK.index.get_level_values('s_').isin(['lan', 'fre'])
qKxrAfskr_sum_rest = qKxrAfskr[mask].groupby(['t']).sum()
qK_sum=qK.groupby(['s_','t']).sum()
qK_sum_rest = qK[mask].groupby(['t']).sum()
rTaxDep=qKxrAfskr_sum/qK_sum
rTaxDep_rest = qKxrAfskr_sum_rest/qK_sum_rest

rBonds_ra = makrodata.rRente['RealKred',:]
rBonds_ra = rBonds_ra[(rBonds_ra.index.get_level_values('t') >= 1992) & (rBonds_ra.index.get_level_values('t') <= 2022)]
tCorp_ra=makrodata.tSelskab[:]
tCorp_ra = tCorp_ra[(tCorp_ra.index.get_level_values('t') >= 1992) & (tCorp_ra.index.get_level_values('t') <= 2022)]

rBonds = P.copy()
rBonds['Pt'] = rBonds.index.get_level_values('TID').map(rBonds_ra)

tCorp = P.copy()
tCorp['Pt'] = tCorp.index.get_level_values('TID').map(tCorp_ra)