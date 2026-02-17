import pandas as pd
# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output
ii= 'BRANCHE' #Aggregeret branche på 69 opdeling

# Indlæs data
df = pd.read_csv('../Nationalregnskab/Data/landbrugsdata_mængdeindeks.csv')
df_priser = pd.read_csv('../Nationalregnskab/Data/landbrugsdata_prisindeks.csv')
df_timer = pd.read_csv('../Nationalregnskab/Data/Timer_landbrugsdata.csv')
df_timeLon = pd.read_csv('../Nationalregnskab/Data/TimeLon_landbrugsdata.csv')
df_kapital = pd.read_csv('../Nationalregnskab/Data/Kapital_landbrugsdata.csv')

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
# Opdateret liste med de korte navne
brancher = ['010000', '100010', '100030', '100040x100050', 'REST']

# Materialer
import_data = df[(df['TILGANG1'] == 'Import inkl. told') & (df[i].isin(brancher))]
M_F = import_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

indenlandsk_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i].isin(brancher))]
M_D = indenlandsk_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

told_data = df[(df['TILGANG1'] == 'Told') & (df[i].isin(brancher))]
M_T = told_data.pivot_table(index=[i, j, t], values='Xt').fillna(0)

# Y-vektoren (Produktionsværdi i alt for de 5 brancher)
produktionsværdi_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Anvendelse, i alt-(Anvendelse)')]
Y = produktionsværdi_data.pivot_table(index=[j, t], values='Xt').fillna(0)
Y.index.names = [i, t]


# #Investeringer
# investerings_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Faste bruttoinvesteringer + Lagerforøgelse + Værdigenstande (Anvendelse)')]
# I = investerings_data.pivot_table(index=[j, t], values='Xt').fillna(0)
# I.index.names = [i, t]

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
L = timer_data.pivot_table(index=[i, t], values='TIMER').fillna(0)
L.index.names = [i, t]

#kapitalapparat
kapitalapparat_data = df_kapital[(df_kapital['BEHOLD'] == 'AN.11 Faste aktiver, nettobeholdning ultimo året')]
K = kapitalapparat_data.pivot_table(index=[ii, t], values='INDHOLD').fillna(0)
K.index.names = [ii, t]

#Bruttoinvesteringer
bruttoinvesterings_data = df_kapital[(df_kapital['BEHOLD'] == 'P.51g Faste bruttoinvesteringer')]
I = bruttoinvesterings_data.pivot_table(index=[ii, t], values='INDHOLD').fillna(0)
I.index.names = [ii, t]

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
w.index.names = [i, t]
