import pandas as pd
# Definition af dimensioner
t = 'TID'         # Tid
j = 'TILGANG2'    # Input
i = 'ANVENDELSE'  # Output

# Indlæs data
df = pd.read_csv('../Nationalregnskab/Data/landbrugsdata_faste_priser.csv')


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

# Erstat navnene i de to kolonner
df['TILGANG2'] = df['TILGANG2'].replace(mapping)
df['ANVENDELSE'] = df['ANVENDELSE'].replace(mapping)

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


#Investeringer
investerings_data = df[(df['TILGANG1'] == 'Dansk produktion') & (df[i] == 'Faste bruttoinvesteringer + Lagerforøgelse + Værdigenstande (Anvendelse)')]
I = investerings_data.pivot_table(index=[j, t], values='Xt').fillna(0)
I.index.names = [i, t]

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