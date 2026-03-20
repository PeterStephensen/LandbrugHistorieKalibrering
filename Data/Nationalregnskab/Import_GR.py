import pandas as pd
# 1. Indlæs filen. 
# 'skiprows' skal matche antallet af tomme/tekst-rækker før dine årstal (ser ud til ca. 4 i billedet)
df_rente = pd.read_excel('Data/Nationalregnskab/Data69/konjunktur_juni2025.xlsx')
df_rente = df_rente.dropna(how='all', axis=0)
# 2. Find rækken baseret på teksten i den første kolonne (B i dit ark)
# Vi antager at kolonnen med teksten nu er din index-kolonne eller første kolonne
row_name = 'RealKred'
mask = df_rente.stack().str.contains('RealKred', na=False).unstack().any(axis=1)
price_row = df_rente[mask]

price_row
# 3. Formater til en pæn DataFrame (transpose den, så år er rækker)
# Vi fjerner den første kolonne (teksten) og vender tabellen om
final_df = price_row.iloc[:, 2:].transpose()

final_df.columns = ['INDHOLD']
final_df.index.name = 'TID'
final_df.reset_index(inplace=True)

# 🔽 NYT
final_df['TID'] = pd.to_numeric(final_df['TID'], errors='coerce')
final_df = final_df[final_df['TID'] < 2023]

print(final_df)

final_df.to_csv('Data/Nationalregnskab/Data69/rente_landbrugsdata.csv', index=False)