import calib as calib
import numpy as np
import pandas as pd
import sys
sys.path.append('../Indlæsning af data')
import calib69 as calib69
import data_formater as df
import data_formater69 as df69
import matplotlib.pyplot as plt
import aggregater as agg
import aggregater69 as agg69

i = 'ANVENDELSE'
t = 'TID'

def to_series(obj):
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if "Xt" in obj.columns:
            return obj["Xt"]
        if len(obj.columns) == 1:
            return obj.iloc[:, 0]
        raise ValueError(
            f"DataFrame for variablen har flere kolonner ({list(obj.columns)}). "
            "Vælg én kolonne manuelt."
        )
    raise TypeError(f"Ukendt datatype: {type(obj)}")

def plot_comparison(var_name, branche_69='01000', branche_117='010000'):
    def extract(ds_list, branche):
        for ds in ds_list:
            if hasattr(ds, var_name):
                s = to_series(getattr(ds, var_name))
                return s.xs(branche, level=i)
        raise KeyError(f"'{var_name}' ikke fundet")

    s1 = extract([df69, calib69], branche_69)
    s2 = extract([df, calib], branche_117)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s1.index, s1.values, label=f'69-opdeling ({branche_69})', marker='o', markersize=3)
    ax.plot(s2.index, s2.values, label=f'117-opdeling ({branche_117})', marker='o', markersize=3, linestyle='--')    
    ax.set_title(var_name)
    ax.set_xlabel('År')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

variables_to_export = ["Y", "Mtot", "K", "L", "P", "w", "P_K", "M_D_tot","tau_MD","P_MxM_tot","tau_MF"]

# Kør for alle variable
for var in variables_to_export:
    try:
        plot_comparison(var)
    except Exception as e:
        print(f"Kunne ikke plotte {var}: {e}")



# --- Figur 4: Landbrugssektor ---
d = df69.M_D.loc[('01000','01000', slice(1994, 2022))].copy()
p = df.M_D.loc[('010000','010000', slice(1994, 2022))].copy()
years = d.index.get_level_values('TID')
plt.figure(figsize=(10,5))
plt.plot(years, d, label=r'$\theta^{MD}_{it}$ Danske materialer')
plt.plot(years, p, label=r'$\theta^{MF}_{it}$ Importerede materialer')
plt.xlabel("År")
plt.ylabel("CES-andel")
plt.title(f"Materiale-CES – Landbrugssektor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# --- Figur 5: Fødevaresektor ---
d = df69.M_D.loc[('01000','REST', slice(1994, 2022))].copy()
p = df.M_D.loc[('010000','REST', slice(1994, 2022))].copy()
plt.figure(figsize=(10,5))
plt.plot(years, d, label=r'$\theta^{MD}_{it}$ Danske materialer')
plt.plot(years, p, label=r'$\theta^{MF}_{it}$ Importerede materialer')
plt.xlabel("År")
plt.ylabel("CES-andel")
plt.title(f"Materiale-CES – Fødevaresektor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()




def get_serie(obj, branche):
    # Sæt index hvis TID og ANVENDELSE er kolonner
    if isinstance(obj, pd.DataFrame):
        cols = obj.columns.tolist()
        idx_cols = [c for c in ['TILGANG2', 'ANVENDELSE', 'TID'] if c in cols]
        if idx_cols:
            obj = obj.set_index(idx_cols)
        if 'ANVENDELSE' in obj.index.names:
            obj = obj.xs(branche, level='ANVENDELSE')
        # Hent første værdikolonne
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj
    if isinstance(obj, pd.Series):
        if 'ANVENDELSE' in obj.index.names:
            return obj.xs(branche, level='ANVENDELSE')
        return obj
    raise ValueError(f"Kan ikke håndtere {type(obj)}")

materiale_vars = {
    'P_Mtot':  (agg.df_Pt_Mit,   agg69.df_Pt_Mit),
    'M_tot':   (agg.df_Xt_Mit,   agg69.df_Xt_Mit),
    'P_MDtot': (agg.df_Pt_MDit,  agg69.df_Pt_MDit),
    'MD_tot':  (agg.df_Xt_MDit,  agg69.df_Xt_MDit),
    'P_MFtot': (agg.df_Pt_MFit,  agg69.df_Pt_MFit),
    'MF_tot':  (agg.df_Xt_MFit,  agg69.df_Xt_MFit),
    'P_KL':    (agg.df_Pt_KLit,  agg69.df_Pt_KLit),
    'KL':      (agg.df_Xt_KLit,  agg69.df_Xt_KLit),
}

for navn, (raw_117, raw_69) in materiale_vars.items():
    try:
        s117 = get_serie(raw_117, '010000')
        s69  = get_serie(raw_69,  '01000')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s117.index, s117.values, label='117-opdeling (010000)', marker='o', markersize=2)
        ax.plot(s69.index,  s69.values,  label='69-opdeling (01000)',   marker='o', markersize=2, linestyle='--')
        ax.set_title(navn)
        ax.set_xlabel('År')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Kunne ikke plotte {navn}: {e}")


def get_serie(obj, branche):
    if isinstance(obj, pd.DataFrame):
        cols = obj.columns.tolist()
        idx_cols = [c for c in ['TILGANG2', 'ANVENDELSE', 'TID'] if c in cols]
        if idx_cols:
            obj = obj.set_index(idx_cols)
        if 'ANVENDELSE' in obj.index.names:
            obj = obj.xs(branche, level='ANVENDELSE')
        # Hvis TILGANG2 stadig er i index, returner DataFrame
        if isinstance(obj, pd.DataFrame):
            return obj
        return obj
    if isinstance(obj, pd.Series):
        if 'ANVENDELSE' in obj.index.names:
            return obj.xs(branche, level='ANVENDELSE')
        return obj
    raise ValueError(f"Kan ikke håndtere {type(obj)}")


materialex_vars = {
    'Mjit': (agg.df_Xt_Mjit, agg69.df_Xt_Mjit),
    'df_materialer_aggregat_loebende': (agg.df_materialer_aggregat_loebende, agg69.df_materialer_aggregat_loebende),
    'df_materialer_aggregat_for': (agg.df_materialer_aggregat_for, agg69.df_materialer_aggregat_for),
    'df_tau_MF': (agg.df_tau_MF, agg69.df_tau_MF),
    'df_tau_MD': (agg.df_tau_MD, agg69.df_tau_MD),
    'df_materialer_MD_for': (agg.df_materialer_MD_for, agg69.df_materialer_MD_for),
    'df_materialer_MF_for': (agg.df_materialer_MF_for, agg69.df_materialer_MF_for),
}

for navn, (raw_117, raw_69) in materialex_vars.items():
    tilgang_filter_117 = '010000'
    tilgang_filter_69  = '01000'
    try:
        s117 = get_serie(raw_117, '010000')
        s69  = get_serie(raw_69,  '01000')

        fig, ax = plt.subplots(figsize=(10, 4))

        if isinstance(s117, pd.DataFrame):
            for tilgang, gruppe in s117.groupby(level='TILGANG2'):
                if tilgang_filter_117 and tilgang != tilgang_filter_117:
                    continue
                ax.plot(gruppe.index.get_level_values('TID'), gruppe.values, label=f'117 {tilgang}', marker='o', markersize=2)

            for tilgang, gruppe in s69.groupby(level='TILGANG2'):
                if tilgang_filter_69 and tilgang != tilgang_filter_69:
                    continue
                ax.plot(gruppe.index.get_level_values('TID'), gruppe.values, label=f'69 {tilgang}', marker='o', markersize=2, linestyle='--')        
        else:
            ax.plot(s117.index, s117.values, label='117-opdeling (010000)', marker='o', markersize=2)
            ax.plot(s69.index,  s69.values,  label='69-opdeling (01000)',   marker='o', markersize=2, linestyle='--')

        ax.set_title(navn)
        ax.set_xlabel('År')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Kunne ikke plotte {navn}: {e}")