"""
kalibrering.py
==============
Kalibrerer CES-strukturens teknologiparametre (theta) og cost-shares (mu)
for 69- og 117-aggregeringsniveau.

Pipeline-rækkefølge
-------------------
Der er en afhængighed imellem aggregater og kalibrering:
  - aggregater.py bruger tau_MD og P_J som inputs
  - kalibrering.py bruger KL, JKL, P_KL, P_JKL (outputs fra aggregater)

tau_MD og P_J afhænger dog KUN af omregning-resultater og eksternt data
(ikke af aggregater), så de kan beregnes i en Fase 1 inden aggregater kører.

Korrekt rækkefølge:
    # Fase 1 – ingen aggregater nødvendige
    tau_MD_69  = beregn_tau_MD(data, results, MAPPING_69,  BRANCHER_69)
    tau_MD_117 = beregn_tau_MD(data, results, MAPPING_117, BRANCHER_117)
    P_J        = beregn_P_J(data, df_realkredit, makrodata)

    # Fase 2 – bruger tau_MD og P_J fra fase 1
    agg = beregn_alle_aggregater(data, results, tau_MD_69, tau_MD_117, P_J)

    # Fase 3 – bruger aggregater fra fase 2
    v69   = byg_variable_69(data, results, agg, df_realkredit, makrodata)
    v117  = byg_variable_117(data, results, agg, opsplit, df_realkredit, makrodata)
    kp69  = kalibrering_69(v69)
    kp117 = kalibrering_117(v117, v69, kp69)
"""

import pandas as pd
import numpy as np

# Dimensionsnavne
t  = "TID"
j  = "TILGANG2"
i  = "ANVENDELSE"

# Basisår for theta-indeks
BASIS_AAR = 1994


# ===========================================================================
# Hjælpefunktioner
# ===========================================================================

def _indekser(serie, basisaar=BASIS_AAR):
    """Indekserer en serie så basisår = 1."""
    basis = serie.xs(basisaar, level=t)
    return serie / basis


def map_69_til_117(serie_69, brancher_117=None):
    """
    Mapper en serie fra 69-opdeling til 117-opdeling.
      01000  → 010000             (1-til-1)
      10120  → 100010, 100030, 100040x100050  (1-til-mange, samme værdi)
      REST   → REST               (1-til-1)

    Parametre
    ----------
    serie_69    : Series med MultiIndex (ANVENDELSE, TID)
    brancher_117: liste over brancher der skal med (None = alle)
    """
    mapping = {
        "01000": ["010000"],
        "10120": ["100010", "100030", "100040x100050"],
        "REST":  ["REST"],
    }
    rows = []
    for b69, b117_liste in mapping.items():
        if b69 not in serie_69.index.get_level_values(i):
            continue
        vaerdier = serie_69.xs(b69, level=i)
        for b117 in b117_liste:
            if brancher_117 and b117 not in brancher_117:
                continue
            temp = vaerdier.copy()
            temp.index = pd.MultiIndex.from_arrays(
                [[b117] * len(temp), temp.index], names=[i, t]
            )
            rows.append(temp)
    return pd.concat(rows).sort_index()


def _expand_til_ij(serie_i, M_index):
    """
    Broadcaster en (ANVENDELSE, TID)-serie ned på (ANVENDELSE, TILGANG2, TID).
    Bruges til at matche Mtot / P_Mtot med M's index.
    """
    df_i = serie_i.reset_index(name="_val")
    M_idx_df = M_index.to_frame(index=False)
    merged = M_idx_df.merge(df_i, on=[i, t], how="left").fillna(0)
    return pd.Series(
        merged["_val"].values,
        index=pd.MultiIndex.from_frame(merged[[i, j, t]]),
    )


# ===========================================================================
# 69-niveau: brugerpriser og kalibrering
# ===========================================================================

def _beregn_ektax(v, r):
    """Beregner skattemæssig kapitalomkostningsfordel (EKtax) pr. branche."""
    EKtax_lan  = v["tCorp"]["Pt"].loc["01000"]  * v["rTaxDep"].loc["lan"]  / (r.loc["01000"]  + v["rTaxDep"].loc["lan"])
    EKtax_fre  = v["tCorp"]["Pt"].loc["10120"]  * v["rTaxDep"].loc["fre"]  / (r.loc["10120"]  + v["rTaxDep"].loc["fre"])
    EKtax_rest = v["tCorp"]["Pt"].loc["REST"]    * v["rTaxDep_rest"]        / (r.loc["REST"]    + v["rTaxDep_rest"])

    for ser, name in [(EKtax_lan, "01000"), (EKtax_fre, "10120"), (EKtax_rest, "REST")]:
        ser.index.name = t
        ser.index = pd.MultiIndex.from_product([[name], ser.index], names=[i, t])

    EKtax = pd.concat([EKtax_lan, EKtax_fre, EKtax_rest]).sort_index()
    EKtax.index.names = [i, t]
    return EKtax


def _beregn_forventet_inflation(P_I, gamma=0.6):
    """Koyck-distribueret forventet inflation i investeringspriserne."""
    inf = P_I["Pt"] / P_I.groupby(i).shift(1)["Pt"]
    E_inf = inf.copy()
    E_inf.loc[pd.IndexSlice[:, :1993]] = np.nan
    for anv, s in inf.groupby(level=i):
        if (anv, 1994) not in inf.index:
            continue
        E_inf.loc[(anv, 1994)] = inf.loc[(anv, 1994)]
        for yr in [y for y in s.index.get_level_values(t) if y >= 1995]:
            E_inf.loc[(anv, yr)] = gamma * E_inf.loc[(anv, yr - 1)] + (1 - gamma) * inf.loc[(anv, yr)]
    return E_inf


def beregn_brugerpris_kapital_69(v, r):
    """
    Beregner brugerprisen på kapital (P_K) for 69-niveau.
    Bygger på Hall-Jorgenson med selskabsskat og forventet inflation.

    Parametre
    ----------
    v : variabel-dict fra vars_fra_data_formater(df69)
    r : rentevektor (Series med brancheindeks)

    Returnerer
    ----------
    P_K : Series med MultiIndex (ANVENDELSE, TID)
    delta : Series med MultiIndex (ANVENDELSE, TID)
    """
    K_prev  = v["K"].groupby(i)["Xt"].shift(1)
    delta   = (K_prev - v["K"]["Xt"] + v["I"]["Xt"]) / K_prev
    delta.index.names = [i, t]

    EKtax      = _beregn_ektax(v, r)
    EKtax_prev = EKtax.groupby(i).shift(1)

    P_I_prev = v["P_I"].groupby(i).shift(1)
    E_inf    = _beregn_forventet_inflation(v["P_I"])

    P_K = (
        (1 + r) * (P_I_prev["Pt"] - P_I_prev["Pt"] * EKtax_prev)
        - (1 - delta) * (P_I_prev["Pt"] - P_I_prev["Pt"] * EKtax) * E_inf
        - ((r - v["rBonds"]["Pt"] * (1 - v["tCorp"]["Pt"])) * 0.6 * P_I_prev["Pt"])
    ) / (1 - v["tCorp"]["Pt"])

    return P_K, delta


def beregn_brugerpris_jord_69(v, r):
    """
    Beregner brugerprisen på jord (P_J) for landbrug (69-niveau).

    Returnerer
    ----------
    P_J : Series indekseret på TID
    """
    P_Jord_prev = v["P_Jord"].shift(1)
    P_Jord_2022 = v["P_Jord"].loc[2022]
    P_Jord_1982 = v["P_Jord"].loc[1982]
    inf_J = (P_Jord_2022 / P_Jord_1982) ** (1 / (2022 - 1982))

    s_hektar = v["hektarstotte"]["INDHOLD"] / v["J"]["Xt"].loc["01000"]
    s_hektar.index.name = t

    grundskyld_korr = v["grundskyld"]["INDHOLD"].copy()
    grundskyld_korr.loc[grundskyld_korr.index >= 2009] /= 1.349
    grundskyld_hektar = grundskyld_korr / v["J"]["Xt"].loc["01000"]

    P_J = (
        (
            (1 + r.loc["01000"]) * P_Jord_prev["Pt"]
            - P_Jord_prev["Pt"] * inf_J["Pt"]
            - ((r.loc["01000"] - v["rBonds"]["Pt"].loc["01000"] * (1 - v["tCorp"]["Pt"].loc["01000"])) * 0.6 * P_Jord_prev["Pt"])
        ) / (1 - v["tCorp"]["Pt"].loc["01000"])
        - s_hektar + grundskyld_hektar
    )
    P_J.index.names = [t]
    return P_J


def _beregn_P_O(v, P_KLxKL, PxJ, landbrug_kode):
    """
    Beregner output-kostindeks (P_O).
    Jord-leddet tilføjes kun for landbrugsbranchen.
    """
    target = v["Y"]["Xt"].index
    P_MxM_adj = v["P_MxM_tot"].reindex(target, fill_value=0)
    P_KL_adj  = P_KLxKL.reindex(target, fill_value=0)

    PxJ_final = pd.Series(0.0, index=target)
    mask      = target.get_level_values(i) == landbrug_kode
    years     = target.get_level_values(t)[mask]
    PxJ_final.loc[mask] = PxJ.reindex(years).values

    return (P_MxM_adj + P_KL_adj + PxJ_final) / v["Y"]["Xt"]


def _beregn_subsidier_adj(v, landbrug_kode):
    """Justerer subsidier for hektarstøtte (efter 2005) og grundskyld (efter 2009)."""
    subs = v["subsidier"]["INDHOLD"].copy()
    mask_lan  = subs.index.get_level_values(i) == landbrug_kode
    mask_tid  = subs.index.get_level_values(t) >= 2005
    mask_grnd = subs.index.get_level_values(t) >= 2009

    mask       = mask_lan & mask_tid
    mask_g     = mask_lan & mask_grnd
    tid_v      = subs.index.get_level_values(t)[mask]
    tid_g      = subs.index.get_level_values(t)[mask_g]

    hek_ri = v["hektarstotte"]["INDHOLD"].reindex(tid_v, fill_value=0)
    gnd_ri = v["grundskyld"]["INDHOLD"].reindex(tid_g, fill_value=0)

    subs.loc[mask]  = subs.loc[mask]  + hek_ri.values
    subs.loc[mask_g] = subs.loc[mask_g] - 0.349 * gnd_ri.values
    return subs


def _beregn_mu_theta(
    Y, P_O, Mtot, P_Mtot, KL, P_KL, JKL, P_JKL,
    M, P_M, M_D, P_D, tau_MD, M_F, P_F, tau_MF,
    K_prev, P_K, L, w, J_prev, P_J,
    EY, EMtot, EM, EKL, EJKL,
    landbrug_kode,
):
    """Beregner mu (cost-shares) og theta (teknologiparametre) for hele CES-strukturen."""

    # ---- Øverste niveau: Y → Mtot / KL ----
    mu_Y_Mtot = (Mtot["Xt"] / Y["Xt"]) * (P_Mtot["Pt"] / P_O) ** EY
    mu_Y_KL   = (KL["Xt"]   / Y["Xt"]) * (P_KL["Pt"]   / P_O) ** EY
    mu_Y_Mtot.index.names = [i, t]
    mu_Y_KL.index.names   = [i, t]

    # JKL erstatter KL for landbrug
    mu_Y_JKL  = mu_Y_KL.copy()
    mask_lan  = mu_Y_JKL.index.get_level_values(i) == landbrug_kode
    tid_lan   = mu_Y_JKL.loc[mask_lan].index.get_level_values(t)
    mu_Y_JKL.loc[mask_lan] = (
        (JKL["Xt"].loc[landbrug_kode].reindex(tid_lan).values / Y["Xt"].loc[landbrug_kode].reindex(tid_lan).values)
        * (P_JKL["Pt"].loc[landbrug_kode].reindex(tid_lan).values / P_O.loc[landbrug_kode].reindex(tid_lan).values) ** EY
    )
    mu_Y_JKL.index.names = [i, t]

    # ---- Materiale-niveau: Mtot → Mjit ----
    Mtot_exp   = _expand_til_ij(Mtot["Xt"],   M.index)
    P_Mtot_exp = _expand_til_ij(P_Mtot["Pt"], M.index)
    mu_Mtot_M  = (M["Xt"] / Mtot_exp) * (P_M["P_M"] / P_Mtot_exp) ** EMtot
    mu_Mtot_M.index.names = [i, j, t]

    # ---- Import vs. dansk ----
    mu_MD = (M_D["Xt"] / M["Xt"]) * (((1 + tau_MD) * P_D["Pt"]) / P_M["P_M"]) ** EM
    mu_MF = (M_F["Xt"] / M["Xt"]) * (((1 + tau_MF["tau"]) * P_F["Pt"]) / P_M["P_M"]) ** EM
    mu_MD.index.names = [i, j, t]
    mu_MF.index.names = [i, j, t]

    # ---- KL-niveau: KL → K / L ----
    mu_KL_K = (K_prev    / KL["Xt"]) * (P_K     / P_KL["Pt"]) ** EKL
    mu_KL_L = (L["Xt"]   / KL["Xt"]) * (w["Pt"] / P_KL["Pt"]) ** EKL

    # ---- JKL-niveau: JKL → J / KL (kun landbrug) ----
    mu_JKL_J  = (J_prev.loc[landbrug_kode] / JKL["Xt"].loc[landbrug_kode]) * (P_J / P_JKL["Pt"].loc[landbrug_kode]) ** EJKL
    mu_JKL_KL = (KL["Xt"].loc[landbrug_kode] / JKL["Xt"].loc[landbrug_kode]) * (P_KL["Pt"].loc[landbrug_kode] / P_JKL["Pt"].loc[landbrug_kode]) ** EJKL
    mu_JKL_J.index  = pd.MultiIndex.from_product([[landbrug_kode], mu_JKL_J.index],  names=[i, t])
    mu_JKL_KL.index = pd.MultiIndex.from_product([[landbrug_kode], mu_JKL_KL.index], names=[i, t])

    # ---- Theta = mu^(1/(E-1)), indekseret til BASIS_AAR ----
    def _theta(mu, E):
        return mu ** (1 / (E - 1)) if E != 1 else mu

    thetas = {
        "theta_Y_KL":    _indekser(_theta(mu_Y_KL,   EY)),
        "theta_Y_JKL":   _indekser(_theta(mu_Y_JKL,  EY)),
        "theta_Y_Mtot":  _indekser(_theta(mu_Y_Mtot, EY)),
        "theta_Mtot_M":  _indekser(_theta(mu_Mtot_M, EMtot)),
        "theta_MD":      _indekser(_theta(mu_MD, EM)),
        "theta_MF":      _indekser(_theta(mu_MF, EM)),
        "theta_KL_K":    _indekser(_theta(mu_KL_K, EKL)),
        "theta_KL_L":    _indekser(_theta(mu_KL_L, EKL)),
        "theta_JKL_J":   _indekser(_theta(mu_JKL_J,  EJKL)),
        "theta_JKL_KL":  _indekser(_theta(mu_JKL_KL, EJKL)),
    }

    return {
        "mu_Y_KL": mu_Y_KL, "mu_Y_JKL": mu_Y_JKL, "mu_Y_Mtot": mu_Y_Mtot,
        "mu_Mtot_M": mu_Mtot_M,
        "mu_MD": mu_MD, "mu_MF": mu_MF,
        "mu_KL_K": mu_KL_K, "mu_KL_L": mu_KL_L,
        "mu_JKL_J": mu_JKL_J, "mu_JKL_KL": mu_JKL_KL,
        **thetas,
    }


# ===========================================================================
# Fase 1: tau_MD og P_J – kan beregnes FØR aggregater
# ===========================================================================

def beregn_tau_MD(data, results, mapping, brancher):
    """
    Beregner tau_MD = afgift / samlet dansk produktion (løbende priser).

    Kræver KUN omregning-resultater – ingen aggregater.
    Skal kaldes i Fase 1 så aggregater.py kan bruge den.

    Parametre
    ----------
    data     : dict fra import_data.load_all()
    results  : dict fra omregning.beregn_alle()
    mapping  : MAPPING_69 eller MAPPING_117
    brancher : BRANCHER_69 eller BRANCHER_117

    Returnerer
    ----------
    DataFrame med kolonner: ANVENDELSE, TID, tau
    """
    # Vælg afgift og io baseret på hvilke brancher der er til stede
    is_69 = any(b in ("01000", "10120") for b in brancher)
    afgift_key = "afgift"     if (is_69 or "afgift_117"  not in results) else "afgift_117"
    afgift = afgift[afgift[i].isin(brancher)].pivot_table(
        index=[i, t], values="afgift"
    ).fillna(0)
    afgift.index.names = [i, t]

    io_key    = "io"     if (is_69 or "io_117"      not in data)    else "io_117"

    # Samlet dansk produktion (løbende priser) – summer over TILGANG2
    io_lob = _m(
        data[io_key][
            (data[io_key]["PRISENHED"] == "Løbende priser") &
            (data[io_key]["TILGANG1"]  == "Dansk produktion")
        ],
        ["TILGANG2", "ANVENDELSE"], mapping
    )
    io_lob = io_lob[io_lob[i].isin(brancher)]
    M_D_tot = io_lob.groupby([i, t])["INDHOLD"].sum().fillna(0)
    M_D_tot.index.names = [i, t]

    tau = afgift["afgift"] / M_D_tot
    tau = tau.loc[tau.index.get_level_values(t) != 1966]
    tau.index.names = [i, t]
    return tau.reset_index(name="tau")


def beregn_P_J(data, df_realkredit, makrodata, r_landbrug=0.05):
    """
    Beregner brugerprisen på jord (P_J) for landbrugssektoren.

    Kræver KUN jordpris, tilskud, grundskyld og makrodata – ingen aggregater.
    Skal kaldes i Fase 1 så aggregater.py kan bruge den som df_jord_pris.

    Parametre
    ----------
    data        : dict fra import_data.load_all()
    df_realkredit: DataFrame med realkreditrente (TID, INDHOLD)
    makrodata   : dreamtools GDX-objekt
    r_landbrug  : realrente for landbrug (default 0.05)

    Returnerer
    ----------
    DataFrame med kolonner: TID, Pt  (samme format som aggregater forventer)
    """
    # Byg de få variable der er nødvendige
    P_Jord = data["jordpris"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    P_Jord.index.names = [t]
    P_Jord = P_Jord.rename(columns={"INDHOLD": "Pt"})

    hektarstotte = data["tilskud"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    hektarstotte.index.names = [t]

    grundskyld = data["grundskyld"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    grundskyld.index.names = [t]

    jordareal = data["jordareal"].pivot_table(index=[t], values="INDHOLD").fillna(0).squeeze()
    jordareal.index.name = t

    # Makrodata: rBonds og tCorp for landbrug
    rBonds_ra = makrodata.rRente["RealKred", :]
    tCorp_ra  = makrodata.tSelskab[:]

    # Align til samme tidsindeks som P_Jord
    tid = P_Jord.index
    rBonds_s = pd.Series(tid.map(rBonds_ra), index=tid, name=t)
    tCorp_s  = pd.Series(tid.map(tCorp_ra),  index=tid, name=t)

    # Minimal v-dict til beregn_brugerpris_jord_69
    r_ser = pd.Series({"01000": r_landbrug})
    r_ser.index.name = i

    v_minimal = {
        "P_Jord":      P_Jord,
        "hektarstotte": hektarstotte,
        "grundskyld":   grundskyld,
        "J":            _byg_jord_multindeks(data["jordareal"], "01000", ["10120", "REST"]),
        "rBonds":       pd.DataFrame({"Pt": rBonds_s}).rename_axis(index={t: t}),
        "tCorp":        pd.DataFrame({"Pt": tCorp_s}).rename_axis(index={t: t}),
    }

    # Tilpas rBonds og tCorp til MultiIndex-format som beregn_brugerpris_jord_69 forventer
    idx = pd.MultiIndex.from_product([["01000"], tid], names=[i, t])
    v_minimal["rBonds"] = pd.DataFrame(
        {"Pt": [rBonds_s.get(yr, float("nan")) for yr in tid] * 1},
        index=idx
    )
    v_minimal["tCorp"] = pd.DataFrame(
        {"Pt": [tCorp_s.get(yr, float("nan")) for yr in tid] * 1},
        index=idx
    )

    P_J = beregn_brugerpris_jord_69(v_minimal, r_ser)
    return P_J.reset_index(name="Pt")




def kalibrering_69(
    v,
    r_map=None,
    EY=1.3, EMtot=0,
    EM_map=None,
    EKL=0.4, EJKL=0,
):
    """
    Kalibrerer CES-strukturen for 69-aggregeringsniveau.

    Parametre
    ----------
    v       : variabel-dict.  Forventede nøgler:
                K, I, I_lob, P_I, tCorp, rBonds, rTaxDep, rTaxDep_rest,
                Y, Y_lob, Mtot, P_Mtot, KL, P_KL, JKL, P_JKL,
                M, P_M, M_D, M_F, P_D, P_F, tau_MF,
                afgift, M_D_loebende,
                subsidier, hektarstotte, grundskyld,
                J, P_Jord, w, L
    r_map   : dict branche→rente (default: 01000=0.05, 10120=0.07, REST=0.07)
    EM_map  : dict branche→EM-elasticitet

    Returnerer
    ----------
    dict med delta, P_K, P_J, tau_MD, P_O, tau_Y, markup, mu_*, theta_*
    """
    if r_map is None:
        r_map = {"01000": 0.05, "10120": 0.07, "REST": 0.07}
    if EM_map is None:
        EM_map = {"01000": 3.5, "10120": 4.0, "REST": 2.0}

    r  = pd.Series(r_map);  r.index.name = i
    EM = pd.Series(EM_map); EM.index.name = j

    LANDBRUG = "01000"

    # Brugerpris kapital og afskrivning
    P_K, delta = beregn_brugerpris_kapital_69(v, r)

    # Brugerpris jord
    P_J = beregn_brugerpris_jord_69(v, r)

    # tau_MD
    M_D_tot = v["M_D_loebende"].groupby([i, t]).sum()
    M_D_tot.index.names = [i, t]
    tau_MD = v["afgift"]["afgift"] / M_D_tot["INDHOLD"]
    tau_MD = tau_MD.loc[tau_MD.index.get_level_values(t) != 1966]
    tau_MD.index.names = [i, t]

    # P_MxM og P_KLxKL
    K_prev    = v["K"].groupby(i)["Xt"].shift(1)
    P_MxM_F   = (1 + v["tau_MF"]["tau"]) * v["P_F"]["Pt"] * v["M_F"]["Xt"]
    P_MxM_D   = (1 + tau_MD)             * v["P_D"]["Pt"] * v["M_D"]["Xt"]
    P_MxM_tot = (P_MxM_D + P_MxM_F).groupby([i, t]).sum()
    P_MxM_tot.index.names = [i, t]

    P_KLxKL = v["w"]["Pt"] * v["L"]["Xt"] + P_K * K_prev
    v["P_MxM_tot"] = P_MxM_tot.to_frame(name="INDHOLD")  # gør det tilgængeligt for P_O

    J_prev = v["J"].groupby(i)["Xt"].shift(1)
    PxJ    = P_J * J_prev.loc[LANDBRUG]

    P_O = _beregn_P_O(v, P_KLxKL, PxJ, LANDBRUG)

    # tau_Y og markup
    subs_adj = _beregn_subsidier_adj(v, LANDBRUG)
    tau_Y  = subs_adj / (v["Y_lob"]["INDHOLD"] - subs_adj)
    markup = v["P"]["Pt"] / ((1 + tau_Y) * P_O) - 1

    # mu og theta
    mu_theta = _beregn_mu_theta(
        Y=v["Y"], P_O=P_O,
        Mtot=v["Mtot"], P_Mtot=v["P_Mtot"],
        KL=v["KL"],   P_KL=v["P_KL"],
        JKL=v["JKL"], P_JKL=v["P_JKL"],
        M=v["M"],     P_M=v["P_M"],
        M_D=v["M_D"], P_D=v["P_D"], tau_MD=tau_MD,
        M_F=v["M_F"], P_F=v["P_F"], tau_MF=v["tau_MF"],
        K_prev=K_prev, P_K=P_K,
        L=v["L"], w=v["w"],
        J_prev=J_prev, P_J=P_J,
        EY=EY, EMtot=EMtot, EM=EM, EKL=EKL, EJKL=EJKL,
        landbrug_kode=LANDBRUG,
    )

    return {
        "delta": delta, "P_K": P_K, "P_J": P_J,
        "tau_MD": tau_MD, "P_O": P_O, "tau_Y": tau_Y, "markup": markup,
        "K_prev": K_prev, "J_prev": J_prev,
        "P_MxM_tot": P_MxM_tot, "P_KLxKL": P_KLxKL,
        **mu_theta,
    }


def kalibrering_117(
    v,
    v69,
    kp69,
    r_map=None,
    EY=0, EMtot=0,
    EM_map=None,
    EKL=0, EJKL=0,
    g=0.02,
):
    """
    Kalibrerer CES-strukturen for 117-aggregeringsniveau.

    Bruger P_K og delta fra 69-niveauet (kortlagt via map_69_til_117)
    og splitter investeringerne med gamma-vægte fra opsplit_kapital.

    Parametre
    ----------
    v     : variabel-dict for 117-niveau
    v69   : variabel-dict for 69-niveau
    kp69  : output fra kalibrering_69()
    r_map : dict branche→rente (default: 010000=0.05, øvrige=0.07)
    EM_map: dict branche→EM-elasticitet
    g     : langsigtet vækstrate (til initialt kapital)

    Returnerer
    ----------
    dict med K, delta, P_K, P_I, tau_MD, P_O, tau_Y, markup, mu_*, theta_*
    """
    if r_map is None:
        r_map = {"010000": 0.05, "100010": 0.07, "100030": 0.07,
                 "100040x100050": 0.07, "REST": 0.07}
    if EM_map is None:
        EM_map = {"010000": 3.5, "100010": 4.0, "100030": 4.0,
                  "100040x100050": 4.0, "REST": 2.0}

    r  = pd.Series(r_map);  r.index.name = i
    EM = pd.Series(EM_map); EM.index.name = j

    LANDBRUG = "010000"

    # Kortlæg P_K, delta og P_I fra 69 til 117
    P_K   = map_69_til_117(kp69["P_K"])
    P_K.index.names = [i, t]
    delta = map_69_til_117(kp69["delta"])
    delta.index.names = [i, t]
    P_I   = map_69_til_117(v69["P_I"]["Pt"]).to_frame(name="Pt")
    P_I.index.names = [i, t]

    # Split investeringer for fødevarebrancher via gamma
    food_branches     = ["100010", "100030", "100040x100050"]
    non_food_branches = ["010000", "REST"]

    I_non_food = map_69_til_117(v69["I"]["Xt"], non_food_branches)
    I_non_food = I_non_food[I_non_food.index.get_level_values(i).isin(non_food_branches)]
    I_non_food.index.names = [i, t]

    I_lob_10120 = v69["I_lob"].xs("10120", level=0)["Xt"]
    gamma       = v["gamma"]

    I_parts = []
    for br in food_branches:
        g_br = gamma["gamma"].xs(br, level=i) if "gamma" in gamma.columns else gamma.xs(br, level=i)
        prod = g_br * I_lob_10120.reindex(g_br.index).fillna(0)
        idx  = pd.MultiIndex.from_arrays([[br] * len(prod), prod.index], names=[i, t])
        I_parts.append(pd.Series(prod.values, index=idx, name="Xt"))

    I_food_lob = pd.concat(I_parts).sort_index().to_frame("Xt")
    I_food_real = I_food_lob.copy()
    I_food_real["Xt"] = I_food_lob["Xt"] / P_I.reindex(I_food_lob.index).fillna(1)["Pt"]

    I_real = pd.concat([I_food_real, I_non_food.to_frame("Xt")]).sort_index()
    I_real.index.names = [i, t]

    # Kapitalakkumulation
    K0 = I_real.xs(1993, level=t)["Xt"] / (g + delta.xs(1993, level=t))
    year_list    = sorted(I_real.index.get_level_values(t).unique())
    branche_list = I_real.index.get_level_values(i).unique()

    K = pd.DataFrame(index=I_real.index, columns=["Xt"], dtype=float)
    for b in branche_list:
        if (b, 1992) not in I_real.index:
            continue
        K.loc[(b, 1992), "Xt"] = K0[b]
        for yr in [y for y in year_list if y >= 1993]:
            d   = delta.loc[(b, yr)]
            inv = I_real.loc[(b, yr), "Xt"]
            K.loc[(b, yr), "Xt"] = (1 - d) * K.loc[(b, yr - 1), "Xt"] + inv

    K.index.names = [i, t]
    K_prev = K.groupby(i)["Xt"].shift(1)

    # tau_MD
    M_D_tot = v["M_D_loebende"].groupby([i, t]).sum()
    M_D_tot.index.names = [i, t]
    tau_MD = v["afgift"]["afgift"] / M_D_tot["INDHOLD"]
    tau_MD = tau_MD.loc[tau_MD.index.get_level_values(t) != 1966]
    tau_MD.index.names = [i, t]

    # P_MxM og P_KLxKL
    P_MxM_F   = (1 + v["tau_MF"]["tau"]) * v["P_F"]["Pt"] * v["M_F"]["Xt"]
    P_MxM_D   = (1 + tau_MD)             * v["P_D"]["Pt"] * v["M_D"]["Xt"]
    P_MxM_tot = (P_MxM_D + P_MxM_F).groupby([i, t]).sum()
    P_MxM_tot.index.names = [i, t]
    P_KLxKL   = v["w"]["Pt"] * v["L"]["Xt"] + P_K * K_prev
    v["P_MxM_tot"] = P_MxM_tot.to_frame(name="INDHOLD")

    J_prev = v["J"].groupby(i)["Xt"].shift(1)
    P_J    = kp69["P_J"]
    PxJ    = P_J * J_prev.loc[LANDBRUG]

    P_O = _beregn_P_O(v, P_KLxKL, PxJ, LANDBRUG)

    # tau_Y og markup
    subs_adj = _beregn_subsidier_adj(v, LANDBRUG)
    tau_Y  = subs_adj / (v["Y_lob"]["INDHOLD"] - subs_adj)
    markup = v["P"]["Pt"] / ((1 + tau_Y) * P_O) - 1

    # mu og theta
    mu_theta = _beregn_mu_theta(
        Y=v["Y"], P_O=P_O,
        Mtot=v["Mtot"], P_Mtot=v["P_Mtot"],
        KL=v["KL"],     P_KL=v["P_KL"],
        JKL=v["JKL"],   P_JKL=v["P_JKL"],
        M=v["M"],       P_M=v["P_M"],
        M_D=v["M_D"],   P_D=v["P_D"], tau_MD=tau_MD,
        M_F=v["M_F"],   P_F=v["P_F"], tau_MF=v["tau_MF"],
        K_prev=K_prev,  P_K=P_K,
        L=v["L"], w=v["w"],
        J_prev=J_prev, P_J=P_J,
        EY=EY, EMtot=EMtot, EM=EM, EKL=EKL, EJKL=EJKL,
        landbrug_kode=LANDBRUG,
    )

    return {
        "K": K, "I_real": I_real, "delta": delta, "P_K": P_K, "P_I": P_I,
        "P_J": P_J, "tau_MD": tau_MD,
        "P_O": P_O, "tau_Y": tau_Y, "markup": markup,
        "K_prev": K_prev, "J_prev": J_prev,
        "P_MxM_tot": P_MxM_tot, "P_KLxKL": P_KLxKL,
        **mu_theta,
    }


# ===========================================================================
# Mappings: lange DST-navne → korte branche-koder
# ===========================================================================

# 69-niveau: pipeline bruger "010000"-labels (6-cifret) og aggregerede
# fødevare-navne fordi vi altid henter på 117-niveau og aggregerer ned.
MAPPING_69 = {
    "010000 Landbrug og gartneri-(Tilgang)":            "01000",
    "010000 Landbrug og gartneri- (Anvendelse)":        "01000",
    "010000 Landbrug og gartneri":                      "01000",
    "Føde-, drikke- og tobaksvareindustri-(Tilgang)":   "10120",
    "Føde-, drikke- og tobaksvareindustri- (Anvendelse)": "10120",
    "Føde-, drikke- og tobaksvareindustri":             "10120",
    # NABK69 bruger stadig 5-cifret kode
    "01000 Landbrug og gartneri":                       "01000",
    "10120 Føde-, drikke- og tobaksvareindustri":       "10120",
    "REST_TILGANG Øvrige brancher":                     "REST",
    "REST_ANVENDELSE Øvrige brancher":                  "REST",
}

MAPPING_117 = {
    "010000 Landbrug og gartneri-(Tilgang)":            "010000",
    "010000 Landbrug og gartneri- (Anvendelse)":        "010000",
    "010000 Landbrug og gartneri":                      "010000",
    "100010 Slagterier-(Tilgang)":                      "100010",
    "100010 Slagterier- (Anvendelse)":                  "100010",
    "100010 Slagterier":                                "100010",
    "100030 Mejerier-(Tilgang)":                        "100030",
    "100030 Mejerier- (Anvendelse)":                    "100030",
    "100030 Mejerier":                                  "100030",
    "100040x100050 Anden fødevareindustri (100040, 100050)-(Tilgang)":    "100040x100050",
    "100040x100050 Anden fødevareindustri (100040, 100050)- (Anvendelse)": "100040x100050",
    "100040x100050 Anden fødevareindustri (100040, 100050)":              "100040x100050",
    "REST_TILGANG Øvrige brancher":                     "REST",
    "REST_ANVENDELSE Øvrige brancher":                  "REST",
}

BRANCHER_69  = ["01000", "10120", "REST"]
BRANCHER_117 = ["010000", "100010", "100030", "100040x100050", "REST"]

# NABK69 BEHOLD-labels
BEHOLD_NETTO = "AN.11 Faste aktiver, nettobeholdning ultimo året"
BEHOLD_INV   = "P.51g Faste bruttoinvesteringer"
PRIS_2020    = "2020-priser, kædede værdier"


# ===========================================================================
# Interne hjælpere til byg_variable
# ===========================================================================

def _m(df, cols, mapping):
    """Anvender mapping på én eller flere kolonner i en kopi af df."""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].replace(mapping)
    return d


def _piv(df, idx, val, rename=None):
    """Pivot + fillna(0) + index rename."""
    p = df.pivot_table(index=idx, values=val).fillna(0)
    if rename:
        p.rename(columns=rename, inplace=True)
    p.index.names = idx
    return p


def _byg_makro(P_template, makrodata, brancher, start=1992, slut=2022):
    """
    Udtrækker rTaxDep, rTaxDep_rest, rBonds og tCorp fra et dreamtools
    GDX-objekt og returnerer dem i det format kalibrering-funktionerne
    forventer.
    """
    qK = makrodata.qK[:, :, :]
    qK = qK[
        (qK.index.get_level_values("t") >= start) &
        (qK.index.get_level_values("t") <= slut) &
        (qK.index.get_level_values("i_").isin(["iB", "iM"]))
    ].rename_axis(index={"i_": "k"})

    rAfskr = makrodata.rAfskr[:, :, :]
    rAfskr = rAfskr[
        (rAfskr.index.get_level_values("t") >= start) &
        (rAfskr.index.get_level_values("t") <= slut)
    ]

    qKxrAfskr = qK * rAfskr
    mask_rest  = ~qK.index.get_level_values("s_").isin(["lan", "fre"])

    rTaxDep      = qKxrAfskr.groupby(["s_", "t"]).sum() / qK.groupby(["s_", "t"]).sum()
    rTaxDep_rest = (qKxrAfskr[mask_rest].groupby("t").sum() /
                    qK[mask_rest].groupby("t").sum())

    rBonds_ra = makrodata.rRente["RealKred", :]
    rBonds_ra = rBonds_ra[
        (rBonds_ra.index.get_level_values("t") >= start) &
        (rBonds_ra.index.get_level_values("t") <= slut)
    ]
    tCorp_ra = makrodata.tSelskab[:]
    tCorp_ra = tCorp_ra[
        (tCorp_ra.index.get_level_values("t") >= start) &
        (tCorp_ra.index.get_level_values("t") <= slut)
    ]

    rBonds = P_template.copy()
    rBonds["Pt"] = rBonds.index.get_level_values(t).map(rBonds_ra)

    tCorp = P_template.copy()
    tCorp["Pt"] = tCorp.index.get_level_values(t).map(tCorp_ra)

    return rTaxDep, rTaxDep_rest, rBonds, tCorp


def _byg_jord_multindeks(jordareal, landbrug_kode, andre_brancher):
    """Bygger J som MultiIndex (ANVENDELSE, TID) med 0 for alle ikke-landbrug."""
    J_temp  = jordareal.pivot_table(index=[t], values="INDHOLD").fillna(0).squeeze()
    J_index = pd.MultiIndex.from_product([[landbrug_kode], J_temp.index], names=[i, t])
    J_serie = pd.Series(J_temp.values, index=J_index, name="INDHOLD")
    for b in andre_brancher:
        idx = pd.MultiIndex.from_product([[b], J_temp.index], names=[i, t])
        J_serie = pd.concat([J_serie, pd.Series(0, index=idx, name="INDHOLD")])
    return J_serie.to_frame("Xt").sort_index()


def _byg_jkl_multindeks(agg_df, jkl_kol, landbrug_kode, andre_brancher):
    """Bygger JKL som MultiIndex (ANVENDELSE, TID) med 0 for ikke-landbrug."""
    JKL_temp  = agg_df[agg_df[i] == landbrug_kode].pivot_table(index=[t], values=jkl_kol).fillna(0).squeeze()
    JKL_index = pd.MultiIndex.from_product([[landbrug_kode], JKL_temp.index], names=[i, t])
    JKL_serie = pd.Series(JKL_temp.values, index=JKL_index, name="Xt")
    for b in andre_brancher:
        idx = pd.MultiIndex.from_product([[b], JKL_temp.index], names=[i, t])
        JKL_serie = pd.concat([JKL_serie, pd.Series(0, index=idx, name="Xt")])
    return JKL_serie.to_frame("Xt").sort_index()


def _byg_R_geld(df_realkredit, brancher_liste):
    """Broadcaster rentetidsserie til (ANVENDELSE, TID) MultiIndex."""
    R_t   = df_realkredit.pivot_table(index=[t], values="INDHOLD").fillna(0)
    R_idx = pd.MultiIndex.from_product([brancher_liste, R_t.index], names=[i, t])
    return pd.DataFrame(
        R_t.loc[R_idx.get_level_values(t)].values,
        index=R_idx, columns=["Rt"],
    )


# ===========================================================================
# byg_variable_69 og byg_variable_117: erstatter data_formater-filerne
# ===========================================================================

def byg_variable_69(data, results, agg_results, df_realkredit, makrodata):
    """
    Erstatter data_formater69.py.
    Bygger variabel-dict til kalibrering_69() direkte fra pipeline-DataFrames.

    Parametre
    ----------
    data         : dict fra import_data.load_all()
    results      : dict fra omregning.beregn_alle()
    agg_results  : dict fra aggregater.beregn_alle_aggregater()
    df_realkredit: DataFrame med realkreditrente (TID, INDHOLD)
    makrodata    : dreamtools GDX-objekt (konjunktur-fil)

    Returnerer
    ----------
    dict med nøgler svarende til variablerne i data_formater69.py
    """
    mp  = MAPPING_69
    br  = BRANCHER_69
    LAN = "01000"
    AND = ["10120", "REST"]

    # --- Mængdeindeks (Xt) fra IO-tabel ---
    mx = _m(results["mængdeindeks"], ["TILGANG2", "ANVENDELSE"], mp)
    mx_br = mx[mx[i].isin(br)]

    M_F = _piv(mx_br[mx_br["TILGANG1"] == "Import eksklusiv told"], [i, j, t], "Xt")
    M_D = _piv(mx_br[mx_br["TILGANG1"] == "Dansk produktion"],       [i, j, t], "Xt")
    M_T = _piv(mx_br[mx_br["TILGANG1"] == "Told"],                   [i, j, t], "Xt")

    mx_y   = mx[mx["TILGANG1"] == "Dansk produktion"]
    Y      = _piv(mx_y[mx_y[i] == "Anvendelse, i alt-(Anvendelse)"], [j, t], "Xt")
    Y.index.names = [i, t]
    G      = _piv(mx_y[mx_y[i] == "Offentligt forbrug, i alt-(Anvendelse)"], [j, t], "Xt")
    G.index.names = [i, t]
    G      = G.reindex(Y.index).fillna(0)
    C      = _piv(mx_y[mx_y[i] == "Husholdningernes forbrugsudgifter + NPISH (Anvendelse)"], [j, t], "Xt")
    C.index.names = [i, t]
    X      = _piv(mx_y[mx_y[i] == "Eksport - (Anvendelse)"], [j, t], "Xt")
    X.index.names = [i, t]

    # --- Løbende priser fra IO-tabel ---
    io_lob = _m(
        data["io"][data["io"]["PRISENHED"] == "Løbende priser"],
        ["TILGANG2", "ANVENDELSE"], mp
    )
    Y_lob = _piv(
        io_lob[(io_lob["TILGANG1"] == "Dansk produktion") & (io_lob[i] == "Anvendelse, i alt-(Anvendelse)")],
        [j, t], "INDHOLD"
    )
    Y_lob.index.names = [i, t]
    M_D_loebende = _piv(
        io_lob[(io_lob["TILGANG1"] == "Dansk produktion") & io_lob[i].isin(br)],
        [i, j, t], "INDHOLD"
    )

    # --- Prisindeks (Pt) fra IO-tabel ---
    px = _m(results["prisindeks"], ["TILGANG2", "ANVENDELSE"], mp)
    px_br = px[px[i].isin(br)]

    P     = _piv(px[(px["TILGANG1"] == "Dansk produktion") & (px[i] == "Anvendelse, i alt-(Anvendelse)")], [j, t], "Pt")
    P.index.names = [i, t]
    P_F   = _piv(px_br[px_br["TILGANG1"] == "Import eksklusiv told"], [i, j, t], "Pt")
    P_D   = _piv(px_br[px_br["TILGANG1"] == "Dansk produktion"],       [i, j, t], "Pt")
    P_T   = _piv(px_br[px_br["TILGANG1"] == "Told"],                   [i, j, t], "Pt")

    # --- Timer og timeløn ---
    tm   = _m(results["timer"],    [i], mp)
    tml  = _m(results["timer_lon"], [i], mp)
    tlon = _m(results["timeløn"],  [i], mp)

    L     = _piv(tm,  [i, t], "TIMER", {"TIMER": "Xt"}) / 1000
    L_lon = _piv(tml, [i, t], "TIMER", {"TIMER": "Xt"}) / 1000
    w     = _piv(tlon, [i, t], "TIMELOEN_KR", {"TIMELOEN_KR": "Pt"})

    # --- Kapital: K (2020-priser), I (mængde), I_lob (løbende), P_I ---
    kap_raw = _m(data["kapital"], ["BRANCHE"], mp)
    kap_raw = kap_raw.rename(columns={"BRANCHE": i})

    K = _piv(
        kap_raw[(kap_raw["BEHOLD"] == BEHOLD_NETTO) & (kap_raw["PRISENHED"] == PRIS_2020)],
        [i, t], "INDHOLD", {"INDHOLD": "Xt"}
    )
    I_lob = _piv(
        kap_raw[(kap_raw["BEHOLD"] == BEHOLD_INV) & (kap_raw["PRISENHED"] == "Løbende priser")],
        [i, t], "INDHOLD", {"INDHOLD": "Xt"}
    )

    kap_mx = _m(results["mængdeindeks_kap"], ["BRANCHE"], mp).rename(columns={"BRANCHE": i})
    I      = _piv(kap_mx[kap_mx["BEHOLD"] == BEHOLD_INV], [i, t], "Xt")

    kap_px = _m(results["prisindeks_kap"], ["BRANCHE"], mp).rename(columns={"BRANCHE": i})
    P_I    = _piv(kap_px[kap_px["BEHOLD"] == BEHOLD_INV], [i, t], "Pt")

    # --- Afgifter og toldsatser ---
    tau_MF = _m(results["toldsats"],    [i, j], mp)
    tau_MF = _piv(tau_MF[tau_MF[i].isin(br)], [i, j, t], "tau")

    afgift = _m(results["afgift"], [i], mp)
    afgift = _piv(afgift[afgift[i].isin(br)], [i, t], "afgift")

    # --- Input: subsidier ---
    inp    = _m(data["input"], [i], mp)
    sub    = inp[(inp["TILGANG1"] == "Andre produktionsskatter, netto") & (inp["PRISENHED"] == "Løbende priser")]
    subsidier = _piv(sub, [i, t], "INDHOLD")

    # --- Aggregater ---
    agg69 = agg_results["aggregater_69"].reset_index()
    agg69 = _m(agg69, [i], mp)
    var69 = agg_results["varegrupper_69"].reset_index()
    var69 = _m(var69, [i, j], mp)

    KL     = _piv(agg69[agg69[i].isin(br)], [i, t], "KL",    {"KL":    "Xt"})
    Mtot   = _piv(agg69[agg69[i].isin(br)], [i, t], "M_tot", {"M_tot": "Xt"})
    MDtot  = _piv(agg69[agg69[i].isin(br)], [i, t], "MD_tot",{"MD_tot":"Xt"})
    MFtot  = _piv(agg69[agg69[i].isin(br)], [i, t], "MF_tot",{"MF_tot":"Xt"})
    M      = _piv(var69[var69[i].isin(br)],  [i, j, t], "M", {"M": "Xt"})
    P_KL   = _piv(agg69[agg69[i].isin(br)], [i, t], "P_KL",  {"P_KL":  "Pt"})
    P_Mtot = _piv(agg69[agg69[i].isin(br)], [i, t], "P_Mtot",{"P_Mtot":"Pt"})
    P_MDtot= _piv(agg69[agg69[i].isin(br)], [i, t], "P_MDtot")
    P_MFtot= _piv(agg69[agg69[i].isin(br)], [i, t], "P_MFtot")
    P_M    = _piv(var69[var69[i].isin(br)],  [i, j, t], "P_M")
    P_JKL  = _piv(agg69[agg69[i].isin(br)], [i, t], "P_JKL", {"P_JKL": "Pt"})

    JKL    = _byg_jkl_multindeks(agg69, "JKL", LAN, AND)
    J      = _byg_jord_multindeks(data["jordareal"], LAN, AND)

    # --- Øvrige tidsserier ---
    P_Jord = data["jordpris"].pivot_table(index=[t], values="INDHOLD").fillna(0).rename(columns={"INDHOLD": "Pt"})
    P_Jord.index.names = [t]

    hektarstotte = data["tilskud"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    hektarstotte.index.names = [t]

    grundskyld = data["grundskyld"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    grundskyld.index.names = [t]

    R_geld = _byg_R_geld(df_realkredit, br)

    # --- Makrodata (GDX) ---
    rTaxDep, rTaxDep_rest, rBonds, tCorp = _byg_makro(P, makrodata, br)

    return dict(
        M_D=M_D, M_F=M_F, M_T=M_T,
        Y=Y, Y_lob=Y_lob, G=G, C=C, X=X,
        L=L, L_lon=L_lon,
        K=K, I=I, I_lob=I_lob,
        M_D_loebende=M_D_loebende,
        KL=KL, M=M, Mtot=Mtot, MDtot=MDtot, MFtot=MFtot, JKL=JKL, J=J,
        subsidier=subsidier, hektarstotte=hektarstotte, grundskyld=grundskyld,
        P=P, w=w, P_I=P_I, P_F=P_F, P_D=P_D, P_T=P_T,
        tau_MF=tau_MF, afgift=afgift,
        P_KL=P_KL, P_M=P_M, P_Mtot=P_Mtot, P_Jord=P_Jord,
        P_JKL=P_JKL, P_MDtot=P_MDtot, P_MFtot=P_MFtot,
        R_geld=R_geld, rBonds=rBonds, tCorp=tCorp,
        rTaxDep=rTaxDep, rTaxDep_rest=rTaxDep_rest,
    )


def byg_variable_117(data, results, agg_results, opsplit_results,
                     df_realkredit, makrodata):
    """
    Erstatter data_formater.py (117-niveau).
    Bygger variabel-dict til kalibrering_117() direkte fra pipeline-DataFrames.

    Parametre
    ----------
    data            : dict fra import_data.load_all()
    results         : dict fra omregning.beregn_alle()
    agg_results     : dict fra aggregater.beregn_alle_aggregater()
    opsplit_results : dict fra opsplit_kapital.beregn_kapital_117()
                      (nøgler: gamma, kapital_117, prisindeks_kap_117,
                               mængdeindeks_kap_117)
    df_realkredit   : DataFrame med realkreditrente (TID, INDHOLD)
    makrodata       : dreamtools GDX-objekt

    Returnerer
    ----------
    dict med nøgler svarende til variablerne i data_formater.py
    """
    mp  = MAPPING_117
    br  = BRANCHER_117
    LAN = "010000"
    AND = ["100010", "100030", "100040x100050", "REST"]

    # --- Mængdeindeks (Xt) ---
    mx = _m(results["mængdeindeks_117"], ["TILGANG2", "ANVENDELSE"], mp)
    mx_br = mx[mx[i].isin(br)]

    M_F = _piv(mx_br[mx_br["TILGANG1"] == "Import eksklusiv told"], [i, j, t], "Xt")
    M_D = _piv(mx_br[mx_br["TILGANG1"] == "Dansk produktion"],       [i, j, t], "Xt")
    M_T = _piv(mx_br[mx_br["TILGANG1"] == "Told"],                   [i, j, t], "Xt")

    mx_y = mx[mx["TILGANG1"] == "Dansk produktion"]
    Y    = _piv(mx_y[mx_y[i] == "Anvendelse, i alt-(Anvendelse)"], [j, t], "Xt")
    Y.index.names = [i, t]
    G    = _piv(mx_y[mx_y[i] == "Offentligt forbrug, i alt-(Anvendelse)"], [j, t], "Xt")
    G.index.names = [i, t]
    G    = G.reindex(Y.index).fillna(0)
    C    = _piv(mx_y[mx_y[i] == "Husholdningernes forbrugsudgifter + NPISH (Anvendelse)"], [j, t], "Xt")
    C.index.names = [i, t]
    X    = _piv(mx_y[mx_y[i] == "Eksport - (Anvendelse)"], [j, t], "Xt")
    X.index.names = [i, t]

    # --- Løbende priser ---
    io_lob = _m(
        data["io_117"][data["io_117"]["PRISENHED"] == "Løbende priser"],
        ["TILGANG2", "ANVENDELSE"], mp
    )
    Y_lob = _piv(
        io_lob[(io_lob["TILGANG1"] == "Dansk produktion") & (io_lob[i] == "Anvendelse, i alt-(Anvendelse)")],
        [j, t], "INDHOLD"
    )
    Y_lob.index.names = [i, t]
    M_D_loebende = _piv(
        io_lob[(io_lob["TILGANG1"] == "Dansk produktion") & io_lob[i].isin(br)],
        [i, j, t], "INDHOLD"
    )

    # --- Prisindeks (Pt) ---
    px    = _m(results["prisindeks_117"], ["TILGANG2", "ANVENDELSE"], mp)
    px_br = px[px[i].isin(br)]

    P   = _piv(px[(px["TILGANG1"] == "Dansk produktion") & (px[i] == "Anvendelse, i alt-(Anvendelse)")], [j, t], "Pt")
    P.index.names = [i, t]
    P_F = _piv(px_br[px_br["TILGANG1"] == "Import eksklusiv told"], [i, j, t], "Pt")
    P_D = _piv(px_br[px_br["TILGANG1"] == "Dansk produktion"],       [i, j, t], "Pt")
    P_T = _piv(px_br[px_br["TILGANG1"] == "Told"],                   [i, j, t], "Pt")

    # --- Timer og timeløn ---
    tm   = _m(results["timer_117"],    [i], mp)
    tml  = _m(results["timer_lon_117"] if "timer_lon_117" in results else results["timer_117"], [i], mp)
    tlon = _m(results["timeløn_117"],  [i], mp)

    L     = _piv(tm,  [i, t], "TIMER", {"TIMER": "Xt"}) / 1000
    L_lon = _piv(tml, [i, t], "TIMER", {"TIMER": "Xt"}) / 1000
    w     = _piv(tlon, [i, t], "TIMELOEN_KR", {"TIMELOEN_KR": "Pt"})

    # --- Kapital (117-niveau, fra opsplit) ---
    kap117 = _m(opsplit_results["kapital_117"], ["BRANCHE"], mp).rename(columns={"BRANCHE": i})

    # K bruges IKKE direkte i 117-calib (genopbygges fra I og delta)
    # men gemmes for fuldstændighedens skyld
    K = _piv(
        kap117[(kap117["BEHOLD"] == BEHOLD_NETTO) & (kap117["PRISENHED"] == PRIS_2020)],
        [i, t], "INDHOLD", {"INDHOLD": "Xt"}
    )
    I_lob = _piv(
        kap117[(kap117["BEHOLD"] == BEHOLD_INV) & (kap117["PRISENHED"] == "Løbende priser")],
        [i, t], "INDHOLD", {"INDHOLD": "Xt"}
    )

    kap_mx = _m(opsplit_results["mængdeindeks_kap_117"], ["BRANCHE"], mp).rename(columns={"BRANCHE": i})
    I      = _piv(kap_mx[kap_mx["BEHOLD"] == BEHOLD_INV], [i, t], "Xt")

    kap_px = _m(opsplit_results["prisindeks_kap_117"], ["BRANCHE"], mp).rename(columns={"BRANCHE": i})
    P_I    = _piv(kap_px[kap_px["BEHOLD"] == BEHOLD_INV], [i, t], "Pt")

    # --- Afgifter og toldsatser ---
    tau_MF = _m(results["toldsats_117"], [i, j], mp)
    tau_MF = _piv(tau_MF[tau_MF[i].isin(br)], [i, j, t], "tau")

    afgift = _m(results["afgift_117"], [i], mp)
    afgift = _piv(afgift[afgift[i].isin(br)], [i, t], "afgift")

    # --- Input: subsidier ---
    inp   = _m(data["input_117"], [i], mp)
    sub   = inp[(inp["TILGANG1"] == "Andre produktionsskatter, netto") & (inp["PRISENHED"] == "Løbende priser")]
    subsidier = _piv(sub, [i, t], "INDHOLD")

    # --- Aggregater ---
    agg117 = agg_results["aggregater_117"].reset_index()
    agg117 = _m(agg117, [i], mp)
    var117 = agg_results["varegrupper_117"].reset_index()
    var117 = _m(var117, [i, j], mp)

    KL     = _piv(agg117[agg117[i].isin(br)], [i, t], "KL",    {"KL":    "Xt"})
    Mtot   = _piv(agg117[agg117[i].isin(br)], [i, t], "M_tot", {"M_tot": "Xt"})
    MDtot  = _piv(agg117[agg117[i].isin(br)], [i, t], "MD_tot",{"MD_tot":"Xt"})
    MFtot  = _piv(agg117[agg117[i].isin(br)], [i, t], "MF_tot",{"MF_tot":"Xt"})
    M      = _piv(var117[var117[i].isin(br)],  [i, j, t], "M", {"M": "Xt"})
    P_KL   = _piv(agg117[agg117[i].isin(br)], [i, t], "P_KL",  {"P_KL":  "Pt"})
    P_Mtot = _piv(agg117[agg117[i].isin(br)], [i, t], "P_Mtot",{"P_Mtot":"Pt"})
    P_MDtot= _piv(agg117[agg117[i].isin(br)], [i, t], "P_MDtot")
    P_MFtot= _piv(agg117[agg117[i].isin(br)], [i, t], "P_MFtot")
    P_M    = _piv(var117[var117[i].isin(br)],  [i, j, t], "P_M")
    P_JKL  = _piv(agg117[agg117[i].isin(br)], [i, t], "P_JKL", {"P_JKL": "Pt"})

    JKL    = _byg_jkl_multindeks(agg117, "JKL", LAN, AND)
    J      = _byg_jord_multindeks(data["jordareal"], LAN, AND)

    # --- Øvrige tidsserier ---
    P_Jord = data["jordpris"].pivot_table(index=[t], values="INDHOLD").fillna(0).rename(columns={"INDHOLD": "Pt"})
    P_Jord.index.names = [t]

    hektarstotte = data["tilskud"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    hektarstotte.index.names = [t]

    grundskyld = data["grundskyld"].pivot_table(index=[t], values="INDHOLD").fillna(0)
    grundskyld.index.names = [t]

    R_geld = _byg_R_geld(df_realkredit, br)

    # --- Makrodata (GDX) ---
    rTaxDep, rTaxDep_rest, rBonds, tCorp = _byg_makro(P, makrodata, br)

    # --- Gamma (splitting-vægte, fra opsplit_kapital) ---
    gdf = opsplit_results["gamma"].copy()
    gdf = _m(gdf, [i], mp)
    gdf = gdf[gdf[i].isin(["100010", "100030", "100040x100050"])]
    gamma = gdf.pivot_table(index=[i, t], values="gamma").fillna(0)
    gamma.index.names = [i, t]

    return dict(
        M_D=M_D, M_F=M_F, M_T=M_T,
        Y=Y, Y_lob=Y_lob, G=G, C=C, X=X,
        L=L, L_lon=L_lon,
        K=K, I=I, I_lob=I_lob,
        M_D_loebende=M_D_loebende,
        KL=KL, M=M, Mtot=Mtot, MDtot=MDtot, MFtot=MFtot, JKL=JKL, J=J,
        subsidier=subsidier, hektarstotte=hektarstotte, grundskyld=grundskyld,
        P=P, w=w, P_I=P_I, P_F=P_F, P_D=P_D, P_T=P_T,
        tau_MF=tau_MF, afgift=afgift,
        P_KL=P_KL, P_M=P_M, P_Mtot=P_Mtot, P_Jord=P_Jord,
        P_JKL=P_JKL, P_MDtot=P_MDtot, P_MFtot=P_MFtot,
        gamma=gamma,
        R_geld=R_geld, rBonds=rBonds, tCorp=tCorp,
        rTaxDep=rTaxDep, rTaxDep_rest=rTaxDep_rest,
    )