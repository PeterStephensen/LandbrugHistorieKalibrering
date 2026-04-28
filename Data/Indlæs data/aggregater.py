"""
aggregater.py
=============
Beregner prisindeks og mængdeindeks for aggregerede inputfaktorer:
  - Mjit  : materialer per varegruppe (TILGANG2 x ANVENDELSE)
  - MD    : dansk materiale-aggregat
  - MF    : importeret materiale-aggregat
  - Mtot  : samlet materiale-aggregat (Paasche over varegrupper)
  - KL    : kapital-loen aggregat
  - JKL   : jord-kapital-loen aggregat

Kører pa begge aggregeringsniveauer (69 og 117) ud fra DataFrames
returneret af import_data.load_all() og omregning.beregn_alle().

Eksterne inputs (ikke i pipeline):
  df_tau_MD  : toldsats pa dansk produktion (ANVENDELSE, TID, tau)
  df_jord_pris: jordprisindeks (TID, Pt)

Brug:
    from import_data import load_all
    from omregning import beregn_alle
    from aggregater import beregn_alle_aggregater

    data    = load_all()
    results = beregn_alle(data)

    # Laes eksterne filer
    df_tau_MD   = pd.read_csv("tau_MD.csv")
    df_jord_pris = pd.read_csv("P_J.csv")

    agg = beregn_alle_aggregater(data, results, df_tau_MD, df_jord_pris)

    df_agg_69  = agg["aggregater_69"]   # P_Mtot, M_tot, P_KL, KL, P_JKL, JKL, ...
    df_mjit_69 = agg["varegrupper_69"]  # P_M, M per TILGANG2 x ANVENDELSE
    df_agg_117  = agg["aggregater_117"]
    df_mjit_117 = agg["varegrupper_117"]
"""

import pandas as pd
import numpy as np

# ===========================================================================
# Branch-mappings
# ===========================================================================

MAPPING_69 = {
    "010000 Landbrug og gartneri-(Tilgang)":            "01000",
    "010000 Landbrug og gartneri- (Anvendelse)":        "01000",
    "010000 Landbrug og gartneri":                      "01000",
    "Føde-, drikke- og tobaksvareindustri-(Tilgang)":   "10120",
    "Føde-, drikke- og tobaksvareindustri- (Anvendelse)":"10120",
    "Føde-, drikke- og tobaksvareindustri":             "10120",
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
    "100040x100050 Anden fødevareindustri (100040, 100050)-(Tilgang)":   "100040x100050",
    "100040x100050 Anden fødevareindustri (100040, 100050)- (Anvendelse)":"100040x100050",
    "100040x100050 Anden fødevareindustri (100040, 100050)":             "100040x100050",
    "REST_TILGANG Øvrige brancher":                     "REST",
    "REST_ANVENDELSE Øvrige brancher":                  "REST",
}

BRANCHER_69  = ["01000", "10120", "REST"]
BRANCHER_117 = ["010000", "100010", "100030", "100040x100050", "REST"]

BEHOLD_KAPITAL = "AN.11 Faste aktiver, nettobeholdning ultimo året"


# ===========================================================================
# Privat hjalpefunktion: kædet prisindeks (selvstaendig kopi)
# ===========================================================================

def _kædet_prisindeks(lob: pd.DataFrame, forp: pd.DataFrame,
                       id_cols: list, base_year: int = 2020) -> pd.DataFrame:
    ratio = (lob["INDHOLD"] / forp["INDHOLD"]).unstack("TID")
    Pt = ratio.copy().astype(float)
    Pt.loc[:, :] = np.nan
    Pt[base_year] = 1.0
    years = sorted(ratio.columns)
    for y in years:
        if y > base_year and y - 1 in Pt.columns:
            Pt[y] = Pt[y - 1] * ratio[y]
    for y in reversed(years):
        if y < base_year and y + 1 in Pt.columns:
            Pt[y] = Pt[y + 1] / ratio[y + 1]
    out = Pt.stack(future_stack=True).reset_index()
    out.columns = id_cols + ["TID", "Pt"]
    return out


def _pt_xt(lob_idx, forp_idx, id_cols):
    """Beregn Pt og Xt fra to allerede-indekserede DataFrames."""
    Pt_df = _kædet_prisindeks(lob_idx, forp_idx, id_cols)
    Pt_idx = Pt_df.set_index(id_cols + ["TID"])
    Xt_df = (lob_idx["INDHOLD"] / Pt_idx["Pt"]).to_frame(name="Xt").reset_index()
    return Pt_df, Xt_df


# ===========================================================================
# Hjaelpefunktion: udvid tau_MD til TILGANG2-niveau
# ===========================================================================

def _expand_tau_md(df_tau_MD: pd.DataFrame,
                   df_md_lob: pd.DataFrame) -> pd.DataFrame:
    """
    tau_MD er kun indekseret pa (ANVENDELSE, TID).
    Materialer er indekseret pa (TILGANG2, ANVENDELSE, TID).
    Denne funktion broadcaster tau ud over alle TILGANG2 ved merge.
    """
    tau_anvend = df_tau_MD.index.get_level_values("ANVENDELSE").unique()
    md_reset   = df_md_lob.reset_index()
    filtered   = md_reset[md_reset["ANVENDELSE"].isin(tau_anvend)]
    expanded   = (
        filtered[["TILGANG2", "ANVENDELSE", "TID"]]
        .merge(df_tau_MD.reset_index(), on=["ANVENDELSE", "TID"], how="left")
        .set_index(["TILGANG2", "ANVENDELSE", "TID"])
    )
    return expanded


# ===========================================================================
# Kernefunktioner – kører pa klargjorte, indekserede DataFrames
# ===========================================================================

def _materialer_mjit(md_lob, md_for, mf_lob, mf_for, tau_md, tau_mf):
    """Prisindeks og mængdeindeks for materialer per varegruppe (Mjit)."""
    tau_md_prev = tau_md.groupby(level=["TILGANG2", "ANVENDELSE"])["tau"].shift(1)
    tau_mf_prev = tau_mf.groupby(level=["TILGANG2", "ANVENDELSE"])["tau"].shift(1)

    lob_val  = (1 + tau_md["tau"]) * md_lob["INDHOLD"] + (1 + tau_mf["tau"]) * mf_lob["INDHOLD"]
    for_val  = (1 + tau_md_prev)   * md_for["INDHOLD"] + (1 + tau_mf_prev)   * mf_for["INDHOLD"]

    lob_df = lob_val.reset_index(name="INDHOLD").set_index(["TILGANG2", "ANVENDELSE", "TID"])
    for_df = for_val.reset_index(name="INDHOLD").set_index(["TILGANG2", "ANVENDELSE", "TID"])
    return _pt_xt(lob_df, for_df, ["TILGANG2", "ANVENDELSE"])


def _materialer_enkelt(lob, forp, id_cols=["ANVENDELSE"]):
    """Prisindeks og mængdeindeks for MD eller MF aggregeret over TILGANG2."""
    lob_agg  = lob.groupby(["ANVENDELSE", "TID"])["INDHOLD"].sum().reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    for_agg  = forp.groupby(["ANVENDELSE", "TID"])["INDHOLD"].sum().reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    return _pt_xt(lob_agg, for_agg, ["ANVENDELSE"])


def _materialer_aggregat(Pt_Mjit, Xt_Mjit, lob_mjit):
    """
    Samlet materiale-aggregat (Mtot) via Paasche-vaegt:
      nævner = sum_j(Xt_j * Pt-1_j)
    """
    løbende = lob_mjit.groupby(["ANVENDELSE", "TID"])["INDHOLD"].sum()

    Pt_prev = Pt_Mjit.set_index(["TILGANG2", "ANVENDELSE", "TID"]).groupby(
        ["TILGANG2", "ANVENDELSE"])["Pt"].shift(1)
    for_val = (
        Xt_Mjit.set_index(["TILGANG2", "ANVENDELSE", "TID"])["Xt"] * Pt_prev
    ).groupby(["ANVENDELSE", "TID"]).sum()

    lob_df = løbende.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    for_df = for_val.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    return _pt_xt(lob_df, for_df, ["ANVENDELSE"])


def _kl_aggregat(df_kap_xt, df_kap_pt, df_timer, df_timeløn):
    """
    KL-aggregat:
      Lobende: w_t * L_t/1000 + K_{t-1} * P_K_t
      Foregående: w_{t-1} * L_t/1000 + P_K_{t-1} * K_{t-1}
    """
    k_prev   = df_kap_xt.groupby(level="ANVENDELSE")["Xt"].shift(1)
    lon_prev = df_timeløn.groupby(level="ANVENDELSE")["TIMELOEN_KR"].shift(1)
    pk_prev  = df_kap_pt.groupby(level="ANVENDELSE")["Pt"].shift(1)

    lob_val = df_timeløn["TIMELOEN_KR"] * df_timer["TIMER"] / 1000 + k_prev * df_kap_pt["Pt"]
    for_val = lon_prev * df_timer["TIMER"] / 1000 + pk_prev * k_prev

    lob_df = lob_val.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    for_df = for_val.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    return _pt_xt(lob_df, for_df, ["ANVENDELSE"]), lob_val, for_val


def _jkl_aggregat(kl_lob, kl_for, df_jord, df_jord_pt):
    """
    JKL-aggregat:
      Lobende: KL_lob + J_{t-1} * P_J_t
      Foregående: KL_for + J_{t-1} * P_J_{t-1}
    """
    J_prev      = df_jord["INDHOLD"].shift(1)
    pj_prev     = df_jord_pt["Pt"].shift(1)

    lob_val = kl_lob + J_prev * df_jord_pt["Pt"]
    for_val = kl_for + J_prev * pj_prev

    lob_df = lob_val.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    for_df = for_val.reset_index(name="INDHOLD").set_index(["ANVENDELSE", "TID"])
    return _pt_xt(lob_df, for_df, ["ANVENDELSE"])


# ===========================================================================
# Forberedelse af inputs – normerer branch-navne og sætter index
# ===========================================================================

def _klargjoer_inputs(df_io, df_tau_MD, df_tau_MF,
                      df_kapital_xt, df_kapital_pt,
                      df_input, df_timer, df_timeløn,
                      df_jord, df_jord_pris,
                      mapping, brancher,
                      kapital_har_behold=False):
    """
    Normerer branch-navne, filtrerer og sætter index pa alle inputs.
    Returnerer en dict med klar-til-brug DataFrames.
    """
    def _map(df, cols):
        d = df.copy()
        for c in cols:
            if c in d.columns:
                d[c] = d[c].replace(mapping)
        return d

    # IO-tabel: MD og MF, lobende og foregående
    io = _map(df_io, ["TILGANG2", "ANVENDELSE"])
    io = io[io["ANVENDELSE"].isin(brancher)]

    def _split_io(tilgang1_val):
        d = io[io["TILGANG1"] == tilgang1_val].copy()
        d = d[d["TID"] != 1966]
        d.drop(columns=["TILGANG1"], inplace=True)
        lob = d[d["PRISENHED"] == "Løbende priser"].drop(columns=["PRISENHED"]).set_index(["TILGANG2", "ANVENDELSE", "TID"]).sort_index()
        forp = d[d["PRISENHED"] != "Løbende priser"].drop(columns=["PRISENHED"]).set_index(["TILGANG2", "ANVENDELSE", "TID"]).sort_index()
        return lob, forp

    md_lob, md_for = _split_io("Dansk produktion")
    mf_lob, mf_for = _split_io("Import eksklusiv told")

    # tau_MF
    tau_mf = _map(df_tau_MF, ["TILGANG2", "ANVENDELSE"])
    tau_mf = tau_mf[tau_mf["ANVENDELSE"].isin(brancher)].set_index(["TILGANG2", "ANVENDELSE", "TID"]).sort_index()

    # tau_MD – udvid til TILGANG2-niveau
    tau_md_in = _map(df_tau_MD, ["ANVENDELSE"]).set_index(["ANVENDELSE", "TID"])
    tau_md = _expand_tau_md(tau_md_in, md_lob)

    # Kapital
    kap_xt = df_kapital_xt.copy()
    kap_pt = df_kapital_pt.copy()
    if kapital_har_behold:
        kap_xt = kap_xt[kap_xt["BEHOLD"] == BEHOLD_KAPITAL].drop(columns=["BEHOLD"])
        kap_pt = kap_pt[kap_pt["BEHOLD"] == BEHOLD_KAPITAL].drop(columns=["BEHOLD"])
        kap_xt = kap_xt.rename(columns={"BRANCHE": "ANVENDELSE"})
        kap_pt = kap_pt.rename(columns={"BRANCHE": "ANVENDELSE"})
    kap_xt = _map(kap_xt, ["ANVENDELSE"]).set_index(["ANVENDELSE", "TID"]).sort_index()
    kap_pt = _map(kap_pt, ["ANVENDELSE"]).set_index(["ANVENDELSE", "TID"]).sort_index()

    # Lonsum, timer, timeløn
    inp   = _map(df_input, ["ANVENDELSE"])
    lonsom = inp[inp["TILGANG1"] == "Aflønning af ansatte"].copy().drop(columns=["TILGANG1"])
    lonsom_lob = lonsom[lonsom["PRISENHED"] == "Løbende priser"].drop(columns=["PRISENHED"]).set_index(["ANVENDELSE", "TID"]).sort_index()

    timer    = _map(df_timer,   ["ANVENDELSE"]).set_index(["ANVENDELSE", "TID"]).sort_index()
    timeløn  = _map(df_timeløn, ["ANVENDELSE"]).set_index(["ANVENDELSE", "TID"]).sort_index()

    # Jord og jordpris
    jord     = df_jord.set_index(["TID"]) if "TID" in df_jord.columns else df_jord
    jord_pt  = df_jord_pris.set_index(["TID"]) if "TID" in df_jord_pris.columns else df_jord_pris

    return dict(md_lob=md_lob, md_for=md_for, mf_lob=mf_lob, mf_for=mf_for,
                tau_md=tau_md, tau_mf=tau_mf,
                kap_xt=kap_xt, kap_pt=kap_pt,
                lonsom_lob=lonsom_lob, timer=timer, timeløn=timeløn,
                jord=jord, jord_pt=jord_pt)


# ===========================================================================
# Hoved-aggregatfunktion
# ===========================================================================

def beregn_aggregater(df_io, df_tau_MD, df_tau_MF,
                      df_kapital_xt, df_kapital_pt,
                      df_input, df_timer, df_timeløn,
                      df_jord, df_jord_pris,
                      mapping, brancher,
                      kapital_har_behold=False):
    """
    Beregner alle aggregater for ét niveau (69 eller 117).

    Parametre
    ----------
    df_io           : data["io"] eller data["io_117"]
    df_tau_MD       : ekstern – dansk produktionstoldsats (ANVENDELSE, TID, tau)
    df_tau_MF       : results["toldsats"] eller ["toldsats_117"]
    df_kapital_xt   : results["mængdeindeks_kap"] eller ["mængdeindeks_kap_117"]
    df_kapital_pt   : results["prisindeks_kap"] eller ["prisindeks_kap_117"]
    df_input        : data["input"] eller ["input_117"]
    df_timer        : results["timer"] eller ["timer_117"]
    df_timeløn      : results["timeløn"] eller ["timeløn_117"]
    df_jord         : data["jordareal"]
    df_jord_pris    : ekstern – jordprisindeks (TID, Pt)
    mapping         : MAPPING_69 eller MAPPING_117
    brancher        : BRANCHER_69 eller BRANCHER_117
    kapital_har_behold : True for 69-niveau (BRANCHE+BEHOLD kolonner)

    Returnerer
    ----------
    dict med:
      "aggregater"  – bred DataFrame: P_Mtot, M_tot, P_KL, KL, P_JKL, JKL,
                                      P_MDtot, MD_tot, P_MFtot, MF_tot
      "varegrupper" – DataFrame: P_M, M per TILGANG2 x ANVENDELSE
    """
    inp = _klargjoer_inputs(
        df_io, df_tau_MD, df_tau_MF,
        df_kapital_xt, df_kapital_pt,
        df_input, df_timer, df_timeløn,
        df_jord, df_jord_pris,
        mapping, brancher, kapital_har_behold
    )

    # Materialer per varegruppe
    Pt_Mjit, Xt_Mjit = _materialer_mjit(
        inp["md_lob"], inp["md_for"],
        inp["mf_lob"], inp["mf_for"],
        inp["tau_md"], inp["tau_mf"]
    )

    # MD- og MF-aggregater
    Pt_MD, Xt_MD = _materialer_enkelt(inp["md_lob"], inp["md_for"])
    Pt_MF, Xt_MF = _materialer_enkelt(inp["mf_lob"], inp["mf_for"])

    # Samlet materiale-aggregat
    lob_mjit = (inp["md_lob"] * 0 + (1 + inp["tau_md"]["tau"]) * inp["md_lob"]["INDHOLD"] +
                (1 + inp["tau_mf"]["tau"]) * inp["mf_lob"]["INDHOLD"]).rename("INDHOLD").reset_index()
    lob_mjit.columns = ["TILGANG2", "ANVENDELSE", "TID", "INDHOLD"]
    lob_mjit = lob_mjit.set_index(["TILGANG2", "ANVENDELSE", "TID"])
    Pt_Mit, Xt_Mit = _materialer_aggregat(Pt_Mjit, Xt_Mjit, lob_mjit)

    # KL-aggregat
    (Pt_KL, Xt_KL), kl_lob, kl_for = _kl_aggregat(
        inp["kap_xt"], inp["kap_pt"], inp["timer"], inp["timeløn"]
    )

    # JKL-aggregat
    Pt_JKL, Xt_JKL = _jkl_aggregat(kl_lob, kl_for, inp["jord"], inp["jord_pt"])

    # Saml aggregater i ét bredt DataFrame
    df_agg = (
        Pt_Mit.set_index(["ANVENDELSE", "TID"]).rename(columns={"Pt": "P_Mtot"})
        .join([
            Xt_Mit.set_index(["ANVENDELSE", "TID"]).rename(columns={"Xt": "M_tot"}),
            Pt_KL.set_index(["ANVENDELSE", "TID"]).rename(columns={"Pt": "P_KL"}),
            Xt_KL.set_index(["ANVENDELSE", "TID"]).rename(columns={"Xt": "KL"}),
            Pt_JKL.set_index(["ANVENDELSE", "TID"]).rename(columns={"Pt": "P_JKL"}),
            Xt_JKL.set_index(["ANVENDELSE", "TID"]).rename(columns={"Xt": "JKL"}),
            Pt_MD.set_index(["ANVENDELSE", "TID"]).rename(columns={"Pt": "P_MDtot"}),
            Xt_MD.set_index(["ANVENDELSE", "TID"]).rename(columns={"Xt": "MD_tot"}),
            Pt_MF.set_index(["ANVENDELSE", "TID"]).rename(columns={"Pt": "P_MFtot"}),
            Xt_MF.set_index(["ANVENDELSE", "TID"]).rename(columns={"Xt": "MF_tot"}),
        ])
    )

    # Varegrupper
    df_vare = (
        Pt_Mjit.set_index(["TILGANG2", "ANVENDELSE", "TID"]).rename(columns={"Pt": "P_M"})
        .join(Xt_Mjit.set_index(["TILGANG2", "ANVENDELSE", "TID"]).rename(columns={"Xt": "M"}))
    )

    return {"aggregater": df_agg, "varegrupper": df_vare}


# ===========================================================================
# Alt-i-en wrapper
# ===========================================================================

def beregn_alle_aggregater(data: dict, results: dict,
                           df_tau_MD_69: pd.DataFrame,
                           df_tau_MD_117: pd.DataFrame,
                           df_jord_pris: pd.DataFrame) -> dict:
    """
    Kører beregn_aggregater for begge niveauer og returnerer alt i en dict.

    Parametre
    ----------
    data          : dict fra import_data.load_all()
    results       : dict fra omregning.beregn_alle()
    df_tau_MD_69  : toldsats pa dansk produktion, 69-niveau (ANVENDELSE, TID, tau)
    df_tau_MD_117 : toldsats pa dansk produktion, 117-niveau
    df_jord_pris  : jordprisindeks (TID, Pt)

    Returnerer
    ----------
    dict med nøglerne:
      "aggregater_69", "varegrupper_69",
      "aggregater_117", "varegrupper_117"
    """
    print("Beregner aggregater (69-niveau) ...")
    res_69 = beregn_aggregater(
        df_io           = data["io"],
        df_tau_MD       = df_tau_MD_69,
        df_tau_MF       = results["toldsats"],
        df_kapital_xt   = results["mængdeindeks_kap"],
        df_kapital_pt   = results["prisindeks_kap"],
        df_input        = data["input"],
        df_timer        = results["timer"],
        df_timeløn      = results["timeløn"],
        df_jord         = data["jordareal"],
        df_jord_pris    = df_jord_pris,
        mapping         = MAPPING_69,
        brancher        = BRANCHER_69,
        kapital_har_behold = True,
    )

    print("Beregner aggregater (117-niveau) ...")
    res_117 = beregn_aggregater(
        df_io           = data["io_117"],
        df_tau_MD       = df_tau_MD_117,
        df_tau_MF       = results["toldsats_117"],
        df_kapital_xt   = results["mængdeindeks_kap_117"],
        df_kapital_pt   = results["prisindeks_kap_117"],
        df_input        = data["input_117"],
        df_timer        = results["timer_117"],
        df_timeløn      = results["timeløn_117"],
        df_jord         = data["jordareal"],
        df_jord_pris    = df_jord_pris,
        mapping         = MAPPING_117,
        brancher        = BRANCHER_117,
        kapital_har_behold = False,
    )

    print("Færdig!")
    return {
        "aggregater_69":   res_69["aggregater"],
        "varegrupper_69":  res_69["varegrupper"],
        "aggregater_117":  res_117["aggregater"],
        "varegrupper_117": res_117["varegrupper"],
    }


if __name__ == "__main__":
    from import_data import load_all
    from omregning import beregn_alle
    from opsplit_kapital import beregn_kapital_117

    data    = load_all()
    results = beregn_alle(data)

    # Tilføj 117-niveau kapital fra opsplit_kapital.py
    kap117 = beregn_kapital_117(data)
    results["mængdeindeks_kap_117"] = kap117["mængdeindeks_kap_117"]
    results["prisindeks_kap_117"]    = kap117["prisindeks_kap_117"]


    agg = beregn_alle_aggregater(data, results, df_tau_MD_69, df_tau_MD_117, df_jord_pris)
    for name, df in agg.items():
        print(f"\n{'='*50}\n  {name}: {df.shape}")
        print(df.head(3))