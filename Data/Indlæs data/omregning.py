"""
omregning.py
============
Beregner afledte størrelser fra de rå DST-data indlæst af import_data.py.
Ingen CSV-filer læses eller skrives – alt arbejder på pandas DataFrames.

Brug:
    from import_data import load_all
    from omregning import beregn_alle

    data    = load_all()
    results = beregn_alle(data)

    # 69-niveau
    df_prisindeks       = results["prisindeks"]        # Pt  – IO-tabel
    df_mængdeindeks     = results["mængdeindeks"]      # Xt  – IO-tabel
    df_toldsats         = results["toldsats"]          # tau – toldrate
    df_timer_lon        = results["timer_lon"]         # lønmodtager-timer
    df_timer            = results["timer"]             # alle timer (inkl. selvstændige)
    df_timeløn          = results["timeløn"]           # timeløn (kr.)
    df_afgift           = results["afgift"]            # produktskatter + moms

    # 117-niveau (fødevareundersektorer)
    df_prisindeks_117   = results["prisindeks_117"]    # Pt  – IO-tabel
    df_mængdeindeks_117 = results["mængdeindeks_117"]  # Xt  – IO-tabel
    df_toldsats_117     = results["toldsats_117"]      # tau – toldrate
    df_timer_lon_117    = results["timer_lon_117"]     # lønmodtager-timer
    df_timer_117        = results["timer_117"]         # alle timer
    df_timeløn_117      = results["timeløn_117"]       # timeløn (kr.)
    df_afgift_117       = results["afgift_117"]        # produktskatter + moms

    # Kapital (ét niveau – NABK69 er ikke disaggregeret)
    df_prisindeks_kap   = results["prisindeks_kap"]    # Pt  – kapital
    df_mængdeindeks_kap = results["mængdeindeks_kap"]  # Xt  – kapital
"""

import pandas as pd
import numpy as np

# Kort branche-kode (bruges som fælles nøgle på tværs af datasæt)
# Opdateres her ét sted hvis branch-navne ændrer sig i import_data.py
ANDEN_FODE_NAVN = "100040x100050 Anden fødevareindustri (100040, 100050)"

# Kort branche-kode – bruges som fælles nøgle når NAIO3 (ANVENDELSE) og
# NAIO5 (BRANCHE) skal alignes. Opdateres ét sted hvis import_data.py ændrer navne.
BRANCHE_MAPPING = {
    # ---- 69-niveau ----
    "010000 Landbrug og gartneri":                          "01000",
    "010000 Landbrug og gartneri- (Anvendelse)":            "01000",
    "Føde-, drikke- og tobaksvareindustri":                 "10120",
    "Føde-, drikke- og tobaksvareindustri- (Anvendelse)":   "10120",
    "REST_TILGANG Øvrige brancher":                         "REST",
    "REST_ANVENDELSE Øvrige brancher":                      "REST",
    # ---- 117-niveau ----
    # Landbrug (samme navn, men eksplicit her for klarhedens skyld)
    "100010 Slagterier":                                    "100010",
    "100010 Slagterier- (Anvendelse)":                      "100010",
    "100030 Mejerier":                                      "100030",
    "100030 Mejerier- (Anvendelse)":                        "100030",
    ANDEN_FODE_NAVN:                                        "anden_fode",
}


# ===========================================================================
# Fælles hjælpefunktion: kædet prisindeks
# ===========================================================================

def _beregn_prisindeks(
    df_loebende: pd.DataFrame,
    df_for: pd.DataFrame,
    id_cols: list[str],
    base_year: int = 2020,
) -> pd.DataFrame:
    """
    Beregner kædet prisindeks (Pt) forankret i `base_year` = 1.

    Metode:
      - Pt/Pt-1  =  løbende priser(t) / foregående års priser(t)
      - Fremad fra base_year: Pt = Pt-1 * (Pt/Pt-1)
      - Bagud fra base_year: Pt = Pt+1 / (Pt/Pt-1 i næste år)

    Parametre
    ----------
    df_loebende : DataFrame med løbende priser (index = id_cols + TID)
    df_for      : DataFrame med foregående års priser (samme index)
    id_cols     : kolonner der identificerer en tidsserie (alt undtagen TID)
    base_year   : ankerpunkt, sættes til 1.0

    Returnerer
    ----------
    DataFrame med kolonner = id_cols + ['TID', 'Pt']
    """
    ratio = (df_loebende["INDHOLD"] / df_for["INDHOLD"]).unstack("TID")

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

    df_out = (
        Pt.stack(future_stack=True)
        .reset_index()
    )
    df_out.columns = id_cols + ["TID", "Pt"]
    return df_out


def _beregn_mængdeindeks(
    df_loebende: pd.DataFrame,
    df_Pt: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """
    Xt = løbende priser / Pt.

    Parametre
    ----------
    df_loebende : DataFrame med løbende priser (index = id_cols + TID)
    df_Pt       : output fra _beregn_prisindeks (sættes som index her)
    id_cols     : kolonner der identificerer en tidsserie

    Returnerer
    ----------
    DataFrame med kolonner = id_cols + ['TID', 'Xt']
    """
    df_Pt_idx = df_Pt.set_index(id_cols + ["TID"])
    Xt = (df_loebende["INDHOLD"] / df_Pt_idx["Pt"]).to_frame(name="Xt")
    return Xt.reset_index()


# ===========================================================================
# IO-tabel (69-niveau): prisindeks, mængdeindeks og toldsats
# ===========================================================================

def beregn_prisindeks_io(df_io: pd.DataFrame) -> pd.DataFrame:
    """
    Kædet prisindeks (Pt) for input-output tabellen, forankret i 2020 = 1.
    Input: data["io"] fra load_all().
    """
    id_cols = ["TILGANG1", "TILGANG2", "ANVENDELSE"]
    idx     = id_cols + ["TID"]

    lob = (
        df_io.loc[(df_io["PRISENHED"] == "Løbende priser") & (df_io["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    forp = (
        df_io.loc[df_io["PRISENHED"] != "Løbende priser"]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    return _beregn_prisindeks(lob, forp, id_cols)


def beregn_mængdeindeks_io(df_io: pd.DataFrame) -> pd.DataFrame:
    """
    Mængdeindeks (Xt) for input-output tabellen.
    Input: data["io"] fra load_all().
    """
    id_cols = ["TILGANG1", "TILGANG2", "ANVENDELSE"]
    idx     = id_cols + ["TID"]

    lob = (
        df_io.loc[(df_io["PRISENHED"] == "Løbende priser") & (df_io["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    df_Pt = beregn_prisindeks_io(df_io)
    return _beregn_mængdeindeks(lob, df_Pt, id_cols)


def beregn_toldsats(df_io: pd.DataFrame) -> pd.DataFrame:
    """
    Toldsats: tau = Told / Import eksklusiv told (løbende priser).
    Input: data["io"] fra load_all().
    """
    id_cols = ["TILGANG2", "ANVENDELSE"]
    idx     = id_cols + ["TID"]

    lob = (
        df_io.loc[(df_io["PRISENHED"] == "Løbende priser") & (df_io["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(["TILGANG1"] + idx)
    )

    told   = lob.loc[lob.index.get_level_values("TILGANG1") == "Told",           "INDHOLD"].droplevel("TILGANG1")
    xtold  = lob.loc[lob.index.get_level_values("TILGANG1") == "Import eksklusiv told", "INDHOLD"].droplevel("TILGANG1")

    tau = (told / xtold).fillna(0)
    return tau.reset_index(name="tau")


# ===========================================================================
# Kapital: prisindeks og mængdeindeks
# ===========================================================================

def beregn_prisindeks_kapital(df_kapital: pd.DataFrame) -> pd.DataFrame:
    """
    Kædet prisindeks (Pt) for kapitalapparatet, forankret i 2020 = 1.
    Input: data["kapital"] fra load_all().
    """
    id_cols = ["BEHOLD", "BRANCHE"]
    idx     = id_cols + ["TID"]

    lob = (
        df_kapital.loc[(df_kapital["PRISENHED"] == "Løbende priser") & (df_kapital["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    forp = (
        df_kapital.loc[df_kapital["PRISENHED"] == "Forrige års priser"]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    return _beregn_prisindeks(lob, forp, id_cols)


def beregn_mængdeindeks_kapital(df_kapital: pd.DataFrame) -> pd.DataFrame:
    """
    Mængdeindeks (Xt) for kapitalapparatet.
    Input: data["kapital"] fra load_all().
    """
    id_cols = ["BEHOLD", "BRANCHE"]
    idx     = id_cols + ["TID"]

    lob = (
        df_kapital.loc[(df_kapital["PRISENHED"] == "Løbende priser") & (df_kapital["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    df_Pt = beregn_prisindeks_kapital(df_kapital)
    return _beregn_mængdeindeks(lob, df_Pt, id_cols)


# ===========================================================================
# Løntimer og timeløn
# ===========================================================================

def beregn_timer_og_timeløn(
    df_input: pd.DataFrame,
    df_lontimer: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Beregner:
      - timer_lon : lønmodtager-timer pr. branche/år
      - timer     : alle timer pr. branche/år (inkl. selvstændige)
      - timeløn   : timeløn i kr. (= aflønning af ansatte / lønmodtager-timer)

    Parametre
    ----------
    df_input    : data["input"] fra load_all()  (NAIO3, 69-niveau)
    df_lontimer : data["lontimer"] fra load_all() (NAIO5, 69-niveau)

    Returnerer
    ----------
    dict med nøgler: "timer_lon", "timer", "timeløn"
    """
    # Lonsum: aflønning af ansatte, løbende priser
    lonsum = (
        df_input
        .loc[(df_input["TILGANG1"] == "Aflønning af ansatte") &
             (df_input["PRISENHED"] == "Løbende priser")]
        .drop(columns=["PRISENHED", "TILGANG1"])
        .copy()
    )

    # Timer for lønmodtagere
    timer_lon = (
        df_lontimer
        .loc[df_lontimer["SOCIO"] == "Præsterede timer for lønmodtagere (1000 timer)"]
        .drop(columns=["SOCIO"])
        .copy()
    )

    # Alle timer (inkl. selvstændige)
    timer_ialt = (
        df_lontimer
        .loc[df_lontimer["SOCIO"] != "Præsterede timer for lønmodtagere (1000 timer)"]
        .drop(columns=["SOCIO"])
        .copy()
    )

    # Normér branche-navne til korte koder
    lonsum["ANVENDELSE"]    = lonsum["ANVENDELSE"].replace(BRANCHE_MAPPING)
    timer_lon["BRANCHE"]    = timer_lon["BRANCHE"].replace(BRANCHE_MAPPING)
    timer_ialt["BRANCHE"]   = timer_ialt["BRANCHE"].replace(BRANCHE_MAPPING)

    # Sæt index
    lonsum.set_index(["ANVENDELSE", "TID"], inplace=True)
    timer_lon.set_index(["BRANCHE", "TID"], inplace=True)
    timer_ialt.set_index(["BRANCHE", "TID"], inplace=True)

    # Align index-navne til fælles akse
    timer_lon.index.names  = ["ANVENDELSE", "TID"]
    timer_ialt.index.names = ["ANVENDELSE", "TID"]

    # Timeløn = aflønning (mio. kr.) / timer (1000 t) = kr./time * 1000
    timeløn = (lonsum["INDHOLD"] / (timer_lon["INDHOLD"] / 1000)).to_frame(name="TIMELOEN_KR")

    return {
        "timer_lon": timer_lon.reset_index().rename(columns={"INDHOLD": "TIMER"}),
        "timer":     timer_ialt.reset_index().rename(columns={"INDHOLD": "TIMER"}),
        "timeløn":   timeløn.reset_index(),
    }


# ===========================================================================
# Afgifter (produktskatter + moms)
# ===========================================================================

def beregn_afgift(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Samlet afgift = Produktskatter (netto ekskl. told og moms) + Moms.
    Input: data["input"] fra load_all() (NAIO3, 69-niveau).

    Returnerer
    ----------
    DataFrame med kolonner: ANVENDELSE, TID, afgift
    """
    common = ["ANVENDELSE", "TID"]

    afgift = (
        df_input
        .loc[(df_input["TILGANG1"] == "Produktskatter, netto ekskl. told og moms") &
             (df_input["PRISENHED"] == "Løbende priser")]
        .drop(columns=["PRISENHED", "TILGANG1"])
    )
    moms = (
        df_input
        .loc[(df_input["TILGANG1"] == "Moms") &
             (df_input["PRISENHED"] == "Løbende priser")]
        .drop(columns=["PRISENHED", "TILGANG1"])
    )

    merged = afgift.merge(moms, on=common, how="outer", suffixes=("_afgift", "_moms"))
    out = merged[common].copy()
    out["afgift"] = merged["INDHOLD_afgift"].fillna(0) + merged["INDHOLD_moms"].fillna(0)
    return out


# ===========================================================================
# Samlet beregningsfunktion
# ===========================================================================



def _beregn_io_suite(df_io: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Kører prisindeks, mængdeindeks og toldsats på én IO-DataFrame."""
    id_cols = ["TILGANG1", "TILGANG2", "ANVENDELSE"]
    idx     = id_cols + ["TID"]
    lob = (
        df_io.loc[(df_io["PRISENHED"] == "Løbende priser") & (df_io["TID"] != 1966)]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    forp = (
        df_io.loc[df_io["PRISENHED"] != "Løbende priser"]
        .drop(columns=["PRISENHED"])
        .set_index(idx)
    )
    pt = _beregn_prisindeks(lob, forp, id_cols)
    xt = _beregn_mængdeindeks(lob, pt, id_cols)
    tau = beregn_toldsats(df_io)
    return {"prisindeks": pt, "mængdeindeks": xt, "toldsats": tau}


def _beregn_input_suite(
    df_input: pd.DataFrame,
    df_lontimer: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Kører timer, timeløn og afgift på ét sæt input/lontimer DataFrames."""
    lon = beregn_timer_og_timeløn(df_input, df_lontimer)
    afgift = beregn_afgift(df_input)
    return {**lon, "afgift": afgift}


# ===========================================================================
# Samlet beregningsfunktion
# ===========================================================================

def beregn_alle(data: dict) -> dict[str, pd.DataFrame]:
    """
    Kører alle beregninger på begge aggregeringsniveauer og returnerer dem i en dict.

    Parametre
    ----------
    data : dict returneret af import_data.load_all()

    Returnerer
    ----------
    69-niveau  : "prisindeks", "mængdeindeks", "toldsats",
                 "timer_lon", "timer", "timeløn", "afgift"
    117-niveau : samme navne med "_117"-suffiks
    Kapital    : "prisindeks_kap", "mængdeindeks_kap"  (ét niveau)
    """
    print("Beregner IO-suite (69-niveau) ...")
    io_69 = _beregn_io_suite(data["io"])

    print("Beregner IO-suite (117-niveau) ...")
    io_117 = _beregn_io_suite(data["io_117"])

    print("Beregner kapital ...")
    pt_kap = beregn_prisindeks_kapital(data["kapital"])
    xt_kap = beregn_mængdeindeks_kapital(data["kapital"])

    print("Beregner timer, timeløn og afgift (69-niveau) ...")
    inp_69 = _beregn_input_suite(data["input"], data["lontimer"])

    print("Beregner timer, timeløn og afgift (117-niveau) ...")
    inp_117 = _beregn_input_suite(data["input_117"], data["lontimer_117"])

    print("Beregner kapital ...")
    pt_kap = beregn_prisindeks_kapital(data["kapital"])
    xt_kap = beregn_mængdeindeks_kapital(data["kapital"])

    print("Beregner timer, timeløn og afgift (69-niveau) ...")
    return {
        # 69-niveau
        **io_69,
        **inp_69,
        # 117-niveau
        **{f"{k}_117": v for k, v in io_117.items()},
        **{f"{k}_117": v for k, v in inp_117.items()},
        # Kapital
        "prisindeks_kap":     pt_kap,
        "mængdeindeks_kap":   xt_kap,
    }


if __name__ == "__main__":
    from import_data import load_all
    data    = load_all()
    results = beregn_alle(data)
    for name, df in results.items():
        print(f"\n{'='*50}")
        print(f"  {name}: {df.shape[0]} rækker x {df.shape[1]} kolonner")
        print(df.head(3).to_string(index=False))