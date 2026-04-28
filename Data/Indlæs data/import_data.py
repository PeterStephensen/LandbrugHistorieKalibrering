"""
import_data.py
=============
Indlæser alle datasæt fra Danmarks Statistiks API og returnerer dem
som pandas DataFrames. Ingen CSV-filer gemmes.

Data hentes altid på 117-niveau (mest granulart). 69-niveau-versionerne
er afledt ved at aggregere fødevareindustriens undersektorer til ét samlet tal –
ingen ekstra API-kald.

Brug:
    from import_data import load_all

    data = load_all()

    # 117-niveau (fødevareindustri opdelt i undersektorer)
    df_io_117       = data["io_117"]        # NAIO1 – input-output
    df_input_117    = data["input_117"]     # NAIO3 – indkomstdannelse
    df_lontimer_117 = data["lontimer_117"]  # NAIO5 – løntimer

    # 69-niveau (fødevareindustri som ét aggregat, afledt uden ekstra API-kald)
    df_io           = data["io"]            # NAIO1 – input-output
    df_input        = data["input"]         # NAIO3 – indkomstdannelse
    df_lontimer     = data["lontimer"]      # NAIO5 – løntimer

    # Øvrige (ikke afhængige af aggregeringsniveau)
    df_kapital      = data["kapital"]       # NABK69 – kapitalapparat
    df_jordareal    = data["jordareal"]     # AFG6   – jordareal
    df_jordpris     = data["jordpris"]      # Excel  – jordpris
    df_grundskyld   = data["grundskyld"]   # Excel  – grundskyld
    df_tilskud      = data["tilskud"]       # TILSKUD + TILSKUD1 – EU-tilskud
"""

import requests
import pandas as pd
import numpy as np
from io import StringIO

DST_API = "https://api.statbank.dk/v1/data"


# ===========================================================================
# Hjælpefunktioner
# ===========================================================================

def _fetch_dst(table: str, variables: list) -> pd.DataFrame:
    """Henter data fra DST BULK-endpoint og returnerer en rå DataFrame."""
    params = {"table": table, "format": "BULK", "variables": variables}
    r = requests.post(DST_API, json=params)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep=";")


def _parse_indhold(df: pd.DataFrame) -> pd.DataFrame:
    """Konverterer INDHOLD-kolonnen fra dansk talformat til float."""
    df = df.copy()
    df["INDHOLD"] = (
        df["INDHOLD"].astype(str)
        .str.replace("\u00a0", "", regex=False)  # non-breaking space
        .str.replace(" ",      "", regex=False)  # tusindtals-separator (mellemrum)
        .str.replace(".",      "", regex=False)  # tusindtals-separator (punktum)
        .str.replace(",",      ".", regex=False) # decimal-komma -> punktum
    )
    df["INDHOLD"] = pd.to_numeric(df["INDHOLD"], errors="coerce")
    return df


def _add_rest_gruppe(df, kolonne, exclude, ny_vaerdi):
    """Summer alle rækker UNDTAGEN dem i `exclude` og tilføj som ny gruppe."""
    group_cols = [c for c in df.columns if c not in [kolonne, "INDHOLD"]]
    new_rows = (
        df.loc[~df[kolonne].isin(exclude)]
        .groupby(group_cols, as_index=False)["INDHOLD"]
        .sum()
    )
    new_rows[kolonne] = ny_vaerdi
    return pd.concat([df, new_rows[df.columns]], ignore_index=True)


def _add_aggregat_gruppe(df, kolonne, koder, ny_vaerdi):
    """Summer præcis de rækker i `koder` og tilføj som ny gruppe (originaler bevares)."""
    group_cols = [c for c in df.columns if c not in [kolonne, "INDHOLD"]]
    new_rows = (
        df.loc[df[kolonne].isin(koder)]
        .groupby(group_cols, as_index=False)["INDHOLD"]
        .sum()
    )
    new_rows[kolonne] = ny_vaerdi
    return pd.concat([df, new_rows[df.columns]], ignore_index=True)


def _collapse_til_aggregat(df, kolonne, koder, ny_vaerdi):
    """
    Summer rækker i `koder` til én ny gruppe og FJERN de originale rækker.
    Bruges til at konvertere fra 117-niveau til 69-niveau.
    """
    df = _add_aggregat_gruppe(df, kolonne, koder, ny_vaerdi)
    return df[~df[kolonne].isin(koder)].reset_index(drop=True)


# ===========================================================================
# Branchekoder og navne
# 117-niveau er kilden – 69-niveau afledes herfra uden ekstra API-kald.
# ===========================================================================

# API-koder til NAIO1 TILGANG2 (117-niveau)
_FODE_T   = ["T100010", "T100020", "T100030", "T100040", "T100050", "T110000", "T120000"]
_OVRIGE_T = ["TCB","TCC","TCD","TCE","TCF","TCG","TCH",
             "TCI","TCJ","TCK","TCL","TCM","TD_E","TF","TG_I",
             "TJ","TK","TLA","TLB","TM_N","TO_Q","TR_S"]
NAIO1_TILGANG2 = ["T010000","T020000","T030000","TB"] + _FODE_T + _OVRIGE_T

# API-koder til NAIO1 og NAIO3 ANVENDELSE (117-niveau)
_FODE_A   = ["A100010","A100020","A100030","A100040","A100050","A110000","A120000"]
_OVRIGE_A = ["ACB","ACC","ACD","ACE","ACF","ACG","ACH",
             "ACI","ACJ","ACK","ACL","ACM","AD_E","AF","AG_I",
             "AJ","AK","ALA","ALB","AM_N","AO_Q","AR_S"]
NAIO3_ANVENDELSE = ["A010000","A020000","A030000","AB"] + _FODE_A + _OVRIGE_A
NAIO1_ANVENDELSE = NAIO3_ANVENDELSE + [
    "ACPT","ANPISH","ACO","ABI","AL5200","AV5300","AE6000","AA00000",
]

# API-koder til NAIO5 BRANCHE (117-niveau)
_FODE_V   = ["V100010","V100020","V100030","V100040","V100050","V110000","V120000"]
_OVRIGE_V = ["VCB","VCC","VCD","VCE","VCF","VCG","VCH",
             "VCI","VCJ","VCK","VCL","VCM","VD_E","VF","VG_I",
             "VJ","VK","VLA","VLB","VM_N","VO_Q","VR_S"]
NAIO5_BRANCHE = ["V010000","V020000","V030000","VB"] + _FODE_V + _OVRIGE_V

# ---- Tekst-navne som API-svaret returnerer (bruges i filtrering/aggregering) ----

# Alle 7 fødevare-undersektorer (til kollaps fra 117 → 69)
FODE_NAVNE_T = [
    "100010 Slagterier-(Tilgang)",
    "100020 Fiskeindustri-(Tilgang)",
    "100030 Mejerier-(Tilgang)",
    "100040 Bagerier, brødfabrikker mv.-(Tilgang)",
    "100050 Anden fødevareindustri-(Tilgang)",
    "110000 Drikkevareindustri-(Tilgang)",
    "120000 Tobaksindustri-(Tilgang)",
]
FODE_NAVNE_A = [
    "100010 Slagterier- (Anvendelse)",
    "100020 Fiskeindustri- (Anvendelse)",
    "100030 Mejerier- (Anvendelse)",
    "100040 Bagerier, brødfabrikker mv.- (Anvendelse)",
    "100050 Anden fødevareindustri- (Anvendelse)",
    "110000 Drikkevareindustri- (Anvendelse)",
    "120000 Tobaksindustri- (Anvendelse)",
]
FODE_NAVNE_V = [
    "100010 Slagterier",
    "100020 Fiskeindustri",
    "100030 Mejerier",
    "100040 Bagerier, brødfabrikker mv.",
    "100050 Anden fødevareindustri",
    "110000 Drikkevareindustri",
    "120000 Tobaksindustri",
]

# Sektorer der vises individuelt på 117-niveau (resten lægges i REST)
# Fiskeindustri, drikkevarer og tobak lægges i REST på 117-niveau.
FODE_117_T = ["010000 Landbrug og gartneri-(Tilgang)",
              "100010 Slagterier-(Tilgang)",
              "100030 Mejerier-(Tilgang)",
              "100040 Bagerier, brødfabrikker mv.-(Tilgang)",
              "100050 Anden fødevareindustri-(Tilgang)"]
FODE_117_A = ["010000 Landbrug og gartneri- (Anvendelse)",
              "100010 Slagterier- (Anvendelse)",
              "100030 Mejerier- (Anvendelse)",
              "100040 Bagerier, brødfabrikker mv.- (Anvendelse)",
              "100050 Anden fødevareindustri- (Anvendelse)"]
FODE_117_V = ["010000 Landbrug og gartneri",
              "100010 Slagterier",
              "100030 Mejerier",
              "100040 Bagerier, brødfabrikker mv.",
              "100050 Anden fødevareindustri"]

# Sub-aggregat på 117-niveau: bagerier + anden fødevareindustri
ANDEN_FODE_T = ["100040 Bagerier, brødfabrikker mv.-(Tilgang)",
                "100050 Anden fødevareindustri-(Tilgang)"]
ANDEN_FODE_A = ["100040 Bagerier, brødfabrikker mv.- (Anvendelse)",
                "100050 Anden fødevareindustri- (Anvendelse)"]
ANDEN_FODE_V = ["100040 Bagerier, brødfabrikker mv.",
                "100050 Anden fødevareindustri"]
ANDEN_FODE_NAVN = "100040x100050 Anden fødevareindustri (100040, 100050)"

# Navne på det fulde fødevare-aggregat på 69-niveau
FODE_69_T = "Føde-, drikke- og tobaksvareindustri-(Tilgang)"
FODE_69_A = "Føde-, drikke- og tobaksvareindustri- (Anvendelse)"
FODE_69_V = "Føde-, drikke- og tobaksvareindustri"

# Efterspørgselskomponenter i NAIO1 ANVENDELSE
_EFTERSP = ["Husholdningernes forbrugsudgifter (Anvendelse)",
            "NPISH i alt - (Anvendelse)",
            "Offentligt forbrug, i alt-(Anvendelse)",
            "Faste bruttoinvesteringer ialt - (Anvendelse)",
            "Lagre - (Anvendelse)",
            "Værdigenstande - (Anvendelse)",
            "Eksport - (Anvendelse)",
            "Anvendelse, i alt-(Anvendelse)"]


# ===========================================================================
# Private processorer – fælles logik, returnerer (df_117, df_69)
# ===========================================================================

def _process_io(df):
    df = _parse_indhold(df)

    # Beregn import eksklusiv told
    grp = [c for c in df.columns if c not in ["TILGANG1", "INDHOLD"]]
    imp  = df[df["TILGANG1"] == "Import inkl. told"].groupby(grp, as_index=False)["INDHOLD"].sum()
    told = df[df["TILGANG1"] == "Told"].groupby(grp, as_index=False)["INDHOLD"].sum()
    merged = imp.merge(told, on=grp, how="outer", suffixes=("_i", "_t"))
    imp_ex = merged[grp].copy()
    imp_ex["INDHOLD"] = merged["INDHOLD_i"].fillna(0) - merged["INDHOLD_t"].fillna(0)
    imp_ex["TILGANG1"] = "Import eksklusiv told"
    df = pd.concat([df, imp_ex[df.columns]], ignore_index=True)

    # Efterspørgsels-aggregater (fælles for begge niveauer)
    df = _add_aggregat_gruppe(df, "ANVENDELSE",
        ["Husholdningernes forbrugsudgifter (Anvendelse)", "NPISH i alt - (Anvendelse)"],
        "Husholdningernes forbrugsudgifter + NPISH (Anvendelse)")
    df = _add_aggregat_gruppe(df, "ANVENDELSE",
        ["Faste bruttoinvesteringer ialt - (Anvendelse)", "Lagre - (Anvendelse)", "Værdigenstande - (Anvendelse)"],
        "Faste bruttoinvesteringer + Lagerforøgelse + Værdigenstande (Anvendelse)")

    # ---- 117-niveau ----
    d117 = _add_rest_gruppe(df,  "TILGANG2",   FODE_117_T, "REST_TILGANG Øvrige brancher")
    d117 = _add_rest_gruppe(d117,"ANVENDELSE", FODE_117_A + _EFTERSP, "REST_ANVENDELSE Øvrige brancher")
    d117 = _add_aggregat_gruppe(d117, "TILGANG2",   ANDEN_FODE_T, ANDEN_FODE_NAVN)
    d117 = _add_aggregat_gruppe(d117, "ANVENDELSE", ANDEN_FODE_A, ANDEN_FODE_NAVN)

    sel_t117 = ["010000 Landbrug og gartneri-(Tilgang)",
                "100010 Slagterier-(Tilgang)",
                "100030 Mejerier-(Tilgang)",
                ANDEN_FODE_NAVN,
                "REST_TILGANG Øvrige brancher"]
    sel_a117 = ["010000 Landbrug og gartneri- (Anvendelse)",
                "100010 Slagterier- (Anvendelse)",
                "100030 Mejerier- (Anvendelse)",
                ANDEN_FODE_NAVN,
                "REST_ANVENDELSE Øvrige brancher",
                "Husholdningernes forbrugsudgifter + NPISH (Anvendelse)",
                "Offentligt forbrug, i alt-(Anvendelse)",
                "Faste bruttoinvesteringer + Lagerforøgelse + Værdigenstande (Anvendelse)",
                "Eksport - (Anvendelse)",
                "Anvendelse, i alt-(Anvendelse)"]
    df_117 = d117[d117["TILGANG2"].isin(sel_t117) & d117["ANVENDELSE"].isin(sel_a117)].copy()

    # ---- 69-niveau: kollaps alle fødevare-undersektorer ----
    d69 = _collapse_til_aggregat(df, "TILGANG2",   FODE_NAVNE_T, FODE_69_T)
    d69 = _collapse_til_aggregat(d69,"ANVENDELSE", FODE_NAVNE_A, FODE_69_A)
    d69 = _add_rest_gruppe(d69, "TILGANG2",
        ["010000 Landbrug og gartneri-(Tilgang)", FODE_69_T],
        "REST_TILGANG Øvrige brancher")
    d69 = _add_rest_gruppe(d69, "ANVENDELSE",
        ["010000 Landbrug og gartneri- (Anvendelse)", FODE_69_A] + _EFTERSP,
        "REST_ANVENDELSE Øvrige brancher")

    sel_t69 = ["010000 Landbrug og gartneri-(Tilgang)", FODE_69_T, "REST_TILGANG Øvrige brancher"]
    sel_a69 = ["010000 Landbrug og gartneri- (Anvendelse)",
               FODE_69_A,
               "REST_ANVENDELSE Øvrige brancher",
               "Husholdningernes forbrugsudgifter + NPISH (Anvendelse)",
               "Offentligt forbrug, i alt-(Anvendelse)",
               "Faste bruttoinvesteringer + Lagerforøgelse + Værdigenstande (Anvendelse)",
               "Eksport - (Anvendelse)",
               "Anvendelse, i alt-(Anvendelse)"]
    df_69 = d69[d69["TILGANG2"].isin(sel_t69) & d69["ANVENDELSE"].isin(sel_a69)].copy()

    return df_117, df_69


def _process_input(df):
    df = _parse_indhold(df)

    # 117-niveau
    d117 = _add_rest_gruppe(df.copy(), "ANVENDELSE", FODE_117_A, "REST_ANVENDELSE Øvrige brancher")
    d117 = _add_aggregat_gruppe(d117, "ANVENDELSE", ANDEN_FODE_A, ANDEN_FODE_NAVN)
    sel_117 = ["010000 Landbrug og gartneri- (Anvendelse)",
               "100010 Slagterier- (Anvendelse)",
               "100030 Mejerier- (Anvendelse)",
               ANDEN_FODE_NAVN,
               "REST_ANVENDELSE Øvrige brancher"]
    df_117 = d117[d117["ANVENDELSE"].isin(sel_117)].copy()

    # 69-niveau
    d69 = _collapse_til_aggregat(df, "ANVENDELSE", FODE_NAVNE_A, FODE_69_A)
    d69 = _add_rest_gruppe(d69, "ANVENDELSE",
        ["010000 Landbrug og gartneri- (Anvendelse)", FODE_69_A],
        "REST_ANVENDELSE Øvrige brancher")
    sel_69 = ["010000 Landbrug og gartneri- (Anvendelse)", FODE_69_A, "REST_ANVENDELSE Øvrige brancher"]
    df_69 = d69[d69["ANVENDELSE"].isin(sel_69)].copy()

    return df_117, df_69


def _process_lontimer(df):
    df = _parse_indhold(df)

    # 117-niveau
    d117 = _add_rest_gruppe(df.copy(), "BRANCHE", FODE_117_V, "REST_ANVENDELSE Øvrige brancher")
    d117 = _add_aggregat_gruppe(d117, "BRANCHE", ANDEN_FODE_V, ANDEN_FODE_NAVN)
    sel_117 = ["010000 Landbrug og gartneri",
               "100010 Slagterier",
               "100030 Mejerier",
               ANDEN_FODE_NAVN,
               "REST_ANVENDELSE Øvrige brancher"]
    df_117 = d117[d117["BRANCHE"].isin(sel_117)].copy()

    # 69-niveau
    d69 = _collapse_til_aggregat(df, "BRANCHE", FODE_NAVNE_V, FODE_69_V)
    d69 = _add_rest_gruppe(d69, "BRANCHE",
        ["010000 Landbrug og gartneri", FODE_69_V],
        "REST_ANVENDELSE Øvrige brancher")
    sel_69 = ["010000 Landbrug og gartneri", FODE_69_V, "REST_ANVENDELSE Øvrige brancher"]
    df_69 = d69[d69["BRANCHE"].isin(sel_69)].copy()

    return df_117, df_69


# ===========================================================================
# Offentlige loaders – kan bruges enkeltvist
# ===========================================================================

def load_io_117():
    """NAIO1 – input-output, 117-niveau."""
    raw = _fetch_dst("NAIO1", [
        {"code": "TILGANG1",   "values": ["*"]},
        {"code": "TILGANG2",   "values": NAIO1_TILGANG2},
        {"code": "ANVENDELSE", "values": NAIO1_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    return _process_io(raw)[0]

def load_io():
    """NAIO1 – input-output, 69-niveau."""
    raw = _fetch_dst("NAIO1", [
        {"code": "TILGANG1",   "values": ["*"]},
        {"code": "TILGANG2",   "values": NAIO1_TILGANG2},
        {"code": "ANVENDELSE", "values": NAIO1_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    return _process_io(raw)[1]

def load_input_117():
    """NAIO3 – indkomstdannelse, 117-niveau."""
    raw = _fetch_dst("NAIO3", [
        {"code": "TILGANG1",   "values": ["D214X31","D211","D29X39","D1","B2A3G"]},
        {"code": "ANVENDELSE", "values": NAIO3_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    return _process_input(raw)[0]

def load_input():
    """NAIO3 – indkomstdannelse, 69-niveau."""
    raw = _fetch_dst("NAIO3", [
        {"code": "TILGANG1",   "values": ["D214X31","D211","D29X39","D1","B2A3G"]},
        {"code": "ANVENDELSE", "values": NAIO3_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    return _process_input(raw)[1]

def load_lontimer_117():
    """NAIO5 – løntimer, 117-niveau."""
    raw = _fetch_dst("NAIO5", [
        {"code": "SOCIO",   "values": ["EMPH_DC","SALH_DC"]},
        {"code": "BRANCHE", "values": NAIO5_BRANCHE},
        {"code": "TID",     "values": ["*"]},
    ])
    return _process_lontimer(raw)[0]

def load_lontimer():
    """NAIO5 – løntimer, 69-niveau."""
    raw = _fetch_dst("NAIO5", [
        {"code": "SOCIO",   "values": ["EMPH_DC","SALH_DC"]},
        {"code": "BRANCHE", "values": NAIO5_BRANCHE},
        {"code": "TID",     "values": ["*"]},
    ])
    return _process_lontimer(raw)[1]


def load_kapital():
    """NABK69 – kapitalapparat med REST-gruppe og foregående års priser."""
    df = _fetch_dst("NABK69", [
        {"code": "BEHOLD",    "values": ["LEN","P51G"]},
        {"code": "AKTIV",     "values": ["N11"]},
        {"code": "BRANCHE",   "values": ["*"]},
        {"code": "PRISENHED", "values": ["*"]},
        {"code": "TID",       "values": ["*"]},
    ])
    df = _parse_indhold(df)
    df = df.loc[
        ~df["BRANCHE"].isin(["V","VMEMO"]) &
        ~df["BRANCHE"].str.contains("Heraf: Offentlig forvaltning og service", na=False) &
        ~df["BRANCHE"].str.contains("I alt", na=False)
    ]
    df.drop(columns=["AKTIV"], inplace=True)

    df_for = df.loc[df["PRISENHED"] != "Løbende priser"].drop(columns=["PRISENHED"])
    df_lob = df.loc[df["PRISENHED"] == "Løbende priser"].drop(columns=["PRISENHED"])
    df_lob = df_lob.loc[df_lob["TID"] != 1966]

    idx = ["BEHOLD","BRANCHE","TID"]
    df_for_i = df_for.set_index(idx)
    df_lob_i = df_lob.set_index(idx)
    maengdeindeks = df_for_i["INDHOLD"] / df_for_i.groupby(["BEHOLD","BRANCHE"])["INDHOLD"].shift(1)
    forrige = df_lob_i.groupby(["BEHOLD","BRANCHE"])["INDHOLD"].shift(1) * maengdeindeks
    forrige = forrige.fillna(0).reset_index(name="INDHOLD")
    forrige["PRISENHED"] = "Forrige års priser"
    forrige = forrige[df.columns]
    df = pd.concat([df, forrige], ignore_index=True)

    exclude = ["01000 Landbrug og gartneri","10120 Føde-, drikke- og tobaksvareindustri"]
    df = _add_rest_gruppe(df, "BRANCHE", exclude, "REST_ANVENDELSE Øvrige brancher")
    udvalgte = exclude + ["REST_ANVENDELSE Øvrige brancher"]
    return df[df["BRANCHE"].isin(udvalgte)].copy()


def load_jordareal():
    """AFG6 – jordareal i millioner ha (fratrukket brak og graes)."""
    df = _fetch_dst("AFG6", [
        {"code": "AFGRØDE", "values": ["000","245","255"]},
        {"code": "ENHED",    "values": ["HA"]},
        {"code": "AREAL1",   "values": ["AIALT"]},
        {"code": "TID",      "values": ["*"]},
    ])
    df = _parse_indhold(df)
    idx = ["ENHED","AREAL1","TID"]
    tot  = df[df["AFGRØDE"] == "Landbrug og gartneri i alt"].set_index(idx)
    gras = df[df["AFGRØDE"] == "8. Græs uden for omdrift"].set_index(idx)
    brak = df[df["AFGRØDE"] == "10. Braklægning"].set_index(idx)
    return (
        (tot["INDHOLD"] - gras["INDHOLD"] - brak["INDHOLD"]) / 1_000_000
    ).reset_index(name="INDHOLD")[["TID","INDHOLD"]]


def load_jordpris(fil="Data69/Jordpris.xlsx"):
    """Jordpris fra lokal Excel-fil. Returnerer TID/INDHOLD DataFrame."""
    df = pd.read_excel(fil, skiprows=2).dropna(how="all", axis=0)
    mask = (
        df.stack()
        .str.contains(r"Arable land, buying price \(DKK per ha\)", na=False)
        .unstack().any(axis=1)
    )
    row = df[mask].iloc[:, 2:].transpose()
    row.columns = ["INDHOLD"]
    row.index.name = "TID"
    return row.reset_index()


def load_grundskyld(fil="Data69/Grundskyld Landbruget.xlsx"):
    """Grundskyld fra lokal Excel-fil. Returnerer TID/INDHOLD (i 1000 DKK)."""
    df = pd.read_excel(fil, header=0).dropna(how="all", axis=0).dropna(how="all", axis=1)
    row = df[df.iloc[:, 0] == 10000]
    out = row.iloc[:, 1:].T.reset_index()
    out.columns = ["TID","INDHOLD"]
    out["INDHOLD"] = out["INDHOLD"] / 1000
    return out


def load_tilskud():
    """TILSKUD + TILSKUD1 – EU-tilskud, tre perioder sammensyet til én tidsserie."""
    df_f = _fetch_dst("TILSKUD",  [{"code":"TYPE","values":["010","085","090","095"]},{"code":"TID","values":["*"]}])
    df_e = _fetch_dst("TILSKUD1", [{"code":"TILSKUDSART","values":["170","175"]},{"code":"TID","values":["*"]}])
    df_f = _parse_indhold(df_f)
    df_e = _parse_indhold(df_e)

    def _get(df, col, val):
        return df[df[col] == val].drop(columns=[col]).set_index("TID")

    hek  = _get(df_f, "TYPE", "Hektarstøtte i alt")
    brak = _get(df_f, "TYPE", "Braklægningsstøtte")
    ekst = _get(df_f, "TYPE", "Ekstensiveringspræmie")
    p1 = (hek["INDHOLD"] + brak["INDHOLD"] + ekst["INDHOLD"]).reset_index(name="INDHOLD")
    p1 = p1[p1["TID"] < 2005]

    p2 = _get(df_f, "TYPE", "Enkeltbetalingsordning, arealtilskud")["INDHOLD"].reset_index(name="INDHOLD")
    p2 = p2[(p2["TID"] >= 2005) & (p2["TID"] < 2015)]

    grund = _get(df_e, "TILSKUDSART", "2.1 Grundbetaling")
    gron  = _get(df_e, "TILSKUDSART", "2.2 Grønne krav")
    p3 = (grund["INDHOLD"] + gron["INDHOLD"]).reset_index(name="INDHOLD")
    p3 = p3[p3["TID"] >= 2015]

    return pd.concat([p1, p2, p3], ignore_index=True).sort_values("TID").reset_index(drop=True)


# ===========================================================================
# load_all – henter én gang per tabel, returnerer begge niveauer
# ===========================================================================

def load_all(
    jordpris_fil="Jordpris.xlsx",
    grundskyld_fil="Grundskyld Landbruget.xlsx",
):
    """
    Indlæser alle datasæt og returnerer dem i en dict.
    NAIO1, NAIO3 og NAIO5 hentes kun én gang (117-niveau);
    69-niveau afledes uden ekstra API-kald.

    Returnerer dict med nøgler:
      117-niveau : "io_117", "input_117", "lontimer_117"
      69-niveau  : "io",     "input",     "lontimer"
      Ovrige     : "kapital", "jordareal", "jordpris", "grundskyld", "tilskud"
    """
    print("Henter NAIO1  (input-output) ...")
    raw_io = _fetch_dst("NAIO1", [
        {"code": "TILGANG1",   "values": ["*"]},
        {"code": "TILGANG2",   "values": NAIO1_TILGANG2},
        {"code": "ANVENDELSE", "values": NAIO1_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    io_117, io_69 = _process_io(raw_io)

    print("Henter NAIO3  (indkomstdannelse) ...")
    raw_inp = _fetch_dst("NAIO3", [
        {"code": "TILGANG1",   "values": ["D214X31","D211","D29X39","D1","B2A3G"]},
        {"code": "ANVENDELSE", "values": NAIO3_ANVENDELSE},
        {"code": "PRISENHED",  "values": ["*"]},
        {"code": "TID",        "values": ["*"]},
    ])
    inp_117, inp_69 = _process_input(raw_inp)

    print("Henter NAIO5  (lontimer) ...")
    raw_lon = _fetch_dst("NAIO5", [
        {"code": "SOCIO",   "values": ["EMPH_DC","SALH_DC"]},
        {"code": "BRANCHE", "values": NAIO5_BRANCHE},
        {"code": "TID",     "values": ["*"]},
    ])
    lon_117, lon_69 = _process_lontimer(raw_lon)

    print("Henter NABK69 (kapital) ...")
    kapital = load_kapital()

    print("Henter AFG6   (jordareal) ...")
    jordareal = load_jordareal()

    print("Henter TILSKUD/TILSKUD1 (tilskud) ...")
    tilskud = load_tilskud()

    print("Laeder Excel  (jordpris) ...")
    jordpris = load_jordpris(jordpris_fil)

    print("Laeder Excel  (grundskyld) ...")
    grundskyld = load_grundskyld(grundskyld_fil)

    print("Faerdig!")
    return {
        "io_117":       io_117,
        "input_117":    inp_117,
        "lontimer_117": lon_117,
        "io":           io_69,
        "input":        inp_69,
        "lontimer":     lon_69,
        "kapital":      kapital,
        "jordareal":    jordareal,
        "jordpris":     jordpris,
        "grundskyld":   grundskyld,
        "tilskud":      tilskud,
    }


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"\n{'='*50}")
        print(f"  {name}: {df.shape[0]} raekker x {df.shape[1]} kolonner")
        print(df.head(3).to_string(index=False))