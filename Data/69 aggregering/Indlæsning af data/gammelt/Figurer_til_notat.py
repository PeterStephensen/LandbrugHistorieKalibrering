%load_ext autoreload
%autoreload 2

import data_formater69 as df
import calib69 as calib
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.ticker as mticker
import numpy as np

# ── Farvepalette inspireret af Word-dokumentets grønne tema ──────────────────
DREAM_COLORS = [
    "#27AE60",  # grøn
    "#2980B9",  # blå
    "#C0392B",  # rød
    "#E67E22",  # orange
    "#8E44AD",  # lilla (hvis du har en 5. serie)
]

GRAY       = "#000000"
LIGHT_GRAY = "#E8E8E8"
FONT       = "Calibri"   # serif – på din egen maskine kan du skifte til "Georgia" eller "Calibri"

def dream_plot(d_wide, ylabel):
    """
    d_wide : DataFrame med år som index og branche som kolonner
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Linjer ───────────────────────────────────────────────────────────────
    for i, branche in enumerate(d_wide.columns):
        ax.plot(
            d_wide.index,
            d_wide[branche],
            color=DREAM_COLORS[i % len(DREAM_COLORS)],
            linewidth=1.8,
            label=branche,
        )
    # ── Akser ────────────────────────────────────────────────────────────────
    ax.set_xlabel("År", fontsize=9, color=GRAY, fontfamily=FONT)
    ax.set_ylabel(ylabel, fontsize=9, color=GRAY, fontfamily=FONT)
    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GRAY)
    ax.spines["bottom"].set_color(LIGHT_GRAY)
    ax.yaxis.grid(True, color=LIGHT_GRAY, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

    # ── Legende ──────────────────────────────────────────────────────────────
    ax.legend(
        title="Branche",
        title_fontsize=8,
        fontsize=8,
        frameon=False,
        loc="upper right",
        labelcolor=GRAY,
    )

    return fig, ax


# ── Eksempel med simuleret data (erstat med din calib-kode) ──────────────────
# Produktionsværdi
if __name__ == "__main__":
    np.random.seed(42)
    years = np.arange(1994, 2023)
    brancher = ["Landbrug", "Fødevareindustri"]

    import pandas as pd
    d = df.Y['Xt'].loc[(slice(None), slice(1993, 2022))]*df.P['Pt'].loc[(slice(None), slice(1994, 2022))]
    d_wide = d.unstack('ANVENDELSE').sort_index()
    d_wide = d_wide.drop(columns=['REST'])
    d_wide.columns = brancher  # sætter "Landbrug" og "Fødevareindustri"
    d_wide.index.name = "År"

    fig, ax = dream_plot(
        d_wide,
        ylabel="Mio. kr.",
    )

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    years = np.arange(1994, 2023)
    brancher = ["Landbrug", "Fødevareindustri", "Resten af økonomien"]

    import pandas as pd
    d =df.K['Xt'].loc[(slice(None), slice(1993, 2022))]/ df.Y['Xt'].loc[(slice(None), slice(1994, 2022))]
    d_wide = d.unstack('ANVENDELSE').sort_index()
    d_wide.columns = brancher  # sætter "Landbrug" og "Fødevareindustri"
    d_wide.index.name = "År"

    fig, ax = dream_plot(
        d_wide,
        ylabel="Kapital-output-ratio (K/Y)",
    )

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    years = np.arange(1994, 2023)
    brancher = ["Landbrug", "Fødevareindustri", "Resten af økonomien"]

    import pandas as pd
    d =df.L['Xt'].loc[(slice(None), slice(1993, 2022))]/ df.Y['Xt'].loc[(slice(None), slice(1994, 2022))]
    d_wide = d.unstack('ANVENDELSE').sort_index()
    d_wide.columns = brancher  # sætter "Landbrug" og "Fødevareindustri"
    d_wide.index.name = "År"

    fig, ax = dream_plot(
        d_wide,
        ylabel="Arbejdskraft-output-ratio (L/Y)",
    )

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    years = np.arange(1994, 2023)
    brancher = ["Landbrug", "Fødevareindustri", "Resten af økonomien"]

    import pandas as pd
    d = df.K['Xt'].loc[(slice(None), slice(1993, 2022))] / df.L['Xt'].loc[(slice(None), slice(1994, 2022))]    
    d_wide = d.unstack('ANVENDELSE').sort_index()
    d_wide.columns = brancher  # sætter "Landbrug" og "Fødevareindustri"
    d_wide.index.name = "År"

    fig, ax = dream_plot(
        d_wide,
        ylabel="Kapital per arbejdstime (K/L)",
    )

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    years = np.arange(1994, 2023)
    brancher = ["Landbrug", "Fødevareindustri", "Resten af økonomien"]

    import pandas as pd
    d =df.Mtot['Xt'].loc[(slice(None), slice(1993, 2022))]/ df.Y['Xt'].loc[(slice(None), slice(1994, 2022))]
    d_wide = d.unstack('ANVENDELSE').sort_index()
    d_wide.columns = brancher  # sætter "Landbrug" og "Fødevareindustri"
    d_wide.index.name = "År"

    fig, ax = dream_plot(
        d_wide,
        ylabel="Materiale-output-ratio (Mtot/Y)",
    )

    plt.show()