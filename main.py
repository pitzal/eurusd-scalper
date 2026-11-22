
import pandas as pd
import yfinance as yf

# =====================================
# EURUSD SCALPER V1
# Prima versione - Download dati M1
# =====================================

CAPITALE_INIZIALE = 96.00
SIMBOLO = "EURUSD=X"
INTERVALLO = "1m"
PERIODO = "5d"


def scarica_dati():
    print("Scaricamento dati EUR/USD...")

    df = yf.download(
        SIMBOLO,
        period=PERIODO,
        interval=INTERVALLO,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError(
            "Nessun dato ricevuto dalla fonte."
        )

    # Gestisce le colonne multilivello
    # restituite da alcune versioni di yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [
        str(c).lower() for c in df.columns
    ]

    if "datetime" in df.columns:
        df = df.rename(
            columns={"datetime": "timestamp"}
        )
    elif "date" in df.columns:
        df = df.rename(
            columns={"date": "timestamp"}
        )

    colonne = [
        "timestamp",
        "open",
        "high",
        "low",
        "close"
    ]

    df = df[colonne].dropna()
    df = df.sort_values("timestamp")

    return df


def calcola_indicatori(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # RSI con gestione dei casi senza perdite
    denom = avg_gain + avg_loss

    df["rsi"] = 50.0

    valid = denom > 0
    df.loc[valid, "rsi"] = (
        100 * avg_gain[valid] / denom[valid]
    )

    return df


def genera_segnali(df):
    df = df.copy()
    df["signal"] = "WAIT"

    long_cross = (
        (df["close"].shift(1) <=
         df["ema20"].shift(1)) &
        (df["close"] > df["ema20"])
    )

    short_cross = (
        (df["close"].shift(1) >=
         df["ema20"].shift(1)) &
        (df["close"] < df["ema20"])
    )

    long_condition = (
        long_cross &
        (df["ema20"] > df["ema50"]) &
        df["rsi"].between(50, 70)
    )

    short_condition = (
        short_cross &
        (df["ema20"] < df["ema50"]) &
        df["rsi"].between(30, 50)
    )

    df.loc[long_condition, "signal"] = "BUY"
    df.loc[short_condition, "signal"] = "SELL"

    return df


def main():
    print("=== EURUSD SCALPER V1 ===")
    print(f"Capitale virtuale: {CAPITALE_INIZIALE} EUR")

    df = scarica_dati()
    df = calcola_indicatori(df)
    df = genera_segnali(df)

    df.to_csv(
        "eurusd_m1_signals.csv",
        index=False
    )

    segnali = df[
        df["signal"] != "WAIT"
    ]

    print(f"Candele scaricate: {len(df)}")
    print(f"Segnali trovati: {len(segnali)}")

    if not segnali.empty:
        print("\nUltimi 10 segnali:")
        print(
            segnali[
                ["timestamp", "close", "rsi", "signal"]
            ].tail(10).to_string(index=False)
        )

    print("\nFile salvato: eurusd_m1_signals.csv")
    print("Nessun ordine reale eseguito.")


if __name__ == "__main__":
    main()
