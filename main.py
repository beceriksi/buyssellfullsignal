# main.py — PRO Futures Sinyal Botu (MEXC USDT Perp)
# BUY: hacim + trend + momentum  |  SELL: trend kırılım + momentum + ADX (hacime bağlı değil)
# Gerekenler: pip install ccxt pandas numpy requests

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt

# ====== Ayarlar (ENV ile değiştirilebilir) ======
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

SCAN_LIMIT     = int(os.getenv("SCAN_LIMIT", "300"))     # taranacak max market
TF_LIST        = os.getenv("TF_LIST", "1h,4h").split(",")# ["1h","4h"]
MAX_WORKERS    = int(os.getenv("MAX_WORKERS", "12"))

# BUY (pump)
BUY_MIN_TURNOVER   = float(os.getenv("BUY_MIN_TURNOVER", "10000"))  # zayıf hacimli pump’ları da kaçırma
BUY_VOL_RATIO_MIN  = float(os.getenv("BUY_VOL_RATIO_MIN", "1.10"))  # EMA hacim oranı
BUY_RSI_MIN        = float(os.getenv("BUY_RSI_MIN", "50.0"))
BUY_ADX_MIN        = float(os.getenv("BUY_ADX_MIN", "16.0"))        # çok düşük trend gücünü ele

# SELL (dump) — hacime bakmıyoruz
SELL_RSI_MAX       = float(os.getenv("SELL_RSI_MAX", "48.0"))
SELL_ADX_MIN       = float(os.getenv("SELL_ADX_MIN", "18.0"))
GAP_PCT_MAX        = float(os.getenv("GAP_PCT_MAX", "0.10"))        # anormal gap’leri ele (10%)

MEXC_SPOT_API      = "https://api.mexc.com"          # sadece klines için qv okumada lazım değil, futures ayrı
MEXC_CONTRACT_API  = "https://contract.mexc.com"     # klines & funding

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- HTTP ----------
def jget(url, params=None, retries=3, timeout=12):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200: return r.json()
        except: time.sleep(0.4)
    return None

def telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("\n" + text + "\n"); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15
        )
    except: pass

# ---------- İndikatörler ----------
def ema(x,n): return x.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n, adjust=False).mean()/(dn.ewm(alpha=1/n, adjust=False).mean()+1e-12)
    return 100-(100/(1+rs))

def adx(df,n=14):
    up=df['high'].diff(); dn=-df['low'].diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr1=df['high']-df['low']; tr2=(df['high']-df['close'].shift()).abs(); tr3=(df['low']-df['close'].shift()).abs()
    tr=pd.DataFrame({'a':tr1,'b':tr2,'c':tr3}).max(axis=1)
    atr=tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di=100*pd.Series(plus).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    minus_di=100*pd.Series(minus).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    dx=((plus_di-minus_di).abs()/((plus_di+minus_di)+1e-12))*100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def vol_ratio(turnover, n=10):
    base=turnover.ewm(span=n, adjust=False).mean()
    return float(turnover.iloc[-1]/(base.iloc[-2]+1e-12))

def gap_ok(c, pct):
    if len(c) < 2: return False
    return abs(float(c.iloc[-1] / c.iloc[-2] - 1.0)) <= pct

# ---------- Coin listesi (MEXC USDT Perp, 1. bot mantığının pro hali) ----------
def get_mexc_usdt_perp_symbols(limit=500):
    """ccxt ile USDT linear perpetual piyasalarını eksiksiz al."""
    try:
        ex = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        ex.load_markets()
        syms = []
        for s, m in ex.markets.items():
            # m['id'] örn: 'BTC_USDT' (MEXC contract API ile birebir uyumlu)
            if m.get('active') and m.get('contract') and m.get('linear') and m.get('quote')=='USDT':
                syms.append(m['id'])
        syms = sorted(set(syms))
        return syms[:limit]
    except Exception as e:
        print("CCXT MEXC symbol fetch error:", e)
        return []

# ---------- MEXC Contract verisi ----------
def kline_contract(sym, interval, limit):
    d=jget(f"{MEXC_CONTRACT_API}/api/v1/contract/kline/{sym}", {"interval": interval, "limit": limit})
    if not d or "data" not in d: return None
    df=pd.DataFrame(d["data"], columns=["ts","open","high","low","close","volume","turnover"]).astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64","turnover":"float64"}
    )
    return df

def funding(sym):
    d=jget(f"{MEXC_CONTRACT_API}/api/v1/contract/funding_rate", {"symbol": sym})
    try: return float(d["data"]["fundingRate"])
    except: return None

# ---------- Sinyal mantığı ----------
def analyze_one(sym, tf):
    # veri çek
    limit = 220 if tf in ("1h","4h") else 200
    df = kline_contract(sym, tf, limit)
    if df is None or len(df) < 60: 
        return None

    c,h,l,t = df["close"], df["high"], df["low"], df["turnover"]
    if not gap_ok(c, GAP_PCT_MAX):
        return None

    # hesaplar
    e20 = float(ema(c,20).iloc[-1]); e50 = float(ema(c,50).iloc[-1])
    rr  = float(rsi(c,14).iloc[-1])
    ax  = float(adx(pd.DataFrame({"high":h,"low":l,"close":c}),14).iloc[-1])
    trend_up = e20 > e50
    last_dir_up = c.iloc[-1] > c.iloc[-2]
    vr = vol_ratio(t, 10)

    # BUY koşulları (hacim + trend + momentum)
    buy = (
        (t.iloc[-1] >= BUY_MIN_TURNOVER) and        # alt hacim sınırı
        trend_up and                                 # EMA20 > EMA50
        (rr >= BUY_RSI_MIN) and                      # RSI 50+
        (vr >= BUY_VOL_RATIO_MIN) and                # hacim ema oranı
        (ax >= BUY_ADX_MIN)                          # trend gücü
    )

    # SELL koşulları (hacim YOK) — trend kırılım + momentum + ADX
    # Son 2 mum altında kapanış:
    break2 = (c.iloc[-1] < c.iloc[-2]) and (c.iloc[-1] < c.iloc[-3])
    sell = (
        (not trend_up) and                           # EMA20 < EMA50
        (rr <= SELL_RSI_MAX) and                     # RSI düşük
        break2 and                                   # son 2 mum altı
        (ax >= SELL_ADX_MIN)                         # trend gücü
    )

    if not (buy or sell):
        return None

    # güven skoru (0-100)
    conf = 0.0
    if buy:
        conf = (min(1.0, (t.iloc[-1]/max(1.0, BUY_MIN_TURNOVER))) * 20.0) \
             + (max(0.0, rr-50.0)*1.0) + (min(2.0, vr)*25.0) + (ax/3.0)
    else:  # sell
        conf = (max(0.0, 50.0-rr)*1.0) + (ax/2.5) + (20.0 if break2 else 0.0)
    conf = int(max(0, min(100, round(conf))))

    fr = funding(sym); frtxt=""
    if fr is not None:
        if fr > 0.01: frtxt = f" | Funding:+{fr:.3f}"
        elif fr < -0.01: frtxt = f" | Funding:{fr:.3f}"

    side = "BUY" if buy else "SELL"
    arrow = "↑" if trend_up else "↓"
    return {
        "symbol": sym, "tf": tf.upper(), "side": side, "conf": conf,
        "rsi": rr, "adx": ax, "vr": vr, "trend": arrow, "last_up": last_dir_up,
        "price": float(c.iloc[-1]), "turnover": float(t.iloc[-1]), "funding": frtxt
    }

def run_scan():
    syms = get_mexc_usdt_perp_symbols(SCAN_LIMIT)
    if not syms:
        telegram("⛔ MEXC USDT perpetual sembol listesi alınamadı.")
        return

    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(analyze_one, s, tf) for s in syms for tf in TF_LIST]
        for f in as_completed(futs):
            try:
                r = f.result()
                if r: results.append(r)
            except: pass

    buys  = [x for x in results if x["side"]=="BUY"]
    sells = [x for x in results if x["side"]=="SELL"]

    lines = [
        f"⚡ *MEXC USDT Perp Sinyal Taraması*",
        f"⏱ {ts()}",
        f"🧪 Taranan market: {len(syms)} | Zaman dilimi: {', '.join([t.upper() for t in TF_LIST])}",
        f"🟢 BUY: {len(buys)}  |  🔴 SELL: {len(sells)}",
    ]

    if buys:
        lines.append("\n🟢 *BUY — En yüksek güven (ilk 12)*")
        for x in sorted(buys, key=lambda z: z["conf"], reverse=True)[:12]:
            lines.append(f"- {x['symbol']} | {x['tf']} | Güven:{x['conf']} | RSI:{x['rsi']:.1f} | ADX:{x['adx']:.0f} | VolEMA x{max(0.0,x['vr']):.2f} | {x['trend']} | {x['price']:.6g}{x['funding']}")

    if sells:
        lines.append("\n🔴 *SELL — En yüksek güven (ilk 12)*")
        for x in sorted(sells, key=lambda z: z["conf"], reverse=True)[:12]:
            lines.append(f"- {x['symbol']} | {x['tf']} | Güven:{x['conf']} | RSI:{x['rsi']:.1f} | ADX:{x['adx']:.0f} | Break2✅ | {x['trend']} | {x['price']:.6g}{x['funding']}")

    if not buys and not sells:
        lines.append("\nℹ️ Şu an kriterlere uyan sinyal yok.")

    lines.append(f"\n⏳ Süre: {int(time.time()-start)} sn")
    telegram("\n".join(lines))

if __name__ == "__main__":
    run_scan()
