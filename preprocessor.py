import pandas as pd
import ta
import numpy as np
import warnings

def parse_volume(value):
    """
    Mengubah kolom Volume seperti '1.2K', '3M', '1,234' menjadi angka numerik.
    """
    if isinstance(value, str):
        # 1. Bersihkan spasi di awal/akhir
        value = value.strip()
        
        # 2. HAPUS pemisah ribuan (koma)
        #    Contoh: '1,234.5K' -> '1234.5K'
        #    Contoh: '1,234' -> '1234'
        value = value.replace(',', '') 
        
        # 3. Proses K, M, B
        if value.endswith('K'):
            # float('1.2') * 1000
            return float(value[:-1]) * 1_000
        elif value.endswith('M'):
            # float('3.5') * 1000000
            return float(value[:-1]) * 1_000_000
        elif value.endswith('B'):
            return float(value[:-1]) * 1_000_000_000
        elif value == '-': # Menangani jika ada data volume '-'
            return np.nan
        else:
            # Ini untuk menangani angka biasa yang terbaca sbg string (misal '1234')
            try:
                return float(value)
            except ValueError:
                return np.nan
                
    # Kembalikan nilai jika inputnya sudah numerik (bukan string)
    return value

def DEMA(series, period=14):
    """ Menghitung Double Exponential Moving Average (DEMA) """
    ema = series.ewm(span=period, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=period, adjust=False).mean()
    return dema

def kategori_candlestick(x):
    """ Fungsi helper untuk kategori candlestick pattern """
    if x < 0.1:
        return "Sangat Rendah"
    elif x < 0.3:
        return "Rendah"
    elif x < 0.6:
        return "Sedang"
    else:
        return "Tinggi"

# --- Fungsi Pemrosesan Utama ---

def process_dataframe(df_input):
    """
    Menerapkan semua langkah preprocessing dan feature engineering pada satu DataFrame.
    """
    # Salin agar tidak mengubah dataframe asli (best practice)
    df = df_input.copy()

    df.columns = df.columns.str.strip()

    # 1. Urutkan dari Date terlama ke terbaru
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Peringatan: Gagal mengurai tanggal. Error: {e}")

    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # 2. Menyesuaikan kolom Volume
    df['Volume'] = df['Volume'].apply(parse_volume)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # 3. Indikator Teknikal
    # Pastikan tidak ada error, gunakan warning filter jika perlu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        
        df['MA']    = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
        df['EMA']   = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
        df['DEMA']  = DEMA(df['Close'], period=5)
        df['KAMA']  = ta.momentum.KAMAIndicator(df['Close'], window=5).kama()
        df['SMA']   = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
        df['SAR']   = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()

        df['ADX']   = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=10).adx()
        df['APO']   = df['Close'] - df['Close'].ewm(span=10, adjust=False).mean()
        df['BOP']   = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, 1e-10)
        df['CCI']   = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=10).cci()
        
        macd_custom = ta.trend.MACD(
            close=df['Close'], window_slow=10, window_fast=5, window_sign=9
        )
        df['MACD'] = macd_custom.macd()
        
        df['MFI']   = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=10).money_flow_index()
        df['MOM'] = df['Close'] - df['Close'].shift(10)
        df['RSI']   = ta.momentum.RSIIndicator(df['Close'], window=10).rsi()

        df['AD']    = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
        df['ADOSC'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=10).chaikin_money_flow()
        df['OBV']   = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        df['Prev_Close'] = df['Close'].shift(1)
        df['TRANGE'] = df[['High', 'Low', 'Prev_Close']].apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Prev_Close']),
                abs(row['Low'] - row['Prev_Close'])
            ), axis=1
        )
        df['ATR'] = df['TRANGE'].ewm(span=10, adjust=False).mean()
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        df.drop(columns=['Prev_Close'], inplace=True, errors='ignore')

    # 4. Candlestick Pattern
    df['Body'] = (df['Close'] - df['Open']).abs()
    df['Range'] = df['High'] - df['Low']
    df['Upper_Shadow'] = df['High'] - df[['Close','Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close','Open']].min(axis=1) - df['Low']

    df['P_Body'] = (df['Body'] / df['Range'].replace(0, np.nan)).fillna(0)
    df['P_Upper'] = (df['Upper_Shadow'] / df['Range'].replace(0, np.nan)).fillna(0)
    df['P_Lower'] = (df['Lower_Shadow'] / df['Range'].replace(0, np.nan)).fillna(0)

    df['K_Body'] = df['P_Body'].apply(kategori_candlestick)
    df['K_Upper'] = df['P_Upper'].apply(kategori_candlestick)
    df['K_Lower'] = df['P_Lower'].apply(kategori_candlestick)

    conditions_candle = [
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])) & (df['Close'] < df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])) & (df['Close'] > df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sangat Rendah', 'Rendah'])) & (df['Close'] < df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sangat Rendah', 'Rendah'])) & (df['Close'] > df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])) & (df['Close'] < df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])) & (df['Close'] > df['Open']),
        (df['K_Body'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])),
        (df['K_Body'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Upper'].isin(['Sedang', 'Tinggi'])) & (df['K_Lower'].isin(['Sangat Rendah', 'Rendah'])),
        (df['K_Body'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Upper'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Lower'].isin(['Sedang', 'Tinggi'])),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Lower'].isin(['Sangat Rendah', 'Rendah'])) & (df['Close'] < df['Open']),
        (df['K_Body'].isin(['Sedang', 'Tinggi'])) & (df['K_Upper'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Lower'].isin(['Sangat Rendah', 'Rendah'])) & (df['Close'] > df['Open']),
        (df['K_Body'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Upper'] == 'Sangat Rendah') & (df['K_Lower'] == 'Sangat Rendah') & (df['Close'] <= df['Open']),
        (df['K_Body'].isin(['Sangat Rendah', 'Rendah'])) & (df['K_Upper'] == 'Sangat Rendah') & (df['K_Lower'] == 'Sangat Rendah') & (df['Close'] >= df['Open'])
    ]
    choices_candle = [
        'Spinning Top Bearish', 'Spinning Top Bullish', 'Shooting Star', 'Inverted Hammer',
        'Hanging Man', 'Hammer', 'Doji', 'Gravestone Doji', 'Dragonfly Doji',
        'Marubozu Bearish', 'Marubozu Bullish', 'Bearish Full Marubozu', 'Bullish Full Marubozu'
    ]
    df['CandlePattern'] = np.select(conditions_candle, choices_candle, default='Uncategorized')

    # 5. Eight-Trigram
    conditions_trigram = [
        (df['Close'] >= df['Close'].shift(1)) & (df['High'] >= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1)),
        (df['Close'] <= df['Close'].shift(1)) & (df['High'] <= df['High'].shift(1)) & (df['Low'] <= df['Low'].shift(1)),
        (df['Close'] >= df['Close'].shift(1)) & (df['High'] >= df['High'].shift(1)) & (df['Low'] <= df['Low'].shift(1)),
        (df['Close'] <= df['Close'].shift(1)) & (df['High'] >= df['High'].shift(1)) & (df['Low'] <= df['Low'].shift(1)),
        (df['Close'] >= df['Close'].shift(1)) & (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1)),
        (df['Close'] <= df['Close'].shift(1)) & (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1)),
        (df['Close'] <= df['Close'].shift(1)) & (df['High'] >= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1)),
        (df['Close'] >= df['Close'].shift(1)) & (df['High'] <= df['High'].shift(1)) & (df['Low'] <= df['Low'].shift(1))
    ]
    values_trigram = [
        "BullishHigh", "BearLow", "BullishHorn", "BearHorn",
        "BullishHarami", "BearHarami", "BearHigh", "BullishLow"
    ]
    df['Pattern'] = np.select(conditions_trigram, values_trigram, default="Uncategorized")
    df['Pattern'] = df['Pattern'].fillna("Uncategorized")

    # 6. Hapus Kolom yang Tidak Dibutuhkan
    cols_to_drop = [
        "Date", "Close", "Open", "High", "Low", "Volume","Change %",
        "Body", "Range", "Upper_Shadow", "Lower_Shadow", "P_Body", 
        "P_Upper", "P_Lower", "K_Body", "K_Upper", "K_Lower"
    ]
    # Gunakan errors='ignore' untuk menghindari error jika kolom sudah terhapus
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 7. Hapus Baris yang Mengandung Missing Value
    df = df.dropna()
    
    return df