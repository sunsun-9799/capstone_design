import pandas as pd

# =========================
# 함수 정의
# =========================
def load_data():
    bkpf = pd.read_csv('bkpf.csv')
    bseg = pd.read_csv('bseg.csv')
    return bkpf, bseg


def merge_data(bkpf, bseg):
    bkpf = bkpf.drop_duplicates(subset=['bukrs', 'belnr', 'gjahr'])
    return pd.merge(
        bkpf,
        bseg,
        on=['bukrs', 'belnr', 'gjahr'],
        how='inner'
    )


def select_columns(df):
    keep_cols = [
        'bukrs', 'belnr', 'gjahr',
        'blart', 'bldat', 'budat',
        'cpudt', 'cputm',
        'usnam', 'tcode',
        'waers', 'monat', 'glvor',
        'buzei', 'bschl', 'shkzg', 'koart',
        'hkont', 'dmbtr', 'wrbtr',
        'lifnr', 'kunnr', 'matnr',
        'menge', 'vorgn', 'ktosl',
        'ebeln', 'ebelp'
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    return df[existing_cols]


def save_data(df):
    df.to_csv('merged_cleaned.csv', index=False)


# =========================
# 실행 영역
# =========================
if __name__ == "__main__":
    bkpf, bseg = load_data()
    merged = merge_data(bkpf, bseg)
    cleaned = select_columns(merged)
    save_data(cleaned)

    print("완료!")