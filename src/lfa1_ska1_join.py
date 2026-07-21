import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. 데이터 로드
# =========================
print("=== 1. 데이터 로드 ===")
df   = pd.read_csv(r'C:\Users\gandi\capde\featured.csv', low_memory=False)
lfa1 = pd.read_csv(r'C:\Users\gandi\capde\lfa1.csv', low_memory=False)
ska1 = pd.read_csv(r'C:\Users\gandi\capde\ska1.csv', low_memory=False)

print(f"featured.csv: {df.shape}")
print(f"lfa1:         {lfa1.shape}")
print(f"ska1:         {ska1.shape}")
print()

# =========================
# 2. LFA1 조인 (벤더 마스터)
# =========================
print("=== 2. LFA1 조인 ===")

lfa1_cols  = ['lifnr', 'land1', 'ktokk', 'name1', 'erdat']
lfa1_clean = lfa1[lfa1_cols].drop_duplicates(subset=['lifnr']).copy()

# lifnr 포맷 통일 (leading zero 제거)
def normalize_lifnr(s):
    try:
        return str(int(float(str(s).strip())))
    except:
        return str(s).strip()

lfa1_clean['lifnr_key'] = lfa1_clean['lifnr'].apply(normalize_lifnr)
df['lifnr_key']         = df['lifnr'].apply(normalize_lifnr)

df = df.merge(
    lfa1_clean[['lifnr_key', 'land1', 'ktokk', 'name1', 'erdat']],
    on='lifnr_key', how='left'
)

matched = df['land1'].notna().sum()
print(f"LFA1 매칭: {matched:,}건 ({matched/len(df)*100:.1f}%)")
print(f"국가(land1) 분포:")
print(df['land1'].value_counts().head(5))
print()

# =========================
# 3. SKA1 조인 (계정 마스터)
# =========================
print("=== 3. SKA1 조인 ===")

ska1_cols  = ['saknr', 'ktoks', 'gvtyp', 'xbilk']
ska1_clean = ska1[ska1_cols].drop_duplicates(subset=['saknr']).copy()

# hkont, saknr 포맷 통일 (leading zero 제거)
df['hkont_key']          = df['hkont'].astype(str).str.strip().str.lstrip('0')
ska1_clean['saknr_key']  = ska1_clean['saknr'].astype(str).str.strip().str.lstrip('0')

df = df.merge(
    ska1_clean[['saknr_key', 'ktoks', 'gvtyp', 'xbilk']],
    left_on='hkont_key', right_on='saknr_key', how='left'
)

matched2 = df['ktoks'].notna().sum()
print(f"SKA1 매칭: {matched2:,}건 ({matched2/len(df)*100:.1f}%)")
print(f"계정유형(ktoks) 분포:")
print(df['ktoks'].value_counts().head(5))
print()

# =========================
# 4. 마스터 데이터 피처 생성
# =========================
print("=== 4. 마스터 데이터 피처 생성 ===")

# 벤더 국가 (결측 → UNKNOWN)
df['vendor_country'] = df['land1'].fillna('UNKNOWN')

# 벤더 계정 그룹 (결측 → UNKNOWN)
df['vendor_group'] = df['ktokk'].fillna('UNKNOWN')

# 재무상태표 계정 여부
df['is_balance_sheet_acct'] = (df['xbilk'] == 'X').astype(int)

print(f"vendor_country 분포:")
print(df['vendor_country'].value_counts().head(5))
print(f"is_balance_sheet_acct=1: {df['is_balance_sheet_acct'].sum():,}건")
print()

# =========================
# 5. 불필요 컬럼 정리
# =========================
drop_cols = ['lifnr_key', 'hkont_key', 'saknr_key', 'saknr']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# =========================
# 6. 저장
# =========================
print(f"=== 최종 shape: {df.shape} ===")
df.to_csv(r'C:\Users\gandi\capde\featured_v2.csv', index=False)
print("저장 완료: featured_v2.csv")
print("다음 단계: inject_anomalies.py 또는 autoencoder_final.py 실행")
