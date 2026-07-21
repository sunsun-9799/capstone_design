import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. 데이터 로드
# =========================
print("=== 1. 데이터 로드 ===")
df = pd.read_csv(r'C:\Users\gandi\capde\merged_cleaned.csv', low_memory=False)
print(f"shape: {df.shape}")
print()

# =========================
# 2. 날짜/시간 변환
# =========================
df['budat'] = pd.to_datetime(df['budat'], errors='coerce')
df['bldat'] = pd.to_datetime(df['bldat'], errors='coerce')
df['cpudt'] = pd.to_datetime(df['cpudt'], errors='coerce')
df['cputm'] = pd.to_datetime(df['cputm'], format='%H:%M:%S', errors='coerce')
df['hour']  = df['cputm'].dt.hour

# =========================
# 3. 시간 기반 피처
# =========================
print("=== 3. 시간 기반 피처 ===")

df['is_off_hours']  = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
df['is_weekend']    = (df['budat'].dt.dayofweek >= 5).astype(int)
df['is_month_end']  = (df['budat'].dt.month.isin([12, 3])).astype(int)
df['date_diff']     = (df['budat'] - df['bldat']).dt.days

# 사용자별 평균 전기 시각 및 이탈 정도
user_hour_stats = df.groupby('usnam')['hour'].agg(['mean', 'std'])\
                    .rename(columns={'mean': 'user_avg_hour', 'std': 'user_std_hour'})
df = df.join(user_hour_stats, on='usnam')
df['user_std_hour']  = df['user_std_hour'].fillna(1)
df['hour_deviation'] = (df['hour'] - df['user_avg_hour']).abs() / df['user_std_hour'].clip(lower=1)

print("완료")
print()

# =========================
# 4. 금액 기반 피처
# =========================
print("=== 4. 금액 기반 피처 ===")

df['log_dmbtr']       = np.log1p(df['dmbtr'].clip(lower=0))
df['is_round_amount'] = ((df['dmbtr'] % 1000 == 0) & (df['dmbtr'] > 0)).astype(int)

# 문서유형(blart) 내 금액 z-score
blart_stats = df.groupby('blart')['dmbtr'].agg(['mean', 'std'])\
                .rename(columns={'mean': 'blart_amt_mean', 'std': 'blart_amt_std'})
df = df.join(blart_stats, on='blart')
df['blart_amt_std']          = df['blart_amt_std'].fillna(1)
df['amount_zscore_by_blart'] = (df['dmbtr'] - df['blart_amt_mean']) / df['blart_amt_std'].clip(lower=1)

# 현지통화 vs 문서통화 차이
df['dmbtr_wrbtr_diff'] = (df['dmbtr'] - df['wrbtr']).abs()
df['has_currency_gap'] = (df['dmbtr_wrbtr_diff'] > 0.01).astype(int)

print("완료")
print()

# =========================
# 5. 관계 기반 피처
# =========================
print("=== 5. 관계 기반 피처 ===")

# 벤더별 거래 건수 및 평균 금액
vendor_stats = df.groupby('lifnr').agg(
    vendor_tx_count=('dmbtr', 'count'),
    vendor_amt_mean=('dmbtr', 'mean')
).reset_index()
df = df.merge(vendor_stats, on='lifnr', how='left')

# 사용자-tcode 조합 빈도
user_tcode_freq = df.groupby(['usnam', 'tcode']).size().reset_index(name='user_tcode_freq')
df = df.merge(user_tcode_freq, on=['usnam', 'tcode'], how='left')

# glvor-blart 조합 빈도 및 희귀 여부
combo_freq = df.groupby(['glvor', 'blart']).size().reset_index(name='glvor_blart_freq')
df = df.merge(combo_freq, on=['glvor', 'blart'], how='left')
df['is_rare_combo'] = (df['glvor_blart_freq'] < 10).astype(int)

# 문서 내 차변/대변 균형
df['signed_amt'] = df.apply(lambda x: x['dmbtr'] if x['shkzg'] == 'S' else -x['dmbtr'], axis=1)
doc_balance = df.groupby(['bukrs', 'belnr', 'gjahr'])['signed_amt'].sum()\
                .reset_index(name='doc_balance')
df = df.merge(doc_balance, on=['bukrs', 'belnr', 'gjahr'], how='left')
df['is_unbalanced'] = (df['doc_balance'].abs() > 0.01).astype(int)

print("완료")
print()

# =========================
# 6. 중간 컬럼 정리
# =========================
drop_cols = ['user_avg_hour', 'user_std_hour', 'blart_amt_mean', 'blart_amt_std', 'signed_amt']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f"=== Feature Engineering 완료 ===")
print(f"최종 shape: {df.shape}")
print()

# =========================
# 7. 저장
# =========================
df.to_csv(r'C:\Users\gandi\capde\featured.csv', index=False)
print("저장 완료: featured.csv")
print("다음 단계: lfa1_ska1_join.py 실행")
