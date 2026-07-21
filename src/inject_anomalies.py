import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('/home/claude/featured_v2.csv', low_memory=False)
df['budat'] = pd.to_datetime(df['budat'], errors='coerce')
df['bldat'] = pd.to_datetime(df['bldat'], errors='coerce')
df['cpudt'] = pd.to_datetime(df['cpudt'], errors='coerce')

df['is_anomaly'] = 0
df['anomaly_type'] = 'normal'

anomaly_rows = []

# =============================================
# 유형 1 — 사용자 행동 이상 (900건)
# 평소 9~13시 전기하는 BHUSHAN/MOUNIKAPALI가
# 새벽 0~4시 + 희귀 tcode + 고액 거래
# =============================================
target_users = ['BHUSHAN', 'MOUNIKAPALI']
rare_tcodes = ['FBZ1', 'MIGO_GI', 'VL09']  # 해당 사용자가 거의 안 쓰는 tcode
high_amount_threshold = df['dmbtr'].quantile(0.95)  # 상위 5% 금액 기준

base_rows = df[df['usnam'].isin(target_users)].sample(900, random_state=42).copy()

for i, row in base_rows.iterrows():
    new_row = row.copy()
    new_row['belnr'] = f'ANA1{str(i).zfill(8)}'
    new_row['hour'] = np.random.randint(0, 5)
    new_row['cputm'] = f"{new_row['hour']:02d}:{np.random.randint(0,59):02d}:00"
    new_row['tcode'] = np.random.choice(rare_tcodes)
    new_row['dmbtr'] = round(np.random.uniform(high_amount_threshold,
                                                 high_amount_threshold * 3), 2)
    new_row['log_dmbtr'] = np.log1p(new_row['dmbtr'])
    new_row['is_off_hours'] = 1
    # hour_deviation 재계산 (BHUSHAN 평균 9.3, MOUNIKAPALI 평균 11.0)
    avg = 9.3 if new_row['usnam'] == 'BHUSHAN' else 11.0
    std = 1.3 if new_row['usnam'] == 'BHUSHAN' else 1.4
    new_row['hour_deviation'] = abs(new_row['hour'] - avg) / std
    new_row['user_tcode_freq'] = 1  # 희귀 조합
    new_row['is_anomaly'] = 1
    new_row['anomaly_type'] = 'user_behavior'
    anomaly_rows.append(new_row)

print(f"유형1 생성 완료: {len(anomaly_rows)}건")

# =============================================
# 유형 2 — glvor-blart 조합 불일치 (800건)
# RMWE인데 blart=RE, 또는 RMRP인데 blart=WE
# =============================================
n_before = len(anomaly_rows)

rmwe_rows = df[df['glvor'] == 'RMWE'].sample(400, random_state=42).copy()
rmrp_rows = df[df['glvor'] == 'RMRP'].sample(400, random_state=42).copy()

for i, row in rmwe_rows.iterrows():
    new_row = row.copy()
    new_row['belnr'] = f'ANA2{str(i).zfill(8)}'
    new_row['blart'] = 'RE'  # RMWE인데 RE 문서유형 — 불일치
    new_row['tcode'] = 'MIRO'
    new_row['glvor_blart_freq'] = 0
    new_row['is_rare_combo'] = 1
    new_row['is_anomaly'] = 1
    new_row['anomaly_type'] = 'glvor_blart_mismatch'
    anomaly_rows.append(new_row)

for i, row in rmrp_rows.iterrows():
    new_row = row.copy()
    new_row['belnr'] = f'ANA3{str(i).zfill(8)}'
    new_row['blart'] = 'WE'  # RMRP인데 WE 문서유형 — 불일치
    new_row['tcode'] = 'MB01'
    new_row['glvor_blart_freq'] = 0
    new_row['is_rare_combo'] = 1
    new_row['is_anomaly'] = 1
    new_row['anomaly_type'] = 'glvor_blart_mismatch'
    anomaly_rows.append(new_row)

print(f"유형2 생성 완료: {len(anomaly_rows) - n_before}건")

# =============================================
# 유형 3 — 벤더 집중 단기 폭증 (600건)
# 벤더 916521과의 거래가 특정 2주에 집중
# =============================================
n_before = len(anomaly_rows)

target_vendor = '916521'
base_rows = df[df['lifnr'].astype(str).str.strip().str.lstrip('0') == target_vendor].sample(600, random_state=42).copy()

# 특정 2주 기간으로 날짜 집중
burst_start = pd.Timestamp('2021-06-01')
burst_end = pd.Timestamp('2021-06-14')

for i, row in base_rows.iterrows():
    new_row = row.copy()
    new_row['belnr'] = f'ANA4{str(i).zfill(8)}'
    burst_date = burst_start + pd.Timedelta(days=np.random.randint(0, 14))
    new_row['budat'] = burst_date
    new_row['bldat'] = burst_date
    new_row['cpudt'] = burst_date
    new_row['monat'] = burst_date.month
    new_row['is_anomaly'] = 1
    new_row['anomaly_type'] = 'vendor_burst'
    anomaly_rows.append(new_row)

print(f"유형3 생성 완료: {len(anomaly_rows) - n_before}건")

# =============================================
# 유형 4 — just-below-threshold (500건)
# 동일 벤더에서 99,999 / 49,999 / 9,999 반복
# =============================================
n_before = len(anomaly_rows)

thresholds = [99999, 49999, 9999]
vendor_pool = ['916521', '623106', '447487']

base_rows = df[df['lifnr'].astype(str).str.strip().str.lstrip('0').isin(vendor_pool)].sample(500, random_state=42).copy()

for i, row in base_rows.iterrows():
    new_row = row.copy()
    new_row['belnr'] = f'ANA5{str(i).zfill(8)}'
    threshold = np.random.choice(thresholds)
    # 한도 바로 아래 금액 (±50 범위)
    new_row['dmbtr'] = threshold - np.random.randint(1, 50)
    new_row['wrbtr'] = new_row['dmbtr']
    new_row['log_dmbtr'] = np.log1p(new_row['dmbtr'])
    new_row['is_round_amount'] = 0
    blart_mean = df[df['blart'] == new_row['blart']]['dmbtr'].mean()
    blart_std = df[df['blart'] == new_row['blart']]['dmbtr'].std()
    new_row['amount_zscore_by_blart'] = (new_row['dmbtr'] - blart_mean) / max(blart_std, 1)
    new_row['dmbtr_wrbtr_diff'] = 0
    new_row['has_currency_gap'] = 0
    new_row['is_anomaly'] = 1
    new_row['anomaly_type'] = 'just_below_threshold'
    anomaly_rows.append(new_row)

print(f"유형4 생성 완료: {len(anomaly_rows) - n_before}건")



# =============================================
# 최종 합치기
# =============================================
anomaly_df = pd.DataFrame(anomaly_rows)
final_df = pd.concat([df, anomaly_df], ignore_index=True)

print()
print(f'=== 주입 결과 ===')
print(f'원본: {len(df):,}건')
print(f'이상치 주입: {len(anomaly_df):,}건')
print(f'최종: {len(final_df):,}건')
print()
print('=== 유형별 분포 ===')
print(final_df['anomaly_type'].value_counts())
print()
print(f'이상치 비율: {final_df["is_anomaly"].mean()*100:.2f}%')

final_df.to_csv('/home/claude/featured_with_anomalies.csv', index=False)
print()
print('저장 완료: featured_with_anomalies.csv')
