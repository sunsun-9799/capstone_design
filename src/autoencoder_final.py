import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 1. 데이터 로드
# =========================
print("=== 1. 데이터 로드 ===")
df = pd.read_csv(r'C:\Users\gandi\capde\featured_v2.csv', low_memory=False)
print(f"전체: {len(df):,}건")
print()

# =========================
# 2. 피처 선택 및 전처리
# =========================
print("=== 2. 전처리 ===")

NUMERIC_FEATURES = [
    'hour', 'is_off_hours', 'is_weekend', 'is_month_end', 'date_diff',
    'hour_deviation', 'log_dmbtr', 'is_round_amount', 'amount_zscore_by_blart',
    'dmbtr_wrbtr_diff', 'has_currency_gap', 'vendor_tx_count', 'vendor_amt_mean',
    'user_tcode_freq', 'glvor_blart_freq', 'is_rare_combo', 'doc_balance',
    'is_unbalanced', 'is_balance_sheet_acct', 'monat', 'bschl', 'buzei',
    'menge', 'ebeln'
]

CATEGORICAL_FEATURES = [
    'blart', 'usnam', 'tcode', 'waers', 'glvor',
    'shkzg', 'koart', 'vorgn', 'ktosl', 'ktoks', 'vendor_country'
]

# 범주형 Label Encoding
le_dict = {}
for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype(str).fillna('UNKNOWN')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 수치형 결측 → -1
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = df[ALL_FEATURES].values.astype(np.float32)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"피처 수: {X_scaled.shape[1]}개")
print(f"전처리 완료: {X_scaled.shape}")
print()

# =========================
# 3. Autoencoder 모델 정의
# =========================
print("=== 3. Autoencoder 모델 구성 ===")

input_dim = X_scaled.shape[1]

def build_autoencoder(input_dim):
    inputs = keras.Input(shape=(input_dim,))

    # Encoder: 입력 압축
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(16, activation='relu', name='latent')(x)

    # Decoder: 복원
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation='linear', name='output')(x)

    autoencoder = keras.Model(inputs, decoded, name='autoencoder')
    return autoencoder

autoencoder = build_autoencoder(input_dim)
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)
autoencoder.summary()
print()

# =========================
# 4. 학습 (전체 데이터)
# =========================
print("=== 4. 학습 ===")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=2048,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print()

# =========================
# 5. 재구성 오차 계산
# =========================
print("=== 5. 재구성 오차 계산 ===")

X_reconstructed = autoencoder.predict(X_scaled, batch_size=2048, verbose=0)
reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
df['ae_error'] = reconstruction_errors

print(f"평균 재구성 오차: {reconstruction_errors.mean():.6f}")
print(f"최대 재구성 오차: {reconstruction_errors.max():.6f}")
print(f"95%ile 오차:     {np.percentile(reconstruction_errors, 95):.6f}")
print(f"99%ile 오차:     {np.percentile(reconstruction_errors, 99):.6f}")
print()

# =========================
# 6. 임계값 설정 및 이상치 탐지
# =========================
print("=== 6. 이상치 탐지 ===")

# 전체 오차의 95%ile을 임계값으로 설정
# → 상위 5%를 이상 의심 거래로 분류
threshold = np.percentile(reconstruction_errors, 95)
df['ae_pred'] = (reconstruction_errors > threshold).astype(int)

detected = df[df['ae_pred'] == 1]
print(f"임계값 (95%ile): {threshold:.6f}")
print(f"이상 의심 거래: {len(detected):,}건 (전체의 {len(detected)/len(df)*100:.1f}%)")
print()

# =========================
# 7. Rule-based 설명
# =========================
print("=== 7. 탐지 거래 Rule-based 설명 ===")

def explain_anomaly(row):
    reasons = []
    if row['is_off_hours'] == 1:
        reasons.append(f"업무시간 외 전기 ({int(row['hour'])}시)")
    if row['hour_deviation'] > 3:
        reasons.append(f"사용자 평균 시각 대비 {row['hour_deviation']:.1f}σ 이탈")
    if row['is_weekend'] == 1:
        reasons.append("주말 전기")
    if row['is_month_end'] == 1:
        reasons.append("결산월 전기")
    if abs(row['amount_zscore_by_blart']) > 3:
        reasons.append(f"문서유형 내 금액 이상 (z={row['amount_zscore_by_blart']:.1f})")
    if row['is_round_amount'] == 1:
        reasons.append("비정상 반올림 금액")
    if row['has_currency_gap'] == 1:
        reasons.append("현지/문서 통화 금액 불일치")
    if row['is_rare_combo'] == 1:
        reasons.append("비정상 glvor-blart 조합")
    if row['date_diff'] < -30:
        reasons.append(f"날짜 역전 ({int(row['date_diff'])}일)")
    if not reasons:
        reasons.append("복합 패턴 이상 (단일 규칙으로 설명 불가)")
    return " | ".join(reasons)

detected_df = df[df['ae_pred'] == 1].copy()
detected_df['explanation'] = detected_df.apply(explain_anomaly, axis=1)

# 설명 비율 통계
explained = (detected_df['explanation'] != "복합 패턴 이상 (단일 규칙으로 설명 불가)").sum()
print(f"규칙으로 설명 가능: {explained:,}건 ({explained/len(detected_df)*100:.1f}%)")
print(f"복합 패턴 (설명 어려움): {len(detected_df)-explained:,}건")
print()

# 설명 유형별 빈도
from collections import Counter
all_reasons = []
for exp in detected_df['explanation']:
    all_reasons.extend(exp.split(' | '))
reason_counts = Counter(all_reasons)
print("탐지 이유 빈도:")
for reason, count in reason_counts.most_common():
    print(f"  {reason}: {count:,}건")
print()

# 상위 10건 샘플 출력
print("탐지 거래 샘플 (재구성 오차 상위 10건):")
sample = detected_df.nlargest(10, 'ae_error')[
    ['belnr', 'usnam', 'blart', 'dmbtr', 'ae_error', 'explanation']
]
for _, row in sample.iterrows():
    print(f"  문서번호: {row['belnr']}  사용자: {row['usnam']}  금액: {row['dmbtr']:,.0f}")
    print(f"  재구성오차: {row['ae_error']:.6f}")
    print(f"  탐지이유: {row['explanation']}")
    print()

# =========================
# 8. 시각화
# =========================
print("=== 8. 시각화 저장 ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Autoencoder — Anomaly Detection Results', fontsize=15, fontweight='bold')

# (1) 학습 곡선
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Train Loss', color='#2563EB')
ax1.plot(history.history['val_loss'], label='Val Loss', color='#EF4444', linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.legend()

# (2) 재구성 오차 분포
ax2 = axes[1]
ax2.hist(reconstruction_errors, bins=200, color='#2563EB', alpha=0.7, density=True)
ax2.axvline(threshold, color='#EF4444', linestyle='--', linewidth=2,
            label=f'Threshold (95%ile)\n= {threshold:.4f}')
ax2.set_xlabel('Reconstruction Error (MSE)')
ax2.set_ylabel('Density')
ax2.set_title('Reconstruction Error Distribution', fontweight='bold')
ax2.set_xlim(0, np.percentile(reconstruction_errors, 99.5))
ax2.legend()

# (3) 탐지 이유 빈도
ax3 = axes[2]
top_reasons = reason_counts.most_common(7)
labels = [r[0][:20] for r in top_reasons]
counts = [r[1] for r in top_reasons]
bars = ax3.barh(labels, counts, color='#2563EB', alpha=0.85)
ax3.set_xlabel('Count')
ax3.set_title('Top Detection Reasons', fontweight='bold')
for bar, count in zip(bars, counts):
    ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{count:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(r'C:\Users\gandi\capde\ae_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("시각화 저장 완료: ae_results.png")
print()

# =========================
# 9. 결과 저장
# =========================
print("=== 9. 결과 저장 ===")

# 전체 스코어 저장
df.to_csv(r'C:\Users\gandi\capde\ae_scored.csv', index=False)

# 탐지된 이상 거래 저장
detected_df[['belnr', 'bukrs', 'gjahr', 'usnam', 'blart',
             'dmbtr', 'ae_error', 'explanation']].to_csv(
    r'C:\Users\gandi\capde\ae_detected.csv', index=False
)

print("저장 완료:")
print(f"  ae_scored.csv   — 전체 거래 + 재구성 오차")
print(f"  ae_detected.csv — 이상 의심 거래 + 탐지 이유")
print(f"  ae_results.png  — 시각화")
