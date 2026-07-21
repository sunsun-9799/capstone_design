import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

import os
from matplotlib import font_manager, rcParams

# =========================
# 한글 폰트 설정
# =========================
def set_korean_font():
    font_candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rcParams["font.family"] = font_name
            rcParams["axes.unicode_minus"] = False
            print(f"한글 폰트 적용: {font_name}")
            return font_name
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic"]:
        if font_name in installed:
            rcParams["font.family"] = font_name
            rcParams["axes.unicode_minus"] = False
            print(f"한글 폰트 적용: {font_name}")
            return font_name
    return None

set_korean_font()

# =========================
# 설정
# =========================
DATA_PATH  = r'C:\Users\gandi\capde\ae_scored.csv'
OUTPUT_DIR = r'C:\Users\gandi\capde'
PERCENTILE = 99          # ← 95버전과의 유일한 차이 (95 → 99)
VERSION    = 'v99'       # 파일명 suffix

# =========================
# 1. 데이터 로드 및 임계값 재계산
# =========================
print("=== 1. 데이터 로드 ===")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='utf-8-sig')

# 99%ile 임계값으로 ae_pred 재계산
threshold = df['ae_error'].quantile(PERCENTILE / 100)
df['ae_pred'] = (df['ae_error'] > threshold).astype(int)

print(f"전체: {len(df):,}건")
print(f"임계값 ({PERCENTILE}%ile): {threshold:.4f}")
print(f"탐지된 이상 의심 거래: {df['ae_pred'].sum():,}건 ({df['ae_pred'].mean()*100:.1f}%)")
print()

detected = df[df['ae_pred'] == 1].copy()

# =========================
# 벤더 표시값 정리
# - lifnr 결측은 자재/G/L 거래에서 정상적으로 발생할 수 있으므로
#   벤더 이상 신호로 보지 않고 별도 거래유형으로 분류합니다.
# - 벤더 집중도 그래프에서는 실제 벤더가 존재하는 거래만 사용합니다.
# =========================
def normalize_vendor(v):
    if pd.isna(v):
        return np.nan
    text = str(v).strip()
    if text == '' or text.lower() in ['nan', 'none', 'null', '<na>']:
        return np.nan
    text = text.split('.')[0] if text.endswith('.0') else text
    text = text.lstrip('0')
    return text if text else np.nan

detected['vendor_clean'] = detected['lifnr'].apply(normalize_vendor) if 'lifnr' in detected.columns else np.nan
detected['vendor_group'] = np.where(detected['vendor_clean'].notna(), '벤더 거래', '벤더 없는 거래 유형')

# koart가 있으면 벤더 없는 거래의 성격을 함께 확인할 수 있도록 요약합니다.
if 'koart' in detected.columns:
    no_vendor_koart_summary = detected.loc[detected['vendor_clean'].isna(), 'koart'].value_counts(dropna=False)
else:
    no_vendor_koart_summary = pd.Series(dtype='int64')

# =========================
# 2. Rule-based 설명 (v95와 동일)
# =========================
print("=== 2. Rule-based 설명 생성 ===")

def risk_score(row):
    score = 0
    error_pct = min(row['ae_error'] / df['ae_error'].quantile(0.999), 1.0)
    score += error_pct * 50
    if row['is_rare_combo'] == 1:           score += 20
    if abs(row.get('amount_zscore_by_blart', 0)) > 5:  score += 15
    elif abs(row.get('amount_zscore_by_blart', 0)) > 3: score += 8
    if row.get('has_currency_gap', 0) == 1: score += 7
    if row.get('is_round_amount', 0) == 1:  score += 5
    if row.get('hour_deviation', 0) > 3:    score += 5
    if row.get('date_diff', 0) < -30:       score += 10
    return min(round(score), 100)

def explain_anomaly_v2(row):
    reasons = []
    actions = []
    if row.get('is_rare_combo', 0) == 1:
        reasons.append("업무 프로세스와 문서유형 조합이 전례 없음")
        actions.append("해당 전표의 승인 권한 및 프로세스 경로 확인 필요")
    if row.get('date_diff', 0) < -30:
        reasons.append(f"문서일자가 전기일보다 {abs(int(row['date_diff']))}일 앞선 역전 발생")
        actions.append("회계 기간 소급 전기 여부 확인 필요 (backdating 의심)")
    zscore = abs(row.get('amount_zscore_by_blart', 0))
    if zscore > 5:
        reasons.append(f"동일 문서유형 내 금액이 평균 대비 극단적 이상 (z={zscore:.1f})")
        actions.append("유사 거래 대비 금액 적정성 검토 필요")
    elif zscore > 3:
        reasons.append(f"동일 문서유형 내 금액이 평균 대비 유의미한 이상 (z={zscore:.1f})")
        actions.append("유사 거래 대비 금액 적정성 검토 필요")
    if row.get('has_currency_gap', 0) == 1:
        gap = row.get('dmbtr_wrbtr_diff', 0)
        reasons.append(f"현지통화와 문서통화 금액 차이 발생 (차이: {gap:,.0f})")
        actions.append("환율 적용 기준 및 환차손익 처리 적정성 확인 필요")
    if row.get('is_round_amount', 0) == 1:
        reasons.append("금액이 1,000 단위 정수 (비정상 반올림 의심)")
        actions.append("실제 청구서/계약서 금액과 대조 필요")
    if row.get('hour_deviation', 0) > 3:
        reasons.append(f"해당 사용자의 평소 전기 패턴에서 {row['hour_deviation']:.1f}σ 이탈")
        actions.append("해당 사용자의 접근 권한 및 해당 시간대 업무 필요성 확인 필요")
    if not reasons:
        reasons.append("다중 피처 복합 패턴 — 원본 전표 검토 권고")
        actions.append("전표 전체 라인 및 관련 문서(PO, 청구서) 종합 검토 필요")
    return " | ".join(reasons), " / ".join(actions)

detected[['reason', 'action']] = detected.apply(
    lambda r: pd.Series(explain_anomaly_v2(r)), axis=1
)
detected['risk_score'] = detected.apply(risk_score, axis=1)

def risk_level(score):
    if score >= 70: return '고위험'
    elif score >= 40: return '중위험'
    else: return '저위험'

detected['risk_level'] = detected['risk_score'].apply(risk_level)

high_risk = detected[detected['risk_score'] >= 70]
mid_risk  = detected[detected['risk_score'].between(40, 69)]
low_risk  = detected[detected['risk_score'] < 40]

print(f"고위험: {len(high_risk):,}건")
print(f"중위험: {len(mid_risk):,}건")
print(f"저위험: {len(low_risk):,}건")
print()

# =========================
# 3. 감사인용 시각화 보고서
# =========================
print("=== 3. 감사인용 시각화 보고서 생성 ===")

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#F8F9FA')

fig.text(0.5, 0.97, 'Journal Entry Test — 이상 거래 탐지 감사 보고서',
         ha='center', fontsize=20, fontweight='bold', color='#1E3A5F')
fig.text(0.5, 0.955,
         f'분석 대상: {len(df):,}건  |  이상 의심: {len(detected):,}건 ({len(detected)/len(df)*100:.1f}%)  |  임계값: {PERCENTILE}%ile ({threshold:.4f})',
         ha='center', fontsize=13, color='#6B7280')

border = plt.Rectangle((0.01, 0.01), 0.98, 0.96,
    fill=False, edgecolor='#CBD5E1', linewidth=1.5, transform=fig.transFigure)
fig.add_artist(border)

# (1) 위험도 등급 파이차트
ax1 = fig.add_axes([0.05, 0.75, 0.25, 0.17])
risk_counts = detected['risk_level'].value_counts()
colors_pie = {'고위험': '#EF4444', '중위험': '#F59E0B', '저위험': '#10B981'}
pie_colors = [colors_pie.get(r, '#6B7280') for r in risk_counts.index]
wedges, texts, autotexts = ax1.pie(
    risk_counts.values, labels=list(risk_counts.index), colors=pie_colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(10); at.set_color('white'); at.set_fontweight('bold')
ax1.set_title('위험도 등급 분포', fontsize=13, fontweight='bold', pad=10, color='#1E3A5F')

# (2) 탐지 이유 분포
ax2 = fig.add_axes([0.38, 0.75, 0.56, 0.17])
all_reasons = []
for r in detected['reason']:
    all_reasons.extend(str(r).split(' | '))
reason_counter = Counter(all_reasons)
reason_labels_map = {
    '업무 프로세스와 문서유형 조합이 전례 없음': 'glvor-blart 조합 이상',
    '다중 피처 복합 패턴 — 원본 전표 검토 권고': '다중 피처 복합 패턴',
    '현지통화와 문서통화 금액 차이 발생': '통화 금액 불일치',
    '금액이 1,000 단위 정수 (비정상 반올림 의심)': '반올림 금액 이상',
    '해당 사용자의 평소 전기 패턴에서': '사용자 패턴 이탈',
}
top_reasons, top_counts = [], []
for reason, count in reason_counter.most_common(7):
    short = next((v for k, v in reason_labels_map.items() if k in reason), reason[:25])
    top_reasons.append(short); top_counts.append(count)
bar_colors = ['#EF4444' if '조합' in r or '역전' in r
              else '#F59E0B' if '금액' in r or '통화' in r
              else '#3B82F6' for r in top_reasons]
bars = ax2.barh(top_reasons[::-1], top_counts[::-1], color=bar_colors[::-1], alpha=0.85)
ax2.set_xlabel('건수', fontsize=11)
ax2.set_title('탐지 이유별 분포', fontsize=13, fontweight='bold', color='#1E3A5F')
for bar, cnt in zip(bars, top_counts[::-1]):
    ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f'{cnt:,}건', va='center', fontsize=10)
ax2.set_xlim(0, max(top_counts) * 1.2)

# (3) 재구성 오차 분포
ax3 = fig.add_axes([0.05, 0.54, 0.40, 0.17])
err_clip = df['ae_error'].clip(upper=df['ae_error'].quantile(0.995))
ax3.hist(err_clip[df['ae_pred']==0], bins=100, color='#3B82F6', alpha=0.6,
         label='정상 거래', density=True)
ax3.hist(err_clip[df['ae_pred']==1], bins=100, color='#EF4444', alpha=0.7,
         label='이상 의심', density=True)
ax3.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
            label=f'임계값 ({threshold:.3f})')
ax3.set_xlabel('재구성 오차 (MSE)', fontsize=11)
ax3.set_ylabel('밀도', fontsize=11)
ax3.set_title('재구성 오차 분포', fontsize=13, fontweight='bold', color='#1E3A5F')
ax3.legend(fontsize=10)

# (4) 벤더별 집중도
ax4 = fig.add_axes([0.55, 0.54, 0.40, 0.17])
# lifnr 결측은 벤더 이상 신호가 아니라 벤더가 없는 거래 유형으로 처리하고,
# 실제 벤더가 존재하는 거래만 그래프에 표시합니다.
vendor_counts = detected.loc[detected['vendor_clean'].notna(), 'vendor_clean'].value_counts().head(8)
if len(vendor_counts) > 0:
    ax4.bar(range(len(vendor_counts)), vendor_counts.values, color='#F59E0B', alpha=0.85)
    ax4.set_xticks(range(len(vendor_counts)))
    ax4.set_xticklabels([f'벤더\n{str(v)[:8]}' for v in vendor_counts.index], fontsize=9)
    ax4.set_ylabel('이상 의심 건수', fontsize=11)
    ax4.set_title('벤더별 이상 거래 집중도 (실제 벤더 Top 8)', fontsize=13, fontweight='bold', color='#1E3A5F')
    for i, cnt in enumerate(vendor_counts.values):
        ax4.text(i, cnt + max(vendor_counts.values)*0.02, f'{cnt:,}', ha='center', fontsize=9)
else:
    ax4.text(0.5, 0.55, '실제 벤더가 존재하는 이상 의심 거래가 없습니다.',
             ha='center', va='center', fontsize=12, color='#6B7280', transform=ax4.transAxes)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title('벤더별 이상 거래 집중도 (실제 벤더 Top 8)', fontsize=13, fontweight='bold', color='#1E3A5F')
no_vendor_cnt = int(detected['vendor_clean'].isna().sum())
ax4.text(0.01, -0.20, f'※ lifnr 결측 {no_vendor_cnt:,}건은 자재/G/L 등 벤더 없는 거래 유형으로 분류하여 제외',
         transform=ax4.transAxes, fontsize=9, color='#6B7280')

# (5) Top 20 테이블
ax5 = fig.add_axes([0.05, 0.27, 0.90, 0.23])
ax5.axis('off')
ax5.set_title('즉시 검토 필요 문서 Top 20 (문서번호 중복 제거 · 위험도 기준)',
              fontsize=13, fontweight='bold', color='#1E3A5F', pad=10, loc='left')
# BSEG는 하나의 문서가 여러 라인으로 구성되므로 감사 검토 단위에 맞춰
# 문서번호 기준으로 중복 제거하고, 탐지된 라인 수를 함께 표시합니다.
doc_top = (detected
    .sort_values(['risk_score', 'ae_error'], ascending=False)
    .groupby('belnr', as_index=False)
    .agg(usnam=('usnam', 'first'),
         blart=('blart', 'first'),
         dmbtr=('dmbtr', 'sum'),
         ae_error=('ae_error', 'max'),
         risk_score=('risk_score', 'max'),
         risk_level=('risk_level', 'first'),
         reason=('reason', 'first'),
         line_count=('belnr', 'size'))
    .sort_values(['risk_score', 'ae_error'], ascending=False)
    .head(20)
    .reset_index(drop=True))
top20 = doc_top
col_headers = ['순위','문서번호','라인수','사용자','유형','금액합계','최대오차','위험도','탐지 이유']
col_widths   = [0.03,  0.10,    0.05,   0.06,  0.05, 0.09,    0.07,    0.08,   0.47]
y_header = 0.95
x = 0.0
for h, w in zip(col_headers, col_widths):
    ax5.text(x + w/2, y_header, h, ha='center', va='center',
             fontsize=9.5, fontweight='bold', color='white', transform=ax5.transAxes)
    ax5.add_patch(plt.Rectangle((x, y_header-0.06), w, 0.1,
        transform=ax5.transAxes, color='#1E3A5F', zorder=2))
    x += w
for i, row in top20.iterrows():
    y = y_header - 0.11 - i * 0.043
    bg = '#FEF3C7' if row['risk_score'] >= 70 else '#F9FAFB' if i % 2 == 0 else 'white'
    x = 0.0
    vals = [str(i+1), str(row['belnr'])[:12], f"{int(row['line_count']):,}",
            str(row['usnam'])[:8], str(row['blart'])[:5], f"{row['dmbtr']:,.0f}",
            f"{row['ae_error']:.3f}", row['risk_level'],
            str(row['reason'])[:58] + ('...' if len(str(row['reason'])) > 58 else '')]
    for val, w in zip(vals, col_widths):
        ax5.add_patch(plt.Rectangle((x, y-0.03), w, 0.042,
            transform=ax5.transAxes, color=bg, zorder=1, linewidth=0.3, edgecolor='#E5E7EB'))
        color = '#DC2626' if val == '고위험' else '#374151'
        ax5.text(x + w/2, y, val, ha='center', va='center',
                 fontsize=8, color=color, transform=ax5.transAxes)
        x += w

# (6) 감사인 조치 사항
ax6 = fig.add_axes([0.05, 0.05, 0.90, 0.18])
ax6.axis('off')
ax6.set_title('감사인 조치 사항 요약',
              fontsize=13, fontweight='bold', color='#1E3A5F', pad=10, loc='left')
summary_items = [
    ('즉시 조치 필요', f"{len(high_risk):,}건",
     f"재구성 오차 및 위험 피처가 복합적으로 높은 거래입니다.\n"
     f"특히 glvor-blart 조합 불일치({detected['is_rare_combo'].sum()}건)와 날짜 역전 거래는\n"
     f"승인 경로 및 소급 전기 여부를 즉시 확인하십시오.",
     '#FEE2E2', '#DC2626'),
    ('우선 검토 권고', f"{len(mid_risk):,}건",
     f"금액 이상 또는 통화 불일치가 발견된 거래입니다.\n"
     f"유사 거래 금액 대비 적정성을 검토하고\n"
     f"원천 증빙 서류(청구서, PO)와 대조하십시오.",
     '#FEF3C7', '#D97706'),
    ('표본 검토', f"{len(low_risk):,}건",
     f"명시적 단일 규칙보다 다중 피처 조합에서 이탈한 거래입니다.\n"
     f"표본 추출 후 정기 감사 주기에 포함하여\n"
     f"추세 변화 모니터링을 권고합니다.",
     '#D1FAE5', '#065F46'),
]
for (title, count, desc, bg_color, text_color), x_pos in zip(summary_items, [0.01, 0.34, 0.67]):
    ax6.add_patch(plt.Rectangle((x_pos, 0.05), 0.30, 0.88,
        transform=ax6.transAxes, color=bg_color, linewidth=1, edgecolor=text_color, zorder=1))
    ax6.text(x_pos+0.15, 0.85, title, ha='center', va='center',
             fontsize=11, fontweight='bold', color=text_color, transform=ax6.transAxes)
    ax6.text(x_pos+0.15, 0.68, count, ha='center', va='center',
             fontsize=18, fontweight='bold', color=text_color, transform=ax6.transAxes)
    for j, line in enumerate(desc.split('\n')):
        ax6.text(x_pos+0.15, 0.48-j*0.14, line, ha='center', va='center',
                 fontsize=9, color='#374151', transform=ax6.transAxes)

# 저장 (v99 suffix)
img_path = os.path.join(OUTPUT_DIR, f'audit_report_{VERSION}.png')
plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"보고서 저장 완료: audit_report_{VERSION}.png")

# =========================
# 4. 상세 결과 저장 (v99 suffix)
# =========================
print("=== 4. 상세 결과 저장 ===")

base_output_cols = ['belnr','bukrs','gjahr','usnam','blart','budat',
                    'dmbtr','lifnr','vendor_clean','vendor_group','ae_error',
                    'risk_score','risk_level','reason','action']
output_cols = [c for c in base_output_cols if c in detected.columns]
result_df = detected[output_cols].sort_values('risk_score', ascending=False)

# 문서번호 기준 요약 결과도 별도 저장합니다.
doc_result_df = (detected
    .sort_values(['risk_score', 'ae_error'], ascending=False)
    .groupby('belnr', as_index=False)
    .agg(line_count=('belnr', 'size'),
         bukrs=('bukrs', 'first'),
         gjahr=('gjahr', 'first'),
         usnam=('usnam', 'first'),
         blart=('blart', 'first'),
         budat=('budat', 'first'),
         dmbtr_sum=('dmbtr', 'sum'),
         dmbtr_max=('dmbtr', 'max'),
         vendor_group=('vendor_group', 'first'),
         ae_error_max=('ae_error', 'max'),
         risk_score_max=('risk_score', 'max'),
         risk_level=('risk_level', 'first'),
         reason=('reason', 'first'),
         action=('action', 'first'))
    .sort_values(['risk_score_max', 'ae_error_max'], ascending=False))

csv_path  = os.path.join(OUTPUT_DIR, f'ae_detected_{VERSION}.csv')
xlsx_path = os.path.join(OUTPUT_DIR, f'ae_detected_{VERSION}.xlsx')
doc_csv_path  = os.path.join(OUTPUT_DIR, f'ae_detected_doc_summary_{VERSION}.csv')
doc_xlsx_path = os.path.join(OUTPUT_DIR, f'ae_detected_doc_summary_{VERSION}.xlsx')

result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
doc_result_df.to_csv(doc_csv_path, index=False, encoding='utf-8-sig')
try:
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='line_level', index=False)
        doc_result_df.to_excel(writer, sheet_name='document_summary', index=False)
    doc_result_df.to_excel(doc_xlsx_path, index=False, engine='openpyxl')
    print(f"저장 완료: ae_detected_{VERSION}.xlsx")
    print(f"저장 완료: ae_detected_doc_summary_{VERSION}.xlsx")
except Exception as e:
    print(f"xlsx 저장 실패: {e}")
print(f"저장 완료: ae_detected_{VERSION}.csv")
print(f"저장 완료: ae_detected_doc_summary_{VERSION}.csv")
print()
print("=== 최종 요약 ===")
print(f"전체 분석 거래:  {len(df):,}건")
print(f"이상 의심 탐지:  {len(detected):,}건 ({len(detected)/len(df)*100:.1f}%)")
print(f"  고위험 (즉시 조치): {len(high_risk):,}건")
print(f"  중위험 (우선 검토): {len(mid_risk):,}건")
print(f"  저위험 (표본 검토): {len(low_risk):,}건")
print(f"벤더 없는 거래 유형: {detected['vendor_clean'].isna().sum():,}건")
print(f"문서번호 기준 이상 의심 문서: {detected['belnr'].nunique():,}건")
