"""
이상거래탐지 결과(ae_detected_doc_summary_v99.csv) + 회계감사기준 코퍼스(standards_corpus.json)
를 결합한 RAG 파이프라인.

흐름:
  1. 탐지된 전표(문서 단위)의 reason(룰 기반 탐지 사유)을 쿼리로 사용
  2. TF-IDF 코사인 유사도로 294개 기준서 문단 중 관련 조항 top-k 검색   <- 검색(Retrieval)
  3. 전표 데이터 + 검색된 조항을 프롬프트에 넣어 GPT 호출               <- 생성(Generation)
  4. 결과를 JSON/CSV로 저장

주의: LLM이 근거 문단에 없는 사실을 지어내지 않도록 시스템 프롬프트에서 강하게 제약함.
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드

client = OpenAI()  # 환경변수 OPENAI_API_KEY 자동 인식

MODEL = "gpt-4o-mini"          # 비용 대비 성능 좋은 모델로 시작
DOC_SUMMARY_PATH = "ae_detected_doc_summary_v99.csv"
CORPUS_PATH = "standards_corpus/standards_corpus.json"
OUTPUT_PATH = "audit_explanations_v99.json"

TOP_K = 3            # 전표 1건당 참고할 기준서 문단 개수
RISK_LEVEL_FILTER = "고위험"   # None으로 두면 전체 851건 처리, 지금은 12건만 우선 테스트

# ---------------------------------------------------------------------------
# 탐지 유형 -> 감사 실무 용어 매핑
# 룰 기반 reason은 통계 용어(z-score, 반올림 등)로 쓰여 있어 기준서 문구와 겹치지 않음.
# 각 유형을 "감사인이 실제로 쓰는 언어"로 번역한 쿼리로 바꿔서 검색 정확도를 높임.
# fallback(복합 패턴, 단일 사유 없음)은 억지로 매핑하지 않고 별도 처리한다.
# ---------------------------------------------------------------------------
REASON_TYPE_QUERIES = {
    "rare_combo":      ("조합이 전례 없음", "저널엔트리 처리 관련 부적절하거나 이례적인 활동에 대한 질의"),
    "date_reversal":    ("역전 발생",         "마감 전후 저널엔트리 검토 및 기간귀속 오류"),
    "amount_zscore":    ("평균 대비",          "유의적인 이례거래의 사업적 타당성 평가 및 뒷받침 문서 확인"),
    "currency_gap":     ("통화와 문서통화",     "거래 기록의 정확성 및 통화환산 검증"),
    "round_amount":     ("반올림 의심",         "이례적인 회계추정치 및 반올림 금액에 대한 편의 검토"),
    "hour_deviation":   ("전기 패턴에서",       "직무분리 및 승인 권한, 저널엔트리 처리 관련 이례적 활동 질의"),
}
FALLBACK_KEYWORD = "복합 패턴"


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------

def load_transactions():
    df = pd.read_csv(DOC_SUMMARY_PATH, encoding="utf-8-sig")
    if RISK_LEVEL_FILTER:
        df = df[df["risk_level"] == RISK_LEVEL_FILTER].copy()
    return df


def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)  # [{"standard": "240", "text": "..."}, ...]


# ---------------------------------------------------------------------------
# 2. 검색 (TF-IDF 기반, 임베딩 API 안 씀 -> 비용 없음, 로컬 처리)
#    한글은 형태소 분석기 없이 char n-gram으로 처리 (konlpy 등 설치 부담 없이 충분히 동작)
# ---------------------------------------------------------------------------

class StandardsRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.texts = [c["text"] for c in corpus]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=TOP_K):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {
                "standard": self.corpus[i]["standard"],
                "text": self.corpus[i]["text"],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# 3. 프롬프트 구성 + GPT 호출
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 회계법인의 감사 데이터 분석 담당자를 보조하는 어시스턴트입니다.
아래 원칙을 반드시 지키세요.

1. 오직 [전표 데이터]와 [관련 감사기준서 조항]에 있는 내용만 근거로 사용하세요.
   조항에 없는 내용을 추측하거나 지어내지 마세요 (환각 금지).
2. [관련 감사기준서 조항]은 유사도 검색으로 자동 추출된 "후보"일 뿐, 항상 실제로 관련 있다는
   보장은 없습니다. 각 조항을 읽고, 전표 상황과 실제로 논리적 연관이 있는지 스스로 판단하세요.
   - 관련이 있다고 판단되면: "감사기준서 OOO 문단 관련"처럼 출처를 명확히 밝혀 인용
   - 관련이 없거나 억지로 연결하는 것이라 판단되면: standard_basis에
     "검색된 후보 조항 중 실제로 관련성 높은 조항 없음"이라고 정직하게 기재하고,
     없는 관련성을 지어내지 마세요.
3. 확신할 수 없는 부분은 "추가 확인 필요"라고 명시하세요.
4. 출력은 반드시 아래 JSON 형식을 따르세요:
{
  "summary": "한 줄 요약",
  "reasons": ["근거1", "근거2"],
  "standard_basis": "관련 감사기준서 조항 인용(문단 수준) 또는 관련성 없음 명시",
  "recommended_action": "감사인이 확인해야 할 후속 절차",
  "risk_tag": "상/중/하"
}
"""


def build_user_prompt(row, retrieved_chunks):
    if retrieved_chunks:
        standard_refs = "\n\n".join(
            f"[감사기준서 {c['standard']}]\n{c['text'][:800]}"  # 문단이 너무 길면 앞부분만
            for c in retrieved_chunks
        )
    else:
        standard_refs = (
            "(해당 없음 — 이 전표는 개별 룰이 아니라 Autoencoder의 종합 재구성오차로 "
            "탐지되었습니다. 특정 감사기준서 조항을 억지로 연결하지 말고, "
            "standard_basis 필드에는 '특정 조항 근거 없음 — 통계적 종합판단(AE 재구성오차 상위 1%)에 "
            "따른 탐지'라고만 기재하세요.)"
        )
    return f"""[전표 데이터]
- 전표번호: {row['belnr']}
- 회사코드: {row['bukrs']}, 회계연도: {row['gjahr']}
- 작성자: {row['usnam']}, 문서유형: {row['blart']}, 전기일: {row['budat']}
- 금액 합계: {row['dmbtr_sum']}, 최대 라인 금액: {row['dmbtr_max']}
- 거래처 그룹: {row['vendor_group']}
- 탐지된 라인 수: {row['line_count']}
- 이상 점수(최대): {row['risk_score_max']} / 위험 등급: {row['risk_level']}
- 룰 기반 탐지 사유: {row['reason']}
- 기존 권고 조치: {row['action']}

[관련 감사기준서 조항 (검색 결과)]
{standard_refs}

위 정보를 바탕으로 지정된 JSON 형식으로 감사 설명을 작성하세요.
"""


def classify_reason(reason_text):
    """reason 문자열에 어떤 룰 타입들이 포함돼 있는지 식별."""
    matched = [
        rtype for rtype, (kw, _) in REASON_TYPE_QUERIES.items() if kw in reason_text
    ]
    return matched


def build_retrieval_query(reason_text):
    """
    개별 전표의 reason을 감사 실무 용어 쿼리로 변환.
    매칭되는 룰 타입이 하나도 없으면(fallback) None을 반환 -> 검색 자체를 생략.
    """
    matched_types = classify_reason(reason_text)
    if not matched_types and FALLBACK_KEYWORD in reason_text:
        return None  # fallback: 근거 조항 검색하지 않음
    if not matched_types:
        return None  # 예상 밖 패턴 - 안전하게 검색 생략
    queries = [REASON_TYPE_QUERIES[t][1] for t in matched_types]
    return " ".join(queries)


def generate_explanation(row, retriever):
    reason_text = str(row["reason"])
    retrieval_query = build_retrieval_query(reason_text)

    if retrieval_query is None:
        # fallback 케이스: 근거 조항을 억지로 찾지 않고, 그 사실 자체를 프롬프트에 명시
        retrieved = []
    else:
        retrieved = retriever.search(retrieval_query)

    user_prompt = build_user_prompt(row, retrieved)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # 감사 설명이라 창의성보다 일관성 우선
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_output": content, "parse_error": True}

    return {
        "belnr": row["belnr"],
        "risk_level": row["risk_level"],
        "retrieved_standards": [
            {"standard": c["standard"], "score": round(c["score"], 4)} for c in retrieved
        ],
        "llm_explanation": parsed,
    }


# ---------------------------------------------------------------------------
# 4. 메인 실행
# ---------------------------------------------------------------------------

def main():
    transactions = load_transactions()
    corpus = load_corpus()
    retriever = StandardsRetriever(corpus)

    print(f"처리 대상: {len(transactions)}건 (risk_level={RISK_LEVEL_FILTER or '전체'})")

    results = []
    for i, (_, row) in enumerate(transactions.iterrows(), 1):
        print(f"[{i}/{len(transactions)}] belnr={row['belnr']} 처리 중...")
        try:
            result = generate_explanation(row, retriever)
            results.append(result)
        except Exception as e:
            print(f"  오류 발생 (belnr={row['belnr']}): {e}")
            results.append({"belnr": row["belnr"], "error": str(e)})
        time.sleep(0.5)  # rate limit 여유

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}건 결과를 {OUTPUT_PATH}에 저장했습니다.")


if __name__ == "__main__":
    main()
