"""
회계감사기준 전문(全文) PDF에서 이상거래탐지 프로젝트에 필요한
4개 기준서(240, 315, 330, 1100)만 추출해서 RAG용 코퍼스로 저장.

입력: 0__회계감사기준_전문_2025_개정_.pdf (KICPA, 2025년 11월 개정, 974페이지)
출력: standards_corpus/ 폴더에 기준서별 txt 파일 4개
      + 검색 편의를 위한 통합 JSON 파일(standards_corpus.json)
"""

import os
import re
import json
from pypdf import PdfReader

PDF_PATH = "data/0__회계감사기준_전문_2025_개정_.pdf"   # repo 최상위 기준 상대경로
OUTPUT_DIR = "standards_corpus"
# 주의: 이 스크립트는 repo 최상위 폴더에서 실행하는 것을 전제로 합니다.
#   예: python src/extract_audit_standards.py

# 목차에서 직접 확인한 페이지 범위 (1-indexed, PDF 표기 그대로)
# pypdf는 0-indexed라 아래에서 -1 보정함
SECTIONS = [
    {"code": "240",  "title": "재무제표감사에서 부정에 관한 감사인의 책임", "start": 81,  "end": 124},
    {"code": "315",  "title": "중요왜곡표시위험의 식별과 평가",             "start": 192, "end": 303},
    {"code": "330",  "title": "평가된 위험에 대한 감사인의 대응",           "start": 312, "end": 334},
    {"code": "1100", "title": "내부회계관리제도의 감사",                   "start": 851, "end": 906},
]


def extract_section_text(reader, start_page, end_page):
    """start_page, end_page는 PDF 표기 페이지 번호(1-indexed), 양끝 포함."""
    texts = []
    for page_num in range(start_page - 1, end_page):  # pypdf는 0-indexed
        page = reader.pages[page_num]
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def clean_text(text):
    # 페이지 하단의 "n / 974" 같은 페이지 표시, 과도한 빈 줄 정리
    text = re.sub(r"\d+\s*/\s*974", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text, code):
    """
    문단번호(예: "17.", "17-25" 같은 문단 구간)를 기준으로 대략적으로 쪼갬.
    완벽한 파싱은 아니고, RAG 검색 단위를 문단 수준으로 맞추기 위한 간단한 청킹.
    """
    # "숫자." 로 시작하는 줄을 새 문단의 시작으로 간주
    raw_chunks = re.split(r"\n(?=\d{1,3}\.\s)", text)
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:  # 너무 짧은 조각(목차 잔여물 등)은 제외
            continue
        chunks.append({
            "standard": code,
            "text": chunk
        })
    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reader = PdfReader(PDF_PATH)

    all_chunks = []
    for section in SECTIONS:
        print(f"추출 중: 감사기준서 {section['code']} ({section['title']}) "
              f"- p.{section['start']}~{section['end']}")

        raw_text = extract_section_text(reader, section["start"], section["end"])
        text = clean_text(raw_text)

        # 기준서별 개별 txt 저장 (사람이 직접 읽고 검수하기 편하도록)
        out_path = os.path.join(OUTPUT_DIR, f"standard_{section['code']}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# 감사기준서 {section['code']} - {section['title']}\n\n")
            f.write(text)
        print(f"  저장 완료: {out_path} ({len(text):,}자)")

        # 검색용 문단 단위 청크 생성
        chunks = split_into_paragraphs(text, section["code"])
        all_chunks.extend(chunks)
        print(f"  문단 청크 수: {len(chunks)}개")

    # 통합 JSON (다음 단계 RAG 검색 로직에서 이 파일을 로드해서 씀)
    json_path = os.path.join(OUTPUT_DIR, "standards_corpus.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(all_chunks)}개 문단 청크를 {json_path}에 저장했습니다.")
    print("이 파일이 다음 단계(RAG 검색)에서 근거 문서 코퍼스로 사용됩니다.")


if __name__ == "__main__":
    main()
