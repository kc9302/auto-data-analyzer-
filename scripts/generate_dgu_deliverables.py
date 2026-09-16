"""
One-click CLI script to generate all Dongguk University (DGU) deliverables:
1. dist/dgu_recommendation_feature_journey.xlsx
2. dist/dgu_data_landscape_and_api_wbs.xlsx
3. dist/dgu_executive_presentation.pptx
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dgu.dgu_recsys_generator import DGURecSysDeliverablesGenerator
from scripts.generate_dgu_dataset import generate_dgu_data


def main():
    print("=" * 75)
    print(" 🏛️  동국대학교 맞춤형 추천 시스템 - 핵심 산출물 및 임원 보고 패키징 가동")
    print("=" * 75)

    # 1. Ensure DGU Dataset exists
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "dgu_student_features.csv")
    if not os.path.exists(csv_path):
        print("[1/2] 동국대 학사·비교과 모의 피처 데이터셋 생성 중 (3,500행)...")
        df = generate_dgu_data(n_samples=3500)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"      ✓ 데이터셋 생성 완료: {csv_path} ({len(df):,}행)")
    else:
        print(f"[1/2] 동국대 학사·비교과 모의 데이터셋 확인 완료: {csv_path}")

    # 2. Build Deliverables
    print("[2/2] 엑셀 2종 및 16:9 와이드스크린 PPTX 생성 중...")
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "dist")
    generator = DGURecSysDeliverablesGenerator(output_dir=dist_dir)
    results = generator.generate_all()

    print("\n" + "=" * 75)
    print(" ✨  동국대학교 맞춤형 추천 시스템 3대 산출물 패키징 완료:")
    print("=" * 75)
    print(f" 📊 [산출물 1] 추천 기능별 피처엔지니어링 & 비교모델 엑셀:\n    ➔ {results['feature_journey_excel']}")
    print(f" 📑 [산출물 2] 전체 데이터 현황 및 API 개발 WBS/공수(18.5 M/M) 엑셀:\n    ➔ {results['data_landscape_wbs_excel']}")
    print(f" 📽️  [산출물 3] 동국대 데이터 패턴 분석 & '우린 이런것도 할수있다' 16:9 장표:\n    ➔ {results['executive_pptx']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
