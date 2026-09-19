"""
Theme Manager & Business Translator Module
Provides decoupled corporate styling injection (colors, typography, logo)
and translates technical ML/statistical metrics into intuitive Korean business labels.
"""
import os
import yaml
from typing import Dict, Any, Tuple, Optional


class ThemeManager:
    """
    Manages corporate presentation styling, branding, and color palettes.
    Supports graceful fallback to default corporate themes if YAML is missing.
    """

    DEFAULT_THEME = {
        "theme": {
            "company_name": "AI Data Insights Lab",
            "department": "Advanced Analytics Team",
            "logo_path": ""
        },
        "palette": {
            "primary": "#1E293B",     # Slate Navy
            "secondary": "#334155",   # Subtitles, Card Headers
            "accent": "#0EA5E9",      # Sky Blue (Key Highlights)
            "success": "#10B981",     # Emerald (Clean Data / Normal)
            "warning": "#F59E0B",     # Amber (Moderate Issues)
            "danger": "#F43F5E",      # Rose (Severe Outliers / Critical)
            "card_bg": "#FFFFFF",     # White
            "slide_bg": "#F8FAFC",    # Ultra-light gray/slate
            "border": "#E2E8F0"
        },
        "typography": {
            "font_family_korean": "Malgun Gothic",
            "font_family_latin": "Segoe UI",
            "title_size_pt": 24,
            "header_size_pt": 16,
            "body_size_pt": 11,
            "caption_size_pt": 9
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "theme_corporate.yaml"
        )
        self.theme_data = self._load_theme()

    def _load_theme(self) -> Dict[str, Any]:
        """Loads YAML config or falls back to DEFAULT_THEME."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        # Merge with defaults
                        merged = self.DEFAULT_THEME.copy()
                        for section in ["theme", "palette", "typography"]:
                            if section in data and isinstance(data[section], dict):
                                merged[section] = {**merged[section], **data[section]}
                        return merged
            except Exception:
                pass
        return self.DEFAULT_THEME.copy()

    @property
    def company_name(self) -> str:
        return self.theme_data["theme"].get("company_name", "AI Analytics Lab")

    @property
    def department(self) -> str:
        return self.theme_data["theme"].get("department", "Data Team")

    @property
    def logo_path(self) -> str:
        return self.theme_data["theme"].get("logo_path", "")

    def get_hex(self, color_key: str) -> str:
        """Returns hex color string (e.g. '#1E293B')."""
        palette = self.theme_data.get("palette", {})
        return palette.get(color_key, self.DEFAULT_THEME["palette"].get(color_key, "#1E293B"))

    def get_rgb(self, color_key: str) -> Tuple[int, int, int]:
        """Converts hex color to RGB integer tuple (R, G, B)."""
        hex_code = self.get_hex(color_key).lstrip("#")
        if len(hex_code) == 6:
            return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return (30, 41, 59)

    def get_pptx_rgb(self, color_key: str):
        """Returns python-pptx RGBColor object if available, else (r, g, b)."""
        r, g, b = self.get_rgb(color_key)
        try:
            from pptx.dml.color import RGBColor
            return RGBColor(r, g, b)
        except ImportError:
            return (r, g, b)

    @property
    def font_korean(self) -> str:
        return self.theme_data["typography"].get("font_family_korean", "Malgun Gothic")

    @property
    def font_latin(self) -> str:
        return self.theme_data["typography"].get("font_family_latin", "Segoe UI")


class BusinessTranslator:
    """
    Translates technical ML & statistics terms into plain Korean business language.
    Eliminates communication friction with non-technical stakeholders.
    """

    @staticmethod
    def translate_skewness(skew_val: float) -> Dict[str, str]:
        val = abs(skew_val)
        if val <= 0.5:
            return {
                "status": "정상 대칭",
                "badge": "🟢 양호",
                "description": "데이터가 좌우 대칭을 이루어 정규분포 가정을 충족하며, 추가 변환 없이 모델링에 즉각 투입 가능합니다."
            }
        elif val <= 1.5:
            return {
                "status": "경미한 비대칭",
                "badge": "🟡 주의",
                "description": "특정 방향으로 완만한 치우침이 있으나, 트리 모델 학습에 유의미한 왜곡을 주지 않는 안정적 수준입니다."
            }
        else:
            return {
                "status": "극심한 쏠림",
                "badge": "🚨 자동 교정 완료",
                "description": "이상치 밀집 및 롱테일 쏠림을 식별하여 log1p 수학적 변환으로 데이터 스케일을 안정화했습니다."
            }

    @staticmethod
    def translate_ks_test(p_val: float) -> Dict[str, str]:
        if p_val >= 0.05:
            return {
                "status": "분포 무결성 100%",
                "badge": "✅ 안전",
                "description": "결측 대체 조치 후에도 원천 데이터의 통계적 고유 분포를 온전히 보존함을 검증 완료했습니다."
            }
        elif p_val >= 0.01:
            return {
                "status": "경미한 분포 변동",
                "badge": "🟡 모니터링",
                "description": "미세한 분포 이동이 감지되었으나 허용 통계적 유의수준 범위 내에서 안정적으로 제어 중입니다."
            }
        else:
            return {
                "status": "분포 왜곡 위험",
                "badge": "⚠️ 주의",
                "description": "결측 대체로 인한 데이터 분포 왜곡을 방지하고자 결측 여부 지시자(Indicator) 플래그를 추가 반영했습니다."
            }

    @staticmethod
    def translate_overfitting(gap: float) -> Dict[str, str]:
        if gap <= 0.05:
            return {
                "status": "일반화 성능 최상",
                "badge": "🟢 우수",
                "description": "학습 점수와 교차 검증 점수가 이상적 균형을 유지하여 실운영 서빙 환경에서도 견고한 일반화 성능을 발휘합니다."
            }
        elif gap <= 0.12:
            return {
                "status": "적정 일반화",
                "badge": "🟡 양호",
                "description": "통상적인 머신러닝 학습-검증 격차 허용치 내에 안착하여 안정적 성능 유지가 가능합니다."
            }
        else:
            return {
                "status": "과적합(Overfitting) 위험",
                "badge": "🚨 규제 권고",
                "description": "학습 데이터 편향 위험이 포착되어 정규화 가중치(L1/L2) 상향 및 트리 최대 깊이 제약 조치를 권고합니다."
            }

    @staticmethod
    def translate_health_score(score: float) -> Dict[str, str]:
        if score >= 85:
            return {
                "grade": "A등급 (우수)",
                "badge": "🟢 운영 적합",
                "summary": "결측치 및 중복 레코드가 극소 수준이며, 전반적인 데이터 품질과 정합성이 매우 우수합니다."
            }
        elif score >= 70:
            return {
                "grade": "B등급 (양호)",
                "badge": "🟡 전처리 권장",
                "summary": "일부 결측치와 이상치가 식별되었으나 자동 정제 파이프라인을 통해 결함을 안정적으로 해소 가능합니다."
            }
        else:
            return {
                "grade": "C등급 (주의)",
                "badge": "🚨 수집 체계 보완",
                "summary": "결측률 상회 및 중복 레코드 집중으로 원천 수집 파이프라인의 무결성 점검 및 보완을 권고합니다."
            }

    @staticmethod
    def translate_model_name(model_name: str) -> str:
        name_map = {
            "LGBMClassifier": "LightGBM (초고속 그래디언트 부스팅)",
            "LGBMRegressor": "LightGBM 회귀 (초고속 부스팅)",
            "RandomForestClassifier": "Random Forest (안정적 배깅 앙상블)",
            "RandomForestRegressor": "Random Forest 회귀 (배깅 앙상블)",
            "LogisticRegression": "로지스틱 회귀 (해석력 높은 선형 분류)",
            "Ridge": "릿지 회귀 (L2 규제 선형 모델)",
            "MLPClassifier": "Tabular DeepNet (다층 신경망 분류)",
            "MLPRegressor": "Tabular DeepNet (다층 신경망 회귀)"
        }
        return name_map.get(model_name, f"{model_name} (ML 모델)")


def get_theme_manager(config_path: Optional[str] = None) -> ThemeManager:
    return ThemeManager(config_path=config_path)


def get_business_translator() -> BusinessTranslator:
    return BusinessTranslator()
