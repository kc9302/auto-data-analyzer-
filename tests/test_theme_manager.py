import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.presenter.theme_manager import (
    ThemeManager,
    BusinessTranslator,
    get_theme_manager,
    get_business_translator
)
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder


def test_theme_manager_loading_and_fallback():
    # 1. Default loading from existing theme_corporate.yaml
    mgr = get_theme_manager()
    assert mgr.company_name == "AI Data Insights Lab"
    assert mgr.department == "Advanced Analytics Team"
    
    # Check color retrieval
    hex_p = mgr.get_hex("primary")
    assert hex_p.startswith("#")
    rgb_p = mgr.get_rgb("primary")
    assert len(rgb_p) == 3
    assert all(0 <= c <= 255 for c in rgb_p)

    # Check PPTX color conversion
    pptx_color = mgr.get_pptx_rgb("accent")
    assert hasattr(pptx_color, "color_type") or isinstance(pptx_color, tuple)

    # 2. Fallback when path does not exist
    fallback_mgr = ThemeManager(config_path="non_existent_file.yaml")
    assert fallback_mgr.company_name == "AI Data Insights Lab"
    assert fallback_mgr.get_hex("primary") == "#1E293B"


def test_business_translator_rules():
    translator = get_business_translator()

    # 1. Skewness
    res_symm = translator.translate_skewness(0.2)
    assert "양호" in res_symm["badge"]
    res_skew = translator.translate_skewness(2.4)
    assert "자동 교정" in res_skew["badge"]

    # 2. KS-test
    res_ks_ok = translator.translate_ks_test(0.35)
    assert "안전" in res_ks_ok["badge"]
    res_ks_warn = translator.translate_ks_test(0.005)
    assert "주의" in res_ks_warn["badge"]

    # 3. Overfitting
    res_fit_good = translator.translate_overfitting(0.02)
    assert "우수" in res_fit_good["badge"]
    res_fit_risk = translator.translate_overfitting(0.18)
    assert "규제 권고" in res_fit_risk["badge"]

    # 4. Health score
    assert "A등급" in translator.translate_health_score(92)["grade"]
    assert "B등급" in translator.translate_health_score(75)["grade"]
    assert "C등급" in translator.translate_health_score(50)["grade"]

    # 5. Model name
    name_lgbm = translator.translate_model_name("LGBMClassifier")
    assert "LightGBM" in name_lgbm
    assert "초고속" in name_lgbm


def test_presenter_integration_with_custom_theme():
    # Verify PptxDeckBuilder & HtmlReportBuilder accept custom theme seamlessly
    custom_mgr = ThemeManager()
    custom_mgr.theme_data["theme"]["company_name"] = "Custom Corp Inc"

    deck_builder = PptxDeckBuilder(theme_config=custom_mgr)
    assert deck_builder.theme_mgr.company_name == "Custom Corp Inc"

    html_builder = HtmlReportBuilder(theme_config=custom_mgr)
    assert html_builder.theme_mgr.company_name == "Custom Corp Inc"
