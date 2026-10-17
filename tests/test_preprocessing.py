"""Unit tests verifying pure Arabic text preprocessing primitives and pipeline composition."""

import pytest

from src.preprocessing import (
    ARABIC_STOPWORDS,
    compose_pipeline,
    get_pipeline,
    isri_stem,
    normalize_hamza_alef,
    remove_diacritics,
    remove_non_arabic_punct,
    remove_stopwords,
    remove_tatweel,
)


def test_remove_diacritics_strips_short_vowels() -> None:
    text_with_tashkeel = "كَتَبَ الطَّالِبُ الدَّرْسَ فِي المَدْرَسَةِ"
    expected = "كتب الطالب الدرس في المدرسة"
    assert remove_diacritics(text_with_tashkeel) == expected


def test_remove_tatweel_removes_kashida() -> None:
    text_with_kashida = "جـامـعـة الـقـاهـرة"
    expected = "جامعة القاهرة"
    assert remove_tatweel(text_with_kashida) == expected


def test_normalize_hamza_alef_unifies_variants() -> None:
    variants = "أحمد وإبراهيم في مستشفى وٱمرأة آمنة"
    expected = "احمد وابراهيم في مستشفي وامراه امنه"
    assert normalize_hamza_alef(variants) == expected


def test_remove_non_arabic_punct_isolates_arabic_words() -> None:
    mixed_text = "مرحبا بكم! Welcome to 2026. هل أنتم بخير؟ 100% #رياضة"
    expected = "مرحبا بكم هل أنتم بخير رياضة"
    assert remove_non_arabic_punct(mixed_text) == expected


def test_remove_stopwords_filters_common_words() -> None:
    text = "كتب الطالب في المدرسة عن التاريخ وكان مسرورا"
    cleaned = remove_stopwords(text)
    assert "في" not in cleaned.split()
    assert "عن" not in cleaned.split()
    assert "الطالب" in cleaned.split()
    assert "المدرسة" in cleaned.split()


def test_isri_stem_reduces_to_root() -> None:
    assert isri_stem("كتابات") == "كتب"
    assert isri_stem("مكتبة") == "كتب"


def test_compose_pipeline_sequential_execution() -> None:
    pipeline = compose_pipeline(["remove_diacritics", "remove_tatweel", "normalize_hamza_alef"])
    sample = "إِنَّ الـحُكُومَـةَ فـِي مِصْرَ"
    assert pipeline(sample) == "ان الحكومه في مصر"


def test_compose_pipeline_raises_on_unknown_step() -> None:
    with pytest.raises(KeyError, match="Unknown preprocessing step"):
        compose_pipeline(["non_existent_step"])


def test_get_pipeline_loads_named_pipelines_from_config() -> None:
    raw_char = get_pipeline("raw_char")
    sample_raw = "الـمـَدْرَسـَةُ"
    assert raw_char(sample_raw) == "المدرسة"

    light = get_pipeline("light_normalized")
    sample_light = "إِنَّ الـحُكُومَـةَ فـِي مِصْرَ! (100% نجاح)"
    # 'ان' and 'في' are filtered as stopwords in light_normalized
    assert light(sample_light) == "الحكومه مصر نجاح"

    heavy = get_pipeline("paper_heavy_stemming")
    sample_heavy = "إِنَّ الكِتَابَاتِ الـجَدِيدَةَ فِي المَكْتَبَةِ!"
    result = heavy(sample_heavy)
    assert "كتب" in result.split()


def test_get_pipeline_raises_on_unknown_pipeline_name() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline name"):
        get_pipeline("unregistered_pipeline")


def test_pure_functions_do_not_mutate_input_strings() -> None:
    original = "تَجْرِبَةٌ أُولَى"
    copy = str(original)
    _ = remove_diacritics(original)
    _ = normalize_hamza_alef(original)
    assert original == copy


def test_arabic_stopwords_constant_is_frozen_and_populated() -> None:
    assert isinstance(ARABIC_STOPWORDS, frozenset)
    assert len(ARABIC_STOPWORDS) > 500
    assert "من" in ARABIC_STOPWORDS
    assert "على" in ARABIC_STOPWORDS or "علي" in ARABIC_STOPWORDS
