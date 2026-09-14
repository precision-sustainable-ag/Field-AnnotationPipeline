import pytest

from field_annotation.species import get_class_id


@pytest.fixture
def species_info():
    return {
        "background": {"class_id": 0, "common_name": "background", "alias": []},
        "AMPA": {"class_id": 1, "common_name": "Palmer amaranth", "alias": ["carelessweed"]},
    }


def test_get_class_id_matches_common_name_case_insensitive(species_info):
    assert get_class_id(species_info, "palmer amaranth") == 1
    assert get_class_id(species_info, "Palmer Amaranth") == 1


def test_get_class_id_matches_alias(species_info):
    assert get_class_id(species_info, "Carelessweed") == 1


def test_get_class_id_returns_none_for_unknown_species(species_info):
    assert get_class_id(species_info, "some unlisted weed") is None


def test_get_class_id_returns_none_for_missing_species(species_info):
    assert get_class_id(species_info, None) is None
    assert get_class_id(species_info, "") is None
