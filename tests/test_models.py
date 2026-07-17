"""Registry consistency: no model_id reaches a filename undocumented.

The eo-waterbody-temp analogue of the volume repo's registry test. Our models are
functions/values, not a class hierarchy, so instead of walking __subclasses__ we
assert the same guarantee across three surfaces: the model_id CONSTANTS used in
code, the MODEL_REGISTRY dict, and MODELS.md must all agree.
"""

from pathlib import Path

from eo_waterbody_temp.do_saturation import MODEL_ID as DO_SAT_MODEL_ID
from eo_waterbody_temp.ecostress import PRODUCT as ECOSTRESS_WST_MODEL_ID
from eo_waterbody_temp.models import (
    MODEL_REGISTRY,
    is_registered,
    model_id_from_method,
    variant_from_method,
)

MODELS_MD = Path(__file__).parents[1] / "MODELS.md"
_REQUIRED = ("kind", "product", "units", "description")


def test_code_model_ids_are_registered():
    # every model_id constant the code actually stamps must be in the registry
    for mid in (ECOSTRESS_WST_MODEL_ID, DO_SAT_MODEL_ID):
        assert is_registered(mid), f"{mid} used in code but not registered"


def test_every_registry_entry_has_required_fields():
    for mid, entry in MODEL_REGISTRY.items():
        missing = [k for k in _REQUIRED if k not in entry]
        assert not missing, f"{mid} missing registry fields: {missing}"


def test_every_registry_entry_documented_in_models_md():
    text = MODELS_MD.read_text()
    for mid in MODEL_REGISTRY:
        assert f"`{mid}`" in text, f"{mid} registered but not documented in MODELS.md"


def test_model_id_extraction_from_method():
    assert (
        model_id_from_method("ecostress_l2t_lste_v003:LST")
        == "ecostress_l2t_lste_v003"
    )
    assert (
        model_id_from_method(
            "do_sat_bk1984_freshwater_1atm_v1<-ecostress_l2t_lste_v003:LST"
        )
        == "do_sat_bk1984_freshwater_1atm_v1"
    )


def test_extracted_model_ids_are_registered():
    # the ids the filename builder will actually use must resolve to registry entries
    for method in (
        "ecostress_l2t_lste_v003:LST",
        "ecostress_l2t_lste_v003:SST",
        "do_sat_bk1984_freshwater_1atm_v1<-ecostress_l2t_lste_v003:LST",
    ):
        assert is_registered(model_id_from_method(method))


def test_variant_extraction():
    assert variant_from_method("ecostress_l2t_lste_v003:LST") == "LST"
    assert (
        variant_from_method(
            "do_sat_bk1984_freshwater_1atm_v1<-ecostress_l2t_lste_v003:SST"
        )
        == "SST"
    )
