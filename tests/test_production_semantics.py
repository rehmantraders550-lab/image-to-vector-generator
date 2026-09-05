from pathlib import Path
import json

from poster_vector_rebuilder.production_semantics import (
    build_production_manifest,
    inspect_svg_production_semantics,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_plain_existing_svg_has_no_new_blockers(tmp_path):
    svg = _write(
        tmp_path / "plain.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<g id="00_BACKGROUND"><rect width="100" height="100" fill="#fff"/></g>'
        '</svg>',
    )
    report = inspect_svg_production_semantics(svg)
    assert report["valid"] is True
    assert report["has_spot_colors"] is False
    assert report["has_special_finishes"] is False
    assert report["blockers"] == []


def test_finish_semantics_require_vector_spot_and_overprint(tmp_path):
    svg = _write(
        tmp_path / "bad_finish.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect id="coat" width="80" height="80" data-finish-role="uv-varnish" '
        'opacity="0.5" fill="#ff00ff"/>'
        '</svg>',
    )
    report = inspect_svg_production_semantics(svg)
    codes = {x["code"] for x in report["blockers"]}
    assert report["has_special_finishes"] is True
    assert {"FINISH_HAS_TRANSPARENCY", "FINISH_SPOT_NAME_MISSING", "FINISH_OVERPRINT_NOT_DECLARED"}.issubset(codes)


def test_valid_finish_semantics_are_recognized_but_require_spot_safe_route(tmp_path):
    svg = _write(
        tmp_path / "finish.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path id="coat" d="M10 10H90V90H10Z" fill="#ff00ff" '
        'data-finish-role="printing-lacquer" data-color-role="spot" '
        'data-spot-name="Printing lacquer" data-overprint-fill="true"/>'
        '</svg>',
    )
    report = inspect_svg_production_semantics(svg)
    assert report["valid"] is True
    assert report["has_spot_colors"] is True
    assert report["has_special_finishes"] is True
    assert report["finish_objects"][0]["overprint_fill"] is True
    assert any(x["code"] == "SPOT_PRESERVATION_REQUIRES_DEDICATED_EXPORT" for x in report["warnings"])


def test_manifest_preserves_locked_baseline_defaults(tmp_path):
    svg = _write(
        tmp_path / "plain.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"/>',
    )
    out = tmp_path / "production_manifest.json"
    geometry = {"bleed_mm": 3.0, "target_ppi": 300.0, "trim_width_mm": 100.0, "trim_height_mm": 100.0}
    icc = {"profile": "/profiles/cmyk.icc", "profile_name": "Existing CMYK profile", "rgb_profile": "/profiles/rgb.icc"}
    manifest = build_production_manifest(svg, out, geometry=geometry, icc=icc)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert manifest == payload
    assert payload["locked_baseline"]["default_bleed_mm"] == 3.0
    assert payload["locked_baseline"]["default_target_ppi"] == 300.0
    assert payload["locked_baseline"]["editable_master"] == "SVG"
    assert payload["locked_baseline"]["process_color_default"] == "CMYK"
    assert payload["locked_baseline"]["pdf_policy"] == "PDF/X-4-oriented"
    assert payload["bleed_policy"]["objects_outside_trim_can_be_intentional"] is True
