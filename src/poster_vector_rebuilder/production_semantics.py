from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
import json

SVG_NS = "http://www.w3.org/2000/svg"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_attr(elem: ET.Element, name: str, default: float = 1.0) -> float:
    try:
        return float(elem.get(name, default))
    except (TypeError, ValueError):
        return default


def inspect_svg_production_semantics(svg_path: str | Path) -> dict:
    """Inspect optional production semantics without changing ordinary artwork behavior.

    Production semantics are opt-in SVG metadata. Existing files that do not declare
    spot inks or special finishes continue through the established CMYK workflow.
    """
    root = ET.parse(svg_path).getroot()
    spot_objects = []
    finish_objects = []
    blockers = []
    warnings = []

    for elem in root.iter():
        tag = _local(elem.tag)
        object_id = elem.get("id")
        spot_name = (elem.get("data-spot-name") or "").strip()
        color_role = (elem.get("data-color-role") or "").strip().lower()
        finish_role = (elem.get("data-finish-role") or "").strip().lower()
        prepress_role = (elem.get("data-prepress-role") or "").strip().lower()
        declares_spot = bool(spot_name) or color_role == "spot"
        declares_finish = bool(finish_role) or prepress_role in {
            "varnish", "print-varnish", "uv-varnish", "relief-coating", "printing-lacquer"
        }

        if declares_spot:
            item = {
                "id": object_id,
                "tag": tag,
                "spot_name": spot_name or None,
                "overprint_fill": _truthy(elem.get("data-overprint-fill")),
            }
            spot_objects.append(item)
            if not spot_name:
                blockers.append({
                    "code": "SPOT_NAME_MISSING",
                    "object_id": object_id,
                    "message": "A spot-color object was declared without an authoritative spot-color name."
                })

        if declares_finish:
            opacity = _float_attr(elem, "opacity", 1.0)
            fill_opacity = _float_attr(elem, "fill-opacity", 1.0)
            style = elem.get("style", "")
            has_style_transparency = "opacity:" in style and not (
                "opacity:1" in style or "opacity:1.0" in style or "opacity:100%" in style
            )
            is_vector = tag in {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "g"}
            overprint = _truthy(elem.get("data-overprint-fill"))
            item = {
                "id": object_id,
                "tag": tag,
                "finish_role": finish_role or prepress_role,
                "spot_name": spot_name or None,
                "vector_geometry": is_vector,
                "opacity": opacity,
                "fill_opacity": fill_opacity,
                "overprint_fill": overprint,
            }
            finish_objects.append(item)
            if not is_vector:
                blockers.append({
                    "code": "FINISH_NOT_VECTOR",
                    "object_id": object_id,
                    "message": "Declared print finish must be represented by separate vector geometry."
                })
            if opacity < 0.999 or fill_opacity < 0.999 or has_style_transparency:
                blockers.append({
                    "code": "FINISH_HAS_TRANSPARENCY",
                    "object_id": object_id,
                    "message": "Declared print finish must use solid 100% coverage with no transparency."
                })
            if not spot_name:
                blockers.append({
                    "code": "FINISH_SPOT_NAME_MISSING",
                    "object_id": object_id,
                    "message": "Declared print finish requires a named spot-color separation."
                })
            if not overprint:
                blockers.append({
                    "code": "FINISH_OVERPRINT_NOT_DECLARED",
                    "object_id": object_id,
                    "message": "Declared print finish must explicitly declare overprint fill to avoid unintended knockout."
                })

    if spot_objects:
        warnings.append({
            "code": "SPOT_PRESERVATION_REQUIRES_DEDICATED_EXPORT",
            "message": (
                "Spot colors are present. The established CMYK press conversion must not be claimed "
                "to preserve named spot separations; use a dedicated spot-preserving export/RIP validation path."
            ),
        })

    return {
        "schema": "poster-vector-production-semantics-v1",
        "spot_objects": spot_objects,
        "finish_objects": finish_objects,
        "has_spot_colors": bool(spot_objects),
        "has_special_finishes": bool(finish_objects),
        "blockers": blockers,
        "warnings": warnings,
        "valid": not blockers,
    }


def build_production_manifest(
    svg_path: str | Path,
    output_path: str | Path,
    *,
    geometry: dict,
    icc: dict,
    pdf_target: str = "PDF/X-4-oriented",
) -> dict:
    semantics = inspect_svg_production_semantics(svg_path)
    manifest = {
        "schema": "poster-vector-production-manifest-v1",
        "source_svg": str(svg_path),
        "locked_baseline": {
            "top_level_layer_architecture_preserved": True,
            "default_bleed_mm": 3.0,
            "default_target_ppi": 300.0,
            "editable_master": "SVG",
            "process_color_default": "CMYK",
            "pdf_policy": pdf_target,
            "existing_raster_fallback_policy_preserved": True,
        },
        "geometry": geometry,
        "color_management": {
            "process_space": "CMYK",
            "icc_profile": icc.get("profile"),
            "icc_profile_name": icc.get("profile_name"),
            "rgb_source_profile": icc.get("rgb_profile"),
            "spot_color_policy": (
                "Preserve named spot identity only when explicitly specified by authoritative artwork/job data. "
                "Do not promote an inferred CMYK-to-Pantone match to an authoritative Pantone specification."
            ),
            "proof_policy": (
                "Gamut/output-condition review is required for intense colors, unusual papers, and unusual print surfaces. "
                "Physical/device-specific swatch proofing remains authoritative for critical spot-color matching."
            ),
        },
        "text_policy": {
            "editable_svg": "keep live/editable text where reliably reconstructed",
            "press_pdf": "embed fonts under the established route; outlining remains an explicit delivery option when required",
            "font_identity": "never guess an exact font when the source does not establish it",
        },
        "bleed_policy": {
            "configured_bleed_mm": geometry.get("bleed_mm"),
            "full_bleed_objects_must_extend_through_bleed": True,
            "objects_outside_trim_can_be_intentional": True,
            "preflight_classification": "intentional bleed geometry is not automatically a blocker",
        },
        "production_semantics": semantics,
        "special_finish_policy": {
            "separate_vector_geometry": True,
            "solid_100_percent_no_transparency": True,
            "named_spot_separation": True,
            "overprint_fill_required": True,
            "overprint_simulation_or_RIP_review_required": True,
        },
        "compatibility_policy": {
            "corel_friendly_svg_is_master": True,
            "valid_svg_alone_is_not_full_compatibility_proof": True,
            "coreldraw_open_import_visual_structural_regression_recommended": True,
        },
        "preflight_policy": {
            "warnings_are_classified_by_intent": True,
            "conservative_diagnostics_do_not_automatically_override_validated_acceptance_policy": True,
            "spot_or_finish_jobs_require_separation-aware_validation": True,
        },
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
