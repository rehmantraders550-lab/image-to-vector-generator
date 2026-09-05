from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from .generalized_preflight import run_blocks_1_to_4
from .vector_fit import fit_background_vectors
from .phase24d import recover_hidden_background, run_phase24_acceptance_gate
from .semantic_primitives import reconstruct_semantic_primitives
from .text_reconstruct import reconstruct_text
from .final_assembly import assemble_master_svg
from .prepress import export_prepress_package


def _hex(rgb):
    r,g,b=[int(x) for x in rgb]; return f"#{r:02x}{g:02x}{b:02x}"


def _flat_background(image_path, known_path, output_svg):
    rgb=np.asarray(Image.open(image_path).convert("RGB"),dtype=np.uint8)
    mask=np.asarray(Image.open(known_path).convert("L"))>=128
    sample=rgb[mask] if np.any(mask) else rgb.reshape(-1,3)
    color=np.median(sample,axis=0).astype(np.uint8); h,w=rgb.shape[:2]
    text=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><g id="BACKGROUND_FALLBACK"><rect x="0" y="0" width="{w}" height="{h}" fill="{_hex(color)}"/></g></svg>'
    Path(output_svg).parent.mkdir(parents=True,exist_ok=True); Path(output_svg).write_text(text,encoding="utf-8")
    return {"status":"fallback","svg":str(output_svg),"reason":"constrained background fit unavailable; median authoritative background used"}


def _semantic_mask(foreground_path, text_mask_path, output_path):
    fg=np.asarray(Image.open(foreground_path).convert("L"),dtype=np.uint8)
    txt=np.asarray(Image.open(text_mask_path).convert("L").resize((fg.shape[1],fg.shape[0]),Image.Resampling.NEAREST),dtype=np.uint8)
    out=np.where((fg>=128)&(txt<128),255,0).astype(np.uint8)
    Image.fromarray(out).save(output_path)
    return int(np.count_nonzero(out))


def _photographic_asset(image_path, foreground_path, output_path):
    rgba=np.asarray(Image.open(image_path).convert("RGBA"),dtype=np.uint8).copy()
    fg=np.asarray(Image.open(foreground_path).convert("L").resize((rgba.shape[1],rgba.shape[0]),Image.Resampling.NEAREST),dtype=np.uint8)
    rgba[...,3]=fg
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgba,"RGBA").save(output_path)
    return output_path


def _full_scene_asset(image_path, output_path):
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Image.open(image_path).convert("RGB").save(output_path,optimize=True)
    return output_path


def run_delivery_pipeline(
    input_path: str | Path,
    job_dir: str | Path,
    *,
    max_panels: int=4,
    ocr_confidence: float=80.0,
    trim_width_mm: float | None=None,
    trim_height_mm: float | None=None,
    bleed_mm: float=3.0,
    target_ppi: float=300.0,
    icc_profile: str | Path | None=None,
) -> dict:
    """One-command arbitrary reference image -> editable delivery package.

    Detailed photographic scenes are preserved as an explicitly declared raster scene.
    OCR is still reported, but is not overlaid unless the raster lettering can be removed
    without damaging source pixels. This avoids duplicate/ghost lettering.
    """
    job=Path(job_dir); delivery=job/"delivery"; assets=delivery/"assets"
    delivery.mkdir(parents=True,exist_ok=True); assets.mkdir(parents=True,exist_ok=True)
    stages={}
    prep=run_blocks_1_to_4(input_path,job,max_panels=max_panels); stages["prepare"]=prep
    normalized=Path(prep["outputs"]["normalized_reference"]); meta=job/"metadata"; masks=job/"masks"
    classification=json.loads((meta/"artwork_classification.json").read_text(encoding="utf-8"))

    text=reconstruct_text(normalized,job/"vectors"/"text.svg",report_path=meta/"text_reconstruction.json",exclusion_mask_path=masks/"text_exclusion.png",min_confidence=ocr_confidence)
    stages["text"]=text
    semantic_mask=masks/"semantic_foreground.png"
    semantic_pixels=_semantic_mask(masks/"foreground_mask.png",masks/"text_exclusion.png",semantic_mask)
    semantic_svg=None; photo_href=None; text_svg_for_assembly=text["outputs"]["svg"]
    cleanup_policy={"min_area":10.0,"simplify":0.003,"cleanup_radius":0,"node_budget":12000}
    photographic=classification["primary_class"]=="mixed_or_photographic" and classification["routes"].get("photographic_fallback_possible")

    if photographic:
        photo=assets/"photographic_scene.png"; _full_scene_asset(normalized,photo)
        photo_href="assets/photographic_scene.png"
        text_svg_for_assembly=None
        stages["foreground"]={
            "mode":"full_scene_raster_photographic_fallback","asset":str(photo),"semantic_pixels":semantic_pixels,
            "reason":"classifier identified detailed mixed/photographic content; source scene preserved without pretending soft lighting, texture or depth are vector geometry",
            "ocr_overlay_policy":"suppressed to prevent duplicate/ghost source lettering; OCR remains available in the reconstruction report",
        }
    elif semantic_pixels>0:
        try:
            sem=reconstruct_semantic_primitives(normalized,job/"vectors"/"semantic.svg",mask_path=semantic_mask,report_path=meta/"semantic_primitives.json",colors=12,min_area=cleanup_policy["min_area"],simplify=cleanup_policy["simplify"],cleanup_radius=cleanup_policy["cleanup_radius"])
            semantic_svg=sem["outputs"]["svg"]; stages["foreground"]=sem
        except Exception as exc:
            photo=assets/"photographic_foreground.png"; _photographic_asset(normalized,semantic_mask,photo)
            photo_href="assets/photographic_foreground.png"
            stages["foreground"]={"mode":"raster_fallback_after_semantic_failure","error":f"{type(exc).__name__}: {exc}","asset":str(photo),"note":"high-confidence OCR text excluded from raster alpha"}
    else:
        stages["foreground"]={"mode":"none","semantic_pixels":0}
    stages["hard_graphic_cleanup"]={"status":"applied","policy":cleanup_policy,"note":"semantic contours are simplified before SVG emission; morphology is disabled by default to preserve authoritative visible boundaries"}

    bg_dir=job/"background_fit"; bg_svg=None
    if photographic:
        stages["background_fit"]={"status":"skipped","reason":"full photographic scene is authoritative; a synthetic gradient background would reduce fidelity"}
    else:
        try:
            fit=fit_background_vectors(normalized,masks/"background_known.png",bg_dir,phase24b_report_path=prep["outputs"].get("panel_report"),max_panels=max_panels)
            stages["background_fit"]=fit; bg_svg=fit["outputs"]["svg"]
            recovery=recover_hidden_background(normalized,masks/"background_known.png",fit["outputs"]["report"],job/"background_recovery"); stages["background_recovery"]=recovery
            gate=run_phase24_acceptance_gate(normalized,masks/"background_known.png",fit["outputs"]["report"],bg_svg,job/"background_acceptance"); stages["background_acceptance"]=gate
        except Exception as exc:
            bg_svg=str(job/"background_fit"/"background_fallback.svg")
            stages["background_fit"]=_flat_background(normalized,masks/"background_known.png",bg_svg)
            stages["background_fit"]["error"]=f"{type(exc).__name__}: {exc}"

    with Image.open(normalized) as im: width,height=im.size
    master=delivery/"artwork_master.svg"
    assembly=assemble_master_svg(master,width=width,height=height,background_svg=bg_svg,semantic_svg=semantic_svg,text_svg=text_svg_for_assembly,photographic_href=photo_href,report_path=meta/"final_assembly.json")
    stages["assembly"]=assembly
    prepress=export_prepress_package(
        master,delivery,trim_width_mm=trim_width_mm,trim_height_mm=trim_height_mm,
        bleed_mm=bleed_mm,target_ppi=target_ppi,icc_profile=icc_profile,
    ); stages["prepress"]=prepress
    report={
        "schema":"poster-vector-delivery-v2",
        "status":"complete" if prepress.get("press_ready") else "complete_with_preflight_warnings",
        "source":str(input_path),"classification":classification,"stages":stages,
        "truth_policy":{"visible_source_pixels_authoritative":True,"hidden_background_inference_lower_confidence":True,"photography_never_claimed_as_vector":True,"exact_font_never_guessed":True,"duplicate_text_avoided_on_photographic_fallback":True},
        "outputs":{"master_svg":str(master),"editable_pdf":prepress["outputs"]["editable_pdf"],"press_pdf":prepress["outputs"]["press_pdf"],"proof":prepress["outputs"]["proof"],"production_manifest":prepress["outputs"].get("production_manifest"),"preflight":prepress["outputs"]["report"],"report":str(delivery/"reconstruction_report.json")},
    }
    Path(report["outputs"]["report"]).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
