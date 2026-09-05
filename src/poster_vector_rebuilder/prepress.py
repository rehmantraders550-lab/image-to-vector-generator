from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zlib

import numpy as np
from PIL import Image
import pikepdf

from .production_semantics import build_production_manifest, inspect_svg_production_semantics

REQUIRED_LAYERS=("00_BACKGROUND","10_HERO","20_BRAND","30_DECORATION","40_ICONS","90_PREPRESS")
SVG_NS="http://www.w3.org/2000/svg"
XLINK_NS="http://www.w3.org/1999/xlink"
ET.register_namespace("",SVG_NS)
ET.register_namespace("xlink",XLINK_NS)


def _run(cmd):
    proc=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,check=False)
    return {"command":[str(x) for x in cmd],"returncode":proc.returncode,"stdout":proc.stdout[-4000:],"stderr":proc.stderr[-4000:]}


def _local(tag): return tag.rsplit("}",1)[-1]


def svg_preflight(svg_path: str | Path) -> dict:
    root=ET.parse(svg_path).getroot(); tags=[_local(e.tag) for e in root.iter()]
    layer_ids={e.get("id") for e in root if _local(e.tag)=="g"}
    images=[e for e in root.iter() if _local(e.tag)=="image"]
    raster_status=[]
    for image in images:
        href=image.get("href") or image.get(f"{{{XLINK_NS}}}href") or ""
        raster_status.append({"id":image.get("id"),"href":href,"declared_status":image.get("data-vector-status")})
    checks={"viewbox":bool(root.get("viewBox")),"required_layers":all(x in layer_ids for x in REQUIRED_LAYERS),"no_unsupported_dom":not any(t in {"foreignObject","flowRoot","script"} for t in tags),"node_count_practical":len(tags)<=12000,"raster_images_declared":all(x["declared_status"]=="raster-photographic-fallback" for x in raster_status)}
    return {"checks":checks,"passed":all(checks.values()),"node_count":len(tags),"text_count":tags.count("text"),"raster_images":raster_status,"layer_ids":sorted(x for x in layer_ids if x)}


def _canvas(svg_path: str | Path) -> tuple[ET.Element,int,int]:
    root=ET.parse(svg_path).getroot(); vb=root.get("viewBox")
    if not vb: raise ValueError("Master SVG must have a viewBox")
    values=[float(x) for x in vb.replace(","," ").split()]
    if len(values)!=4 or values[2]<=0 or values[3]<=0: raise ValueError(f"Invalid SVG viewBox: {vb}")
    return root,int(round(values[2])),int(round(values[3]))


def _trim_geometry(width_px:int,height_px:int,*,target_ppi:float,bleed_mm:float,trim_width_mm:float|None,trim_height_mm:float|None) -> dict:
    if target_ppi<=0: raise ValueError("target_ppi must be positive")
    if bleed_mm<0: raise ValueError("bleed_mm cannot be negative")
    aspect=width_px/max(height_px,1); explicit=trim_width_mm is not None or trim_height_mm is not None
    if trim_width_mm is None and trim_height_mm is None:
        trim_width_mm=width_px/target_ppi*25.4; trim_height_mm=height_px/target_ppi*25.4; basis="derived_from_canvas_at_target_ppi"
    elif trim_width_mm is None:
        if trim_height_mm is None or trim_height_mm<=0: raise ValueError("trim_height_mm must be positive")
        trim_width_mm=trim_height_mm*aspect; basis="height_explicit_width_derived_from_aspect"
    elif trim_height_mm is None:
        if trim_width_mm<=0: raise ValueError("trim_width_mm must be positive")
        trim_height_mm=trim_width_mm/aspect; basis="width_explicit_height_derived_from_aspect"
    else:
        if trim_width_mm<=0 or trim_height_mm<=0: raise ValueError("trim dimensions must be positive")
        if abs((trim_width_mm/trim_height_mm)-aspect)/aspect>0.005: raise ValueError("Explicit trim dimensions must preserve source aspect ratio within 0.5%")
        basis="explicit_width_and_height"
    x_ppi=width_px/(float(trim_width_mm)/25.4); y_ppi=height_px/(float(trim_height_mm)/25.4)
    bx=float(bleed_mm)/float(trim_width_mm)*width_px if bleed_mm else 0.0; by=float(bleed_mm)/float(trim_height_mm)*height_px if bleed_mm else 0.0
    render_w=max(1,int(math.ceil(float(trim_width_mm)/25.4*target_ppi))); render_h=max(1,int(math.ceil(float(trim_height_mm)/25.4*target_ppi))); bleed_render=max(0,int(math.ceil(float(bleed_mm)/25.4*target_ppi)))
    return {"trim_width_mm":float(trim_width_mm),"trim_height_mm":float(trim_height_mm),"page_width_mm":float(trim_width_mm)+2*float(bleed_mm),"page_height_mm":float(trim_height_mm)+2*float(bleed_mm),"bleed_mm":float(bleed_mm),"target_ppi":float(target_ppi),"effective_source_ppi_x":float(x_ppi),"effective_source_ppi_y":float(y_ppi),"source_meets_target_ppi":min(x_ppi,y_ppi)+0.5>=target_ppi,"bleed_units_x":bx,"bleed_units_y":by,"page_viewbox_width":width_px+2*bx,"page_viewbox_height":height_px+2*by,"target_trim_render_width_px":render_w,"target_trim_render_height_px":render_h,"target_bleed_render_px":bleed_render,"trim_size_basis":basis,"production_dimensions_confirmed":bool(explicit)}


def _write_physical_svg(root:ET.Element,path:Path,g:dict,width_px:int,height_px:int,*,bleed_data_uri:str|None=None) -> None:
    if bleed_data_uri is None:
        out=ET.Element(f"{{{SVG_NS}}}svg",{"width":f"{g['trim_width_mm']:.6f}mm","height":f"{g['trim_height_mm']:.6f}mm","viewBox":f"0 0 {width_px} {height_px}"})
        for child in root: out.append(deepcopy(child))
    else:
        bx=g["bleed_units_x"]; by=g["bleed_units_y"]; pw=g["page_viewbox_width"]; ph=g["page_viewbox_height"]
        out=ET.Element(f"{{{SVG_NS}}}svg",{"width":f"{g['page_width_mm']:.6f}mm","height":f"{g['page_height_mm']:.6f}mm","viewBox":f"0 0 {pw:.6f} {ph:.6f}"})
        ET.SubElement(out,f"{{{SVG_NS}}}image",{"id":"generated-bleed-extension","x":"0","y":"0","width":f"{pw:.6f}","height":f"{ph:.6f}","preserveAspectRatio":"none","href":bleed_data_uri,f"{{{XLINK_NS}}}href":bleed_data_uri,"data-prepress":"generated-edge-bleed","data-raster-status":"bleed-only"})
        group=ET.SubElement(out,f"{{{SVG_NS}}}g",{"id":"TRIM_ARTWORK","transform":f"translate({bx:.6f},{by:.6f})"})
        for child in root: group.append(deepcopy(child))
    ET.ElementTree(out).write(path,encoding="utf-8",xml_declaration=True)


def _bleed_data_uri(master_svg:Path,inkscape:str,g:dict,temp:Path,commands:list[dict]) -> str:
    render=temp/"trim_target_ppi.png"; rw=g["target_trim_render_width_px"]; rh=g["target_trim_render_height_px"]
    commands.append(_run([inkscape,master_svg,"--export-type=png",f"--export-filename={render}",f"--export-width={rw}",f"--export-height={rh}"]))
    if commands[-1]["returncode"]!=0 or not render.exists(): raise RuntimeError("Target-PPI trim render for bleed construction failed")
    rgb=np.asarray(Image.open(render).convert("RGB"),dtype=np.uint8); b=int(g["target_bleed_render_px"])
    if b<=0: rgba=np.dstack([rgb,np.zeros(rgb.shape[:2],dtype=np.uint8)])
    else:
        padded=np.pad(rgb,((b,b),(b,b),(0,0)),mode="reflect"); alpha=np.full(padded.shape[:2],255,dtype=np.uint8); alpha[b:b+rh,b:b+rw]=0; rgba=np.dstack([padded,alpha])
    bleed=temp/"bleed_only.png"; Image.fromarray(rgba,"RGBA").save(bleed,optimize=True)
    return "data:image/png;base64,"+base64.b64encode(bleed.read_bytes()).decode("ascii")


def _first_existing(paths:list[str|Path]) -> Path|None:
    for p in paths:
        if p and Path(p).is_file(): return Path(p)
    return None


def _icc_profiles(explicit:str|Path|None=None) -> tuple[Path|None,Path|None,str]:
    cmyk=_first_existing([explicit or "",os.environ.get("POSTER_VECTOR_CMYK_ICC",""),"/usr/share/texlive/texmf-dist/tex/generic/colorprofiles/FOGRA39L_coated.icc","/usr/share/color/icc/FOGRA39L_coated.icc","/usr/share/color/icc/ghostscript/default_cmyk.icc"])
    rgb=_first_existing([os.environ.get("POSTER_VECTOR_RGB_ICC",""),"/usr/share/color/icc/ghostscript/srgb.icc","/usr/share/texlive/texmf-dist/tex/generic/colorprofiles/sRGB.icc","/usr/share/color/icc/ghostscript/default_rgb.icc"])
    if cmyk is None: return None,rgb,"unavailable"
    return cmyk,rgb,("FOGRA39L Coated" if "fogra39" in cmyk.name.lower() else cmyk.stem)


def _apply_boxes(page,g:dict) -> None:
    bleed_pt=g["bleed_mm"]/25.4*72.0; pw=g["page_width_mm"]/25.4*72.0; ph=g["page_height_mm"]/25.4*72.0
    media=pikepdf.Array([0,0,pw,ph]); trim=pikepdf.Array([bleed_pt,bleed_pt,pw-bleed_pt,ph-bleed_pt])
    page.MediaBox=media; page.CropBox=media; page.BleedBox=media; page.TrimBox=trim; page.ArtBox=trim


def _apply_output_intent(pdf:pikepdf.Pdf,icc:Path,profile_name:str) -> None:
    icc_stream=pdf.make_stream(icc.read_bytes()); icc_stream["/N"]=4
    intent=pdf.make_indirect(pikepdf.Dictionary({"/Type":pikepdf.Name("/OutputIntent"),"/S":pikepdf.Name("/GTS_PDFX"),"/OutputConditionIdentifier":profile_name,"/Info":f"{profile_name} automated press output intent","/DestOutputProfile":icc_stream}))
    pdf.Root["/OutputIntents"]=pikepdf.Array([intent]); pdf.docinfo["/GTS_PDFXVersion"]="PDF/X-4"; pdf.docinfo["/Creator"]="Poster Vector Rebuilder"; pdf.docinfo["/Subject"]="CMYK press PDF with explicit bleed/trim and ICC output intent; PDF/X-4-oriented"


def _set_pdf_geometry_and_output_intent(source:Path,dest:Path,g:dict,icc:Path,profile_name:str) -> None:
    with pikepdf.open(source) as pdf:
        for page in pdf.pages:
            _apply_boxes(page,g)
            if "/Group" in page.obj:
                group=page.obj["/Group"]
                if isinstance(group,pikepdf.Dictionary) and group.get("/S")==pikepdf.Name("/Transparency"): group["/CS"]=pikepdf.Name("/DeviceCMYK")
        _apply_output_intent(pdf,icc,profile_name); pdf.save(dest,min_version="1.6",force_version="1.6",object_stream_mode=pikepdf.ObjectStreamMode.generate)


def _make_flat_cmyk_pdf(tiff_path:Path,dest:Path,g:dict,icc:Path,profile_name:str) -> None:
    with Image.open(tiff_path) as im:
        cmyk=im.convert("CMYK"); width,height=cmyk.size; raw=cmyk.tobytes()
    pdf=pikepdf.Pdf.new(); pw=g["page_width_mm"]/25.4*72.0; ph=g["page_height_mm"]/25.4*72.0; page=pdf.add_blank_page(page_size=(pw,ph)); _apply_boxes(page,g)
    image=pdf.make_stream(zlib.compress(raw,6)); image["/Type"]=pikepdf.Name("/XObject"); image["/Subtype"]=pikepdf.Name("/Image"); image["/Width"]=width; image["/Height"]=height; image["/ColorSpace"]=pikepdf.Name("/DeviceCMYK"); image["/BitsPerComponent"]=8; image["/Filter"]=pikepdf.Name("/FlateDecode")
    page.obj["/Resources"]=pikepdf.Dictionary({"/XObject":pikepdf.Dictionary({"/Im0":image})}); page.obj["/Contents"]=pdf.make_stream(f"q {pw:.8f} 0 0 {ph:.8f} 0 0 cm /Im0 Do Q\n".encode("ascii"))
    _apply_output_intent(pdf,icc,profile_name); pdf.save(dest,min_version="1.6",force_version="1.6",object_stream_mode=pikepdf.ObjectStreamMode.generate); pdf.close()


def _flatten_press(source:Path,dest:Path,g:dict,gs:str,cmyk_icc:Path,rgb_icc:Path|None,profile_name:str,temp:Path,commands:list[dict]) -> None:
    tif=temp/"press_flat_300ppi.tif"; cmd=[gs,"-q","-dSAFER"]
    for profile in (cmyk_icc,rgb_icc):
        if profile: cmd.append(f"--permit-file-read={profile}")
    cmd += ["-dBATCH","-dNOPAUSE","-sDEVICE=tiff32nc",f"-r{g['target_ppi']}","-dUseCropBox","-sColorConversionStrategy=CMYK","-dProcessColorModel=/DeviceCMYK"]
    if rgb_icc: cmd.append(f"-sDefaultRGBProfile={rgb_icc}")
    cmd += [f"-sDefaultCMYKProfile={cmyk_icc}",f"-sOutputICCProfile={cmyk_icc}",f"-sOutputFile={tif}",source]
    commands.append(_run(cmd))
    if commands[-1]["returncode"]!=0 or not tif.exists(): raise RuntimeError("300 PPI CMYK press flatten fallback failed")
    _make_flat_cmyk_pdf(tif,dest,g,cmyk_icc,profile_name)


def _name_count(obj,target:pikepdf.Name,seen:set[tuple[int,int]]) -> int:
    try:
        og=getattr(obj,"objgen",None)
        if og and og!=(0,0):
            key=(int(og[0]),int(og[1]))
            if key in seen: return 0
            seen.add(key)
        if isinstance(obj,pikepdf.Name): return int(obj==target)
        if isinstance(obj,(pikepdf.Dictionary,pikepdf.Stream)): return sum(_name_count(v,target,seen) for _,v in obj.items())
        if isinstance(obj,pikepdf.Array): return sum(_name_count(v,target,seen) for v in obj)
    except Exception: return 0
    return 0


def _pdfimages_report(pdf:Path,pdfimages:str|None) -> dict:
    if not pdfimages: return {"available":False,"images":[],"min_ppi":None,"all_at_least_300":False}
    run=_run([pdfimages,"-list",pdf]); images=[]
    if run["returncode"]==0:
        for line in run["stdout"].splitlines():
            parts=line.split()
            if len(parts)>=16 and parts[0].isdigit() and parts[1].isdigit() and parts[2] in {"image","smask","mask"}:
                try: images.append({"page":int(parts[0]),"num":int(parts[1]),"type":parts[2],"width":int(parts[3]),"height":int(parts[4]),"color":parts[5],"x_ppi":float(parts[12]),"y_ppi":float(parts[13])})
                except ValueError: pass
    effective=[min(i["x_ppi"],i["y_ppi"]) for i in images if i["type"]=="image"]; minimum=min(effective) if effective else None
    return {"available":True,"command":run,"images":images,"min_ppi":minimum,"all_at_least_300":minimum is None or minimum>=299.0}


def _pdffonts_report(pdf:Path,pdffonts:str|None) -> dict:
    if not pdffonts: return {"available":False,"font_count":None,"all_embedded":False}
    run=_run([pdffonts,pdf]); embedded=[]
    if run["returncode"]==0:
        for line in run["stdout"].splitlines():
            m=re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$",line)
            if m: embedded.append(m.group(1)=="yes")
    return {"available":True,"command":run,"font_count":len(embedded),"all_embedded":all(embedded)}


def _pdf_structural_report(pdf_path:Path,g:dict,pdfimages:str|None,pdffonts:str|None) -> dict:
    with pikepdf.open(pdf_path) as pdf:
        page=pdf.pages[0]; boxes={name:[float(x) for x in page.obj.get(name,[])] for name in ("/MediaBox","/CropBox","/BleedBox","/TrimBox","/ArtBox")}; rgb=sum(_name_count(obj,pikepdf.Name("/DeviceRGB"),set()) for obj in pdf.objects); cmyk=sum(_name_count(obj,pikepdf.Name("/DeviceCMYK"),set()) for obj in pdf.objects); intents=len(pdf.Root.get("/OutputIntents",[])); version=str(pdf.pdf_version); pdfx=str(pdf.docinfo.get("/GTS_PDFXVersion",""))
    expected=g["bleed_mm"]/25.4*72.0; trim=boxes["/TrimBox"]; media=boxes["/MediaBox"]; measured=min(trim[0]-media[0],trim[1]-media[1],media[2]-trim[2],media[3]-trim[3]) if len(trim)==4 and len(media)==4 else -1; images=_pdfimages_report(pdf_path,pdfimages); fonts=_pdffonts_report(pdf_path,pdffonts)
    checks={"pdf_version_at_least_1_6":tuple(int(x) for x in version.split(".")[:2])>=(1,6),"media_bleed_trim_art_boxes_present":all(len(boxes[x])==4 for x in ("/MediaBox","/BleedBox","/TrimBox","/ArtBox")),"bleed_at_least_requested":measured+0.05>=expected,"output_intent_present":intents>=1,"pdfx4_declared":pdfx=="PDF/X-4","device_rgb_absent":rgb==0,"device_cmyk_present":cmyk>0,"raster_images_min_300_ppi":images["all_at_least_300"],"fonts_embedded":fonts["all_embedded"]}
    return {"checks":checks,"passed":all(checks.values()),"pdf_version":version,"pdfx_declaration":pdfx,"boxes":boxes,"measured_bleed_pt":measured,"device_rgb_name_count":rgb,"device_cmyk_name_count":cmyk,"output_intent_count":intents,"images":images,"fonts":fonts}


def export_prepress_package(master_svg:str|Path,output_dir:str|Path,*,proof_dpi:int=150,target_ppi:float=300.0,bleed_mm:float=3.0,trim_width_mm:float|None=None,trim_height_mm:float|None=None,icc_profile:str|Path|None=None) -> dict:
    """Export vector-editable PDF and a CMYK, bleed-aware press PDF.

    If a renderer introduces sub-300-PPI rasterization, only the press PDF is
    deterministically flattened to target PPI CMYK. The master SVG and editable PDF stay vector.
    """
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); master_svg=Path(master_svg); inkscape=shutil.which("inkscape"); gs=shutil.which("gs"); qpdf=shutil.which("qpdf"); pdfcpu=shutil.which("pdfcpu"); pdfimages=shutil.which("pdfimages"); pdffonts=shutil.which("pdffonts")
    if not inkscape: raise RuntimeError("Inkscape is required for vector PDF/proof export")
    if not gs: raise RuntimeError("Ghostscript is required for CMYK press PDF generation")
    cmyk_icc,rgb_icc,profile_name=_icc_profiles(icc_profile)
    if cmyk_icc is None: raise RuntimeError("A CMYK ICC profile is required for press PDF generation")
    root,width_px,height_px=_canvas(master_svg); g=_trim_geometry(width_px,height_px,target_ppi=target_ppi,bleed_mm=bleed_mm,trim_width_mm=trim_width_mm,trim_height_mm=trim_height_mm); svg=svg_preflight(master_svg); production_semantics=inspect_svg_production_semantics(master_svg); source_has_raster=bool(svg["raster_images"])
    editable=output_dir/"artwork_editable.pdf"; press=output_dir/"artwork_press.pdf"; proof=output_dir/"artwork_proof.png"; commands=[]; press_fallback=None
    with tempfile.TemporaryDirectory(prefix="prepress-",dir=output_dir) as td:
        temp=Path(td); trim_svg=temp/"trim.svg"; press_svg=temp/"press.svg"; raw=temp/"press_raw.pdf"; cmyk=temp/"press_cmyk.pdf"; direct=temp/"press_direct.pdf"
        bleed_uri=_bleed_data_uri(master_svg,inkscape,g,temp,commands) if bleed_mm>0 else None; _write_physical_svg(root,trim_svg,g,width_px,height_px); _write_physical_svg(root,press_svg,g,width_px,height_px,bleed_data_uri=bleed_uri)
        commands.append(_run([inkscape,trim_svg,"--export-type=pdf",f"--export-dpi={target_ppi}",f"--export-filename={editable}"]))
        if commands[-1]["returncode"]!=0 or not editable.exists(): raise RuntimeError("Inkscape editable PDF export failed")
        commands.append(_run([inkscape,press_svg,"--export-type=pdf",f"--export-dpi={target_ppi}",f"--export-filename={raw}"]))
        if commands[-1]["returncode"]!=0 or not raw.exists(): raise RuntimeError("Inkscape bleed-aware press PDF export failed")
        gs_cmd=[gs,"-q","-dSAFER"]
        for profile in (cmyk_icc,rgb_icc):
            if profile: gs_cmd.append(f"--permit-file-read={profile}")
        gs_cmd += ["-dBATCH","-dNOPAUSE","-sDEVICE=pdfwrite","-dPDFSETTINGS=/prepress","-dCompatibilityLevel=1.6","-dEmbedAllFonts=true","-dSubsetFonts=true","-sColorConversionStrategy=CMYK","-dProcessColorModel=/DeviceCMYK"]
        if rgb_icc: gs_cmd.append(f"-sDefaultRGBProfile={rgb_icc}")
        gs_cmd += [f"-sDefaultCMYKProfile={cmyk_icc}",f"-sOutputICCProfile={cmyk_icc}","-dBlackPtComp=1","-dKPreserve=2",f"-sOutputFile={cmyk}",raw]; commands.append(_run(gs_cmd))
        if commands[-1]["returncode"]!=0 or not cmyk.exists(): raise RuntimeError("Ghostscript CMYK press conversion failed")
        _set_pdf_geometry_and_output_intent(cmyk,direct,g,cmyk_icc,profile_name); direct_images=_pdfimages_report(direct,pdfimages)
        if direct_images["all_at_least_300"]: shutil.copyfile(direct,press); press_fallback="not_needed"
        else:
            _flatten_press(direct,press,g,gs,cmyk_icc,rgb_icc,profile_name,temp,commands); press_fallback="300ppi_cmyk_flatten_due_renderer_rasterization"
    commands.append(_run([inkscape,master_svg,"--export-type=png",f"--export-filename={proof}",f"--export-width={width_px}",f"--export-height={height_px}"]))
    if commands[-1]["returncode"]!=0 or not proof.exists(): raise RuntimeError("Proof generation failed")
    validations=[{"tool":"ghostscript",**_run([gs,"-q","-dNOPAUSE","-dBATCH","-sDEVICE=nullpage",press])}]; diagnostics=[]
    if qpdf: validations.append({"tool":"qpdf",**_run([qpdf,"--check",press])})
    if pdfcpu: validations.append({"tool":"pdfcpu","mode":"relaxed-default",**_run([pdfcpu,"validate",press])}); diagnostics.append({"tool":"pdfcpu_strict","mode":"strict-diagnostic",**_run([pdfcpu,"validate","-m","strict",press])})
    pdf=_pdf_structural_report(press,g,pdfimages,pdffonts); tool_checks={v["tool"]:v["returncode"]==0 for v in validations}; diagnostic_checks={v["tool"]:v["returncode"]==0 for v in diagnostics}; available={"inkscape":bool(inkscape),"ghostscript":bool(gs),"qpdf":bool(qpdf),"pdfcpu":bool(pdfcpu),"pdfimages":bool(pdfimages),"pdffonts":bool(pdffonts),"pikepdf":True}; missing=[name for name in ("ghostscript","qpdf","pdfcpu","pdfimages","pdffonts") if not available[name]]; warnings=[]
    if not g["production_dimensions_confirmed"]: warnings.append("Trim size was not supplied; physical size was conservatively derived from the canvas at target PPI. Confirm production dimensions before manufacturing.")
    if source_has_raster and not g["source_meets_target_ppi"]: warnings.append(f"Photographic raster is only {min(g['effective_source_ppi_x'],g['effective_source_ppi_y']):.1f} PPI at requested trim size; press-ready status requires at least {target_ppi:.0f} PPI source detail.")
    if press_fallback!="not_needed": warnings.append("Direct vector CMYK conversion introduced sub-300-PPI renderer rasterization; press PDF was flattened to target PPI CMYK. Editable PDF and master SVG remain vector.")
    if pdfcpu and not diagnostic_checks.get("pdfcpu_strict",True): warnings.append("pdfcpu strict diagnostic reported a conformance issue; default validation remains the interoperability gate.")
    for item in production_semantics["warnings"]: warnings.append(item["message"])
    if production_semantics["blockers"]: warnings.extend(item["message"] for item in production_semantics["blockers"])
    if production_semantics["has_spot_colors"]:
        warnings.append("Named spot/finish semantics are present; the established CMYK conversion remains unchanged and is not claimed to preserve spot separations. A dedicated separation-aware export/RIP review is required before press-ready approval.")
    technical=svg["passed"] and pdf["passed"] and all(tool_checks.values()) and not missing and production_semantics["valid"]; source_ok=(not source_has_raster) or g["source_meets_target_ppi"]
    spot_route_safe=not production_semantics["has_spot_colors"]
    icc_data={"profile":str(cmyk_icc),"profile_name":profile_name,"rgb_profile":str(rgb_icc) if rgb_icc else None}
    manifest_path=output_dir/"production_manifest.json"
    production_manifest=build_production_manifest(master_svg,manifest_path,geometry=g,icc=icc_data,pdf_target="PDF/X-4-oriented")
    report={"schema":"poster-vector-preflight-v3","passed":technical,"press_ready":technical and g["production_dimensions_confirmed"] and source_ok and spot_route_safe,"svg":svg,"pdf":pdf,"geometry":g,"production_semantics":production_semantics,"press_rendering":{"mode":press_fallback,"editable_pdf_remains_vector":True},"icc":icc_data,"tool_checks":tool_checks,"diagnostic_checks":diagnostic_checks,"tools":available,"missing_required_validators":missing,"commands":commands,"validations":validations,"diagnostics":diagnostics,"warnings":warnings,"source_rules":{"target":"PDF/X-4-oriented","minimum_raster_ppi":300,"minimum_bleed_mm":3.0,"press_color_space":"CMYK","fonts":"embedded or outlined","spot_colors":"explicit semantics only; dedicated separation-aware export required"},"pdfcpu_policy":"Use pdfcpu's official default relaxed validation for acceptance; retain strict mode as a non-blocking diagnostic.","press_pdf_policy":"CMYK PDF 1.6 with explicit MediaBox/BleedBox/TrimBox/ArtBox, ICC OutputIntent and a PDF/X-4 declaration. External PDF/X certification is not claimed without a dedicated PDF/X validator. Named spot-color preservation is not claimed by this CMYK route.","outputs":{"editable_pdf":str(editable),"press_pdf":str(press),"proof":str(proof),"production_manifest":str(manifest_path),"report":str(output_dir/"preflight_report.json")}}
    Path(report["outputs"]["report"]).write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
