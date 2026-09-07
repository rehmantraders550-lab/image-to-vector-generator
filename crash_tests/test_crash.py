"""Adversarial contracts. Expected rejection is a pass, never a crash finding.

Tests are outside tests/ so the unchanged baseline runs first. Real subprocesses
exercise renderers; fault injection targets only precise dependency boundaries.
"""
import builtins
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
from xml.etree import ElementTree as ET
import zlib

import numpy as np
import pikepdf
from PIL import Image, UnidentifiedImageError
import pytest

from poster_vector_rebuilder import normalize, prepress, segment
from poster_vector_rebuilder.final_assembly import assemble_master_svg


@pytest.fixture
def renderers():
    missing = [name for name in ('inkscape', 'gs') if not prepress.shutil.which(name)]
    if missing:
        if os.environ.get('CRASH_REQUIRE_RENDERERS') == '1':
            pytest.fail('Crash environment missing required renderers: ' + ', '.join(missing))
        pytest.skip('Integration renderer unavailable: ' + ', '.join(missing))


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'reference.png'
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[:] = (25, 110, 210)
    rgb[20:50, 30:60] = (210, 40, 30)
    Image.fromarray(rgb).save(path)
    return path


@pytest.fixture
def master(tmp_path):
    path = tmp_path / 'master.svg'
    assemble_master_svg(path, width=120, height=80)
    return path


@pytest.mark.parametrize('payload', [b'', b'not an image', b'\x89PNG\r\n\x1a\n', b'%PDF-1.6\ntruncated'])
def test_invalid_raster_rejected(tmp_path, payload):
    path = tmp_path / 'bad.png'
    path.write_bytes(payload)
    with pytest.raises((UnidentifiedImageError, OSError, ValueError)) as exc:
        normalize.normalize_reference(path, tmp_path / 'job')
    print(type(exc.value).__name__, str(exc.value))
    assert not (tmp_path / 'job/work/normalized_reference.png').exists()


@pytest.mark.parametrize('value', [None, 42, [], {}])
def test_wrong_path_type_rejected(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        normalize.normalize_reference(value, tmp_path / 'job')


def test_huge_declared_image_rejected_before_decode(tmp_path):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path = tmp_path / 'huge.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 100000, 100000, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b''))
    with pytest.raises(Image.DecompressionBombError) as exc:
        normalize.normalize_reference(path, tmp_path / 'job')
    print(str(exc.value))


def test_large_real_image_bounded_subprocess(tmp_path):
    # 16 megapixels: real allocation/decode/normalization, with an external deadline.
    path = tmp_path / 'large.png'
    Image.new('RGB', (4096, 4096), (40, 110, 210)).save(path)
    run = subprocess.run([sys.executable, '-m', 'poster_vector_rebuilder.cli', 'normalize', str(path), '-o', str(tmp_path / 'large-job')], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)


def test_repeated_run_stable(source, tmp_path):
    job = tmp_path / 'job'
    hashes = []
    for _ in range(4):
        normalize.normalize_reference(source, job)
        segment.segment_reference(job)
        hashes.append({str(p.relative_to(job)): hashlib.sha256(p.read_bytes()).hexdigest() for p in job.rglob('*') if p.is_file()})
    assert all(h == hashes[0] for h in hashes)


def test_independent_dirs_concurrent(source, tmp_path):
    def run(i):
        job = tmp_path / str(i)
        normalize.normalize_reference(source, job)
        segment.segment_reference(job)
        return (job / 'masks/foreground_mask.png').read_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(4)))
    assert all(x == outputs[0] for x in outputs)


def test_same_job_dir_source_consistency(source, tmp_path, monkeypatch):
    # Force a realistic interleaving: B copies after A, before either reads back.
    # Each successful call must describe its own authoritative source, or reject.
    other = tmp_path / 'other.png'
    Image.new('RGB', (63, 47), 'red').save(other)
    expected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, other)}
    copied = threading.Event()
    overwritten = threading.Event()
    original = normalize.shutil.copyfile
    def interleave(src, dst, *args, **kwargs):
        if Path(src) == source:
            result = original(src, dst, *args, **kwargs)
            copied.set()
            assert overwritten.wait(5), 'second writer never arrived'
            return result
        assert copied.wait(5)
        result = original(src, dst, *args, **kwargs)
        overwritten.set()
        return result
    monkeypatch.setattr(normalize.shutil, 'copyfile', interleave)
    def run(path):
        try:
            _, metadata = normalize.preserve_source(path, tmp_path / 'shared')
            return metadata['sha256']
        except (FileExistsError, BlockingIOError):
            overwritten.set()
            return 'busy'
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run, source)
        assert copied.wait(5)
        b = pool.submit(run, other)
        actual = [a.result(), b.result()]
    assert actual[0] in ('busy', expected[source]), {'expected': expected[source], 'actual': actual}
    assert actual[1] in ('busy', expected[other])


@pytest.mark.parametrize('name', ['inkscape', 'gs'])
def test_missing_renderer_clear_rejection(master, tmp_path, monkeypatch, name):
    real = prepress.shutil.which
    monkeypatch.setattr(prepress.shutil, 'which', lambda x: None if x == name else real(x))
    with pytest.raises(RuntimeError, match='required') as exc:
        prepress.export_prepress_package(master, tmp_path / 'out')
    print(str(exc.value))


@pytest.mark.parametrize('backend', ['birefnet_model', 'sam2_model'])
def test_optional_model_dependency_loss(source, tmp_path, monkeypatch, backend):
    real = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'torchvision', 'transformers', 'sam2'}:
            raise ModuleNotFoundError('injected missing optional dependency: ' + name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(RuntimeError, match='requires') as exc:
        segment.segment_reference(tmp_path / 'requested', image_path=source, **{backend: 'unavailable'})
    print(str(exc.value))
    result = segment.segment_reference(tmp_path / 'fallback', image_path=source)
    assert result['backends_used'] == ['opencv-risk']


def test_unicode_long_path(source, tmp_path):
    job = tmp_path / '作品_اردو_é' / ('a' * 100) / ('b' * 100) / ('c' * 100)
    result = normalize.normalize_reference(source, job)
    assert (job / result['normalized_path']).is_file()
    assert (job / result['debug_overlay_path']).is_file()


@pytest.mark.parametrize('text', ['', '<svg', '<svg viewBox="0 0 0 0"/>', '<svg viewBox="0 0 nan inf"/>'])
def test_malformed_svg_rejected(tmp_path, text):
    path = tmp_path / 'bad.svg'
    path.write_text(text)
    with pytest.raises((ET.ParseError, ValueError, OverflowError)) as exc:
        prepress._canvas(path)
    print(type(exc.value).__name__, str(exc.value))


@pytest.mark.parametrize('data', [b'', b'%PDF-1.6\ntruncated', b'wrong type'])
def test_malformed_pdf_rejected(tmp_path, data):
    path = tmp_path / 'bad.pdf'
    path.write_bytes(data)
    with pytest.raises(pikepdf.PdfError) as exc:
        prepress._pdf_structural_report(path, {'bleed_mm': 3}, None, None)
    print(str(exc.value))


def test_malformed_icc_rejected(master, tmp_path, renderers):
    icc = tmp_path / 'invalid.icc'
    icc.write_bytes(b'not an ICC profile')
    with pytest.raises((RuntimeError, ValueError, OSError)) as exc:
        prepress.export_prepress_package(master, tmp_path / 'out', icc_profile=icc)
    print(type(exc.value).__name__, str(exc.value))
    assert not (tmp_path / 'out/preflight_report.json').exists()


@pytest.mark.parametrize('function,field', [('_pdfimages_report', 'all_at_least_300'), ('_pdffonts_report', 'all_embedded')])
def test_failed_inspection_never_reports_success(tmp_path, monkeypatch, function, field):
    monkeypatch.setattr(prepress, '_run', lambda cmd: {'command': list(map(str, cmd)), 'returncode': 17, 'stdout': '', 'stderr': 'injected prepress inspection failure'})
    result = getattr(prepress, function)(tmp_path / 'test.pdf', 'injected-tool')
    assert result[field] is False, result


def test_renderer_nonzero_rejected(master, tmp_path, monkeypatch, renderers):
    monkeypatch.setattr(prepress, '_run', lambda cmd: {'command': list(map(str, cmd)), 'returncode': 23, 'stdout': '', 'stderr': 'injected renderer failure'})
    with pytest.raises(RuntimeError, match='failed'):
        prepress.export_prepress_package(master, tmp_path / 'out')


@pytest.mark.parametrize('function,field', [('_pdfimages_report', 'all_at_least_300'), ('_pdffonts_report', 'all_embedded')])
def test_real_inspector_failure(tmp_path, function, field, renderers):
    tool = 'pdfimages' if function == '_pdfimages_report' else 'pdffonts'
    executable = prepress.shutil.which(tool)
    assert executable, tool
    result = getattr(prepress, function)(tmp_path / 'missing.pdf', executable)
    assert result['command']['returncode'] != 0, result
    assert result[field] is False, result
    print(result)


@pytest.mark.parametrize('function,field', [('_pdfimages_report', 'all_at_least_300'), ('_pdffonts_report', 'all_embedded')])
def test_successful_empty_inspection_keeps_existing_semantics(tmp_path, monkeypatch, function, field):
    monkeypatch.setattr(prepress, '_run', lambda cmd: {'returncode': 0, 'stdout': '', 'stderr': ''})
    assert getattr(prepress, function)(tmp_path / 'no-images-or-fonts.pdf', 'tool')[field] is True


def test_delivery_repeated_and_independent_processes(source, tmp_path, renderers):
    def deliver(job):
        run = subprocess.run([sys.executable, '-m', 'poster_vector_rebuilder.cli', 'deliver', str(source), '-o', str(job)], capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
        report = json.loads((job / 'delivery/reconstruction_report.json').read_text())
        assert report['source'] == str(source)
        for key in ('master_svg', 'editable_pdf', 'press_pdf', 'proof', 'preflight', 'production_manifest'):
            assert Path(report['outputs'][key]).stat().st_size > 0, (key, report)
        with pikepdf.open(report['outputs']['press_pdf']) as pdf:
            assert len(pdf.pages) == 1
        return (job / 'delivery/artwork_master.svg').read_bytes()
    job = tmp_path / 'delivery-job'
    baseline = deliver(job)
    assert deliver(job) == baseline
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(deliver, [tmp_path / 'independent-a', tmp_path / 'independent-b']))
    assert outputs == [baseline, baseline]


def test_busy_delivery_rejected_before_writes(source, tmp_path):
    from poster_vector_rebuilder.job_lock import job_lock
    job = tmp_path / 'busy'
    with job_lock(job):
        before = {p.name: p.read_bytes() for p in job.iterdir()}
        run = subprocess.run([sys.executable, '-m', 'poster_vector_rebuilder.cli', 'deliver', str(source), '-o', str(job)], capture_output=True, text=True, timeout=15)
        assert run.returncode != 0
        assert 'already in use' in run.stderr, run.stderr
        assert {p.name: p.read_bytes() for p in job.iterdir()} == before
