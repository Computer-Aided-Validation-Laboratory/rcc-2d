"""Shared, rectangular-camera render machinery for Experiment 3.

The Exp1/2 helpers intentionally assumed a square, single element camera.
Experiment 3 keeps its geometry and camera conventions explicit here so the
1020 x 252 finite-star case cannot accidentally inherit those assumptions.
"""

from __future__ import annotations

import os
import hashlib
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image
import pyvista as pv
import riley
from scipy.ndimage import gaussian_filter, map_coordinates

from modules.exp1common import build_pv_mesh
from modules.exp2speckint2d import make_speckle_pattern
from modules.exp_common_render import (
    analytic_disk_coverage,
    analytic_gaussian_coverage,
    is_rigid_inverse,
)
from modules.render_outputs import float_and_depths_complete, save_float_and_depths, write_camera_depths
from modules.render_logging import case_label, render_log
from modules.render_selection import uint_textures_enabled
from modules.texture_preview import write_preview_b8
from modules.output_naming import case_name, config_name, output_root
from exp0params_common import TEXGEN_JOBS
from exp3params import (
    BACKGROUND, BIT_DEPTHS, CASE_CAMERA_PIXELS, CASE_ROI_SIZES,
    DEFORMATION_CASES, EGGBOX_PERIOD_FINAL_PX, FORCE_RENDER_OVER, GAMMA, I0,
    MAPPING_MODES, PSF_SIGMA_FINAL_PX, PSF_SUPPORT_SIGMAS, RILEY_RASTER_THREADS,
    SSAA_LEVELS, RILEY_TEXTURE_SAMPLERS, TEX_OVERSAMPLES, TEX_PX_PAD,
    additive_jitter_for, BLACK_AREA_FRACTIONS, RANDOM_SEED,
    GAUSSIAN_CUTOFF_SIGMAS, GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
    GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
)


def coverage_to_intensity(coverage: np.ndarray) -> np.ndarray:
    """Convert post-pixel additive coverage to the final camera intensity.

    Additive textures and Riley's float texture raster stay as raw coverage:
    one isolated blob contributes one and overlaps may exceed one.  Saturation
    is deliberately applied only here, after pixel integration.
    """
    clipped = np.clip(np.asarray(coverage, dtype=np.float64), 0.0, 1.0)
    return np.clip(I0 + GAMMA * (1.0 - 2.0 * clipped), 0.0, 1.0)


def _levels(name: str, defaults: list[int]) -> list[int]:
    value = os.environ.get(name)
    return defaults if not value else [int(v) for v in value.split(",") if v.strip()]


def ssaa_levels() -> list[int]: return _levels("EXP3_SSAA_LEVELS", SSAA_LEVELS)
def oversamples() -> list[int]: return _levels("EXP3_TEX_OVERSAMPLES", TEX_OVERSAMPLES)
def bit_depths() -> list[int]: return _levels("EXP3_BIT_DEPTHS", BIT_DEPTHS)


TEXGEN_WORKERS = max(1, int(os.environ.get("EXP3_TEXGEN_JOBS", str(TEXGEN_JOBS))))
_texgen_pattern = None
_texgen_pattern_type: str | None = None
_texgen_x_start: np.ndarray | None = None
_texgen_y_top: float | None = None
_texgen_texel_size: tuple[float, float] | None = None
_eggbox_x_average: np.ndarray | None = None
_eggbox_y_average: np.ndarray | None = None


def _init_speckle_texgen_worker(
    pattern_type: str,
    speckle_size: float,
    black_fraction: float,
    distribution: str,
    jitter: float,
    bounds: tuple[float, float, float, float],
    x_start: np.ndarray,
    y_top: float,
    sx: float,
    sy: float,
) -> None:
    """Initialise one immutable analytic speckle field per texgen worker."""
    global _texgen_pattern, _texgen_pattern_type, _texgen_x_start
    global _texgen_y_top, _texgen_texel_size
    _texgen_pattern = make_speckle_pattern(
        pattern_type, speckle_size, black_fraction, distribution, jitter,
        RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS, bounds, I0, GAMMA,
        GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
        GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
    )
    _texgen_pattern_type = pattern_type
    _texgen_x_start = x_start
    _texgen_y_top = y_top
    _texgen_texel_size = (sx, sy)


def _integrate_speckle_texgen_rows(task: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    """Return one independently integrated texture row batch."""
    if (
        _texgen_pattern is None or _texgen_pattern_type is None
        or _texgen_x_start is None or _texgen_y_top is None
        or _texgen_texel_size is None
    ):
        raise RuntimeError("Exp3 texture worker was not initialised.")
    start_row, end_row = task
    sx, sy = _texgen_texel_size
    y_start = _texgen_y_top - (np.arange(start_row, end_row) + 1.0) * sy
    xx, yy = np.meshgrid(_texgen_x_start, y_start)
    coverage = (
        _texgen_pattern.evaluate_diskaddsat_box_average(xx, yy, sx, sy)
        if _texgen_pattern_type == "diskaddsat" else
        _texgen_pattern.evaluate_gausscont_box_average(xx, yy, sx, sy)
    )
    return start_row, end_row, coverage


def _init_eggbox_texgen_worker(
    x_average: np.ndarray,
    y_average: np.ndarray,
) -> None:
    """Share immutable 1-D exact box averages with one eggbox worker."""
    global _eggbox_x_average, _eggbox_y_average
    _eggbox_x_average = x_average
    _eggbox_y_average = y_average


def _integrate_eggbox_texgen_rows(task: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    """Evaluate an exact separable eggbox average for one row batch."""
    if _eggbox_x_average is None or _eggbox_y_average is None:
        raise RuntimeError("Exp3 eggbox texture worker was not initialised.")
    start_row, end_row = task
    values = (
        I0 + 0.5 * GAMMA * (1.0 + _eggbox_x_average[None, :])
        * (1.0 + _eggbox_y_average[start_row:end_row, None]) - GAMMA
    )
    return start_row, end_row, values


def selected_cases() -> list[str]:
    value = os.environ.get("EXP3_CASES")
    return DEFORMATION_CASES if not value else [v.strip() for v in value.split(",") if v.strip()]


def force_render() -> bool:
    """Permit explicit subset reruns without changing the shared experiment flag."""
    return FORCE_RENDER_OVER or os.environ.get("EXP3_FORCE_RENDER_OVER") == "1"


def selected_frames(case: str, available: int) -> list[int]:
    value = os.environ.get("EXP3_FRAMES")
    if value:
        return [int(v) for v in value.split(",") if v.strip()]
    return list(range(available)) if "chirp" not in case else [0, min(1, available - 1)]


def eggbox_pitch_world(case: str) -> tuple[float, float]:
    """Return the eggbox period in physical world units for one final camera."""
    width, height = CASE_CAMERA_PIXELS[case]
    roi_x, roi_y = CASE_ROI_SIZES[case]
    return (
        EGGBOX_PERIOD_FINAL_PX * roi_x / width,
        EGGBOX_PERIOD_FINAL_PX * roi_y / height,
    )


def texture_config_dir(case: str, pattern: str, oversamp: int, storage: str) -> Path:
    """Two-level texture output location: case then flat configuration."""
    return output_root(f"exp3_texgen_{pattern}") / case_name(case) / config_name(
        f"{_tag(pattern)}_os{oversamp}_{'f' if storage == 'float' else f'b{bit_depths()[0]}'}"
    )


def load_case(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path("data") / case
    coords = np.loadtxt(root / "coords.csv", delimiter=",")
    connect = np.loadtxt(root / "connectivity.csv", delimiter=",", dtype=np.uintp)
    ux = np.loadtxt(root / "field_disp_x.csv", delimiter=",")
    uy = np.loadtxt(root / "field_disp_y.csv", delimiter=",")
    if connect.ndim == 1: connect = connect[None, :]
    if ux.ndim == 1: ux, uy = ux[:, None], uy[:, None]
    # PyVista's CellArray accepts signed VTK-id arrays reliably; do not pass
    # platform ``uintp`` (which can be rejected for the large manual mesh).
    return coords, np.ascontiguousarray(connect, dtype=np.int64), ux, uy


def case_signature(coords: np.ndarray, connect: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> str:
    """Stable content signature so resumable outputs cannot outlive their mesh."""
    digest = hashlib.sha256()
    for value in (coords, connect, ux, uy):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    # Camera geometry and the final-pixel procedural definition affect every
    # rendered result even when the FE data have not changed.
    digest.update(repr((CASE_CAMERA_PIXELS, CASE_ROI_SIZES, EGGBOX_PERIOD_FINAL_PX)).encode())
    return digest.hexdigest()


def outputs_match_case(root: Path, signature: str) -> bool:
    marker = root / ".exp3_case_sha256"
    return marker.exists() and marker.read_text().strip() == signature


def mark_case_outputs(root: Path, signature: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".exp3_case_sha256").write_text(f"{signature}\n")


def _tag(pattern: str) -> str:
    if pattern == "eggbox": return "eggbox"
    distribution, jitter = additive_jitter_for(pattern)
    return f"{pattern}_blackfrac{BLACK_AREA_FRACTIONS[0]:g}_{distribution}_j{jitter:g}_seed{RANDOM_SEED}"


def texture_path(case: str, pattern: str, oversamp: int, storage: str = "float", bits: int | None = None) -> Path:
    suffix = "float" if storage == "float" else f"b{bits if bits is not None else bit_depths()[0]}"
    return texture_config_dir(case, pattern, oversamp, storage) / f"texture_{suffix}.npy"


def texture_owner_case(case: str) -> str:
    """Return the deterministic texture owner for cases sharing camera space.

    Texture generation is defined in undeformed camera/reference coordinates,
    so rigid and affine cases with equal camera pixels and ROI use precisely
    the same texture.  Prefer the rigid case as the owner; other matching
    cases link to its float asset instead of duplicating multi-GiB arrays.
    """
    geometry = (CASE_CAMERA_PIXELS[case], CASE_ROI_SIZES[case])
    matches = [candidate for candidate in DEFORMATION_CASES
               if (CASE_CAMERA_PIXELS[candidate], CASE_ROI_SIZES[candidate]) == geometry]
    return next((candidate for candidate in matches if "rigid" in candidate), case)


def _link_texture_asset(source: Path, destination: Path) -> None:
    """Create a verified same-filesystem hard link for one shared asset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Shared texture size mismatch: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.linktmp")
    os.link(source, temporary)
    try:
        if temporary.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Shared texture link verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_texture_preview(path: Path, oversamp: int, pattern: str) -> None:
    if oversamp == 1:
        preview = path.with_name(f"{path.stem}_preview_b8.tiff")
        transform = coverage_to_intensity if pattern != "eggbox" else None
        if write_preview_b8(path, preview, transform):
            print(f"  wrote texture preview: {preview.name}")


def generate_texture(case: str, pattern: str, oversamp: int) -> Path:
    """Generate a row-major texture of analytic texel averages.

    Disk and Gaussian source textures represent each texel's pixel-area
    average, rather than a value sampled at its centre.  This makes OS=1 a
    legitimate camera-resolution texture with grey edge texels, consistent
    with Exp2's analytic texture generator and Exp3's analytic references.
    """
    path = texture_path(case, pattern, oversamp)
    texture_signature = hashlib.sha256(repr((
        "raw_coverage_texel_integral_v2", case, pattern, oversamp,
        CASE_CAMERA_PIXELS[case], CASE_ROI_SIZES[case], EGGBOX_PERIOD_FINAL_PX,
    )).encode()).hexdigest()
    marker = path.with_suffix(".sha256")
    width, height = CASE_CAMERA_PIXELS[case]
    tex_w, tex_h = oversamp * (width + 2 * TEX_PX_PAD), oversamp * (height + 2 * TEX_PX_PAD)
    marker_value = marker.read_text().strip() if marker.exists() else ""
    owner = texture_owner_case(case)
    if owner != case:
        # Generate/load the one canonical float source first, then link the
        # case-local logical path.  The marker remains case-local so existing
        # resumable Riley render signatures stay valid.
        owner_path = generate_texture(owner, pattern, oversamp)
        if not path.exists():
            _link_texture_asset(owner_path, path)
        owner_preview = owner_path.with_name(f"{owner_path.stem}_preview_b8.tiff")
        preview = path.with_name(f"{path.stem}_preview_b8.tiff")
        if owner_preview.exists() and not preview.exists():
            _link_texture_asset(owner_preview, preview)
        for bits in bit_depths() if uint_textures_enabled() else ():
            owner_uint = texture_path(owner, pattern, oversamp, "uint", bits)
            current_uint = texture_path(case, pattern, oversamp, "uint", bits)
            if owner_uint.exists() and not current_uint.exists():
                _link_texture_asset(owner_uint, current_uint)
        if not marker.exists():
            marker.write_text(f"{texture_signature}\n")
        return path
    if path.exists() and marker_value == texture_signature and not force_render():
        try:
            texture = np.load(path, mmap_mode="r")
            valid_texture = texture.shape == (tex_h, tex_w) and texture.dtype.kind == "f"
        except (OSError, ValueError):
            valid_texture = False
        if valid_texture:
            _write_texture_preview(path, oversamp, pattern)
            for bits in bit_depths() if uint_textures_enabled() else ():
                uint_path = texture_path(case, pattern, oversamp, "uint", bits)
                if not uint_path.exists():
                    maximum = 2**bits - 1
                    camera_texture = coverage_to_intensity(texture) if pattern != "eggbox" else texture
                    np.save(uint_path, np.rint(np.clip(camera_texture, 0, 1) * maximum).astype(np.uint8 if bits <= 8 else np.uint16))
            return path
    render_log("EXP3", "texgen", case_label(case), f"generating {pattern} OS={oversamp}")
    roi_x, roi_y = CASE_ROI_SIZES[case]
    sx, sy = roi_x / width / oversamp, roi_y / height / oversamp
    x = -roi_x / 2 - TEX_PX_PAD * roi_x / width + (np.arange(tex_w) + .5) * sx
    # Image rows are top-to-bottom to match Riley texture memory.
    y = roi_y / 2 + TEX_PX_PAD * roi_y / height - (np.arange(tex_h) + .5) * sy
    if pattern == "eggbox":
        # Exact separable box average of Riley's eggbox convention.  A texture
        # texel is a finite area, so retain the sinc attenuation rather than a
        # centre sample of the continuous function.
        pitch_x, pitch_y = eggbox_pitch_world(case)
        x_average = np.cos(2 * np.pi * x / pitch_x) * np.sinc(sx / pitch_x)
        y_average = np.cos(2 * np.pi * y / pitch_y) * np.sinc(sy / pitch_y)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        texture = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(tex_h, tex_w))
        rows_per_batch = max(1, int(os.environ.get("EXP3_TEXGEN_CHUNK_PIXELS", "1000000")) // tex_w)
        tasks = [
            (start_row, min(start_row + rows_per_batch, tex_h))
            for start_row in range(0, tex_h, rows_per_batch)
        ]
        workers = min(TEXGEN_WORKERS, len(tasks))
        print(
            f"  exact eggbox texgen: {len(tasks)} row batches, {workers} workers, "
            f"up to {rows_per_batch} rows/batch"
        )
        if workers == 1:
            _init_eggbox_texgen_worker(x_average, y_average)
            for task in tasks:
                start_row, end_row, values = _integrate_eggbox_texgen_rows(task)
                texture[start_row:end_row] = values
        else:
            with multiprocessing.Pool(
                workers,
                initializer=_init_eggbox_texgen_worker,
                initargs=(x_average, y_average),
            ) as pool:
                for start_row, end_row, values in pool.imap_unordered(
                    _integrate_eggbox_texgen_rows, tasks,
                ):
                    # The parent is the sole memmap writer, avoiding races.
                    texture[start_row:end_row] = values
        texture.flush()
        del texture
        temporary.replace(path)
    else:
        distribution, jitter = additive_jitter_for(pattern)
        # A full OS=32 finite-star texture is several GiB.  Write one bounded
        # row batch at a time, so analytic integration does not require a
        # second full texture-sized coordinate/coverage allocation.
        rows_per_batch = max(1, int(os.environ.get("EXP3_TEXGEN_CHUNK_PIXELS", "1000000")) // tex_w)
        tasks = [
            (start_row, min(start_row + rows_per_batch, tex_h))
            for start_row in range(0, tex_h, rows_per_batch)
        ]
        temporary = path.with_suffix(".tmp.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        texture = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(tex_h, tex_w))
        x_start = x - 0.5 * sx
        workers = min(TEXGEN_WORKERS, len(tasks))
        print(
            f"  analytic texgen: {len(tasks)} row batches, {workers} workers, "
            f"up to {rows_per_batch} rows/batch"
        )
        initargs = (
            pattern, 5.0 * roi_x / width, BLACK_AREA_FRACTIONS[0],
            distribution, jitter, (x[0], x[-1], y[-1], y[0]), x_start,
            roi_y / 2 + TEX_PX_PAD * roi_y / height, sx, sy,
        )
        if workers == 1:
            _init_speckle_texgen_worker(*initargs)
            for task in tasks:
                start_row, end_row, coverage = _integrate_speckle_texgen_rows(task)
                texture[start_row:end_row] = coverage
        else:
            with multiprocessing.Pool(workers, initializer=_init_speckle_texgen_worker, initargs=initargs) as pool:
                for start_row, end_row, coverage in pool.imap_unordered(_integrate_speckle_texgen_rows, tasks):
                    texture[start_row:end_row] = coverage
        texture.flush()
        del texture
        temporary.replace(path)
    texture = np.load(path, mmap_mode="r")
    _write_texture_preview(path, oversamp, pattern)
    for bits in bit_depths() if uint_textures_enabled() else ():
        maximum = 2**bits - 1
        camera_texture = coverage_to_intensity(texture) if pattern != "eggbox" else texture
        quant = np.round(np.clip(camera_texture, 0, 1) * maximum)
        dtype = np.uint8 if bits <= 8 else np.uint16
        uint_path = texture_path(case, pattern, oversamp, "uint", bits)
        uint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(uint_path, quant.astype(dtype))
    marker.write_text(f"{texture_signature}\n")
    return path


def _inverse_affine(coords: np.ndarray, ux: np.ndarray, uy: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit the exact global inverse r=Aq+b used by rigid/affine test plates."""
    q = coords[:, :2] + np.column_stack((ux[:, frame], uy[:, frame]))
    design = np.column_stack((q, np.ones(len(q))))
    coeff, *_ = np.linalg.lstsq(design, coords[:, :2], rcond=None)
    if np.max(np.abs(design @ coeff - coords[:, :2])) > 1e-8:
        raise ValueError("Requested affine mapping for a non-affine deformation field.")
    return coeff[:2].T, coeff[2]


def reference_points(case: str, coords: np.ndarray, connect: np.ndarray, ux: np.ndarray, uy: np.ndarray, frame: int, qx: np.ndarray, qy: np.ndarray, *, topology=None, deformed=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target sample positions to reference world positions."""
    if frame == 0:
        return qx, qy, np.ones(qx.size, dtype=bool)
    if MAPPING_MODES[case] == "affine":
        a, b = _inverse_affine(coords, ux, uy, frame)
        refs = np.column_stack((qx, qy)) @ a.T + b
        return refs[:, 0], refs[:, 1], np.ones(qx.size, dtype=bool)
    if deformed is None:
        deformed = np.array(coords, copy=True)
        deformed[:, 0] += ux[:, frame]; deformed[:, 1] += uy[:, frame]
    if MAPPING_MODES[case] == "structured_newton":
        from modules.quad9_structured_newton import inverse_map_structured_quad9
        if topology is None:
            from modules.quad9_structured_newton import build_structured_quad9_topology
            topology = build_structured_quad9_topology(coords, connect)
        return inverse_map_structured_quad9(qx, qy, deformed, coords, topology)
    mesh = build_pv_mesh(deformed, connect)
    mesh.point_data["x_ref"] = coords[:, 0]; mesh.point_data["y_ref"] = coords[:, 1]
    points = np.column_stack((qx, qy, np.zeros(qx.size)))
    sampled = pv.PolyData(points).sample(mesh)
    valid = sampled.point_data["vtkValidPointMask"].astype(bool)
    return sampled.point_data["x_ref"], sampled.point_data["y_ref"], valid


def _sample_texture(texture: np.ndarray, case: str, rx: np.ndarray, ry: np.ndarray, oversamp: int, interp: str) -> np.ndarray:
    width, height = CASE_CAMERA_PIXELS[case]; roi_x, roi_y = CASE_ROI_SIZES[case]
    sx, sy = roi_x / width / oversamp, roi_y / height / oversamp
    col = (rx + roi_x / 2 + TEX_PX_PAD * roi_x / width) / sx - .5
    row = (roi_y / 2 + TEX_PX_PAD * roi_y / height - ry) / sy - .5
    order = {"nearest": 0, "linear": 1, "cubic_catmull_rom": 3}[interp]
    return map_coordinates(texture, [row, col], order=order, mode="nearest", prefilter=order > 1)


def _analytic_eggbox_affine(px: np.ndarray, py: np.ndarray, pixel_x: float, pixel_y: float, a: np.ndarray, b: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    """Closed-form average of Riley's eggbox over an affine target pixel."""
    centre = np.column_stack((px + .5 * pixel_x, py + .5 * pixel_y))
    reference_centre = centre @ a.T + b
    kx, ky = 2.0 * np.pi / pitch[0], 2.0 * np.pi / pitch[1]

    def averaged_cos(vector: np.ndarray) -> np.ndarray:
        phase = reference_centre @ vector
        # numpy.sinc(z) is sin(pi*z)/(pi*z).
        factor_x = np.sinc((vector @ a[:, 0]) * pixel_x / (2.0 * np.pi))
        factor_y = np.sinc((vector @ a[:, 1]) * pixel_y / (2.0 * np.pi))
        return np.cos(phase) * factor_x * factor_y

    cx = averaged_cos(np.array((kx, 0.0)))
    cy = averaged_cos(np.array((0.0, ky)))
    cxy = .5 * (averaged_cos(np.array((kx, ky))) + averaged_cos(np.array((kx, -ky))))
    return I0 - GAMMA + .5 * GAMMA * (1.0 + cx + cy + cxy)


def _save(image: np.ndarray, root: Path, prefix: str, *, overwrite: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Camera files are top-row-first, matching Riley's saved arrays and the
    # existing Exp1/2 output convention.  Rendering maths uses +Y-up.
    image = np.ascontiguousarray(np.flipud(image), dtype=np.float64)
    save_float_and_depths(
        root / f"{prefix}.npy", image, bit_depths(), overwrite=overwrite,
    )


def bespoke_render(case: str, pattern: str, method: str, param: int, *, texture_os: int | None = None, interp: str = "linear", psf: bool = False) -> Path:
    """Chunked custom ortho renderer.  Its frame mapping is independent of Riley."""
    coords, connect, ux, uy = load_case(case)
    signature = case_signature(coords, connect, ux, uy)
    width, height = CASE_CAMERA_PIXELS[case]; roi_x, roi_y = CASE_ROI_SIZES[case]
    root = output_root(
        f"exp3_{'gridint2d' if pattern == 'eggbox' else 'speckint2d'}_render_{method}{'_psf' if psf else ''}"
    ) / case_name(case)
    config = f"{_tag(pattern)}_{'ss' + str(param) if method == 'ssaa' else 'analytic'}"
    if texture_os is not None: config += f"_{interp}_os{texture_os}"
    config += "_f"
    root = root / config_name(config)
    texture = None if texture_os is None else np.load(generate_texture(case, pattern, texture_os), mmap_mode="r")
    speckles = None
    if pattern != "eggbox" and texture is None:
        distribution, jitter = additive_jitter_for(pattern)
        # The pattern is deterministic for a case.  Construct it once per
        # render, not once per pixel chunk (the latter dominated test mode).
        speckles = make_speckle_pattern(
            pattern, 5 * roi_x / width, BLACK_AREA_FRACTIONS[0], distribution,
            jitter, RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS,
            (-roi_x / 2 - 8, roi_x / 2 + 8, -roi_y / 2 - 8, roi_y / 2 + 8),
            I0, GAMMA, GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
            GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
        )
    topology = None
    if MAPPING_MODES[case] == "structured_newton":
        from modules.quad9_structured_newton import build_structured_quad9_topology
        topology = build_structured_quad9_topology(coords, connect)
    # Analytic is only a function evaluation at pixel centres here.  The
    # closed-form affine eggbox integral is intentionally kept as a separate
    # reference, while numerical SSAA is the comparison model.
    samples = 1 if method == "analytic" else param
    offsets = (np.arange(samples) + .5) / samples
    dx, dy = np.meshgrid(offsets * roi_x / width, offsets * roi_y / height)
    weights = 1.0 / (samples * samples)
    chunk = max(1, int(os.environ.get("EXP3_CHUNK_PIXELS", "32768")))
    analytic_speckle = method == "analytic" and pattern in {"diskaddsat", "gausscont"} and texture is None
    for frame in selected_frames(case, ux.shape[1]):
        prefix = f"frame{frame:02d}"
        integral_marker = root / ".exp3_analytic_speckle_integral_v2"
        float_path = root / f"{prefix}.npy"
        if float_path.exists() and outputs_match_case(root, signature) and (not analytic_speckle or integral_marker.exists()) and not force_render():
            write_camera_depths(float_path, bit_depths())
            if float_and_depths_complete(float_path, bit_depths()):
                print(f"  {case} {prefix}: float image exists; camera depths complete; skipping."); continue
        renderer = "gridint2d" if pattern == "eggbox" else "speckint2d"
        render_log(
            "EXP3", renderer, case_label(case),
            f"rendering pattern={pattern}; method={method}; value={param}; frame={frame:02d}; psf={psf}",
        )
        if analytic_speckle and MAPPING_MODES[case] != "affine":
            raise ValueError(f"{case}: analytic speckle integration is unavailable for {MAPPING_MODES[case]} mapping.")
        analytic_a = analytic_b = None
        if analytic_speckle:
            analytic_a, analytic_b = _inverse_affine(coords, ux, uy, frame)
            if pattern == "diskaddsat" and not np.all(is_rigid_inverse(analytic_a[None, :, :])):
                raise ValueError(
                    f"{case} frame {frame:02d}: exact disk integration requires rigid motion; "
                    "use SSAA for affine deformation."
                )
        flat = np.empty(width * height)
        deformed = None
        if frame and MAPPING_MODES[case] == "structured_newton":
            deformed = np.array(coords, copy=True)
            deformed[:, 0] += ux[:, frame]; deformed[:, 1] += uy[:, frame]
        for start in range(0, flat.size, chunk):
            ids = np.arange(start, min(start + chunk, flat.size))
            px = -roi_x / 2 + (ids % width) * roi_x / width
            py = -roi_y / 2 + (ids // width) * roi_y / height
            if method == "analytic" and pattern == "eggbox" and texture is None:
                a, b = _inverse_affine(coords, ux, uy, frame)
                flat[ids] = _analytic_eggbox_affine(px, py, roi_x / width, roi_y / height, a, b, eggbox_pitch_world(case))
                continue
            if analytic_speckle:
                if speckles is None or analytic_a is None or analytic_b is None:
                    raise RuntimeError("Analytic speckle state was not initialised.")
                centres = np.column_stack((px + .5 * roi_x / width, py + .5 * roi_y / height))
                ref_centres = centres @ analytic_a.T + analytic_b
                maps = np.broadcast_to(analytic_a, (len(ids), 2, 2))
                coverage = (
                    analytic_disk_coverage(ref_centres, maps, speckles, roi_x / width)
                    if pattern == "diskaddsat" else
                    analytic_gaussian_coverage(ref_centres, maps, speckles, roi_x / width)
                )
                flat[ids] = coverage
                continue
            qx = (px[:, None] + dx.ravel()).ravel(); qy = (py[:, None] + dy.ravel()).ravel()
            rx, ry, valid = reference_points(case, coords, connect, ux, uy, frame, qx, qy, topology=topology, deformed=deformed)
            if texture is None:
                pitch_x, pitch_y = eggbox_pitch_world(case)
                values = I0 + .5 * GAMMA * (1 + np.cos(2*np.pi*rx/pitch_x)) * (1 + np.cos(2*np.pi*ry/pitch_y)) - GAMMA if pattern == "eggbox" else BACKGROUND + np.zeros_like(rx)
                if pattern != "eggbox":
                    if speckles is None:
                        raise RuntimeError("Speckle pattern was not initialised.")
                    values = speckles.evaluate_coverage(rx, ry)
            else:
                values = _sample_texture(texture, case, rx, ry, texture_os, interp)
            values[~valid] = 0.0 if pattern != "eggbox" else BACKGROUND
            flat[ids] = values.reshape(len(ids), samples * samples).sum(axis=1) * weights
        image = flat.reshape(height, width)
        if psf:
            image = gaussian_filter(image, PSF_SIGMA_FINAL_PX, mode="constant", cval=0.0 if pattern != "eggbox" else BACKGROUND, radius=round(PSF_SUPPORT_SIGMAS * PSF_SIGMA_FINAL_PX))
        if pattern != "eggbox":
            image = coverage_to_intensity(image)
        # Reaching this point means the output failed its resumability check,
        # so replace an existing stale image rather than retaining it.
        _save(image, root, prefix, overwrite=True)
        print(f"  {case} {prefix}: rendered.")
    mark_case_outputs(root, signature)
    if analytic_speckle:
        (root / ".exp3_analytic_speckle_integral_v2").write_text("true analytic pixel integral\n")
    return root


def _mesh_type(width: int) -> riley.MeshType:
    return {3:riley.MeshType.tri3,4:riley.MeshType.quad4ibi,6:riley.MeshType.tri6,8:riley.MeshType.quad8,9:riley.MeshType.quad9}[width]


def _uvs(coords: np.ndarray, case: str, oversamp: int) -> np.ndarray:
    width,height=CASE_CAMERA_PIXELS[case]; roi_x,roi_y=CASE_ROI_SIZES[case]
    tw,th=oversamp*(width+2*TEX_PX_PAD),oversamp*(height+2*TEX_PX_PAD)
    sx,sy=roi_x/width/oversamp,roi_y/height/oversamp
    result=np.empty((len(coords),2)); result[:,0]=((coords[:,0]+roi_x/2+TEX_PX_PAD*roi_x/width)/sx-.5)/(tw-1)
    result[:,1]=((roi_y/2+TEX_PX_PAD*roi_y/height-coords[:,1])/sy-.5)/(th-1)
    return np.ascontiguousarray(result)


def riley_render(case: str, pattern: str, shader: str, ssaa: int, *, texture_os: int | None = None, interp: str = "linear", storage: str = "float", source_bits: int | None = None, psf: bool = False) -> Path:
    coords, connect, ux, uy = load_case(case); width,height=CASE_CAMERA_PIXELS[case]; roi_x,roi_y=CASE_ROI_SIZES[case]
    signature = case_signature(coords, connect, ux, uy)
    if texture_os is not None:
        # Include the texel-integration convention in the render signature.
        # Otherwise a corrected source texture could leave stale Riley images
        # looking complete and silently bypass the rerender.
        generated_texture = generate_texture(case, pattern, texture_os)
        texture_marker = generated_texture.with_suffix(".sha256").read_text().strip()
        signature = hashlib.sha256(f"{signature}:{texture_marker}".encode()).hexdigest()
    root = output_root(f"exp3_riley_render_{shader}{'_psf' if psf else ''}") / case_name(case)
    tag = f"{_tag(pattern)}"
    if texture_os is None:
        tag += f"_func_ss{ssaa}_f"
    else:
        tag += f"_{interp}_os{texture_os}_ss{ssaa}_{'f' if storage == 'float' else f'b{source_bits}'}"
    root = root / config_name(tag)
    frames = selected_frames(case, ux.shape[1])
    source_bits = source_bits if storage == "uint" else bit_depths()[0]
    expected=[root/f"image_c00_f{frame:02d}.npy" for frame in frames]
    if all(p.exists() for p in expected) and outputs_match_case(root, signature) and not force_render():
        for path in expected:
            write_camera_depths(path, bit_depths())
        print(f"  {case} {tag}: float images exist; camera depths complete; skipping."); return root
    disp=np.zeros((ux.shape[1],len(coords),3)); disp[:,:,0]=ux.T; disp[:,:,1]=uy.T
    kwargs={"mesh_type":_mesh_type(connect.shape[1]),"coords":coords,"connect":connect,"disp":disp,"bits":source_bits,"scaling_type":riley.ScaleStrategy.fixed,"scaling_min":0.,"scaling_max":1.}
    if shader == "func":
        # Physical-coordinate function avoids any UV aspect-ratio ambiguity.
        kwargs.update(shader_type=riley.ShaderType.func,uvs=_uvs(coords,case,1),func_shader_builtin=riley.FuncShaderBuiltin.eggbox,func_shader_coord_mode=riley.FuncCoordMode.world_reference,func_shader_params=riley.FuncShaderParams(eggbox_mean=I0,eggbox_contrast=GAMMA,eggbox_pitch=eggbox_pitch_world(case),eggbox_phase=(0.,0.)))
    else:
        texture=np.load(texture_path(case, pattern, texture_os, "uint" if storage=="uint" else "float", source_bits))
        texture_storage = riley.TextureStorage.floating if storage == "float" else (riley.TextureStorage.u8 if texture.dtype == np.uint8 else riley.TextureStorage.u16)
        if storage == "float" and pattern != "eggbox":
            # Match Exp2: preserve unbounded floating coverage through the
            # raster.  Clamp/scale only after Riley has integrated a pixel.
            kwargs["scaling_type"] = riley.ScaleStrategy.none
        else:
            kwargs["scaling_max"] = float(2**source_bits - 1)
        if interp not in RILEY_TEXTURE_SAMPLERS:
            raise ValueError(f"Unsupported Riley sampler {interp!r}; choose from {', '.join(RILEY_TEXTURE_SAMPLERS)}")
        kwargs.update(shader_type=riley.ShaderType.tex,uvs=_uvs(coords,case,texture_os),texture=np.ascontiguousarray(texture),texture_storage=texture_storage,sample=RILEY_TEXTURE_SAMPLERS[interp],sample_mode=riley.TextureSampleMode.direct)
    mesh=riley.Mesh(**kwargs)
    roi=np.array([[-roi_x/2,-roi_y/2,0],[roi_x/2,-roi_y/2,0],[roi_x/2,roi_y/2,0],[-roi_x/2,roi_y/2,0.]],float)
    psf_kwargs = ({"psf_type": riley.PsfType.gaussian, "psf_sigma_x": PSF_SIGMA_FINAL_PX, "psf_sigma_y": PSF_SIGMA_FINAL_PX, "psf_support_rad": PSF_SIGMA_FINAL_PX * PSF_SUPPORT_SIGMAS, "psf_separable": 1} if psf else {})
    camera=riley.Camera(pixels_num=(width,height),pixels_size=(roi_x/width,roi_y/height),pos_world=riley.pos_fill_frame_from_rot(roi,(width,height),(roi_x/width,roi_y/height),1000.,(0,0,0),1.),rot_world=(0,0,0),roi_cent_world=tuple(riley.roi_cent_from_coords(roi)),focal_length=1000.,sub_sample=ssaa,coord_sys=riley.CameraCoordSys.opengl,**psf_kwargs)
    config=riley.create_raster_config(num_frames=ux.shape[1],total_threads=RILEY_RASTER_THREADS,save_strategy=riley.SaveStrategy.memory); config.frame_batch_size_per_group=1;config.max_geom_jobs_in_flight_per_group=1;config.max_geom_workers_per_job=1;config.max_raster_workers_per_job=RILEY_RASTER_THREADS;config.tile_size_min=1
    detail = f"pattern={pattern}; shader={shader}; SSAA={ssaa}; psf={psf}"
    if texture_os is not None:
        detail += f"; interp={interp}; OS={texture_os}; storage={storage}"
    render_log("EXP3", "riley", case_label(case), f"starting {detail}")
    root.mkdir(parents=True,exist_ok=True); images=riley.raster([mesh],[camera],config,out_dir=str(root))
    if images is not None:
        for frame in frames:
            rendered = np.asarray(images[0,frame,0], dtype=np.float64)
            if shader.startswith("tex") and storage == "float" and pattern != "eggbox":
                np.save(root / f"image_c00_f{frame:02d}_raw.npy", rendered)
                rendered = coverage_to_intensity(rendered)
            elif storage == "uint":
                rendered /= float(2**source_bits - 1)
            else:
                rendered /= float(2**source_bits - 1)
            save_float_and_depths(
                root / f"image_c00_f{frame:02d}.npy", rendered, bit_depths(),
                # The case/texture signature check above has already marked
                # this render stale, so an existing float image must be
                # replaced even when this is not a force-render invocation.
                overwrite=True,
            )
    mark_case_outputs(root, signature)
    return root
