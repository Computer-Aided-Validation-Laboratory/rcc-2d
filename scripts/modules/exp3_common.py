"""Shared, rectangular-camera render machinery for Experiment 3.

The Exp1/2 helpers intentionally assumed a square, single element camera.
Experiment 3 keeps its geometry and camera conventions explicit here so the
1020 x 252 finite-star case cannot accidentally inherit those assumptions.
"""

from __future__ import annotations

import os
import hashlib
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
from modules.render_selection import uint_textures_enabled
from exp3params import (
    BACKGROUND, BIT_DEPTHS, CASE_CAMERA_PIXELS, CASE_ROI_SIZES,
    DEFORMATION_CASES, EGGBOX_PERIOD_FINAL_PX, FORCE_RENDER_OVER, GAMMA, I0,
    MAPPING_MODES, PSF_SIGMA_FINAL_PX, PSF_SUPPORT_SIGMAS, RILEY_RASTER_THREADS,
    SSAA_LEVELS, RILEY_TEXTURE_SAMPLERS, TEX_OVERSAMPLES, TEX_PX_PAD,
    additive_jitter_for, BLACK_AREA_FRACTIONS, RANDOM_SEED,
    GAUSSIAN_CUTOFF_SIGMAS, GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
    GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
)


def _levels(name: str, defaults: list[int]) -> list[int]:
    value = os.environ.get(name)
    return defaults if not value else [int(v) for v in value.split(",") if v.strip()]


def ssaa_levels() -> list[int]: return _levels("EXP3_SSAA_LEVELS", SSAA_LEVELS)
def oversamples() -> list[int]: return _levels("EXP3_TEX_OVERSAMPLES", TEX_OVERSAMPLES)
def bit_depths() -> list[int]: return _levels("EXP3_BIT_DEPTHS", BIT_DEPTHS)


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
    width, height = CASE_CAMERA_PIXELS[case]
    return (
        Path("out") / f"exp3_texgen_{pattern}_im{width}x{height}" / case
        / f"{_tag(pattern)}_os{oversamp}_{'f' if storage == 'float' else f'b{bit_depths()[0]}'}"
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


def generate_texture(case: str, pattern: str, oversamp: int) -> Path:
    """Generate a row-major texture at texel centres including the 4-pixel pad."""
    path = texture_path(case, pattern, oversamp)
    texture_signature = hashlib.sha256(repr((case, pattern, oversamp, CASE_CAMERA_PIXELS[case], CASE_ROI_SIZES[case], EGGBOX_PERIOD_FINAL_PX)).encode()).hexdigest()
    marker = path.with_suffix(".sha256")
    width, height = CASE_CAMERA_PIXELS[case]
    tex_w, tex_h = oversamp * (width + 2 * TEX_PX_PAD), oversamp * (height + 2 * TEX_PX_PAD)
    marker_value = marker.read_text().strip() if marker.exists() else ""
    if path.exists() and marker_value == texture_signature and not force_render():
        try:
            texture = np.load(path, mmap_mode="r")
            valid_texture = texture.shape == (tex_h, tex_w) and texture.dtype.kind == "f"
        except (OSError, ValueError):
            valid_texture = False
        if valid_texture:
            for bits in bit_depths() if uint_textures_enabled() else ():
                uint_path = texture_path(case, pattern, oversamp, "uint", bits)
                if not uint_path.exists():
                    maximum = 2**bits - 1
                    np.save(uint_path, np.rint(np.clip(texture, 0, 1) * maximum).astype(np.uint8 if bits <= 8 else np.uint16))
            return path
    print(
        f"  {case} {_tag(pattern)} OS={oversamp}: generating texture.",
        flush=True,
    )
    roi_x, roi_y = CASE_ROI_SIZES[case]
    sx, sy = roi_x / width / oversamp, roi_y / height / oversamp
    x = -roi_x / 2 - TEX_PX_PAD * roi_x / width + (np.arange(tex_w) + .5) * sx
    # Image rows are top-to-bottom to match Riley texture memory.
    y = roi_y / 2 + TEX_PX_PAD * roi_y / height - (np.arange(tex_h) + .5) * sy
    xx, yy = np.meshgrid(x, y)
    if pattern == "eggbox":
        # This is Riley's built-in eggbox convention, retained verbatim so the
        # function and texture paths have the same continuous source field.
        pitch_x, pitch_y = eggbox_pitch_world(case)
        texture = I0 + .5 * GAMMA * (1 + np.cos(2 * np.pi * xx / pitch_x)) * (1 + np.cos(2 * np.pi * yy / pitch_y)) - GAMMA
    else:
        distribution, jitter = additive_jitter_for(pattern)
        speckles = make_speckle_pattern(
            pattern, 5.0 * roi_x / width, BLACK_AREA_FRACTIONS[0], distribution,
            jitter, RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS,
            (x[0], x[-1], y[-1], y[0]), I0, GAMMA,
            GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION, GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
        )
        coverage = speckles.evaluate_coverage(xx, yy)
        texture = speckles.intensity_from_coverage(coverage)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(texture, dtype=np.float64))
    for bits in bit_depths() if uint_textures_enabled() else ():
        maximum = 2**bits - 1
        quant = np.round(np.clip(texture, 0, 1) * maximum)
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


def _save(image: np.ndarray, root: Path, prefix: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Camera files are top-row-first, matching Riley's saved arrays and the
    # existing Exp1/2 output convention.  Rendering maths uses +Y-up.
    image = np.ascontiguousarray(np.flipud(image), dtype=np.float64)
    save_float_and_depths(root / f"{prefix}.npy", image, bit_depths())


def bespoke_render(case: str, pattern: str, method: str, param: int, *, texture_os: int | None = None, interp: str = "linear", psf: bool = False) -> Path:
    """Chunked custom ortho renderer.  Its frame mapping is independent of Riley."""
    coords, connect, ux, uy = load_case(case)
    signature = case_signature(coords, connect, ux, uy)
    width, height = CASE_CAMERA_PIXELS[case]; roi_x, roi_y = CASE_ROI_SIZES[case]
    root = Path("out") / f"exp3_{'gridint2d' if pattern == 'eggbox' else 'speckint2d'}_render_{method}{'_psf' if psf else ''}_im{width}x{height}" / case
    config = f"{_tag(pattern)}_{'ss' + str(param) if method == 'ssaa' else 'analytic'}"
    if texture_os is not None: config += f"_{interp}_os{texture_os}"
    config += "_f"
    root = root / config
    texture = None if texture_os is None else np.load(generate_texture(case, pattern, texture_os), mmap_mode="r")
    speckles = None
    if pattern != "eggbox" and texture is None:
        distribution, jitter = additive_jitter_for(pattern)
        # The pattern is deterministic for a case.  Construct it once per
        # render, not once per pixel chunk (the latter dominated TEST_RUN).
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
        print(f"  {case} {config} {prefix}: rendering.", flush=True)
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
                flat[ids] = speckles.intensity_from_coverage(coverage)
                continue
            qx = (px[:, None] + dx.ravel()).ravel(); qy = (py[:, None] + dy.ravel()).ravel()
            rx, ry, valid = reference_points(case, coords, connect, ux, uy, frame, qx, qy, topology=topology, deformed=deformed)
            if texture is None:
                pitch_x, pitch_y = eggbox_pitch_world(case)
                values = I0 + .5 * GAMMA * (1 + np.cos(2*np.pi*rx/pitch_x)) * (1 + np.cos(2*np.pi*ry/pitch_y)) - GAMMA if pattern == "eggbox" else BACKGROUND + np.zeros_like(rx)
                if pattern != "eggbox":
                    if speckles is None:
                        raise RuntimeError("Speckle pattern was not initialised.")
                    values = speckles.intensity_from_coverage(speckles.evaluate_coverage(rx, ry))
            else:
                values = _sample_texture(texture, case, rx, ry, texture_os, interp)
            values[~valid] = BACKGROUND
            flat[ids] = values.reshape(len(ids), samples * samples).sum(axis=1) * weights
        image = flat.reshape(height, width)
        if psf:
            image = gaussian_filter(image, PSF_SIGMA_FINAL_PX, mode="constant", cval=BACKGROUND, radius=round(PSF_SUPPORT_SIGMAS * PSF_SIGMA_FINAL_PX))
        _save(image, root, prefix)
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
    root=Path("out")/f"exp3_riley_render_{shader}{'_psf' if psf else ''}_im{width}x{height}"/case
    tag = f"{_tag(pattern)}"
    if texture_os is None:
        tag += f"_func_ss{ssaa}_f"
    else:
        tag += f"_{interp}_os{texture_os}_ss{ssaa}_{'f' if storage == 'float' else f'b{source_bits}'}"
    root=root/tag
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
        generate_texture(case, pattern, texture_os)
        texture=np.load(texture_path(case, pattern, texture_os, "uint" if storage=="uint" else "float", source_bits))
        texture_storage = riley.TextureStorage.floating if storage == "float" else (riley.TextureStorage.u8 if texture.dtype == np.uint8 else riley.TextureStorage.u16)
        if storage == "uint":
            kwargs["scaling_max"] = float(2**source_bits - 1)
        if interp not in RILEY_TEXTURE_SAMPLERS:
            raise ValueError(f"Unsupported Riley sampler {interp!r}; choose from {', '.join(RILEY_TEXTURE_SAMPLERS)}")
        kwargs.update(shader_type=riley.ShaderType.tex,uvs=_uvs(coords,case,texture_os),texture=np.ascontiguousarray(texture),texture_storage=texture_storage,sample=RILEY_TEXTURE_SAMPLERS[interp],sample_mode=riley.TextureSampleMode.direct)
    mesh=riley.Mesh(**kwargs)
    roi=np.array([[-roi_x/2,-roi_y/2,0],[roi_x/2,-roi_y/2,0],[roi_x/2,roi_y/2,0],[-roi_x/2,roi_y/2,0.]],float)
    psf_kwargs = ({"psf_type": riley.PsfType.gaussian, "psf_sigma_x": PSF_SIGMA_FINAL_PX, "psf_sigma_y": PSF_SIGMA_FINAL_PX, "psf_support_rad": PSF_SIGMA_FINAL_PX * PSF_SUPPORT_SIGMAS, "psf_separable": 1} if psf else {})
    camera=riley.Camera(pixels_num=(width,height),pixels_size=(roi_x/width,roi_y/height),pos_world=riley.pos_fill_frame_from_rot(roi,(width,height),(roi_x/width,roi_y/height),1000.,(0,0,0),1.),rot_world=(0,0,0),roi_cent_world=tuple(riley.roi_cent_from_coords(roi)),focal_length=1000.,sub_sample=ssaa,coord_sys=riley.CameraCoordSys.opengl,**psf_kwargs)
    config=riley.create_raster_config(num_frames=ux.shape[1],total_threads=RILEY_RASTER_THREADS,save_strategy=riley.SaveStrategy.memory); config.frame_batch_size_per_group=1;config.max_geom_jobs_in_flight_per_group=1;config.max_geom_workers_per_job=1;config.max_raster_workers_per_job=RILEY_RASTER_THREADS;config.tile_size_min=1
    print(f"  {case} {tag}: rendering with Riley (SSAA={ssaa}).", flush=True)
    root.mkdir(parents=True,exist_ok=True); images=riley.raster([mesh],[camera],config,out_dir=str(root))
    if images is not None:
        maximum = float(2**source_bits - 1)
        for frame in frames:
            save_float_and_depths(root/f"image_c00_f{frame:02d}.npy", np.asarray(images[0,frame,0], dtype=np.float64) / maximum, bit_depths())
    mark_case_outputs(root, signature)
    return root
