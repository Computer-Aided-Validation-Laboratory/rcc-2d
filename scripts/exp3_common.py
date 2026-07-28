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

from exp1common import build_pv_mesh
from exp2speckint2d import make_speckle_pattern
from exp3params import (
    BACKGROUND, BIT_DEPTHS, CASE_CAMERA_PIXELS, CASE_ROI_SIZES,
    DEFORMATION_CASES, EGGBOX_PX_PERIOD, FORCE_RENDER_OVER, GAMMA, I0,
    MAPPING_MODES, PSF_SIGMA_FINAL_PX, PSF_SUPPORT_SIGMAS, RILEY_RASTER_THREADS,
    SSAA_LEVELS, TEX_INTERPOLATORS, TEX_OVERSAMPLES, TEX_PX_PAD,
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


def texture_root(pattern: str) -> Path:
    return Path("out") / f"exp3_texgen_{pattern}"


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


def texture_path(case: str, pattern: str, oversamp: int, storage: str = "float") -> Path:
    width, height = CASE_CAMERA_PIXELS[case]
    suffix = "" if storage == "float" else f"_b{bit_depths()[0]}"
    return texture_root(pattern) / f"{_tag(pattern)}_im{width}x{height}_pad{TEX_PX_PAD}_os{oversamp}{suffix}.npy"


def generate_texture(case: str, pattern: str, oversamp: int) -> Path:
    """Generate a row-major texture at texel centres including the 4-pixel pad."""
    path = texture_path(case, pattern, oversamp)
    if path.exists() and not force_render():
        return path
    width, height = CASE_CAMERA_PIXELS[case]
    roi_x, roi_y = CASE_ROI_SIZES[case]
    tex_w, tex_h = oversamp * (width + 2 * TEX_PX_PAD), oversamp * (height + 2 * TEX_PX_PAD)
    sx, sy = roi_x / width / oversamp, roi_y / height / oversamp
    x = -roi_x / 2 - TEX_PX_PAD * roi_x / width + (np.arange(tex_w) + .5) * sx
    # Image rows are top-to-bottom to match Riley texture memory.
    y = roi_y / 2 + TEX_PX_PAD * roi_y / height - (np.arange(tex_h) + .5) * sy
    xx, yy = np.meshgrid(x, y)
    if pattern == "eggbox":
        # This is Riley's built-in eggbox convention, retained verbatim so the
        # function and texture paths have the same continuous source field.
        texture = I0 + .5 * GAMMA * (1 + np.cos(2 * np.pi * xx / EGGBOX_PX_PERIOD)) * (1 + np.cos(2 * np.pi * yy / EGGBOX_PX_PERIOD)) - GAMMA
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
    for bits in bit_depths():
        maximum = 2**bits - 1
        quant = np.round(np.clip(texture, 0, 1) * maximum)
        dtype = np.uint8 if bits <= 8 else np.uint16
        np.save(texture_root(pattern) / f"{_tag(pattern)}_im{width}x{height}_pad{TEX_PX_PAD}_os{oversamp}_b{bits}.npy", quant.astype(dtype))
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
        from quad9_structured_newton import inverse_map_structured_quad9
        if topology is None:
            from quad9_structured_newton import build_structured_quad9_topology
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


def _analytic_eggbox_affine(px: np.ndarray, py: np.ndarray, pixel_x: float, pixel_y: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closed-form average of Riley's eggbox over an affine target pixel."""
    centre = np.column_stack((px + .5 * pixel_x, py + .5 * pixel_y))
    reference_centre = centre @ a.T + b
    k = 2.0 * np.pi / EGGBOX_PX_PERIOD

    def averaged_cos(vector: np.ndarray) -> np.ndarray:
        phase = reference_centre @ vector
        # numpy.sinc(z) is sin(pi*z)/(pi*z).
        factor_x = np.sinc((vector @ a[:, 0]) * pixel_x / (2.0 * np.pi))
        factor_y = np.sinc((vector @ a[:, 1]) * pixel_y / (2.0 * np.pi))
        return np.cos(phase) * factor_x * factor_y

    cx = averaged_cos(np.array((k, 0.0)))
    cy = averaged_cos(np.array((0.0, k)))
    cxy = .5 * (averaged_cos(np.array((k, k))) + averaged_cos(np.array((k, -k))))
    return I0 - GAMMA + .5 * GAMMA * (1.0 + cx + cy + cxy)


def _save(image: np.ndarray, root: Path, prefix: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Camera files are top-row-first, matching Riley's saved arrays and the
    # existing Exp1/2 output convention.  Rendering maths uses +Y-up.
    image = np.ascontiguousarray(np.flipud(image), dtype=np.float64)
    np.save(root / f"{prefix}.npy", image)
    for bits in bit_depths():
        encoded = np.round(np.clip(image, 0, 1) * (2**bits - 1)).astype(np.uint8 if bits <= 8 else np.uint16)
        Image.fromarray(encoded).save(root / f"{prefix}_b{bits}.tiff")


def bespoke_render(case: str, pattern: str, method: str, param: int, *, texture_os: int | None = None, interp: str = "nearest", psf: bool = False) -> Path:
    """Chunked custom ortho renderer.  Its frame mapping is independent of Riley."""
    coords, connect, ux, uy = load_case(case)
    signature = case_signature(coords, connect, ux, uy)
    width, height = CASE_CAMERA_PIXELS[case]; roi_x, roi_y = CASE_ROI_SIZES[case]
    root = Path("out") / f"exp3_{'gridint2d' if pattern == 'eggbox' else 'speckint2d'}_render_{method}{'_psf' if psf else ''}_im{width}x{height}" / case
    prefix_base = f"{_tag(pattern)}_{'ss' + str(param) if method == 'ssaa' else 'analytic'}"
    if texture_os is not None: prefix_base += f"_os{texture_os}_{interp}"
    texture = None if texture_os is None else np.load(generate_texture(case, pattern, texture_os), mmap_mode="r")
    topology = None
    if MAPPING_MODES[case] == "structured_newton":
        from quad9_structured_newton import build_structured_quad9_topology
        topology = build_structured_quad9_topology(coords, connect)
    # Analytic is only a function evaluation at pixel centres here.  The
    # closed-form affine eggbox integral is intentionally kept as a separate
    # reference, while numerical SSAA is the comparison model.
    samples = 1 if method == "analytic" else param
    offsets = (np.arange(samples) + .5) / samples
    dx, dy = np.meshgrid(offsets * roi_x / width, offsets * roi_y / height)
    weights = 1.0 / (samples * samples)
    chunk = max(1, int(os.environ.get("EXP3_CHUNK_PIXELS", "32768")))
    for frame in selected_frames(case, ux.shape[1]):
        prefix = f"{prefix_base}_frame{frame:02d}"
        if (root / f"{prefix}.npy").exists() and outputs_match_case(root, signature) and not force_render():
            print(f"  {case} {prefix}: exists; skipping."); continue
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
                flat[ids] = _analytic_eggbox_affine(px, py, roi_x / width, roi_y / height, a, b)
                continue
            qx = (px[:, None] + dx.ravel()).ravel(); qy = (py[:, None] + dy.ravel()).ravel()
            rx, ry, valid = reference_points(case, coords, connect, ux, uy, frame, qx, qy, topology=topology, deformed=deformed)
            if texture is None:
                values = I0 + .5 * GAMMA * (1 + np.cos(2*np.pi*rx/EGGBOX_PX_PERIOD)) * (1 + np.cos(2*np.pi*ry/EGGBOX_PX_PERIOD)) - GAMMA if pattern == "eggbox" else BACKGROUND + np.zeros_like(rx)
                if pattern != "eggbox":
                    distribution, jitter = additive_jitter_for(pattern)
                    speckles = make_speckle_pattern(pattern, 5*roi_x/width, BLACK_AREA_FRACTIONS[0], distribution, jitter, RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS, (-roi_x/2-8,roi_x/2+8,-roi_y/2-8,roi_y/2+8), I0,GAMMA,GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,GAUSSIAN_CONTINUOUS_TAIL_SIGMAS)
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


def riley_render(case: str, pattern: str, shader: str, ssaa: int, *, texture_os: int | None = None, interp: str = "nearest", storage: str = "float", psf: bool = False) -> Path:
    coords, connect, ux, uy = load_case(case); width,height=CASE_CAMERA_PIXELS[case]; roi_x,roi_y=CASE_ROI_SIZES[case]
    signature = case_signature(coords, connect, ux, uy)
    root=Path("out")/f"exp3_riley_render_{shader}{'_psf' if psf else ''}_im{width}x{height}"/case
    tag=f"{_tag(pattern)}_ss{ssaa}" + (f"_os{texture_os}_{interp}_{storage}" if texture_os else "")
    root=root/tag; expected=[root/f"image_c00_f{frame:02d}.npy" for frame in selected_frames(case,ux.shape[1])]
    if all(p.exists() for p in expected) and outputs_match_case(root, signature) and not force_render():
        print(f"  {case} {tag}: exists; skipping."); return root
    disp=np.zeros((ux.shape[1],len(coords),3)); disp[:,:,0]=ux.T; disp[:,:,1]=uy.T
    kwargs={"mesh_type":_mesh_type(connect.shape[1]),"coords":coords,"connect":connect,"disp":disp,"bits":bit_depths()[0],"scaling_type":riley.ScaleStrategy.fixed,"scaling_min":0.,"scaling_max":1.}
    if shader == "func":
        # Physical-coordinate function avoids any UV aspect-ratio ambiguity.
        kwargs.update(shader_type=riley.ShaderType.func,uvs=_uvs(coords,case,1),func_shader_builtin=riley.FuncShaderBuiltin.eggbox,func_shader_coord_mode=riley.FuncCoordMode.world_reference,func_shader_params=riley.FuncShaderParams(eggbox_mean=I0,eggbox_contrast=GAMMA,eggbox_pitch=(EGGBOX_PX_PERIOD,EGGBOX_PX_PERIOD),eggbox_phase=(0.,0.)))
    else:
        generate_texture(case, pattern, texture_os)
        texture=np.load(texture_path(case, pattern, texture_os, "uint" if storage=="uint" else "float"))
        texture_storage = riley.TextureStorage.floating if storage == "float" else (riley.TextureStorage.u8 if texture.dtype == np.uint8 else riley.TextureStorage.u16)
        if storage == "uint":
            kwargs["scaling_max"] = float(2**bit_depths()[0] - 1)
        kwargs.update(shader_type=riley.ShaderType.tex,uvs=_uvs(coords,case,texture_os),texture=np.ascontiguousarray(texture),texture_storage=texture_storage,sample=TEX_INTERPOLATORS[interp],sample_mode=riley.TextureSampleMode.direct)
    mesh=riley.Mesh(**kwargs)
    roi=np.array([[-roi_x/2,-roi_y/2,0],[roi_x/2,-roi_y/2,0],[roi_x/2,roi_y/2,0],[-roi_x/2,roi_y/2,0.]],float)
    psf_kwargs = ({"psf_type": riley.PsfType.gaussian, "psf_sigma_x": PSF_SIGMA_FINAL_PX, "psf_sigma_y": PSF_SIGMA_FINAL_PX, "psf_support_rad": PSF_SIGMA_FINAL_PX * PSF_SUPPORT_SIGMAS, "psf_separable": 1} if psf else {})
    camera=riley.Camera(pixels_num=(width,height),pixels_size=(roi_x/width,roi_y/height),pos_world=riley.pos_fill_frame_from_rot(roi,(width,height),(roi_x/width,roi_y/height),1000.,(0,0,0),1.),rot_world=(0,0,0),roi_cent_world=tuple(riley.roi_cent_from_coords(roi)),focal_length=1000.,sub_sample=ssaa,coord_sys=riley.CameraCoordSys.opengl,**psf_kwargs)
    config=riley.create_raster_config(num_frames=ux.shape[1],total_threads=RILEY_RASTER_THREADS,save_strategy=riley.SaveStrategy.both); config.frame_batch_size_per_group=1;config.max_geom_jobs_in_flight_per_group=1;config.max_geom_workers_per_job=1;config.max_raster_workers_per_job=RILEY_RASTER_THREADS;config.tile_size_min=1;config.save_format=riley.ImageFormat.tiff;config.save_bits=8;config.save_scaling=riley.ScaleStrategy.none
    root.mkdir(parents=True,exist_ok=True); images=riley.raster([mesh],[camera],config,out_dir=str(root))
    if images is not None:
        for frame in selected_frames(case,ux.shape[1]): np.save(root/f"image_c00_f{frame:02d}.npy",images[0,frame,0])
    mark_case_outputs(root, signature)
    return root
