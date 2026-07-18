# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Shared implementation for the four Exp3 high-accuracy texture renders."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import riley

from exp3_riley_common import (
    camera_for_roi,
    complete,
    compute_texture_world_uvs,
    load_case,
    load_digitised_texture,
    load_float_texture,
    make_config,
    mesh_type,
    raster_plan,
    selected_ints,
    selected_names,
    texture_shape,
)
from exp3params import (
    ANALYTIC_SPECKLE_TYPES,
    BIT_DEPTHS,
    BLACK_AREA_FRACTIONS,
    DEFORMATION_CASES,
    FORCE_RENDER_OVER,
    GAMMA,
    I0,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    RANDOM_SEED,
    RILEY_SSAA_LEVELS,
    TEX_INTERPOLATORS,
    TEX_OVERSAMPLES,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
    TARG_PX_X,
)
from script_timing import ScriptTimer


TextureKind = Literal["eggbox", "speckle"]
InputKind = Literal["texf", "texu"]


def _pattern_tag(
    pattern_type: str, black_fraction: float, distribution: str, fraction: float
) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def _source_path(
    texture_kind: TextureKind,
    input_kind: InputKind,
    oversamp: int,
    bits: int | None,
    tag: str | None,
) -> Path:
    if texture_kind == "eggbox":
        base = f"tex_px{TARG_PX_X}_int_analytic_param_0"
        if input_kind == "texf":
            return TEXTURE_OUTPUT_DIR / f"{base}_pad{TEX_PX_PAD}_oversamp{oversamp}.npy"
        return TEXTURE_OUTPUT_DIR / f"{base}_b{bits}_pad{TEX_PX_PAD}_oversamp{oversamp}.tiff"
    assert tag is not None
    base = f"tex_px{TARG_PX_X}_{tag}_pad{TEX_PX_PAD}_oversamp{oversamp}_analytic"
    return TEXTURE_OUTPUT_DIR / (f"{base}.npy" if input_kind == "texf" else f"{base}_b{bits}.tiff")


def _output_root(texture_kind: TextureKind, input_kind: InputKind) -> Path:
    return Path(f"./out/exp3_riley_render_{texture_kind}_{input_kind}")


def _result_suffixes(texture_kind: TextureKind, input_kind: InputKind) -> tuple[str, ...]:
    if texture_kind == "speckle" and input_kind == "texf":
        return ("_raw", "_clamped")
    return ("",)


def _clamp_speckle_coverage(coverage: np.ndarray) -> np.ndarray:
    inverse = 1.0 - np.clip(coverage, 0.0, 1.0)
    return np.clip(I0 + GAMMA * (2.0 * inverse - 1.0), 0.0, 1.0)


def _render_one(
    *,
    coords: np.ndarray,
    connect: np.ndarray,
    disp: np.ndarray,
    roi_size: float,
    case_out: Path,
    texture_path: Path,
    texture_kind: TextureKind,
    input_kind: InputKind,
    interpolator: str,
    oversamp: int,
    ssaa: int,
    bits: int | None,
) -> None:
    shape = texture_shape(oversamp)
    if input_kind == "texf":
        texture = load_float_texture(texture_path, shape)
        storage = riley.TextureStorage.floating
        storage_bytes = np.dtype(np.float64).itemsize
        scaling_type = (
            riley.ScaleStrategy.none
            if texture_kind == "speckle"
            else riley.ScaleStrategy.fixed
        )
    else:
        texture, storage = load_digitised_texture(texture_path, shape)
        storage_bytes = texture.dtype.itemsize
        scaling_type = riley.ScaleStrategy.fixed

    plan = raster_plan(oversamp, ssaa, storage_bytes)
    print(
        f"    memory plan: texture={plan.texture_bytes / 2**30:.2f} GiB, "
        f"scratch/worker={plan.scratch_bytes_per_worker / 2**30:.2f} GiB, "
        f"raster workers={plan.threads}"
    )
    mesh = riley.Mesh(
        mesh_type=mesh_type(connect.shape[1]),
        coords=coords,
        connect=connect,
        disp=disp,
        shader_type=riley.ShaderType.tex,
        uvs=compute_texture_world_uvs(
            coords, roi_size, TARG_PX_X, oversamp
        ),
        texture=texture,
        texture_storage=storage,
        sample=TEX_INTERPOLATORS[interpolator],
        sample_mode=riley.TextureSampleMode.direct,
        bits=16 if bits is None else bits,
        scaling_type=scaling_type,
        scaling_min=0.0,
        scaling_max=1.0,
    )
    images = riley.raster(
        [mesh], [camera_for_roi(roi_size, ssaa)], make_config(disp.shape[0], plan)
    )
    if images is None:
        raise RuntimeError("Riley returned no in-memory image data.")
    case_out.mkdir(parents=True, exist_ok=True)
    for frame in range(disp.shape[0]):
        image = np.asarray(images[0, frame, 0], dtype=np.float64)
        if texture_kind == "speckle" and input_kind == "texf":
            np.save(case_out / f"image_c00_f{frame:02d}_raw.npy", image)
            np.save(
                case_out / f"image_c00_f{frame:02d}_clamped.npy",
                _clamp_speckle_coverage(image),
            )
        else:
            np.save(case_out / f"image_c00_f{frame:02d}.npy", image)


def main(texture_kind: TextureKind, input_kind: InputKind) -> None:
    """Run an Exp3 texture sweep; configuration is overrideable by EXP3_* envs."""
    print(f"Experiment 3: Riley {texture_kind} {input_kind} texture render")
    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / name for name in DEFORMATION_CASES]
    oversamples = selected_ints("EXP3_TEX_OVERSAMPLES", TEX_OVERSAMPLES)
    ssaas = selected_ints("EXP3_RILEY_SSAA_LEVELS", RILEY_SSAA_LEVELS)
    interpolators = selected_names("EXP3_TEX_INTERPOLATORS", TEX_INTERPOLATORS)
    bit_depths = selected_ints("EXP3_BIT_DEPTHS", BIT_DEPTHS) if input_kind == "texu" else [None]
    root = _output_root(texture_kind, input_kind)
    root.mkdir(parents=True, exist_ok=True)
    timer = ScriptTimer(sys.argv[0])

    tags: list[str | None]
    if texture_kind == "eggbox":
        tags = [None]
    else:
        tags = [
            _pattern_tag(pattern_type, black_fraction, distribution, fraction)
            for pattern_type in ANALYTIC_SPECKLE_TYPES
            for black_fraction in BLACK_AREA_FRACTIONS
            for distribution in PERTURBATION_DISTRIBUTIONS
            for fraction in PERTURBATION_FRACTIONS
        ]

    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist; skipping.")
            continue
        coords, connect, disp, roi_size, frames = load_case(case_path)
        for tag in tags:
            for interpolator in interpolators:
                for oversamp in oversamples:
                    for ssaa in ssaas:
                        for bits in bit_depths:
                            path = _source_path(
                                texture_kind, input_kind, oversamp, bits, tag
                            )
                            config_tag = f"ss{ssaa}_oversamp{oversamp}"
                            if bits is not None:
                                config_tag += f"_srcb{bits}"
                            name = f"{case_path.name}_{interpolator}"
                            if tag is not None:
                                name += f"_{tag}"
                            case_out = root / name / config_tag
                            if not FORCE_RENDER_OVER and complete(
                                case_out, frames, _result_suffixes(texture_kind, input_kind)
                            ):
                                print(f"  {case_out}: outputs exist; skipping.")
                                continue
                            if not path.exists():
                                print(f"  Missing texture: {path}; skipping.")
                                continue
                            print(
                                f"  {case_path.name}, interp={interpolator}, "
                                f"oversamp={oversamp}, SSAA={ssaa}"
                                + ("" if bits is None else f", source bits={bits}")
                            )
                            with timer.case(str(case_out.relative_to(root))):
                                _render_one(
                                    coords=coords, connect=connect, disp=disp,
                                    roi_size=roi_size, case_out=case_out,
                                    texture_path=path, texture_kind=texture_kind,
                                    input_kind=input_kind, interpolator=interpolator,
                                    oversamp=oversamp, ssaa=ssaa, bits=bits,
                                )
