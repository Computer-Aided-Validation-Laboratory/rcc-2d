# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Conservative RAM planner for Experiment 2 render sweeps.

Edit ``NUM_CORES``, ``NOM_RAM_GB`` and ``OS_AND_OTHER_GB`` for the planned
machine.  Disk-additive and Gaussian-additive textures have the same memory
footprint at a given image size, oversampling and storage type.
"""

from __future__ import annotations

from exp2params import (
    AFFINE_MAX_POINTS_PER_CHUNK,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    VTK_MAX_POINTS_PER_CHUNK,
)


# User settings -------------------------------------------------------------
NUM_CORES = 12
NOM_RAM_GB = 32.0
OS_AND_OTHER_GB = 8.0

# Conservative implementation allowances ----------------------------------
RILEY_BYTES_PER_SUBPIXEL = 154
RILEY_FIXED_GB = 1.0
RILEY_TEXTURE_COPIES = 2
CUSTOM_FIXED_PER_WORKER_GB = 0.25
# Exp2 retains coordinate, coverage and lattice-index temporaries in addition
# to the raw point arrays, unlike Exp1's scalar eggbox kernel.
AFFINE_BYTES_PER_ACTIVE_POINT = 384
VTK_BYTES_PER_ACTIVE_POINT = 768
CUSTOM_BYTES_PER_SAMPLE = 48


def _gib(value: float) -> float:
    return value / 2**30


def _largest_power_of_two(predicate) -> int:
    if not predicate(1):
        return 0
    value = 1
    while predicate(value * 2):
        value *= 2
    return value


def _custom_bytes(ssaa: int, *, mapping: str) -> int:
    samples = ssaa * ssaa
    point_cap = VTK_MAX_POINTS_PER_CHUNK if mapping == "vtk" else AFFINE_MAX_POINTS_PER_CHUNK
    per_point = VTK_BYTES_PER_ACTIVE_POINT if mapping == "vtk" else AFFINE_BYTES_PER_ACTIVE_POINT
    active_points = min(point_cap, TARG_PX_X * TARG_PX_Y * samples)
    per_worker = (
        CUSTOM_FIXED_PER_WORKER_GB * 2**30
        + per_point * active_points
        + CUSTOM_BYTES_PER_SAMPLE * samples
    )
    return int(NUM_CORES * per_worker)


def _riley_float_bytes(ssaa: int, oversamp: int) -> int:
    tex_w = oversamp * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h = oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD)
    texture = tex_w * tex_h * 8  # f64 raw coverage texture
    scratch = NUM_CORES * RILEY_BYTES_PER_SUBPIXEL * ssaa * ssaa
    return int(RILEY_FIXED_GB * 2**30 + RILEY_TEXTURE_COPIES * texture + scratch)


def main() -> None:
    budget = int((NOM_RAM_GB - OS_AND_OTHER_GB) * 2**30)
    if budget <= 0:
        raise ValueError("NOM_RAM_GB must exceed OS_AND_OTHER_GB.")
    print("Experiment 2 RAM planner")
    print(f"  Image: {TARG_PX_X}x{TARG_PX_Y}; pad: {TEX_PX_PAD}; workers: {NUM_CORES}")
    print(f"  Nominal RAM: {NOM_RAM_GB:g} GiB; reserve: {OS_AND_OTHER_GB:g} GiB; render budget: {_gib(budget):.2f} GiB")
    affine_max = _largest_power_of_two(
        lambda ssaa: _custom_bytes(ssaa, mapping="affine") <= budget
    )
    vtk_max = _largest_power_of_two(
        lambda ssaa: _custom_bytes(ssaa, mapping="vtk") <= budget
    )
    print("\nBespoke orthographic renderer (rectangular/Gauss SSAA)")
    print(f"  Affine cap: {AFFINE_MAX_POINTS_PER_CHUNK:,}; maximum SSAA: {affine_max} ({_gib(_custom_bytes(affine_max, mapping='affine')):.2f} GiB)")
    print(f"  VTK cap:    {VTK_MAX_POINTS_PER_CHUNK:,}; maximum SSAA: {vtk_max} ({_gib(_custom_bytes(vtk_max, mapping='vtk')):.2f} GiB)")
    print("\nRiley f64 texture (diskaddsat and gausscont have the same footprint)")
    print("  OS    maximum feasible SSAA    estimated peak GiB")
    coupled_max = _largest_power_of_two(lambda value: _riley_float_bytes(value, value) <= budget)
    for oversamp in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        max_ssaa = _largest_power_of_two(lambda ssaa: _riley_float_bytes(ssaa, oversamp) <= budget)
        if max_ssaa and oversamp <= coupled_max * 2:
            print(f"  {oversamp:4d}  {max_ssaa:22d}    {_gib(_riley_float_bytes(max_ssaa, oversamp)):8.2f}")
    print(f"  Conservative coupled limit (SSAA=OS): {coupled_max} ({_gib(_riley_float_bytes(coupled_max, coupled_max)):.2f} GiB)")


if __name__ == "__main__":
    main()
