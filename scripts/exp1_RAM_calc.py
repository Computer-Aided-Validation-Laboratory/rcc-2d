# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Conservative RAM planner for Experiment 1 render sweeps.

Edit the three user settings below for the machine being planned.  The model
uses the renderer's documented 154 B/sub-pixel Riley scratch estimate and
includes two resident copies of a Riley texture (the Python/C-ABI hand-off).
It intentionally leaves a fixed process-overhead allowance as well.
"""

from __future__ import annotations

import math

from exp1params import TARG_PX_X, TARG_PX_Y, TEX_PX_PAD


# User settings -------------------------------------------------------------
NUM_CORES = 12
NOM_RAM_GB = 32.0
OS_AND_OTHER_GB = 8.0

# Conservative implementation allowances ----------------------------------
RILEY_BYTES_PER_SUBPIXEL = 154
RILEY_FIXED_GB = 1.0
RILEY_TEXTURE_COPIES = 2
CUSTOM_FIXED_PER_WORKER_GB = 0.25
CUSTOM_BYTES_PER_ACTIVE_POINT = 128
CUSTOM_BYTES_PER_SAMPLE = 48
CUSTOM_MAX_POINTS_PER_CHUNK = 10_000_000


def _gib(value: float) -> float:
    return value / 2**30


def _largest_power_of_two(predicate) -> int:
    if not predicate(1):
        return 0
    value = 1
    while predicate(value * 2):
        value *= 2
    return value


def _custom_bytes(ssaa: int) -> int:
    """Peak estimate for the multiprocessing bespoke rectangular renderer."""
    samples = ssaa * ssaa
    active_points = min(CUSTOM_MAX_POINTS_PER_CHUNK, TARG_PX_X * TARG_PX_Y * samples)
    per_worker = (
        CUSTOM_FIXED_PER_WORKER_GB * 2**30
        + CUSTOM_BYTES_PER_ACTIVE_POINT * active_points
        + CUSTOM_BYTES_PER_SAMPLE * samples
    )
    return int(NUM_CORES * per_worker)


def _riley_bytes(ssaa: int, oversamp: int, texture_item_bytes: int) -> int:
    """Peak estimate for one Riley texture render, including C-ABI copies."""
    tex_w = oversamp * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h = oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD)
    texture = tex_w * tex_h * texture_item_bytes
    scratch = NUM_CORES * RILEY_BYTES_PER_SUBPIXEL * ssaa * ssaa
    return int(RILEY_FIXED_GB * 2**30 + RILEY_TEXTURE_COPIES * texture + scratch)


def _riley_function_bytes(ssaa: int) -> int:
    return int(RILEY_FIXED_GB * 2**30 + NUM_CORES * RILEY_BYTES_PER_SUBPIXEL * ssaa * ssaa)


def _report_texture_case(name: str, item_bytes: int, budget: int) -> None:
    max_os_converged = _largest_power_of_two(
        lambda value: _riley_bytes(value, value, item_bytes) <= budget
    )
    print(f"\n{name}")
    print("  OS    maximum feasible SSAA    estimated peak GiB")
    for oversamp in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        max_ssaa = _largest_power_of_two(
            lambda ssaa: _riley_bytes(ssaa, oversamp, item_bytes) <= budget
        )
        if max_ssaa and oversamp <= max_os_converged * 2:
            print(f"  {oversamp:4d}  {max_ssaa:22d}    {_gib(_riley_bytes(max_ssaa, oversamp, item_bytes)):8.2f}")
    print(
        f"  Conservative coupled limit (SSAA=OS): {max_os_converged} "
        f"({_gib(_riley_bytes(max_os_converged, max_os_converged, item_bytes)):.2f} GiB)"
    )


def main() -> None:
    budget = int((NOM_RAM_GB - OS_AND_OTHER_GB) * 2**30)
    if budget <= 0:
        raise ValueError("NOM_RAM_GB must exceed OS_AND_OTHER_GB.")
    print("Experiment 1 RAM planner")
    print(f"  Image: {TARG_PX_X}x{TARG_PX_Y}; pad: {TEX_PX_PAD}; workers: {NUM_CORES}")
    print(f"  Nominal RAM: {NOM_RAM_GB:g} GiB; reserve: {OS_AND_OTHER_GB:g} GiB; render budget: {_gib(budget):.2f} GiB")

    custom_max = _largest_power_of_two(lambda ssaa: _custom_bytes(ssaa) <= budget)
    func_max = _largest_power_of_two(lambda ssaa: _riley_function_bytes(ssaa) <= budget)
    print("\nBespoke orthographic renderer (rectangular SSAA)")
    print(f"  Maximum conservative SSAA: {custom_max} ({_gib(_custom_bytes(custom_max)):.2f} GiB)")
    print("\nRiley function shader")
    print(f"  Maximum conservative SSAA: {func_max} ({_gib(_riley_function_bytes(func_max)):.2f} GiB)")
    _report_texture_case("Riley float texture (f64)", 8, budget)
    _report_texture_case("Riley uint8 texture", 1, budget)
    _report_texture_case("Riley uint16 texture", 2, budget)


if __name__ == "__main__":
    main()
