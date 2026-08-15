"""Fast regression and API tests for the distributed PixInt2D package."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pixint2d import (
    AdditiveSpeckles, AnalyticRule, Camera2D, DisplacementSeries, Eggbox,
    GaussRule, Grid2D, MappingMode, Mesh2D, RectRule, RenderOptions, Speck2D,
    quantise_image,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "src" / "pixint2d" / "data" / "single_elem"
GOLD = ROOT / "gold"
CAMERA = Camera2D((32, 32), (32.0, 32.0))


def case(name: str) -> tuple[Mesh2D, DisplacementSeries]:
    path = DATA / name
    return Mesh2D.from_csv(path), DisplacementSeries.from_csv(path)


def disk() -> AdditiveSpeckles:
    return AdditiveSpeckles.jittered_lattice(
        kind="disk", speckle_diameter=5.0, black_area_fraction=.6,
        jitter_pdf="uniform", jitter=.25, seed=3, bounds=(-20., 20., -20., 20.),
    )


def gaussian() -> AdditiveSpeckles:
    return AdditiveSpeckles.jittered_lattice(
        kind="gaussian", speckle_diameter=5.0, black_area_fraction=.6,
        jitter_pdf="gaussian", jitter=.12, seed=3, bounds=(-20., 20., -20., 20.),
        gaussian_edge_fraction=.4, tail_sigmas=8.,
    )


def gold(case_name: str, name: str) -> np.ndarray:
    return np.load(GOLD / case_name / f"{name}.npy")


def test_digitisation_uses_full_camera_code_range() -> None:
    assert np.array_equal(quantise_image(np.array([0., .5, 1.]), 8), np.array([0, 128, 255], dtype=np.uint8))
    assert np.array_equal(quantise_image(np.array([-1., 2.]), 12), np.array([0, 4095], dtype=np.uint16))


def test_grid2d_analytic_rigid_gold() -> None:
    mesh, displacement = case("plate42_cam32_quad9_rigid")
    renderer = Grid2D(mesh, CAMERA, Eggbox(), options=RenderOptions(mapping=MappingMode.AFFINE))
    for frame in (0, 3):
        actual = renderer.render(displacement, frame=frame, integration=AnalyticRule()).image
        assert np.allclose(actual, gold("grid2d_eggbox", f"rigid_f{frame:02d}"), atol=2e-13, rtol=0)


def test_grid2d_analytic_affine_gold() -> None:
    mesh, displacement = case("plate42_cam32_quad9_affine")
    actual = Grid2D(mesh, CAMERA, Eggbox(), options=RenderOptions(mapping=MappingMode.AFFINE)).render(displacement, frame=3, integration=AnalyticRule()).image
    assert np.allclose(actual, gold("grid2d_eggbox", "affine_f03"), atol=2e-13, rtol=0)


def test_speck2d_gaussian_analytic_gold() -> None:
    mesh, displacement = case("plate42_cam32_quad9_rigid")
    renderer = Speck2D(mesh, CAMERA, gaussian(), options=RenderOptions(mapping=MappingMode.AFFINE))
    for frame in (0, 3):
        actual = renderer.render(displacement, frame=frame, integration=AnalyticRule()).image
        assert np.allclose(actual, gold("speck2d_gauss", f"rigid_f{frame:02d}"), atol=2e-13, rtol=0)


def test_speck2d_disk_rectangular_gold_and_parallel_equivalence() -> None:
    mesh, displacement = case("plate42_cam32_quad9_rigid")
    serial = Speck2D(mesh, CAMERA, disk(), options=RenderOptions(mapping=MappingMode.AFFINE, workers=1)).render(displacement, frame=3, integration=RectRule(4)).image
    parallel = Speck2D(mesh, CAMERA, disk(), options=RenderOptions(mapping=MappingMode.AFFINE, workers=2, max_points_per_chunk=64)).render(displacement, frame=3, integration=RectRule(4)).image
    assert np.array_equal(serial, parallel)
    assert np.allclose(serial, gold("speck2d_disk", "rigid_rect4_f03"), atol=1e-14, rtol=0)


def test_quad9_newton_matches_existing_saddle_gold_and_vtk_is_valid() -> None:
    mesh, displacement = case("plate42_cam32_quad9_quadsaddle")
    newton = Grid2D(mesh, CAMERA, Eggbox(), options=RenderOptions(mapping=MappingMode.QUAD9_NEWTON)).render(displacement, frame=3, integration=GaussRule(2)).image
    vtk = Grid2D(mesh, CAMERA, Eggbox(), options=RenderOptions(mapping=MappingMode.VTK)).render(displacement, frame=3, integration=GaussRule(2)).image
    assert np.allclose(newton, gold("grid2d_eggbox", "quadsaddle_gauss2_f03"), atol=2e-13, rtol=0)
    # VTK is deliberately retained as the robust arbitrary-mesh path. Its
    # interpolation convention is not the same as the exact Quad9 inverse.
    assert np.isfinite(vtk).all()


def test_packaged_finite_star_fixture_and_mapping_paths() -> None:
    path = ROOT.parent / "src" / "pixint2d" / "data" / "finite_star_coarse_q9"
    mesh, displacement = Mesh2D.from_csv(path), DisplacementSeries.from_csv(path)
    assert mesh.coords.shape == (825, 2)
    assert mesh.connectivity.shape == (192, 9)
    camera = Camera2D((16, 4), (256., 64.))
    vtk = Grid2D(mesh, camera, Eggbox(), options=RenderOptions(mapping=MappingMode.VTK)).render(displacement, frame=1, integration=GaussRule(2)).image
    structured = Grid2D(mesh, camera, Eggbox(), options=RenderOptions(mapping=MappingMode.STRUCTURED_QUAD9)).render(displacement, frame=1, integration=GaussRule(2)).image
    assert vtk.shape == structured.shape == (4, 16)
    assert np.isfinite(vtk).all() and np.isfinite(structured).all()
