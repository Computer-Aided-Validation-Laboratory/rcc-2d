#!/usr/bin/env python3
"""Generate all paper figures and a single A4 LaTeX preview article."""
from __future__ import annotations

from paper_exp1_figs import (
    exp1_figure_stems, figure_affine_function_difference_maps,
    figure_function_shaders, generate_texture_figures,
    remove_superseded_figures,
)
from paper_exp2_figs import figure_stems as exp2_figure_stems, generate_figures as generate_exp2
from modules.paperfigs import write_latex_preview
from paperparams import FIGURE_CAPTIONS, FIGURE_LABELS


def write_article(stems: tuple[str, ...]) -> list[Path]:
    """Write one editable input block per figure and compile ``article.pdf``."""
    return write_latex_preview(stems, FIGURE_CAPTIONS, FIGURE_LABELS)


def main() -> None:
    remove_superseded_figures()
    written = figure_function_shaders()
    written.extend(generate_texture_figures())
    written.extend(figure_affine_function_difference_maps())
    written.extend(generate_exp2())
    written.extend(write_article((*exp1_figure_stems(), *exp2_figure_stems())))
    print("Wrote paper figures and LaTeX preview:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
