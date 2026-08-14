#!/usr/bin/env python3
"""Generate all paper figures and a single A4 LaTeX preview article."""
from __future__ import annotations

from paper_exp1_figs import (
    exp1_figure_stems, figure_function_shaders, figure_texture_convergence,
    remove_superseded_figures,
)
from paper_exp2_figs import (
    figure_stems as exp2_figure_stems,
    generate_figures as generate_exp2,
)
from paper_exp3_figs import (
    figure_stems as exp3_figure_stems,
    generate_figures as generate_exp3,
)
from paper_ext_figs import generate_figures as generate_extended_figures
from modules.paperfigs import write_latex_preview


def write_article(stems: tuple[str, ...]) -> list[Path]:
    """Write one editable input block per figure and compile ``article.pdf``."""
    return write_latex_preview(stems)


def main() -> None:
    remove_superseded_figures()
    written = figure_function_shaders()
    written.extend(figure_texture_convergence())
    written.extend(generate_exp2())
    written.extend(generate_exp3())
    # Supplementary figures are intentionally kept out of the article and
    # manuscript directory; they are regenerated alongside the paper figures.
    written.extend(generate_extended_figures())
    all_stems = (
        *exp1_figure_stems(),
        *exp2_figure_stems(),
        *exp3_figure_stems(),
    )
    written.extend(write_article(all_stems))
    print("Wrote paper figures and LaTeX preview:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
