#!/usr/bin/env python3
"""Generate all paper figures and a single A4 LaTeX preview article."""
from __future__ import annotations

import subprocess
from pathlib import Path

from paper_exp1_figs import (
    exp1_figure_stems, figure_affine_function_difference_maps,
    figure_function_shaders, generate_texture_figures,
    remove_superseded_figures,
)
from paper_exp2_figs import figure_stems as exp2_figure_stems, generate_figures as generate_exp2
from paperparams import FIGURE_CAPTIONS, FIGURE_LABELS, PAPER_OUTPUT_DIR


def write_article(stems: tuple[str, ...]) -> list[Path]:
    """Write one editable input block per figure and compile ``article.pdf``."""
    PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks: list[Path] = []
    for stem in stems:
        block = PAPER_OUTPUT_DIR / f"{stem}.tex"
        block.write_text(
            "\\begin{figure}[p]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
            f"  \\caption{{{FIGURE_CAPTIONS[stem]}}}\n"
            f"  \\label{{{FIGURE_LABELS[stem]}}}\n"
            "\\end{figure}\n",
            encoding="utf-8",
        )
        blocks.append(block)
    article = PAPER_OUTPUT_DIR / "article.tex"
    inputs = "\n".join(f"\\input{{{block.stem}}}\n\\clearpage" for block in blocks)
    article.write_text(
        "\\documentclass[10pt,a4paper]{article}\n"
        "\\usepackage[a4paper,margin=2.5cm]{geometry}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        f"{inputs}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", article.name],
        cwd=PAPER_OUTPUT_DIR,
        check=True,
    )
    return [*blocks, article, PAPER_OUTPUT_DIR / "article.pdf"]


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
