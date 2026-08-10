# Notes Figures

Can you go through all code for our paper figures in ./scripts/ including anythign touched when we run paper_all_figs.py. Then make these changes:

- exp1, fig5, when we adjust the number values for the difference simages adjust the number of sub panels - at the moment there should be 9 sub panels for this figure but there are still 16 and a bunch of them are blank
- exp1, fig5, make the symmetric colorbar limits for this figure configurable as a CONSTANT in paperparams.py and set it to +/- 10 bits.
- For the exp2 figures where we are using one specific interpolant put the interpolant in the figure file name like cubiccm or cubicbs etc.
- exp2, fig7, make the symmetric colorbar limits for this figure configurable as a CONSTANT in paperparams.py and set it to +/- 100 bits.
