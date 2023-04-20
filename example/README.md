This tutorial supposes we already have results for a power-law + peak mass, and gaussian $\chi_\mathrm{eff}$ population model.
The .ini file sets the desired population `models` for the proposal distribution, as well as the `vt-models` for selection effects.
The `data-file` points to the event posterior samples, a processed `vt-file`, and our gwpopulation `result-file`.
We then specify an `output-file` for our resampled results.
Finally, we set up our added hyper-parameter with a name set with `new-param-name`.
Then, `param-min` and `param-max` act as prior bounds for our new parameter, and `param-bins` can be used to set the number of grid-points at which this prior (and the proposal distribution) is calculated at.

To run, we just need to type `popresample correlated_chieff_example.ini`.