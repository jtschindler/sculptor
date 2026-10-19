"""`FitModel.evaluate_parameters`, removed from `src/sculptor/model.py`
on 2026-10-19 (review §2.3).

It saved one two-parameter corner plot per parameter pair to the current
directory. It no longer worked: it used the pre-1.0 chainconsumer API
(`ChainConsumer.add_chain(DataFrame)`, `plotter.plot(parameters=...)`),
which fails on chainconsumer 1.x with
`AttributeError: 'DataFrame' object has no attribute 'name'`. Nothing
called it. `FitModel.plot_posterior_corner` (one corner plot for all, or a
chosen subset of, parameters) uses the current API.

Kept for reference below, unchanged except dedented to a module-level
function (`self` is the `FitModel`), with the imports it relied on.
"""

import itertools

import chainconsumer as cc
import pandas as pd


def evaluate_parameters(self, discard=2000):

    chain = self.sampler.get_chain(discard=discard, flat=True)

    # Get the parameter names
    param_names = list(self.params_variable.keys())
    # Replace underscores
    param_names_readable = [name.replace("_", " ") for name in param_names]

    # Create a pandas DataFrame with the chain
    df_chain = pd.DataFrame(chain, columns=param_names_readable)

    # Instantiate the chain consumer
    c = cc.ChainConsumer()
    c.add_chain(df_chain)

    # Create individual corner plots
    combs = itertools.combinations(param_names_readable, 2)

    # TODO: Fine tune the corner plot settings to
    #  - include the parameter name nicely
    #  - include the maximum likelihood value
    #  - display the parameter name properly

    for comb in combs:
        filename = "./corner_plot_{}__{}.pdf".format(
            comb[0].replace(" ", "_"), comb[1].replace(" ", "_")
        )
        c.plotter.plot(parameters=comb, filename=filename)
