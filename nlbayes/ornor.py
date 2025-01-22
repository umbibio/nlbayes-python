"""
OR-NOR Bayesian Network Model for TF Activity Inference

This module provides a high-level interface to the OR-NOR Bayesian network model
for inferring transcription factor activities from gene expression data.
"""

from .ModelORNOR import PyModelORNOR
import pandas as pd
import warnings


class ORNOR:
    """OR-NOR Bayesian Network Model for TF Activity Inference.

    Parameters
    ----------
    network : dict
        Network structure where keys are TF names and values are dictionaries
        mapping target gene names to regulation modes (-1 for repression, 1 for activation)
    evidence : dict, optional
        Dictionary mapping gene names to expression states (-1 for down, 1 for up)
    n_graphs : int, default=3
        Number of parallel graphs for convergence assessment
    uniform_prior : bool, default=False
        Whether to use uniform prior for theta parameter
    active_tfs : set, optional
        Set of TFs known to be active (for semi-supervised learning)
    """
    def __init__(self, network, evidence=None, n_graphs=3, uniform_prior=False, active_tfs=None):
        self._model = PyModelORNOR(
            network=network,
            evidence=evidence if evidence is not None else dict(),
            active_tf_set=active_tfs if active_tfs is not None else set(),
            uniform_t=uniform_prior,
            n_graphs=n_graphs
        )
        self._fitted = False

    def fit(self, n_samples=2000, gelman_rubin=1.1, burnin=True):
        """Fit the model using MCMC sampling.

        Parameters
        ----------
        n_samples : int, default=2000
            Number of samples to draw
        gelman_rubin : float, default=1.1
            Gelman-Rubin convergence criterion
        burnin : bool, default=True
            Whether to perform burn-in phase

        Returns
        -------
        self : ORNOR
            The fitted model
        """
        self._model.sample_posterior(N=n_samples, gr_level=gelman_rubin, burnin=burnin)
        self._fitted = True
        return self

    def get_results(self):
        """Get inference results.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing TF inference results with columns:
            - TF_id: TF identifier
            - X: Inferred activity score
            - posterior_p: Posterior probability of activity
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting results")
        return self._model.inference_posterior_df()

    def __getattr__(self, name):
        """Delegate unknown attributes to underlying model."""
        return getattr(self._model, name)
