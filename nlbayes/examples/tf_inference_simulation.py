#!/usr/bin/env python3
"""
TF Inference Simulation Example

This script demonstrates how to use nlbayes to perform transcription factor (TF) inference
using simulated data. It generates a random network and evidence, then uses the OR-NOR model
to infer active TFs.
"""

import numpy as np
from sklearn import metrics
import plotext as plt_ascii

from nlbayes import ORNOR
from nlbayes.utils import gen_network, gen_evidence


def plot_metric_ascii(metric_type, true_labels, scores):
    """Plot ROC or PR curve using ASCII art."""
    if metric_type not in ['roc_curve', 'pr_curve']:
        raise ValueError("metric_type must be either 'roc_curve' or 'pr_curve'")
    
    plt_ascii.clf()  # Clear the plotting area
    plt_ascii.plotsize(60, 15)  # Set fixed width and height
    
    if metric_type == 'roc_curve':
        fpr, tpr, _ = metrics.roc_curve(true_labels, scores)
        auc = metrics.auc(fpr, tpr)
        
        plt_ascii.plot(fpr, tpr)
        plt_ascii.title(f'ROC Curve (AUC = {auc:.2f})')
        plt_ascii.xlabel('False Positive Rate')
        plt_ascii.ylabel('True Positive Rate')
        plt_ascii.show()
        print(f"\nAUC: {auc:.3f}")
        
    else:  # pr_curve
        precision, recall, _ = metrics.precision_recall_curve(true_labels, scores)
        auc = metrics.auc(recall, precision)
        
        plt_ascii.plot(recall, precision)
        plt_ascii.title(f'PR Curve (AUC = {auc:.2f})')
        plt_ascii.xlabel('Recall')
        plt_ascii.ylabel('Precision')
        plt_ascii.show()
        print(f"\nAUC: {auc:.3f}")


def main():
    # Generate a random network
    network = gen_network(
        NX=125,      # total number of transcription factors
        NY=2500,     # total number of target genes
        AvgNTF=5,    # number of TFs that regulate a gene (average)
    )

    # Randomly select 5 TFs as active
    candidates = list(network.keys())
    active_tfs = np.random.choice(candidates, size=5, replace=False)
    print("Selected active TFs:", active_tfs)

    # Generate evidence based on the network and active TFs
    evidence = gen_evidence(
        network=network,
        active_tfs=active_tfs,  # known set of active TFs
        tf_target_fraction=0.2  # only a fraction of a TF's targets will become diff. expr.
    )

    # Create and fit the OR-NOR model
    model = ORNOR(network, evidence, n_graphs=5)
    model.fit(n_samples=2000, gelman_rubin=1.1, burnin=True)

    # Get inference results
    results = model.get_results()
    results['ground_truth'] = results['TF_id'].isin(active_tfs)
    results = results.sort_values('X', ascending=False)
    results.index = range(1, len(results) + 1)
    results.index.name = 'rank'

    # Print top 10 TFs
    print("\nTop 10 inferred active TFs:")
    print(results.head(10))

    # Plot ROC and PR curves
    print("\nROC Curve (ASCII):")
    plot_metric_ascii('roc_curve', results['ground_truth'], results['X'])
    print("\nPR Curve (ASCII):")
    plot_metric_ascii('pr_curve', results['ground_truth'], results['X'])


if __name__ == '__main__':
    main()
