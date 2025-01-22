#!/usr/bin/env python3
"""
TF Inference Simulation Example

This script demonstrates how to use nlbayes to perform transcription factor (TF) inference
using simulated data. It generates a random network and evidence, then uses the OR-NOR model
to infer active TFs.
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import plotext as plt_ascii

import nlbayes
from nlbayes.utils import gen_network, gen_evidence

def plot_metric_ascii(metric_type, model, df, active_tfs):
    """Plot ROC or PR curve for the model predictions using ASCII art."""
    if metric_type not in ['roc_curve', 'pr_curve']:
        raise ValueError("metric_type must be either 'roc_curve' or 'pr_curve'")
    
    y_true = df['gt_act'].values
    y_score = df['X'].values
    
    plt_ascii.clf()  # Clear the plotting area
    plt_ascii.plotsize(60, 15)  # Set fixed width and height
    
    if metric_type == 'roc_curve':
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        
        # Create ASCII plot
        plt_ascii.plot(fpr, tpr)
        plt_ascii.title(f'ROC Curve (AUC = {auc:.2f})')
        plt_ascii.xlabel('False Positive Rate')
        plt_ascii.ylabel('True Positive Rate')
        plt_ascii.show()
        print(f"\nAUC: {auc:.3f}")
        
    else:  # pr_curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        auc = metrics.auc(recall, precision)
        
        # Create ASCII plot
        plt_ascii.plot(recall, precision)
        plt_ascii.title(f'PR Curve (AUC = {auc:.2f})')
        plt_ascii.xlabel('Recall')
        plt_ascii.ylabel('Precision')
        plt_ascii.show()
        print(f"\nAUC: {auc:.3f}")

def plot_metric(metric_type, model, df, active_tfs):
    """Plot ROC or PR curve for the model predictions."""
    if metric_type not in ['roc_curve', 'pr_curve']:
        raise ValueError("metric_type must be either 'roc_curve' or 'pr_curve'")
    
    y_true = df['gt_act'].values
    y_score = df['X'].values
    
    if metric_type == 'roc_curve':
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
    else:  # pr_curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        auc = metrics.auc(recall, precision)
        plt.plot(recall, precision, label=f'PR curve (AUC = {auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

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
        tf_target_fraction=0.2, # only a fraction of a TF's targets will become diff. expr.
    )

    # Create and run the OR-NOR model
    model = nlbayes.ModelORNOR(network, evidence, uniform_t=False, n_graphs=5)
    model.sample_posterior(N=2000, gr_level=1.1, burnin=True)

    # Get inference results as a DataFrame
    df = model.inference_posterior_df()
    df['gt_act'] = df['TF_id'].isin(active_tfs)  # ground truth for active TFs
    df = df.sort_values('X', ascending=False)  # sort by inferred activity
    df.index = range(1, len(df) + 1)
    df.index.name = 'rank'

    # Print top 10 TFs
    print("\nTop 10 inferred active TFs:")
    print(df.head(10))

    # Plot ROC and PR curves in ASCII
    print("\nROC Curve (ASCII):")
    plot_metric_ascii('roc_curve', model, df, active_tfs)
    print("\nPR Curve (ASCII):")
    plot_metric_ascii('pr_curve', model, df, active_tfs)

if __name__ == '__main__':
    main()
