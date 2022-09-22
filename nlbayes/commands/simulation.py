import os
import argparse
import numpy as np
import pandas as pd

import nlbayes
from nlbayes.utils import gen_evidence, gen_network, get_ents_rels_dfs, get_evidence_dict, get_network_dict, get_tests_from_dicts


def make_network(NX, NY, AvgNTF, seed):

    np.random.seed(seed)
    network = gen_network(NX=NX, NY=NY, AvgNTF=AvgNTF)
    np.random.seed()
    
    return network

def make_evidence(network, n_active_tfs, active_tfs_str, background_p, mor_noise_p, tf_target_fraction, seed):

    active_tfs_str = active_tfs_str.replace(',', ' ')
    active_tfs = active_tfs_str.split()
    candidates = [src for src in network.keys() if src not in active_tfs]

    np.random.seed(seed)
    for _ in range(50):
        ### Select random TF as active
        active_tfs = active_tfs_str.split()
        n = n_active_tfs - len(active_tfs)
        if n > 0:
            active_tfs.extend(np.random.choice(candidates, n, False))

        evidence = gen_evidence(
            network,
            n_active_tfs=n_active_tfs,
            active_tfs=active_tfs.copy(),
            background_p=background_p,
            mor_noise_p=mor_noise_p,
            tf_target_fraction=tf_target_fraction,
        )
        if len(evidence) >= 5:
            break
    else:
        raise RuntimeError('Unable to find tf with non-zero differential expression')
    np.random.seed()

    evid, tests = get_tests_from_dicts(network, evidence, active_tfs)

    return evid, tests.reset_index()


def flip_edge_signs_randomly(rels: pd.DataFrame, p: float) -> pd.DataFrame:
    mask = np.random.choice([False, True], p=[1-p, p], size=len(rels))
    rels.loc[mask, 'type'] =  - rels.loc[mask, 'type']

    return rels


def remove_edge_signs_randomly(rels: pd.DataFrame, p: float) -> pd.DataFrame:
    mask = np.random.choice([False, True], p=[1-p, p], size=len(rels))
    rels.loc[mask, 'type'] = 0

    return rels


def randomize_mor(rels: pd.DataFrame, p: float) -> pd.DataFrame:
    mask = np.random.choice([False, True], p=[1-p, p], size=len(rels))
    n = mask.sum()
    choices = [-1, 0, 1]
    rels.loc[mask, 'type'] = np.random.choice(choices, size=n)

    return rels


def randomize_network(ents: pd.DataFrame, rels: pd.DataFrame, p: float) -> pd.DataFrame:
    mask = np.random.choice([False, True], p=[1-p, p], size=len(rels))
    n = mask.sum()
    choices = ents.loc[ents.type == 'mRNA'].uid.values
    rels.loc[mask, 'trguid'] = np.random.choice(choices, size=n)

    return rels


def randomize_evidence(evid: pd.DataFrame, p: float) -> pd.DataFrame:
    n_genes = len(evid)
    n_upreg = (evid.val > 0).sum()
    n_dwreg = (evid.val < 0).sum()

    up_p = n_upreg / n_genes
    dw_p = n_dwreg / n_genes
    no_p = 1. - up_p - dw_p

    mask = np.random.choice([False, True], p=[1-p, p], size=n_genes)
    n = mask.sum()
    evid.loc[mask, 'val'] = np.random.choice([-1, 0, 1], p=[dw_p, no_p, up_p], size=n)

    return evid


def run_model(ents, rels, evid, tests, z0, z1, t_alpha, t_beta, s_leniency, n_graphs, combined_test_th):

    ents, rels, evid, tests = reduce_problem_size(ents, rels, evid, tests, thr=combined_test_th)

    network = get_network_dict(rels)
    evidence =  get_evidence_dict(evid)

    model = nlbayes.ModelORNOR(network, evidence, set(), False,
                               t_alpha, t_beta, z1, z0, s_leniency, n_graphs)
    model.sample_posterior(20000, 1.1, True)

    X = pd.DataFrame([dict(srcuid=srcuid, X=val) for _, srcuid, idx, val in model.get_posterior_means('X') if idx == '1']).set_index('srcuid')
    T = pd.DataFrame([dict(srcuid=srcuid, T=val) for _, srcuid, idx, val in model.get_posterior_means('T') if idx == '0']).set_index('srcuid')
    df = X.merge(T, left_index=True, right_index=True)

    result = tests.set_index('srcuid').merge(df, left_index=True, right_index=True).sort_values('X', ascending=False)

    return result.reset_index()


def reduce_problem_size(ents, rels, evid, tests, thr):
    tests = tests.loc[tests.combined <= thr].copy()
    rels = rels.loc[rels.srcuid.isin(tests.srcuid)].copy()
    evid = evid.loc[evid.uid.isin(rels.trguid)].copy()
    ents = ents.loc[(ents.uid.isin(rels.srcuid))|(ents.uid.isin(rels.trguid))].copy()
    return ents, rels, evid, tests


def simulation(
        net_seed, net_n_tfs, net_n_genes, net_avg_n_tfs, net_rnd_p,
        net_rnd_mor_p, net_rmv_sgn_p, net_flp_sgn_p,
        evd_seed, evd_n_active_tfs, evd_active_tf_uids, evd_background_p,
        evd_mor_noise_p, evd_fraction_targets, evd_rnd_p,
        z0, z1, t_alpha, t_beta, s_leniency, n_graphs, combined_test_th):

    network = make_network(net_n_tfs, net_n_genes, net_avg_n_tfs, net_seed)
    evid, tests = make_evidence(network, evd_n_active_tfs, evd_active_tf_uids,
                                evd_background_p, evd_mor_noise_p,
                                evd_fraction_targets, evd_seed)
    ents, rels = get_ents_rels_dfs(network)

    rels = randomize_network(ents, rels, net_rnd_p)
    rels = randomize_mor(rels, net_rnd_mor_p)
    rels = remove_edge_signs_randomly(rels, net_rmv_sgn_p)
    rels = flip_edge_signs_randomly(rels, net_flp_sgn_p)

    evid = randomize_evidence(evid, evd_rnd_p)

    ents, rels, evid, tests = \
        reduce_problem_size(ents, rels, evid, tests, thr=combined_test_th)

    result = run_model(ents, rels, evid, tests,
                       z0, z1, t_alpha, t_beta, s_leniency, n_graphs,
                       combined_test_th)

    return ents, rels, evid, result


def compute_explained_evidence(rels, evid, result, thr=0.5):
    assert 'srcuid' in result.columns

    srcuid = result.loc[result.X >= thr].reset_index().srcuid
    if len(srcuid) == 0:
        r = result.copy()
        r['expl_ornor'] = 0.
        r['expl_sum'] = 0.
        r['prcnt_expl_ornor'] = 0.
        r['prcnt_expl_sum'] = 0.
        data = {
            'expl_ornor': r['expl_ornor'],
            'expl_sum': r['expl_sum'],
            'prcnt_expl_ornor': r['prcnt_expl_ornor'],
            'prcnt_expl_sum': r['prcnt_expl_sum'],
            'result': r,
        }
        return data
    tmp = rels.loc[rels.srcuid.isin(srcuid)].merge(evid.loc[evid.val != 0, ['uid', 'val']], how='inner', left_on='trguid', right_on='uid')[['uid', 'type', 'val']]
    grp = tmp.groupby('uid')[['type', 'val']]

    tmp = grp.apply(lambda x: x.min())
    expl_ornor = (tmp.type * tmp.val) == 1
    prcnt_expl_ornor = expl_ornor.sum() / len(expl_ornor)
    tmp = grp.apply(lambda x: np.sign(x.sum()))
    expl_sum = (tmp.type * tmp.val) == 1
    prcnt_expl_sum = expl_sum.sum() / len(expl_sum)
    data = {
        'expl_ornor': expl_ornor,
        'expl_sum': expl_sum,
        'prcnt_expl_ornor': prcnt_expl_ornor,
        'prcnt_expl_sum': prcnt_expl_sum,
    }

    return data


def compute_confusion_matrix(r, thr=0.5):
    TP = len(r.loc[(r.gt_act)&(r.X >= thr)])
    FP = len(r.loc[(~r.gt_act)&(r.X >= thr)])
    TN = len(r.loc[(~r.gt_act)&(r.X < thr)])
    FN = len(r.loc[(r.gt_act)&(r.X < thr)])
    return TP, FP, TN, FN


def analyze_result(rels, evid, result):
    out = []
    for thr in np.linspace(0., 1., 101):
        try:
            data = compute_explained_evidence(rels, evid, result, thr=thr)
            prcnt_expl_ornor = data['prcnt_expl_ornor']
            prcnt_expl_sum = data['prcnt_expl_sum']
        except:
            prcnt_expl_ornor = float('nan')
            prcnt_expl_sum = float('nan')
        TP, FP, TN, FN = compute_confusion_matrix(result, thr=thr)
        out.append((thr, TP, FP, TN, FN, prcnt_expl_ornor, prcnt_expl_sum))
    df = pd.DataFrame(out, columns=['thr', 'TP', 'FP', 'TN', 'FN', 'prcnt_expl_ornor', 'prcnt_expl_sum'])

    return df


def main():
    print(f"This is nlbayes version: {nlbayes.__version__}\n")

    parser = argparse.ArgumentParser(description='Specify parameters.')
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--outname', type=str, default='')

    parser.add_argument('--net_seed', type=int, default=0)
    parser.add_argument('--net_n_tfs', type=int, default=250)
    parser.add_argument('--net_n_genes', type=int, default=5000)
    parser.add_argument('--net_avg_n_tfs', type=float, default=6.66)
    parser.add_argument('--net_rnd_p', type=float, default=0.)
    parser.add_argument('--net_rnd_mor_p', type=float, default=0.)
    parser.add_argument('--net_rmv_sgn_p', type=float, default=0.)
    parser.add_argument('--net_flp_sgn_p', type=float, default=0.)

    parser.add_argument('--evd_seed', type=int, default=0)
    parser.add_argument('--evd_n_active_tfs', type=int, default=10)
    parser.add_argument('--evd_active_tf_uids', type=str, default='')
    parser.add_argument('--evd_background_p', type=float, default=0.)
    parser.add_argument('--evd_mor_noise_p', type=float, default=0.)
    parser.add_argument('--evd_fraction_targets', type=float, default=0.1)
    parser.add_argument('--evd_rnd_p', type=float, default=0.)

    parser.add_argument('--z0', type=float, default=0.01)
    parser.add_argument('--z1', type=float, default=0.99)
    parser.add_argument('--t_alpha', type=float, default=1.)
    parser.add_argument('--t_beta', type=float, default=1.)
    parser.add_argument('--s_leniency', type=float, default=0.1)

    parser.add_argument('--n_graphs', type=int, default=3)
    parser.add_argument('--n_replica', type=int, default=1)

    kvargs = vars(parser.parse_args())

    outdir_path = kvargs['outdir']
    out_filename = kvargs['outname']
    n_replica = kvargs['n_replica']
    del(kvargs['outdir'], kvargs['outname'], kvargs['n_replica'])

    if n_replica > 1:
        np.random.seed(kvargs['net_seed'])
        net_seeds = np.random.choice(100_000, size=n_replica, replace=False)
        np.random.seed(kvargs['evd_seed'])
        evd_seeds = np.random.choice(100_000, size=n_replica, replace=False)
        np.random.seed()
    else:
        net_seeds = [kvargs['net_seed']]
        evd_seeds = [kvargs['evd_seed']]

    if out_filename:
        filename_template = f'evaluation___{{net_seed:05d}}_{{evd_seed:05d}}_{out_filename}'
    else:
        filename_template = (
            'evaluation___{net_seed:05d}_{evd_seed:05d}'
            '___{net_n_tfs:04d}_{net_n_genes:05d}_{net_avg_n_tfs:06.2f}_{net_rnd_p:.2f}_{net_rnd_mor_p:.2f}_{net_rmv_sgn_p:.2f}_{net_flp_sgn_p:.2f}'
            '___{evd_n_active_tfs:02d}_{evd_active_tf_uids:X<40}_{evd_background_p:.2f}_{evd_mor_noise_p:.2f}_{evd_fraction_targets:.2f}_{evd_rnd_p:.2f}'
            '___{z0:.4f}_{z1:.4f}_{t_alpha:07.2f}_{t_beta:07.2f}_{s_leniency:.2f}'
            '.csv.gz'
        )
        
    filepath_template = os.path.join(outdir_path, filename_template)
    for net_seed, evd_seed in zip(net_seeds, evd_seeds):
        kvargs['net_seed'] = net_seed
        kvargs['evd_seed'] = evd_seed

        filepath = filepath_template.format(**kvargs).replace(',', 'x')
        if os.path.exists(filepath):
            continue

        try:
            ents, rels, evid, result = simulation(**kvargs, combined_test_th=1.)
        except RuntimeError:
            continue

        df = analyze_result(rels, evid, result)
        df.to_csv(filepath, index=False)



if __name__ == '__main__':

    main()

