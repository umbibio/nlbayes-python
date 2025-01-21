from typing import Tuple
import json
import numpy as np
import pandas as pd
from itertools import chain
from scipy.stats import fisher_exact

from numpy.random import MT19937, RandomState, SeedSequence



def ORNOR_inference(network, evidence):

    if isinstance(network, pd.DataFrame):
        network = get_network_dict(network)
    
    if isinstance(evidence, pd.DataFrame):
        evidence = get_evidence_dict(evidence)
        
    return


def read_network_json(filepath: str) -> dict:
    with open(filepath) as file:
        network = json.load(file)
    return network

def gen_network(NX: int, NY: int, AvgNTF: float=20,
                p_up: float=0.65, p_dw: float=0.35, rng=np.random) -> dict:

    assert p_up + p_dw <= 1.
    r = 1.8

    p = NX/NY*r/AvgNTF
    n_edges = rng.negative_binomial( r, p, size=NX)
    n_edges = np.minimum(NY, n_edges)
    n_edges = np.maximum(3, n_edges)
    n_edges = np.sort(n_edges)[::-1]
    network = {}

    for i in range(NX):
        src = f"TF{i+1:03d}"

        js = np.sort(rng.choice(NY, size=n_edges[i], replace=False))
        trgs = [f"Gene{j+1:05d}" for j in js]

        mors = rng.choice([-1, 0, 1], p=[p_dw, 1. - (p_up + p_dw), p_up], size=len(trgs)).tolist()

        network[src] = dict(zip(trgs, mors))

    return network


def gen_evidence(network: dict, n_active_tfs: int=1, active_tfs: list=[],
                 background_p: float=0.02, mor_noise_p: float=0.05,
                 tf_target_fraction: float=0.1, mor_weights: tuple=(1, 1), 
                 return_active_tfs: bool=False,
                 rng = np.random) -> dict:

    tfs = list(network.keys())
    genes = list(set(chain(*[d.keys() for d in network.values()])))
    tfs.sort()
    genes.sort()

    n_active_tfs = max(n_active_tfs, len(active_tfs))

    ### Select random TF as active
    n = n_active_tfs - len(active_tfs)
    candidates = [src for src in tfs if src not in active_tfs]
    if n > 0:
        active_tfs.extend(rng.choice(candidates, n, False))

    ### Generate a random deg background
    p = 1 - background_p
    vals = rng.choice([-1, 0, 1], len(genes), p=[(1-p)/2, p, (1-p)/2])
    evidence = dict(zip(genes, vals))

    ### We will randomize the actual sign of expression for each target
    probs = [[1.0 - mor_noise_p, 0.9 * mor_noise_p, 0.1 * mor_noise_p],
             [0.5 * mor_noise_p, 1.0 - mor_noise_p, 0.5 * mor_noise_p],
             [0.1 * mor_noise_p, 0.9 * mor_noise_p, 1.0 - mor_noise_p]]

    ### regulation direction may be biased by changing the mor_weight
    ### each TF contribution will be added/substracted algebraically and then will take only sign
    dw_step, up_step = mor_weights

    ### For each active TF, select a fraction of its targets and generate randomized expression
    ### Chance that sign of expression matches the sign of regulation is given by probs
    for act_src_uid in active_tfs:
        trgs = list(network[act_src_uid].keys())
        de_trgs = rng.choice(trgs,
                                   size=int(len(trgs)*tf_target_fraction),
                                   replace=False)
        for trg in de_trgs:
            mor = network[act_src_uid][trg]
            evidence[trg] += rng.choice([-dw_step, 0, up_step],
                                              p=probs[mor+1])
    evidence = {k:np.sign(v) for k, v in evidence.items() if v != 0}
    
    ngtac = len(np.intersect1d(
        [s for s, d in network.items()
         if len(np.intersect1d(list(d.keys()), list(evidence.keys()))) > 0],
        active_tfs))
    ndegs = len(evidence)
    nrels = sum([len(d) for d in network.values()])
    print(f"{ngtac=}, {ndegs=}, {nrels=}")

    if return_active_tfs:
        return evidence, active_tfs

    return evidence


def get_ents_rels_dfs(network: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rels = []
    src_ents = set()
    trg_ents = set()
    for src, trg_dict in network.items():
        src_ents.add(src)
        for trg, mor in trg_dict.items():
            trg_ents.add(trg)
            rels.append({'srcuid': src, 'trguid': trg, 'type': mor})
    src_ents = list(src_ents)
    trg_ents = list(trg_ents)
    src_ents.sort()
    trg_ents.sort()

    rels = pd.DataFrame(rels)
    ents = pd.DataFrame([{'uid': src, 'name': src} for src in src_ents] + [{'uid': trg, 'name': trg} for trg in trg_ents])

    ents = ents.assign(type=['Protein']*len(src_ents) + ['mRNA']*len(trg_ents))

    return ents, rels


def get_evid_df(evidence: dict) -> pd.DataFrame:
    e = pd.DataFrame()
    e['uid'] = e['name'] = evidence.keys()
    e['type'] = 'mRNA'
    e['foldchange'] = [float(evidence[g]) for g in e['uid']]
    e['pvalue'] = 1.-(e['foldchange'] != 0).astype(float)
    e['val'] = np.sign(e['foldchange']).astype(int)

    return e


# def get_network_dict(rels: pd.DataFrame, src_colname: str='srcuid', trg_colname: str='trguid', mor_colname: str='type') -> dict:
#     dff = rels.set_index([src_colname, trg_colname])
#     dff = dff.groupby(level=0).apply(lambda df: df.xs(df.name)[mor_colname].to_dict())
#     return dff.to_dict()


def get_network_dict(network: pd.DataFrame, src_colname: str='', trg_colname: str='', mor_colname: str='') -> dict:

    # clean up the names of the columns to try to match with possible roles.
    # We convert to lowercase and use string translation to remove uninformative
    # characters 
    t = str.maketrans('', '', '.-_ ')
    smpl_cols = [c.lower().translate(t) for c in network.columns]

    # possible matches for each role. The order matters, we select the the first
    # match of the list
    tst_collection = {
        'src': ['src', 'tf', 'factor'],
        'trg': ['trg', 'target', 'gene'],
        'mor': ['mor', 'type', 'sign', 'reg', 'sgn'],
    }

    gcn = {}
    if src_colname:
        assert src_colname in network.columns
        gcn['src'] = src_colname
    if trg_colname:
        assert trg_colname in network.columns
        gcn['trg'] = trg_colname
    if mor_colname:
        assert mor_colname in network.columns
        gcn['mor'] = mor_colname

    missing_columns = []
    for role in tst_collection.keys():
        if role in gcn.keys():
            continue

        # we run each test in order. If we find a match, both loops get broken
        # if no match, the inner loop is complete and we go to the next test
        for tst in tst_collection[role]:
            for scol, col in zip(smpl_cols, network.columns):
                if tst in scol:
                    gcn[role] = col
                    print(f"Using column `{col}` as {role}")
                    break
            else:
                continue
            break
        else:
            missing_columns.append(f"'{role}'")
            print(f"Error: failed to guess columns for '{role}'")

    if missing_columns:
        message = 'Error: missing ' 
        message += 'column ' if len(missing_columns) == 1 else 'columns '
        message += ', '.join(missing_columns)
        raise ValueError(message)

    network = network.loc[:, [gcn['src'], gcn['trg'], gcn['mor']]]
    network = network.dropna()
    network[gcn['src']] = network[gcn['src']].astype(str)
    network[gcn['trg']] = network[gcn['trg']].astype(str)
    network[gcn['mor']] = network[gcn['mor']].apply(np.sign).astype(int)

    dff = network.set_index([gcn['src'], gcn['trg']])
    dff = dff.groupby(level=0).apply(lambda df: df.xs(df.name)[gcn['mor']].to_dict())
    return dff.to_dict()


# def get_evidence_dict(evid: pd.DataFrame, uid_colname: str='uid', deg_colname: str='val') -> dict:
#     dff = evid.loc[evid[deg_colname]!=0].set_index(uid_colname)
#     return dff[deg_colname].to_dict()


def get_evidence_dict(evidence: pd.DataFrame, logfc_threshold: float=0.585, pval_threshold: float=0.05,
                      gene_colname: str='', pval_colname: str='', logfc_colname: str='',
                      network: dict={}) -> dict:

    evidence = evidence.copy()
    genes_in_network = set([g for r in network.values() for g in r.keys()])

    # clean up the names of the columns to try to match with possible roles.
    # We convert to lowercase and use string translation to remove uninformative
    # characters 
    t = str.maketrans('', '', '.-_ ')
    smpl_cols = [c.lower().translate(t) for c in evidence.columns]

    # possible matches for each role. The order matters, we select the the first
    # match of the list
    tst_collection = {
        'gene': ['geneid', 'ncbi', 'entrez', 'ensembl', 'symbol', 'gene', 'name', 'uid'],
        'pval': ['adj', 'pval', 'p-val'],
        'logfc': ['log2fc', 'logfc', 'foldchange', 'val'],
    }

    gcn = {}
    if gene_colname:
        assert gene_colname in evidence.columns
        gcn['gene'] = gene_colname
    if pval_colname:
        assert pval_colname in evidence.columns
        gcn['pval'] = pval_colname
    if logfc_colname:
        assert logfc_colname in evidence.columns
        gcn['logfc'] = logfc_colname


    missing_columns = []
    for role in tst_collection.keys():
        if role in gcn.keys():
            continue

        # we run each test in order. If we find a match, both loops get broken
        # if no match, the inner loop is complete and we go to the next test
        for tst in tst_collection[role]:
            for scol, col in zip(smpl_cols, evidence.columns):
                if tst in scol:
                    gcn[role] = col
                    print(f"Using column `{col}` as {role}")
                    break
            else:
                continue
            break
        else:
            if role == 'pval':
                print(f"Warning: failed to guess columns for 'pval'. Assuming pval == 0.")
                gcn['pval'] = 'pval'
                evidence['pval'] = 0.
            else:
                print(f"Error: failed to guess columns for '{role}'")
                missing_columns.append(f"'{role}'")

    if missing_columns:
        message = 'Error: missing ' 
        message += 'column ' if len(missing_columns) == 1 else 'columns '
        message += ', '.join(missing_columns)
        raise ValueError(message)

    evidence = evidence.loc[:, [gcn['gene'], gcn['pval'], gcn['logfc']]]
    evidence.columns = ['gene', 'pval', 'logfc']
    evidence = evidence.dropna()

    evidence['gene'] = evidence['gene'].astype(str)
    evidence['pval'] = evidence['pval'].astype(float)
    evidence['logfc'] = evidence['logfc'].astype(float)

    evidence['deg'] = evidence['logfc'].apply(np.sign).astype(int)

    evidence = evidence.query('pval <= @pval_threshold')
    evidence = evidence.query('logfc <= -@logfc_threshold or logfc >= @logfc_threshold')

    if genes_in_network:
        evidence = evidence.query('gene in @genes_in_network')

    return evidence.set_index('gene')['deg'].to_dict()


def get_tests_from_dicts(
        network: dict, evidence: dict,
        gt_act_src_uids: list=[]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    _, rels = get_ents_rels_dfs(network)
    trgs = rels.trguid.unique()
    d = dict(zip(trgs,[0]*len(trgs)))
    d.update(evidence)
    evid = get_evid_df(d)
    tests = get_tests(evid, rels, gt_act_src_uids)
    return evid, tests


def get_tests(evid, rels, gt_act_src_uids=[]):
    evid = evid.reset_index().set_index('uid')

    tmp = rels.merge(evid.loc[:, ['val']], how='left', left_on='trguid', right_index=True).fillna(0.)

    all_degp = len(evid.loc[evid.val == 1])    # number of positive deg
    all_degn = len(evid.loc[evid.val == -1])   # number of negative deg

    all_ydeg = all_degp + all_degn             # number of yes deg
    all_ndeg = len(evid.loc[evid.val == 0])    # number of not deg

    all_tot = all_ydeg + all_ndeg              # number of target genes

    # for enrichment
    trg_ydeg = tmp.loc[tmp.val != 0, ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('trg_ydeg')
    trg_ndeg = tmp.loc[tmp.val == 0, ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('trg_ndeg')

    # for mor binary and ternary test

    # mode of reg +1
    morp_degp = tmp.loc[(tmp.type ==  1) & (tmp.val ==  1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morp_degp')
    morp_degn = tmp.loc[(tmp.type ==  1) & (tmp.val == -1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morp_degn')
    # mode of reg -1
    morn_degp = tmp.loc[(tmp.type == -1) & (tmp.val ==  1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morn_degp')
    morn_degn = tmp.loc[(tmp.type == -1) & (tmp.val == -1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morn_degn')

    def enrichment_test(row):

        contingency_table =  [[row.trg_ydeg, all_ydeg - row.trg_ydeg],
                              [row.trg_ndeg, all_ndeg - row.trg_ndeg]]

        return fisher_exact(
           contingency_table,
            alternative = "greater")[1]

    def mor_binary_test(row):

        contingency_table = [[row.morp_degp, row.morn_degp],
                             [row.morp_degn, row.morn_degn]]

        return fisher_exact(
           contingency_table,
            alternative = "greater")[1]

    df = pd.DataFrame([trg_ydeg, trg_ndeg, morp_degp, morp_degn, morn_degp, morn_degn]).T.fillna(0).astype(int)
    df.index.rename('srcuid', True)
    df['enrichment'] = df.apply(enrichment_test, axis=1)
    df['mor_binary'] = df.apply(mor_binary_test, axis=1)

    rho = df.enrichment * df.mor_binary
    df['combined'] = rho.apply(lambda r: r * (1 - np.log(r)) if r > 0 else r)

    trg_tot = df.trg_ndeg + df.trg_ydeg
    tst_score = df.morp_degp + df.morn_degn - df.morp_degn - df.morn_degp
    df['trg_tot'] = trg_tot
    df['score'] = tst_score / df.trg_ydeg
    df['gt_act'] = df.index.isin(gt_act_src_uids)

    return df
