
from typing import Tuple
import numpy as np
import pandas as pd
from itertools import chain
from scipy.stats import fisher_exact



def gen_network(NX: int, NY: int, AvgNTF: float=20) -> dict:
    r = 1.8

    p = NX/NY*r/AvgNTF
    n_edges = np.random.negative_binomial( r, p, size=NX)
    n_edges = np.minimum(NY, n_edges)
    n_edges = np.maximum(3, n_edges)
    n_edges = np.sort(n_edges)[::-1]
    network = {}

    for i in range(NX):
        src = f"TF{i+1:03d}"

        js = np.sort(np.random.choice(NY, size=n_edges[i], replace=False))
        trgs = [f"Gene{j+1:05d}" for j in js]

        mors = np.random.choice([-1, 0, 1], p=[0.35, 0., 0.65], size=len(trgs))

        network[src] = dict(zip(trgs, mors))

    return network


def gen_evidence(network: dict, n_active_tfs: int=1, active_tfs: list=[],
                 background_p: float=0.02, mor_noise_p: float=0.05,
                 tf_target_fraction: float=0.1, mor_weights: tuple=(1, 1),
                 ) -> dict:

    tfs = list(network.keys())
    genes = list(set(chain(*[d.keys() for d in network.values()])))
    tfs.sort()
    genes.sort()

    ### Select random TF as active
    n = n_active_tfs - len(active_tfs)
    candidates = [src for src in tfs if src not in active_tfs]
    if n > 0:
        active_tfs.extend(np.random.choice(candidates, n, False))

    ### Generate a random deg background
    p = 1 - background_p
    vals = np.random.choice([-1, 0, 1], len(genes), p=[(1-p)/2, p, (1-p)/2])
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
        de_trgs = np.random.choice(trgs,
                                   size=int(len(trgs)*tf_target_fraction),
                                   replace=False)
        for trg in de_trgs:
            mor = network[act_src_uid][trg]
            evidence[trg] += np.random.choice([-dw_step, 0, up_step],
                                              p=probs[mor+1])
    evidence = {k:np.sign(v) for k, v in evidence.items() if v != 0}
    
    ngtac = len(np.intersect1d(
        [s for s, d in network.items()
         if len(np.intersect1d(list(d.keys()), list(evidence.keys()))) > 0],
        active_tfs))
    ndegs = len(evidence)
    nrels = sum([len(d) for d in network.values()])
    print(f"{ngtac=}, {ndegs=}, {nrels=}")

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


def get_network_dict(rels: pd.DataFrame, src_colname: str='srcuid', trg_colname: str='trguid', mor_colname: str='type') -> dict:
    dff = rels.set_index([src_colname, trg_colname])
    dff = dff.groupby(level=0).apply(lambda df: df.xs(df.name)[mor_colname].to_dict())
    return dff.to_dict()


def get_evidence_dict(evid: pd.DataFrame, uid_colname: str='uid', deg_colname: str='val') -> dict:
    dff = evid.loc[evid[deg_colname]!=0].set_index(uid_colname)
    return dff[deg_colname].to_dict()


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

