import os
import json
import argparse
import inspect

from hashlib import sha256
import numpy as np
import pandas as pd

from numpy.random import RandomState, MT19937, SeedSequence

import nlbayes
from nlbayes.utils import gen_network, gen_evidence, get_tests_from_dicts



def make_network(net_n_tfs:int, net_n_genes:int, net_avg_n_tfs:float, net_p_up:float, net_p_dw:float, net_seed:int):

    rng = RandomState(MT19937(SeedSequence(net_seed)))
    network = gen_network(NX=net_n_tfs, NY=net_n_genes, AvgNTF=net_avg_n_tfs, p_up=net_p_up, p_dw=net_p_dw, rng=rng)
    
    return network


def make_evidence(network:dict, evd_n_active_tfs:int, evd_active_tf_uids:str, evd_background_p:float, evd_mor_noise_p:float, evd_fraction_targets:float, evd_seed:int):

    active_tfs_str = evd_active_tf_uids.replace(',', ' ')
    active_tfs = active_tfs_str.split()
    candidates = [src for src in network.keys() if src not in active_tfs]

    rng = RandomState(MT19937(SeedSequence(evd_seed)))

    for _ in range(50):
        ### Select random TF as active
        active_tfs = active_tfs_str.split()
        n = evd_n_active_tfs - len(active_tfs)
        if n > 0:
            active_tfs.extend(rng.choice(candidates, n, False))

        evidence = gen_evidence(
            network,
            n_active_tfs=evd_n_active_tfs,
            active_tfs=active_tfs.copy(),
            background_p=evd_background_p,
            mor_noise_p=evd_mor_noise_p,
            tf_target_fraction=evd_fraction_targets,
            rng=rng,
        )
        if len(evidence) >= 5:
            break
    else:
        raise RuntimeError('Unable to find tf with non-zero differential expression')

    return evidence, active_tfs


def main():
    print(f"This is nlbayes version: {nlbayes.__version__}\n")

    parser = argparse.ArgumentParser(description='Specify parameters.')
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--outname', type=str, default='')

    parser.add_argument('--net_seed', type=int, default=0)
    parser.add_argument('--net_n_tfs', type=int, default=250)
    parser.add_argument('--net_n_genes', type=int, default=5000)
    parser.add_argument('--net_avg_n_tfs', type=float, default=6.66)
    parser.add_argument('--net_p_up', type=float, default=0.65)
    parser.add_argument('--net_p_dw', type=float, default=0.35)
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

    kvargs = vars(parser.parse_args())

    outdir_path = kvargs['outdir']
    out_filename = kvargs['outname']

    del(kvargs['outdir'], kvargs['outname'])

    max_int = np.iinfo(int).max

    if not kvargs['net_seed']:
        kvargs['net_seed'] = np.random.randint(1, max_int)
    if not kvargs['evd_seed']:
        kvargs['evd_seed'] = np.random.randint(1, max_int)


    experiment_hash = sha256(json.dumps(kvargs, sort_keys=True).encode()).hexdigest()
    
    if not out_filename:
        out_filename = experiment_hash

    netw_filepath = os.path.join(outdir_path, out_filename, 'network.json')
    evid_filepath = os.path.join(outdir_path, out_filename, 'evidence.tsv')
    test_filepath = os.path.join(outdir_path, out_filename, 'tf_tests.tsv')
    meta_filepath = os.path.join(outdir_path, out_filename, 'metadata.json')

    os.makedirs(os.path.join(outdir_path, out_filename), exist_ok=True)

    with open(meta_filepath, 'w') as file:
        json.dump({**kvargs, 'experiment_hash':experiment_hash}, file, indent=4)


    param_names = inspect.signature(make_network).parameters.keys()
    params = {k:v for k,v in kvargs.items() if k in param_names}
    network = make_network(**params)

    with open(netw_filepath, 'w') as file:
        json.dump(network, file)


    param_names = inspect.signature(make_evidence).parameters.keys()
    params = {k:v for k,v in kvargs.items() if k in param_names}
    evidence, active_tfs = make_evidence(network, **params)

    evidence, tests = get_tests_from_dicts(network, evidence, active_tfs)
    tests = tests.reset_index().sort_values(['gt_act', 'srcuid'], ascending=[False, True])
    
    evidence.to_csv(evid_filepath, index=False, sep='\t')
    tests.to_csv(test_filepath, index=False, sep='\t')


if __name__ == '__main__':
    main()
