import numpy as np
import nlbayes
from nlbayes.utils import gen_network, gen_evidence, get_tests_from_dicts


network = gen_network(125, 2500, 3.33)
gt_act = [f"TF{i:03d}" for i in np.random.choice(125, 3)]
gt_act.sort()
evidence = gen_evidence(network, 3, gt_act, 0.02, 0.05, 0.1)
tests = get_tests_from_dicts(network, evidence, gt_act)

inference_model = nlbayes.ModelORNOR(network, evidence, zy=0.99, s_leniency=0.1, n_graphs=5)
inference_model.sample_posterior(N=5000, gr_level=1.1, burnin=True)

var_name = 'X'
inference_model.get_posterior_mean_stat(var_name, 1)
df = inference_model.get_posterior(var_name).set_index('uid')
df.loc[gt_act]

tests.loc[tests.gt_act]

inference_model.burn_stats()
inference_model.total_sampled
