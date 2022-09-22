import nlbayes

network = dict(
    tf1=dict(gene1= 1,           gene3= 1,            gene5=-1),
    tf2=dict(gene1= 1, gene2= 1,                      gene5=-1),
    tf3=dict(          gene2=-1, gene3= 1, gene4= -1,         ),
)
evidence = dict(gene1=0,gene3=1, gene5=-1)


inference_model = nlbayes.ModelORNOR(network, evidence, zy=0.99, s_leniency=0.1)
inference_model.sample_posterior(N=500, gr_level=1.1, burnin=True)

var_name = 'X'
inference_model.get_posterior_mean_stat(var_name, 1)
inference_model.get_posterior(var_name).set_index('uid')

inference_model.burn_stats()
inference_model.total_sampled
