#include "ModelBase.h"


namespace nlb
{
    ModelBase::ModelBase(unsigned int n_graphs)
    {
        this->n_graphs = n_graphs;
    }
    ModelBase::~ModelBase() {
    }

    std::vector<unsigned int> ModelBase::get_seeds()
    {

        std::vector<unsigned int> seeds;

        for (unsigned int i = 0; i < this->n_graphs; i++)
            seeds.push_back(this->graphs[i]->seed);
        
        return seeds;
    }

    gelman_rubin_vector_t ModelBase::get_gelman_rubin() 
    {
        gelman_rubin_vector_t gelman_rubin_vector;
        double B, W, Vhat, gelman_rubin_stat, max_gelman_rubin_stat;

        // The number of drawn samples so far
        // Make sure all nodes within all graphs have been sampled the same number of times
        double N = this->graphs[0]->random_nodes[0]->chain_length();
        double M = this->n_graphs;

        unsigned int number_nodes = this->graphs[0]->random_nodes.size();
        gelman_rubin_vector.reserve(number_nodes);

        std::string node_id;
        unsigned int node_n_stats;

        if (N > 0){
            gsl_rstat_workspace * rstat4B = gsl_rstat_alloc();
            gsl_rstat_workspace * rstat4W = gsl_rstat_alloc();
            for (unsigned int i = 0; i < number_nodes; i++) { // node
                max_gelman_rubin_stat = 0.;
                node_id = this->graphs[0]->random_nodes[i]->id;
                node_n_stats = this->graphs[0]->random_nodes[i]->n_stats;

                for (unsigned int j = 0;  j < node_n_stats; j++) { // stat
                    for (unsigned int k = 0;  k < M; k++) {// graph
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->mean(j), rstat4B);
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->variance(j), rstat4W);
                    }
                    B = N * gsl_rstat_variance(rstat4B);
                    W = gsl_rstat_mean(rstat4W);
                    Vhat = W * (N - 1.)/N + B * (M + 1.)/(M * N);

                    if (W > 0. && Vhat > 0.)
                        gelman_rubin_stat =  std::sqrt((N + 3.)/(N + 1.) * Vhat / W);
                    else if (Vhat > 0.)
                        gelman_rubin_stat = INFINITY;
                    else
                        gelman_rubin_stat = 1.;

                    max_gelman_rubin_stat = std::max(gelman_rubin_stat, max_gelman_rubin_stat);

                    gsl_rstat_reset(rstat4B);
                    gsl_rstat_reset(rstat4W);
                }
                gelman_rubin_vector.push_back(std::make_pair(node_id, max_gelman_rubin_stat));
            }
            gsl_rstat_free(rstat4B);
            gsl_rstat_free(rstat4W);
        } else {
            for (unsigned int i = 0; i < number_nodes; i++) { // node
                node_id = this->graphs[0]->random_nodes[i]->id;
                gelman_rubin_vector.push_back(std::make_pair(node_id, INFINITY));
            }
        }

        return gelman_rubin_vector;
    }

    double ModelBase::get_max_gelman_rubin() 
    {
        double max_gelman_rubin = 0.;
        for(auto node: this->get_gelman_rubin())
            max_gelman_rubin = std::max(max_gelman_rubin, node.second);
        return max_gelman_rubin;
    }

    void ModelBase::sample(unsigned int N, unsigned int dN)
    {
        unsigned int n;
        double gr = INFINITY;

        for (
            n=0;
            n < N && gr > 1.10;
            dN=std::min(dN, N-n), n+=dN, gr=this->get_max_gelman_rubin()
        )
            this->sample_n(dN);

    }

    void ModelBase::sample_n(unsigned int N)
    {
        #pragma omp parallel for
        for (unsigned int i = 0; i < this->n_graphs; i++)
            this->graphs[i]->sample(N);

        this->total_sampled += N;
    }

    void ModelBase::burn_stats()
    {
        for (unsigned int i = 0; i < this->n_graphs; i++)
            this->graphs[i]->burn_stats();
    }

    void ModelBase::print_stats(std::string node_name)
    {
        gsl_rstat_workspace * posterior = gsl_rstat_alloc();
        for (unsigned int i = 0; i < this->graphs[0]->random_nodes.size(); i++) { // node
            if (node_name == this->graphs[0]->random_nodes[i]->name) {
                printf("\n%s", this->graphs[0]->random_nodes[i]->id.c_str());
                for (unsigned int j = 0;  j < this->graphs[0]->random_nodes[i]->n_stats; j++) { // stat
                    for (unsigned int k = 1;  k < this->n_graphs; k++) // graph
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->mean(j), posterior);
                    printf("\t%6.4f(%6.4f)", gsl_rstat_mean(posterior), gsl_rstat_sd(posterior));
                    gsl_rstat_reset(posterior);
                }
            }
        }
        printf("\n\n");
        gsl_rstat_free(posterior);
    }

    posterior_vector_t ModelBase::get_posterior(std::string node_name)
    {
        posterior_vector_t posterior_vector;
        std::vector<double> var_mean, var_sd;
        std::string var_uid;
        double varstat_mean, varstat_sd;
        varname_varuid_mean_sd_tuple_t varname_varuid_mean_sd_tuple;

        gsl_rstat_workspace * posterior = gsl_rstat_alloc();
        for (unsigned int i = 0; i < this->graphs[0]->random_nodes.size(); i++) { // node
            if (node_name == "" || node_name == this->graphs[0]->random_nodes[i]->name) {
                var_mean.clear();
                var_sd.clear();
                var_uid = this->graphs[0]->random_nodes[i]->uid;
                for (unsigned int j = 0;  j < this->graphs[0]->random_nodes[i]->n_stats; j++) { // stat
                    for (unsigned int k = 1;  k < this->n_graphs; k++) // graph
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->mean(j), posterior);
                    varstat_mean = gsl_rstat_mean(posterior);
                    varstat_sd = gsl_rstat_sd(posterior);
                    var_mean.push_back(varstat_mean);
                    var_sd.push_back(varstat_sd);
                    gsl_rstat_reset(posterior);
                }
                varname_varuid_mean_sd_tuple = varname_varuid_mean_sd_tuple_t(node_name, var_uid, var_mean, var_sd);
                posterior_vector.push_back(varname_varuid_mean_sd_tuple);
            }
        }
        gsl_rstat_free(posterior);

        return posterior_vector;
    }

    posterior_stat_vector_t ModelBase::get_posterior_means(std::string node_name)
    {
        varid_stat_pair_t var_mean;
        posterior_stat_vector_t var_mean_vector;

        std::string varstat_id;
        double varstat_mean;
        gsl_rstat_workspace * posterior = gsl_rstat_alloc();
        for (unsigned int i = 0; i < this->graphs[0]->random_nodes.size(); i++) { // node
            if (node_name == "" || node_name == this->graphs[0]->random_nodes[i]->name) {
                for (unsigned int j = 0;  j < this->graphs[0]->random_nodes[i]->n_stats; j++) { // stat
                    varstat_id = this->graphs[0]->random_nodes[i]->id + "_" + std::to_string(j);
                    for (unsigned int k = 1;  k < this->n_graphs; k++) // graph
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->mean(j), posterior);
                    varstat_mean = gsl_rstat_mean(posterior);
                    var_mean_vector.push_back(varid_stat_pair_t(varstat_id, varstat_mean));
                    gsl_rstat_reset(posterior);
                }
            }
        }
        gsl_rstat_free(posterior);

        return var_mean_vector;
    }

    posterior_stat_vector_t ModelBase::get_posterior_sdevs(std::string node_name)
    {
        varid_stat_pair_t var_sd;
        posterior_stat_vector_t var_sd_vector;

        std::string varstat_id;
        double varstat_sd;
        gsl_rstat_workspace * posterior = gsl_rstat_alloc();
        for (unsigned int i = 0; i < this->graphs[0]->random_nodes.size(); i++) { // node
            if (node_name == this->graphs[0]->random_nodes[i]->name) {
                for (unsigned int j = 0;  j < this->graphs[0]->random_nodes[i]->n_stats; j++) { // stat
                    varstat_id = this->graphs[0]->random_nodes[i]->id + "_" + std::to_string(j);
                    for (unsigned int k = 1;  k < this->n_graphs; k++) // graph
                        gsl_rstat_add(this->graphs[k]->random_nodes[i]->mean(j), posterior);
                    varstat_sd = gsl_rstat_sd(posterior);
                    var_sd_vector.push_back(varid_stat_pair_t(varstat_id, varstat_sd));
                    gsl_rstat_reset(posterior);
                }
            }
        }
        gsl_rstat_free(posterior);

        return var_sd_vector;

    }


}