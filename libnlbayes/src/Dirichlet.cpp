#include <iostream>
#include <string>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "Dirichlet.h"

namespace nlb
{
    void Dirichlet::set_init_attr()
    {
        double sum_alpha = 0.;

        for (unsigned int i = 0; i < this->var_size; i++) 
            sum_alpha += this->prior_alpha[i];

        this->value = new double[this->var_size];
        this->prior_mean = new double[this->var_size];
        this->sample_tmp_var = new double[this->var_size];
        this->sample_prev = new double[this->var_size];

        for (unsigned int i = 0; i < this->var_size; i++)
            this->prior_mean[i] = this->prior_alpha[i] / sum_alpha;
    }

    Dirichlet::Dirichlet (
        std::string name, std::string uid, const unsigned int SIZE,
        const double *prior_alpha, gsl_rng * rng)
    : RVNode (name, uid, false, SIZE, -1) // infinite outcomes
    {
        this->prior_alpha = prior_alpha;
        this->set_init_attr();
        this->sample_from_prior(rng);
    }

    Dirichlet::Dirichlet ( std::string name, std::string uid, const unsigned int SIZE )
    : RVNode (name, uid, false, SIZE, -1) // infinite outcomes
    {
    }

    Dirichlet::~Dirichlet ()
    {
        delete []this->value;
        delete []this->prior_mean;
        delete []this->sample_tmp_var;
        delete []this->sample_prev;
        // std::cout << "Object at " << this << " destroyed, instance of  Dirichlet\t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    double Dirichlet::get_own_likelihood ()
    {
        return gsl_ran_dirichlet_pdf(this->var_size, this->prior_alpha, this->value);
    }

    void Dirichlet::sample_from_prior(gsl_rng * rng)
    {
        gsl_ran_dirichlet(rng, this->var_size, this->prior_alpha, this->value);
    }

    void Dirichlet::metropolis_hastings_with_prior_proposal (gsl_rng * rng)
    {
        unsigned int i;
        double q10, q01, log_q, logratio;
        bool accept;

        // remember value in case we reject proposal
        for (i = 0; i< this->var_size; i++)
            this->sample_prev[i] = this->value[i];
        double prev_loglik = this->get_blanket_loglikelihood();

        // perform random walk
        gsl_ran_dirichlet(rng, this->var_size, this->prior_alpha, this->sample_tmp_var);
        for (i = 0; i< this->var_size; i++)
            this->value[i] += this->sample_tmp_var[i] - this->prior_mean[i];

        double prop_loglik = this->get_blanket_loglikelihood();

        if(!(prop_loglik > -INFINITY)) {
            // allways reject out of range proposals
            for (i = 0; i< this->var_size; i++)
                this->value[i] = this->sample_prev[i];
            return;
        }

        if (prev_loglik > -INFINITY)
        {
            // non-symmetric proposal. compute forward leap probability density
            q10 = gsl_ran_dirichlet_pdf(this->var_size, this->prior_alpha, this->sample_tmp_var);

            for (i = 0; i< this->var_size; i++)
                this->sample_tmp_var[i] = 2*this->prior_mean[i] - this->sample_tmp_var[i];

            // non-symmetric proposal. compute reverse leap probability density
            q01 = gsl_ran_dirichlet_pdf(this->var_size, this->prior_alpha, this->sample_tmp_var);
            log_q = std::log(q01/q10);

            logratio = prop_loglik - prev_loglik + log_q;
            accept = logratio >= 0. || logratio > - gsl_ran_exponential(rng, 1.0);
            if (!accept)
                for (i = 0; i< this->var_size; i++)
                    this->value[i] = this->sample_prev[i];
        }
    }


    void Dirichlet::sample (gsl_rng * rng, bool update_stats)
    {
        this->metropolis_hastings_with_prior_proposal(rng);

        if (update_stats)
            for (unsigned int i = 0; i < this->var_size; i++) {
                // gsl_histogram_increment (this->histogram[i], this->value[i]);
                gsl_rstat_add (this->value[i], this->stats[i]);
            }
        return;
    }

}
