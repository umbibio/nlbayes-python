#include <iostream>
#include <cmath>
#include <gsl/gsl_rng.h>

#include "RVNode.h"


// The random number generator
gsl_rng * rng;

namespace nlb
{
    RVNode::RVNode () {}

    RVNode::~RVNode()
    {
        for (unsigned int i = 0; i < this->n_stats; i++)
            gsl_rstat_free(this->stats[i]);
        delete []this->stats;
        // std::cout << "Object at " << this << " destroyed, instance of  RVNode\t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    RVNode::RVNode (std::string name, std::string uid, bool is_discrete, unsigned int var_size, unsigned int n_outcomes)
    {
        this->is_discrete = is_discrete;
        this->var_size = var_size;
        this->n_outcomes = n_outcomes;

        this->name = name;
        this->uid = uid;
        this->id = name;

        this->id += "_" + uid;

        // this->histogram = new gsl_histogram * [this->var_size];
        if(this->is_discrete) {
            // discrete distributions
            this->n_stats = this->var_size * this->n_outcomes;

            // for (unsigned int i = 0; i < this->var_size; i++) {
            //     this->histogram[i] = gsl_histogram_alloc(this->n_outcomes);
            //     gsl_histogram_set_ranges_uniform (this->histogram[i], 0., (double)this->n_outcomes);
            // }

        } else {
            // continuous distributions
            this->n_stats = this->var_size;

            // // only distributions with support in [0, 1] for now
            // for (unsigned int i = 0; i < this->var_size; i++) {
            //     this->histogram[i] = gsl_histogram_alloc(30);
            //     gsl_histogram_set_ranges_uniform (this->histogram[i], 0., 1.);
            // }
        }

        this->stats = new gsl_rstat_workspace * [this->n_stats];
        for (unsigned int i = 0; i < this->n_stats; i++)
            this->stats[i] = gsl_rstat_alloc();
    }

    void RVNode::burn_stats()
    {
        // for (unsigned int i = 0; i < this->var_size; i++)
        //     gsl_histogram_scale(this->histogram[i], 0.);
        
        for (unsigned int i = 0; i < this->n_stats; i++)
            gsl_rstat_reset(this->stats[i]);
    }

    double RVNode::get_own_likelihood ()
    {
        /* This is a virtual method. Derived class must implement its
        correct behaviour */
        return 1.;
    }

    double RVNode::get_own_loglikelihood ()
    {
        return std::log( this->get_own_likelihood() );
    }

    double RVNode::get_children_loglikelihood ()
    {
        /* This is a virtual method. Derived class must implement its
        correct behaviour */
        return 0.;
    }

    double RVNode::get_blanket_loglikelihood ()
    {
        // Likelihood by conditioning on this variables Markov Blanket
        double loglik = 0.;

        // Conditioned on parents
        loglik += this->get_own_loglikelihood();

        // Conditioned on children's parents
        loglik += this->get_children_loglikelihood();

        return loglik;
    }

    void RVNode::sample(gsl_rng * rng, bool update_stats)
    {
        /* This is a virtual method. Derived class must implement its
        correct behaviour */
        return;
    }

    double RVNode::mean(unsigned int i)
    {
        return gsl_rstat_mean(this->stats[i]);
    }

    double RVNode::variance(unsigned int i)
    {
        return gsl_rstat_variance(this->stats[i]);
    }

    double RVNode::chain_length()
    {
        return gsl_rstat_n(this->stats[0]);
    }
}
