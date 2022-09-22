#include <iostream>
#include <string>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "Beta.h"


namespace nlb
{
    Beta::~Beta()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  Beta   \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    void Beta::set_init_attr(double a, double b)
    {
        this->prior_alpha = a;
        this->prior_beta = b;
        this->prior_mean = a / (a + b);
    }


    Beta::Beta (
        std::string name, std::string uid, double a, double b, double value)
    : RVNode (name, uid, false, 1, -1) // infinite outcomes
    {
        this->set_init_attr(a, b);

        if (!(value < 0. || value > 1.))
            this->value = value;
        else
            throw std::invalid_argument(this->id + ": Provided value is not valid (" + std::to_string(value) + ")");

    }


    Beta::Beta (
        std::string name, std::string uid, double a, double b, gsl_rng * rng)
    : RVNode (name, uid, false, 1, -1) // infinite outcomes
    {
        this->set_init_attr(a, b);
        this->value = this->sample_from_prior(rng);
    }


    double Beta::get_own_likelihood ()
    {
        return gsl_ran_beta_pdf (this->value, this->prior_alpha, this->prior_beta);
    }


    double Beta::sample_from_prior (gsl_rng * rng)
    {
        return gsl_ran_beta(rng, this->prior_alpha, this->prior_beta);
    }


    void Beta::metropolis_hastings_with_prior_proposal (gsl_rng * rng)
    {
        double sample_dx, q10, q01, log_q, logratio;
        bool accept;

        // remember value in case we reject proposal
        double prev = this->value;
        double prev_loglik = this->get_blanket_loglikelihood();

        sample_dx = this->sample_from_prior(rng) - this->prior_mean;
        this->value += sample_dx;

        double prop_loglik = this->get_blanket_loglikelihood();

        if(!(prop_loglik > -INFINITY)) {
            // allways reject out of range proposals
            this->value = prev;
            return;
        }

        if (prev_loglik > -INFINITY)
        {
            q10 = gsl_ran_beta_pdf(this->prior_mean + sample_dx, this->prior_alpha, this->prior_beta);
            q01 = gsl_ran_beta_pdf(this->prior_mean - sample_dx, this->prior_alpha, this->prior_beta);
            log_q = std::log(q01/q10);

            logratio = prop_loglik - prev_loglik + log_q;
            accept = logratio >= 0. || logratio > - gsl_ran_exponential(rng, 1.0);
            if (!accept)
                this->value = prev;
        }
    }


    void Beta::sample (gsl_rng * rng, bool update_stats)
    {
        this->metropolis_hastings_with_prior_proposal(rng);

        if (update_stats){
            // gsl_histogram_increment (this->histogram[0], this->value);
            gsl_rstat_add (this->value, this->stats[0]);
        }

        return;
    }


}
