#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "Multinomial.h"

namespace nlb
{
    void Multinomial::set_init_attr(const double * prob) 
    {
        this->prob = prob;
        this->outcome_likelihood = new double[this->n_outcomes];
        this->cumulative_outcome_likelihood = new double[this->n_outcomes];
    }

    Multinomial::Multinomial () {}
    Multinomial::~Multinomial ()
    {
        delete []this->outcome_likelihood;
        delete []this->cumulative_outcome_likelihood;
        // std::cout << "Object at " << this << " destroyed, instance of  Multinomial\t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    Multinomial::Multinomial (
        std::string name, std::string uid, const unsigned int n_outcomes, const double * prob,
        unsigned int value)
    : RVNode (name, uid, true, 1, n_outcomes)
    {
        this->set_init_attr(prob);

        // for (unsigned int i = 0; i < n_outcomes; i++)
        //     printf("%6.2f", this->prob[i]);
        // std::cout << std::endl;

        if (value < n_outcomes)
            this->value = value;
        else
            throw std::invalid_argument(this->id + ": Provided value is not valid (" + std::to_string(value) + ")");
    }

    Multinomial::Multinomial (
        std::string name, std::string uid, const unsigned int n_outcomes, const double * prob,
        gsl_rng * rng)
    : RVNode (name, uid, true, 1, n_outcomes)
    {
        this->set_init_attr(prob);
        this->sample(rng);
    }

    double Multinomial::get_own_likelihood ()
    {
        return this->prob[this->value];
    }

    void Multinomial::compute_outcome_likelihood ()
    {
        unsigned int current_value = this->value;
        double llik, lik;
        double sum_lik = 0.;

        for (unsigned int i = 0; i < this->n_outcomes; i++)
        {
            // iterate over possible values to compute 
            // make sure we preserve current value by the end of this loop
            this->value = i;
            llik = this->get_blanket_loglikelihood();
            lik = exp(llik);

            this->outcome_likelihood[i] = lik;

            sum_lik += lik;
            this->cumulative_outcome_likelihood[i] = sum_lik;
        }
        // restore the proper value
        this->value = current_value;

        if (!(sum_lik > 0.)) {
            sum_lik = 0.;
            for (unsigned int i = 0; i < this->n_outcomes; i++){
                this->outcome_likelihood[i] = this->prob[i];
                sum_lik += this->prob[i];
                this->cumulative_outcome_likelihood[i] = sum_lik;
            }
        } 
    }

    void Multinomial::sample (gsl_rng * rng, bool update_stats)
    {
        this->compute_outcome_likelihood();

        double dice = gsl_ran_flat(rng, 0., this->cumulative_outcome_likelihood[this->n_outcomes-1]);

        for (unsigned int i = 0; i < this->n_outcomes; i++) {
            if (dice < this->cumulative_outcome_likelihood[i]) {
                this->value = i;
                break;
            }
        }

        if (update_stats) {
            // gsl_histogram_increment (this->histogram[0], (double) this->value);
            for (unsigned int i = 0; i < this->n_outcomes; i++)
                gsl_rstat_add (this->value == i ? 1. : 0., this->stats[i]);
        }

        return;
    }

}
