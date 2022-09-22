#ifndef NLB_DIRICHLET
#define NLB_DIRICHLET


#include <string>


#include "RVNode.h"


namespace nlb
{

    class Dirichlet: public RVNode
    {
        private:
            double *sample_tmp_var, *sample_prev, *prior_mean;

        protected:
            void set_init_attr();

        public:
            double *value;
            const double *prior_alpha;

            Dirichlet (
                std::string, std::string, const unsigned int, const double *, gsl_rng *);
            Dirichlet (
                std::string, std::string, const unsigned int);
            ~Dirichlet () override;

            double get_own_likelihood () override;

            void sample_from_prior(gsl_rng *);
            void metropolis_hastings_with_prior_proposal (gsl_rng *);

            void sample (gsl_rng *, bool=false) override;
    };
}

#endif
