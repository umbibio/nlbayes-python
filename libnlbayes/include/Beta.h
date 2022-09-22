#ifndef NLB_BETA
#define NLB_BETA


#include <string>


#include "RVNode.h"


namespace nlb
{

    class Beta: public RVNode
    {
        private:
            void set_init_attr(double, double);

        protected:
        public:
            double value, prior_alpha, prior_beta, prior_mean;

            ~Beta ();
            // Constructor with provided value
            Beta (
                std::string, std::string, double alpha, double beta, double value);
            
            // Constructor that samples initial value
            Beta (
                std::string, std::string, double alpha, double beta, gsl_rng * rng);

            double get_own_likelihood () override;

            double sample_from_prior (gsl_rng *);
            void metropolis_hastings_with_prior_proposal (gsl_rng *);

            void sample (gsl_rng *, bool=false) override;
    };
}

#endif
