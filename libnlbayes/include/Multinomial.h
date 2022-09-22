#ifndef NLB_MULTINOMIAL
#define NLB_MULTINOMIAL


#include <string>
#include <vector>


#include "RVNode.h"


namespace nlb
{
    class Multinomial: public RVNode
    {
        private:
            void set_init_attr(const double * );

        protected:

        public:

            unsigned int value;

            const double * prob;
            double * outcome_likelihood;
            double * cumulative_outcome_likelihood;

            Multinomial ();
            ~Multinomial () override;
            Multinomial (
                std::string, std::string, const unsigned int, const double *, unsigned int value);
            Multinomial (
                std::string, std::string, const unsigned int, const double *, gsl_rng *);

            double get_own_likelihood () override;

            void compute_outcome_likelihood ();
            void sample (gsl_rng *, bool=false) override;
    };
}

#endif
