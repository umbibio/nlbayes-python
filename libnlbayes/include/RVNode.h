#ifndef NLB_RVNODE
#define NLB_RVNODE


#include <string>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rstat.h>

extern gsl_rng *rng;


namespace nlb
{

    class RVNode
    {
        private:

        protected:

        public:

            unsigned int var_size, n_outcomes, n_stats;
            bool is_discrete;
            std::string name, uid, id;

            // // allow for several histograms in a single node.
            // // useful for multivariate distributions
            // gsl_histogram ** histogram;

            // online statistics
            gsl_rstat_workspace ** stats;

            RVNode ();
            virtual ~RVNode ();
            RVNode (std::string, std::string, bool, unsigned int, unsigned int);

            void print_id ();
            void burn_stats();

            virtual double get_own_likelihood ();
            double get_own_loglikelihood ();
            virtual double get_children_loglikelihood ();
            double get_blanket_loglikelihood ();

            virtual void sample (gsl_rng *, bool=false);

            double mean(unsigned int = 0);
            double variance(unsigned int = 0);
            double chain_length();
    };
}

#endif