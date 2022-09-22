#ifndef NLB_HNODE
#define NLB_HNODE


#include <string>
#include <iostream>


#include "Multinomial.h"


namespace nlb
{
    // forward declaration
    class YDataNode;

    class HNode: public Multinomial
    {
        private:

        protected:

        public:
            YDataNode * data;

            // this switch defines how the likelihood is computed
            // if set to false, this is treated as a normal RV and can be sampled as such
            // this is useful when simulating differential expression data
            bool is_latent = true;

            ~HNode() override;
            HNode(YDataNode *);
            HNode (std::string, const double *, gsl_rng *);
            HNode (std::string, const double *, unsigned int);

            virtual double get_model_likelihood ();
            double get_own_likelihood () override;
            double get_children_loglikelihood () override;
    };
}

#endif
