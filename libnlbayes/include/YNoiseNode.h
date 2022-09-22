#ifndef NLB_YNOISENODE
#define NLB_YNOISENODE


#include "Dirichlet.h"


namespace nlb
{
    // forward declaration
    class YDataNode;

    class YNoiseNode: public Dirichlet
    {
        private:

        protected:

        public:
            std::vector<YDataNode *> children;

            ~YNoiseNode () override;
            YNoiseNode (unsigned int, const double *, gsl_rng *);
            double get_children_loglikelihood () override;
    };
}

#endif
