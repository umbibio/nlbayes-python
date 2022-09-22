#ifndef NLB_TNODE
#define NLB_TNODE


#include "Beta.h"
#include "HParentNode.h"


namespace nlb
{
    class TNode: public Beta, public HParentNode
    {
        private:

        protected:

        public:

            ~TNode () override;
            TNode (std::string, double, double, gsl_rng *);
            TNode (std::string, double, double, double);
            double get_children_loglikelihood () override;
    };
}

#endif
