#ifndef NLB_SNODE
#define NLB_SNODE


#include "Multinomial.h"
#include "HParentNode.h"
#include "XNode.h"
#include "TNode.h"
#include "HNodeORNOR.h"


namespace nlb
{
    class SNode: public Multinomial, public HParentNode
    {
        private:

        protected:

        public:
            ~SNode () override;
            SNode (std::string, const double *, gsl_rng *);
            SNode (std::string, const double *, unsigned int);
            double get_children_loglikelihood () override;
    };
}

#endif
