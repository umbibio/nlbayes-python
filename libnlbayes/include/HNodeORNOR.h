#ifndef NLB_HNODE_ORNOR
#define NLB_HNODE_ORNOR


#include <string>
#include <vector>
#include <tuple>


#include "HNode.h"


namespace nlb
{
    // forward declarations
    class TNode;
    class XNode;
    class SNode;
    class YDataNode;

    // ?? consider to fill this tuple with pointers to values instead of whole nodes
    typedef std::tuple<double *, unsigned int *, unsigned int *> txs_tuple;

    class HNodeORNOR: public HNode
    {
        private:

        protected:

        public:

            double zy_value;
            double zn_value;

            std::vector< txs_tuple > parents;

            ~HNodeORNOR () override;
            HNodeORNOR(YDataNode *, double, double);
            HNodeORNOR (std::string, const double *, gsl_rng *, double, double);
            HNodeORNOR (std::string, const double *, unsigned int, double, double);

            void append_parent(TNode *, XNode *, SNode *);

            double get_model_likelihood () override;
    };
}

#endif
