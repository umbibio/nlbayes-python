#ifndef NLB_GRAPHORNOR
#define NLB_GRAPHORNOR


#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>
#include <stdio.h>


#include <gsl/gsl_rng.h>


#include "RVNode.h"
#include "XNode.h"
#include "HNodeORNOR.h"
#include "YDataNode.h"
#include "SNode.h"
#include "TNode.h"
#include "NodeDictionary.h"
#include "GraphBase.h"
#include "YNoiseNode.h"


namespace nlb
{
    const double SPRIOR[3 * 3] = {0.40, 0.40, 0.20,
                                  0.30, 0.40, 0.30,
                                  0.20, 0.40, 0.40};

    class GraphORNOR: public GraphBase
    {
        private:
        protected:
        public:
            const double XPROB[2] = {0.99, 0.01};
            const double XPROB_ACTIVE[2] = {0.10, 0.90};
            double SPROB[3][3] = {{0.900, 0.090, 0.010},
                                  {0.050, 0.900, 0.050},
                                  {0.010, 0.090, 0.900}};
            double YPROB[3] = {0.05, 0.90, 0.05};
            double YNOISE[3][3] = {{0.945, 0.050, 0.005},
                                   {0.050, 0.900, 0.050},
                                   {0.005, 0.050, 0.945}};

            GraphORNOR ();
            GraphORNOR (unsigned int);
            virtual ~GraphORNOR ();

            void build_structure (network_t, evidence_dict_t, prior_active_tf_set_t,
                                  const double [3 * 3] = SPRIOR, double t_alpha = 2., double t_beta = 2., double z_value = 25., double z0_value = 25.);
    };
}

#endif