#ifndef NLB_MODELORNOR
#define NLB_MODELORNOR


#include <cmath>
#include <time.h>


#include <gsl/gsl_rstat.h>


#include "ModelBase.h"
#include "GraphORNOR.h"


namespace nlb
{

    class ModelORNOR: public ModelBase
    {
        private:
        protected:
        public:
            double t_alpha, t_beta;
            double zy, zn;
            ModelORNOR();
            ~ModelORNOR() override;
            ModelORNOR(const network_t, const evidence_dict_t, const prior_active_tf_set_t = prior_active_tf_set_t(),
                       const double [3 * 3] = nlb::SPRIOR, double t_alpha = 2., double t_beta = 2., double z_value = 1., double z0_value = 1.,
                       unsigned int = 3);
    };
}

#endif