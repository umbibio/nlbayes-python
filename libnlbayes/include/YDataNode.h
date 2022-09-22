#ifndef NLB_YNODEDATA
#define NLB_YNODEDATA


#include <string>
#include <vector>


#include "Multinomial.h"
#include "YNoiseNode.h"
#include "HNode.h"


namespace nlb
{

    class YDataNode: public Multinomial
    {
        private:

        protected:

        public:
            // Future: want to use these values for influencing own likelihood
            double pvalue, abslogfc;
            double *noise [3];

            HNode * hidden_truth;
            // YNoiseNode * noise;

            ~YDataNode () override;
            YDataNode (std::string, const double *, unsigned int);
            YDataNode (HNode *);

            double get_own_likelihood () override;
    };
}

#endif
