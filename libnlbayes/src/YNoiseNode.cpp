#include "YNoiseNode.h"
#include "YDataNode.h"


namespace nlb
{
    YNoiseNode::~YNoiseNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  YNoiseNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    YNoiseNode:: YNoiseNode (unsigned int idx, const double * ALPHA, gsl_rng * rng)
    : Dirichlet ((std::string) "Noise", std::to_string(idx), 3)
    {
        this->prior_alpha = ALPHA;
        this->set_init_attr();
        this->sample_from_prior(rng);
    }

    double YNoiseNode::get_children_loglikelihood ()
    {
        double loglik = 0.;

        for (auto child: this->children)
            loglik += child->get_own_loglikelihood();

        return loglik;
    }
}
