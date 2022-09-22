#include "SNode.h"

namespace nlb
{
    SNode::~SNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  SNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    SNode::SNode (std::string uid, const double * prob, gsl_rng * rng)
    : Multinomial ((std::string) "S", uid, 3, prob, rng) {}
    SNode::SNode (std::string uid, const double * prob, unsigned int value)
    : Multinomial ((std::string) "S", uid, 3, prob, value) {}

    double SNode::get_children_loglikelihood ()
    {
        double loglik = 0.;
        for (auto child: this->children)
            loglik += child->get_own_loglikelihood();
        return loglik;
    }
}
