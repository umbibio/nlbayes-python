#include "XNode.h"


namespace nlb
{
    XNode::~XNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  XNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    XNode::XNode (std::string uid, const double * prob, gsl_rng * rng)
    : Multinomial ((std::string) "X", uid, 2, prob, rng) {}
    XNode::XNode (std::string uid, const double * prob, unsigned int value)
    : Multinomial ((std::string) "X", uid, 2, prob, value) {}

    double XNode::get_children_loglikelihood ()
    {
        double loglik = 0.;
        for (auto child: this->children)
            loglik += child->get_own_loglikelihood();
        return loglik;
    }
}
