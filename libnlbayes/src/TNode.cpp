#include "TNode.h"

namespace nlb
{
    TNode::~TNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  TNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    TNode::TNode (std::string uid, double a, double b, gsl_rng * rng)
    : Beta ((std::string) "T", uid, a, b, rng) {}
    TNode::TNode (std::string uid, double a, double b, double value)
    : Beta ((std::string) "T", uid, a, b, value) {}

    double TNode::get_children_loglikelihood ()
    {
        double loglik = 0.;

        for (auto child: this->children)
            loglik += child->get_own_loglikelihood();

        return loglik;
    }
}
