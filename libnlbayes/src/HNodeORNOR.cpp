#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "HNodeORNOR.h"


#include "YDataNode.h"
#include "XNode.h"
#include "SNode.h"


namespace nlb
{
    HNodeORNOR::~HNodeORNOR ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  HNodeORNOR  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    HNodeORNOR::HNodeORNOR (YDataNode * Y, double zy_value, double zn_value)
    : HNode(Y) {
        this->zy_value = zy_value;
        this->zn_value = zn_value;
    }

    HNodeORNOR::HNodeORNOR ( std::string uid, const double * prob, gsl_rng * rng , double zy_value, double zn_value)
    : HNode (uid, prob, rng) {
        this->zy_value = zy_value;
        this->zn_value = zn_value;
    }

    HNodeORNOR::HNodeORNOR ( std::string uid, const double * prob, unsigned int value , double zy_value, double zn_value)
    : HNode (uid, prob, value) {
        this->zy_value = zy_value;
        this->zn_value = zn_value;
    }

    void HNodeORNOR::append_parent (TNode * t, XNode * x, SNode * s) {
        txs_tuple parent(&t->value, &x->value, &s->value);
        this->parents.push_back(parent);
        t->append_h_child(this);
        x->append_h_child(this);
        s->append_h_child(this);
    }

    double HNodeORNOR::get_model_likelihood ()
    {
        double pr, pr0, pr1, pr2, zcompl_pn, likelihood;

        double zvalue;
        double * tvalue;
        unsigned int * xvalue;
        unsigned int * svalue;

        if (this->data->value != 1)
            zvalue = this->zy_value;
        else
            zvalue = this->zn_value;


        switch (this->value)
        {
        case 0:
            pr0 = 1.;
            zcompl_pn = 1.;
            for (auto txs_nodes: this->parents) {
                std::tie(tvalue, xvalue, svalue) = txs_nodes;
                if (*svalue != 1) {
                    zcompl_pn *= (1. - zvalue);
                    if (*xvalue == 1 && *svalue == 0)
                        pr0 *= (1. - *tvalue) * zvalue;
                }
            }
            pr0 = (1. - pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[0];
            likelihood = pr0;
            break;
        
        case 2:
            pr0 = 1.;
            pr2 = 1.;
            zcompl_pn = 1.;
            for (auto txs_nodes: this->parents) {
                std::tie(tvalue, xvalue, svalue) = txs_nodes;
                if (*svalue != 1) {
                    zcompl_pn *= (1. - zvalue);
                    if (*xvalue == 1) {
                        pr = (1. - *tvalue) * zvalue;
                        if (*svalue == 2) pr2 *= pr;
                        else pr0 *= pr;
                    }
                }
            }
            pr2 = (pr0 - pr2*pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[2];
            likelihood = pr2;
            break;
        
        case 1:
            pr1 = 1.;
            zcompl_pn = 1.;
            for (auto txs_nodes: this->parents) {
                std::tie(tvalue, xvalue, svalue) = txs_nodes;
                if (*svalue != 1){
                    zcompl_pn *= (1. - zvalue);
                    if (*xvalue == 1)
                        pr1 *= (1. - *tvalue) * zvalue;
                }
            }
            pr1 = pr1 * (1. - zcompl_pn) + zcompl_pn * this->prob[1];
            likelihood = pr1;
            break;
        
        default:
            throw std::out_of_range("Current node value is invalid");
            break;
        }

        return likelihood;
    }

}
