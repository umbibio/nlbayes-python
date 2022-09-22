#include "HParentNode.h"

namespace nlb
{
    HParentNode::HParentNode () {}
    HParentNode::~HParentNode () {
        // std::cout << "Object at " << this << " destroyed, instance of  HParentNode\t" << "\t" << typeid(this).name() << std::endl;
    }

    void HParentNode::append_h_child (HNode * h) {
        this->children.insert(h);
    }
}
