#include <stdexcept>
#include "NodeDictionary.h"


namespace nlb
{
    NodeDictionary::NodeDictionary () {}

    void NodeDictionary::include_node(RVNode * node)
    {
        this->dictionary.insert(uid_node_pair_t(node->uid, node));
    }

    RVNode * NodeDictionary::find_node(std::string uid)
    {
        node_dictionary_t::iterator itr;
        itr = this->dictionary.find(uid);
        if (itr != this->dictionary.end())
            return itr->second;
        else
            throw std::out_of_range("Node not found. Make sure to include it before calling this method.");
    }
}
