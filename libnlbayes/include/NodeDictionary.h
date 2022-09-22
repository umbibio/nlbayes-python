#ifndef NLB_DICTIONARY
#define NLB_DICTIONARY


#include <string>
#include <vector>
#include <map>

#include "RVNode.h"


namespace nlb
{
    typedef std::pair<std::string, nlb::RVNode *> uid_node_pair_t;
    typedef std::map<std::string, nlb::RVNode *> node_dictionary_t;

    class NodeDictionary
    {
        private:
        protected:
        public:
            node_dictionary_t dictionary;

            NodeDictionary();

            void include_node(RVNode *);
            RVNode * find_node(std::string);
    };
}

#endif
