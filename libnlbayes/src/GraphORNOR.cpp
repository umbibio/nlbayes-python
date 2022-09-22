#include <math.h>
#include <algorithm>

#include "GraphORNOR.h"


namespace nlb
{

    GraphORNOR::GraphORNOR() : GraphBase() {}
    GraphORNOR::GraphORNOR(unsigned int seed) : GraphBase(seed) {}
    GraphORNOR::~GraphORNOR() {}

    void GraphORNOR::build_structure (
        network_t interaction_network, 
        evidence_dict_t evidence,
        prior_active_tf_set_t active_tf_set,
        const double SPRIOR[3 * 3],
        double t_alpha, double t_beta,
        double zy_value,
        double zn_value
    ) {
        // std::cout << "Active TFs: " << active_tf_set.size() << std::endl;
        // std::cout << "Evidence size: " << evidence.size() << std::endl;
        std::copy(SPRIOR, SPRIOR + 9, &this->SPROB[0][0]);

        // This means that evidence will be sampled from the graph, given the set of active TF
        bool is_simulation = evidence.size() == 0;

        // first, find unique src tfs and trg genes
        std::set<std::string> src_uid, trg_uid;
        std::string src, trg;
        src_trg_pair_t src_trg_pair;
        int mor;
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            src_uid.insert(src);
            trg_uid.insert(trg);
            if ( mor != -1 && mor != 0 && mor != 1) 
                throw std::out_of_range("MOR values can only be either -1, 0 or 1");
        }
        
        // temp variable pointers to create and manipulate nodes
        XNode * X;
        TNode * T;
        HNodeORNOR * H;
        YDataNode * Y;
        SNode * S;

        // collections are a wrapper for a dictionary
        NodeDictionary x_dictionary = NodeDictionary();
        NodeDictionary t_dictionary = NodeDictionary();
        NodeDictionary h_dictionary = NodeDictionary();

        // One X for each TF
        for (auto uid: src_uid) {
            if (is_simulation) {
                if (active_tf_set.count(uid))
                    X = new XNode(uid, this->XPROB_ACTIVE, (unsigned int) 1);
                else
                    X = new XNode(uid, this->XPROB, (unsigned int) 0);
                this->norand_nodes.push_back(X);
            } else {
                if (active_tf_set.count(uid))
                    X = new XNode(uid, this->XPROB_ACTIVE, this->rng);
                else
                    X = new XNode(uid, this->XPROB, this->rng);
                this->random_nodes.push_back(X);
            }

            x_dictionary.include_node(X);
        }
        

        // The observed data and its corresponding hidden true state
        if (is_simulation) {
            // no evidence provided. Treat H and Y as random variables to sample

            for (auto uid: trg_uid) {

                // start out with no deg, so first samples for X are not bumped up
                H = new HNodeORNOR(uid, this->YPROB, (unsigned int) 1, zy_value, zn_value);
                Y = new YDataNode(H);
                for (unsigned int i = 0; i < 3; i++)
                    Y->noise[i] = this->YNOISE[i];

                H->is_latent = false;
                h_dictionary.include_node(H);
                this->random_nodes.push_back(H);
                this->random_nodes.push_back(Y);
            }
        } else {
            // evidence available

            evidence_dict_t::iterator itr;
            unsigned int deg;

            // First compute deg proportions in data
            unsigned int deg_counts[3] = {0};
            for (itr = evidence.begin(); itr != evidence.end(); ++itr) {
                deg = (unsigned int) (itr->second + 1);
                deg_counts[deg] += 1;
            }
            // Now use that for Y and H nodes prior probabilities
            // take into account all target genes present in network
            this->YPROB[0] = (double) deg_counts[0] / trg_uid.size();
            this->YPROB[2] = (double) deg_counts[2] / trg_uid.size();
            this->YPROB[1] = (double) 1. - this->YPROB[0] - this->YPROB[2];

            // Finally initialize Y and H nodes
            for (auto uid: trg_uid) {
                itr = evidence.find(uid);
                if (itr != evidence.end())
                    deg = (unsigned int) (itr->second + 1);
                else
                    deg = 1; // 1 -> not DEG
                
                Y = new YDataNode(uid, this->YPROB, deg);
                for (unsigned int i = 0; i < 3; i++)
                    Y->noise[i] = this->YNOISE[i];

                H = new HNodeORNOR(Y, zy_value, zn_value);
                H->is_latent = true;

                h_dictionary.include_node(H);
                this->norand_nodes.push_back(H);
                this->norand_nodes.push_back(Y);
            }
        }

        // Determine the number of targets for each TF
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            X = (XNode *) x_dictionary.find_node(src);
            X->n_h_child++;
        }

        for (auto& dict_item: x_dictionary.dictionary) {
            X = (XNode *) dict_item.second;
            if (X->n_h_child == 0) throw std::runtime_error("Included TF with no targets");

            double t_mean = t_alpha / (t_alpha + t_beta);
            T = new TNode(X->uid, t_alpha, t_beta, t_mean);

            this->random_nodes.push_back(T);
            t_dictionary.include_node(T);
        }

        // Nodes for interaction S nodes 
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            X = (XNode *) x_dictionary.find_node(src);
            T = (TNode *) t_dictionary.find_node(src);
            H = (HNodeORNOR *) h_dictionary.find_node(trg);
            std::string s_id = X->uid + "-->" + H->uid;

            unsigned int mor_idx = (unsigned int) (mor + 1);
            S = new SNode(s_id, this->SPROB[mor_idx], mor_idx);
            if (is_simulation) {
                this->norand_nodes.push_back(S);
            } else {
                this->random_nodes.push_back(S);
            }

            H->append_parent(T, X, S);
        }
    }
}
