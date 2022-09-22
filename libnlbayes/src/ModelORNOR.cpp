#include "ModelORNOR.h"
#include <random>

namespace nlb
{
    ModelORNOR::ModelORNOR() : ModelBase() {}
    ModelORNOR::~ModelORNOR() {
        for (auto graph: this->graphs)
            delete graph;
    }

    ModelORNOR::ModelORNOR(
        const network_t network, const evidence_dict_t evidence, const prior_active_tf_set_t active_tf_set,
        const double SPRIOR[3 * 3], double t_alpha, double t_beta, double z_value, double z0_value,
        unsigned int n_graphs
    ) : ModelBase(n_graphs) {

        this->network = network;
        this->evidence = evidence;
        this->active_tf_set = active_tf_set;
        this->t_alpha = t_alpha;
        this->t_beta = t_beta;
        this->zy = z_value;
        this->zn = z0_value;

        std::random_device rd;
        GraphORNOR * graph;

        for (unsigned int i = 0; i < this->n_graphs; i++) {
            graph = new GraphORNOR((unsigned int) rd());
            graph->build_structure(network, evidence, active_tf_set, SPRIOR, t_alpha, t_beta, z_value, z0_value);
            this->graphs.push_back(graph);
        }
    }
}
