#include "GraphBase.h"
#include <random>

namespace nlb
{
    GraphBase::GraphBase()
    {
        std::random_device rd;
        this->rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(this->rng, (unsigned int) rd());
    }

    GraphBase::GraphBase(unsigned int seed)
    {
        this->seed = seed;
        this->rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(this->rng, seed);
    }

    GraphBase::~GraphBase()
    {
        for (auto node: this->random_nodes)
            delete node;
        for (auto node: this->norand_nodes)
            delete node;
        gsl_rng_free(this->rng);
        // std::cout << "Object at " << this << " destroyed, instance of  GraphBase\t" << std::endl;
    }

    void GraphBase::sample(unsigned int N)
    {
        for (unsigned int i = 0; i < N; i++) {
            for (auto node: this->random_nodes) node->sample(this->rng, true);
        }
    }

    void GraphBase::burn_stats()
    {
        for (auto node: this->random_nodes) node->burn_stats();
    }

    void GraphBase::print_stats()
    {
        printf("NodeID");
        for (auto node: this->random_nodes) {
            printf("\n%s", node->id.c_str());
            for (unsigned int i = 0;  i < node->n_stats; i++)
                printf("\t%6.2f", gsl_rstat_mean(node->stats[i]));
        }
        printf("\n\n");
    }
}
