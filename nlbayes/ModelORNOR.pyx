# distutils: language = c++

from cpython cimport array
import array
from nlbayes.ModelORNOR cimport *
import pandas as pd
import numpy as np
from tqdm import tqdm
from cysignals.signals cimport sig_check
from cysignals.signals cimport sig_on, sig_off
from pprint import pprint


cdef class PyModelORNOR:
    cdef ModelORNOR *c_model  # Hold a C++ instance which we're wrapping

    def __cinit__(self, network, evidence = dict(), set active_tf_set = set(),
                  uniform_t = True, t_alpha = None, t_beta = None,
                  zy = 0., zn = 0.,
                  s_leniency = 0.1,
                  n_graphs = 3, verbosity = 0):

        zy_value = 1. if zy == 0. else zy

        if len(evidence) > 0 and zn == 0.:
            n_edges = sum(len(d) for d in network.values())
            n_edges_deg = 0

            for src, trg_dict in network.items():
                for trg, mor in trg_dict.items():
                    if (trg in evidence.keys() and evidence[trg] != 0):
                        n_edges_deg += 1
            
            zn_value = n_edges_deg / n_edges / 10

        elif zn == 0.:
            zn_value = 1.

        else:
            zn_value = zn


        if len(evidence) == 0:
            if t_alpha is None and t_beta is None:
                t_alpha = 18.
                t_beta = 2.
            else:
                assert t_alpha is not None and not t_beta is None, (
                    "Must provide both t_alpha and t_beta parameters for "
                    " theta random variable with beta distribution.")
        elif uniform_t:
            t_alpha = 1.
            t_beta = 1.
        else:
            if t_alpha is None and t_beta is None:
                t_alpha = 2.
                t_beta = 2.
            else:
                assert t_alpha is not None and not t_beta is None, (
                    "Must provide both t_alpha and t_beta parameters for "
                    " theta random variable with beta distribution.")

        if verbosity >= 2:
            print("\tT alpha   : ", t_alpha)
            print("\tT beta    : ", t_beta)
            print("\tS leniency: ", s_leniency)
            print("\tZY value  : ", zy_value)
            print("\tZN value  : ", zn_value)
            print("\t# Graphs  : ", n_graphs)

        cdef src_trg_pair_t src_trg
        cdef network_edge_t network_edge
        cdef network_t network_c = network_t()
        for src_uid, trg_dict in network.items():
            for trg_uid, mor in trg_dict.items():
                src_trg = (str(src_uid).encode('utf8'),
                           str(trg_uid).encode('utf8'))
                network_edge = src_trg, mor
                network_c.push_back(network_edge)

        cdef evidence_dict_t evidence_c = evidence_dict_t()
        for trg_uid, trg_de in evidence.items():
            evidence_c.insert((str(trg_uid).encode('utf8'), trg_de))

        cdef prior_active_tf_set_t active_tf_set_c = prior_active_tf_set_t()
        for src_uid in active_tf_set:
            active_tf_set_c.insert(str(src_uid).encode('utf8'))
        
        sprior = [1.0 - s_leniency, 0.9 * s_leniency, 0.1 * s_leniency,
                  0.5 * s_leniency, 1.0 - s_leniency, 0.5 * s_leniency,
                  0.1 * s_leniency, 0.9 * s_leniency, 1.0 - s_leniency]

        cdef double[9] sprior_c = array.array('d', sprior)
        self.c_model = new ModelORNOR(
            network_c, evidence_c, active_tf_set_c,
            sprior_c, t_alpha, t_beta, zy_value, zn_value,
            n_graphs)


    def get_gelman_rubin(self):
        gr_list = self.c_model.get_gelman_rubin()
        return [id.decode('utf8').split('_')[:2]+[gr] for id, gr in gr_list]

    def get_max_gelman_rubin(self):
        return self.c_model.get_max_gelman_rubin()

    def sample_posterior(self, N, gr_level, burnin=False, show_progress=True):
        if burnin:
            print("\nInitializing model burn-in ...")
            converged = False
            while not converged:
                status = self.sample_n(200, 20, 5.0, show_progress)
                converged = status == 0
                if (status == -1):
                    print("Interrupt signal received")
                    return

            self.burn_stats()
            print("Burn-in complete ...")

        n_sampled = 0
        converged = False
        i = 1
        while n_sampled < N and not converged:
            n = min(200 * i, N - n_sampled)
            status = self.sample_n(n, 5, gr_level, show_progress)
            converged = status == 0
            n_sampled = n_sampled + n
            i = i + 1
            if status == -1:
                print("Interrupt signal received")
                return

        if not converged:
            x = dict(self.c_model.get_gelman_rubin())
            n_vars = len(x)
            x = {k:v for k, v in x.items() if v > gr_level}
            n_did_not_converge = len(x)
            print(
                "\nThere are", n_vars, "random variables in the model.",
                n_did_not_converge, "of them did not converge.")
            if n_did_not_converge < 20:
                pprint(x)


    def sample_n(self, N, dN, gr_level, show_progress=True):

        print()
        gr = float('inf')

        status = 0
        n = 0
        try:
            if show_progress:
                progress = tqdm(total=N)
            while n < N and gr > gr_level:
                if show_progress:
                    progress.update(dN)
                dN = min(dN, N-n)

                sig_on()
                self.c_model.sample_n(dN)
                sig_off()

                n += dN
                gr = self.c_model.get_max_gelman_rubin()

            if show_progress:
                progress.total = n
                progress.update(0)

        except KeyboardInterrupt:
            status = -1
        finally:
            if show_progress:
                progress.close()

        converged = gr <= gr_level

        if converged:
            print("Converged after", self.c_model.total_sampled, "samples")

        elif status == 0:
            status = 1
            print("Drawed", self.c_model.total_sampled, "samples so far")

        elif status == -1:
            print("\nProcess interrupted.")
            print("Drawed", self.c_model.total_sampled, "samples so far")

        print("Max Gelman-Rubin statistic is", gr, "(target",
              "was" if converged else "is", gr_level,")")

        return status

    def sample(self, unsigned int N = 50000, unsigned int deltaN = 30):
        self.c_model.sample(N, deltaN)
        return

        # # TODO: KeyboardInterrupt is not working

        # n = 0
        # gr = float('inf')
        # while n < N and gr > 1.10:
        #     try:
        #         self.c_model.sample_n(deltaN)
        #         deltaN = min(deltaN, N-n)
        #         n += deltaN
        #         gr = self.c_model.get_max_gelman_rubin()
        #         # sig_check()

        #     except KeyboardInterrupt:
        #         print("\n\nCaught KeyboardInterrupt, workers have been "
        #               "terminated\n")

        # print("Drawed", n, "samples")
        # print("Max Gelman-Rubin statistics is", gr)


    @property
    def network(self):

        cdef int mor
        cdef std_string src_c, trg_c
        cdef src_trg_pair_t src_trg_pair

        out = {}
        for i in range(self.c_model.network.size()):
            (src_c, trg_c), mor = self.c_model.network[i]
            src = src_c.decode()
            trg = trg_c.decode()
            if not src in out.keys():
                out[src] = {}
            out[src][trg] = mor

        return out

    @property
    def evidence(self):
        return {uid.decode(): val for uid, val in self.c_model.evidence}

    @property
    def active_tf_set(self):
        return {uid.decode(): val for uid, val in self.c_model.active_tf_set}

    @property
    def _config(self):
        return {
            "t_alpha": self.c_model.t_alpha,
            "t_beta": self.c_model.t_beta,
            "zy": self.c_model.zy,
            "zn": self.c_model.zn,
            "n_graphs": self.c_model.n_graphs,
        }

    @property
    def _seeds(self):
        return self.c_model.get_seeds()

    @property
    def total_sampled(self):
        return self.c_model.total_sampled

    @property
    def n_graphs(self):
        return self.c_model.n_graphs

    def print_stats(self):
        self.c_model.print_stats()

    def burn_stats(self):
        self.c_model.burn_stats()
        self.c_model.total_sampled = 0
    
    def get_posterior(self, var_name):
        means = self.get_posterior_means(var_name)
        sdevs = self.get_posterior_sdevs(var_name)

        dff_list = []
        for data in [means, sdevs]:
            A = np.array(data)
            col_names = A.T[0]
            col_uids = A.T[1]
            n_stats = A.T[2].astype(int).max()+1
            index = zip(col_names[::n_stats], col_uids[::n_stats])
            index = pd.MultiIndex.from_tuples(index, names=['name', 'uid'])
            columns = [f"V{i}" for i in range(n_stats)]

            array = A.T[3].reshape((-1, n_stats)).astype(float)
            dff_list.append(pd.DataFrame(array, index=index, columns=columns))

        df = pd.merge(*dff_list, left_index=True, right_index=True, suffixes=['_mean', '_sdev'])

        return df.reset_index()

    def get_posterior_mean_stat(self, var_name, stat):
        A = np.array(self.get_posterior_means(var_name))
        n_stats = A.T[2].astype(int).max()+1

        uids = A[stat::n_stats, 1]
        vals = A[stat::n_stats, 3].astype(float)

        return dict(zip(uids, vals))

    def get_posterior_means(self, var_name):
        posterior_stat = self.c_model.get_posterior_means(var_name.encode('utf8'))
        return [id.decode('utf8').split('_')+[stat] for id, stat in posterior_stat]

    def get_posterior_sdevs(self, var_name):
        posterior_stat = self.c_model.get_posterior_sdevs(var_name.encode('utf8'))
        return [id.decode('utf8').split('_')+[stat] for id, stat in posterior_stat]

    def __dealloc__(self):
        if self.c_model is not NULL:
            del self.c_model
