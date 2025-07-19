#ifndef MODEL_HPP
#define MODEL_HPP

#include <armadillo>
#include <cmath>
#include <cassert>

#include "linalg.hpp"
#include "lattice.hpp"
#include "utility.hpp"

struct GF;

namespace model {

    // Type aliases for better readability
    using Matrix = arma::mat;
    using Vector = arma::vec;
    using IMatrix = arma::imat;
    using GreenFunc = arma::mat;

    class HubbardAttractiveU {
        private:
            // Model parameters
            double t_;        // hopping parameter
            double U_;        // interaction strength
            double mu_;       // chemical potential
            double dtau_;     // imaginary time step
            double alpha_;    // Hubbard-Stratonovich coupling
            double n_flavor_; // Number of factorized DQMC product (spin)
            
            // Lattice info
            const int ns_;     // number of lattice sites
            const int nt_;       // number of time slices
            
            // Matrices
            Matrix expK_; 
            Matrix invexpK_; 
            IMatrix fields_;  

            void init_expK(const Lattice& lat);
            void init_fields();
            void compute_alpha();

        public:
            HubbardAttractiveU(const Lattice& lat, 
                            double t, double U, double mu, 
                            double dtau, int nt);

            // Getters
            const Matrix& expK() const { return expK_; }
            const IMatrix& fields() const { return fields_; }
            int nt() const { return nt_; }
            int ns() const { return ns_; }
            int n_flavor() const { return n_flavor_; }

            // Functions that will be used in the simulation
            Matrix calc_B(int t, int nfl);
            Matrix calc_invB(int t, int nfl);

            double acceptance_ratio(GreenFunc& Gtt, double delta, int i);
            void update_fields(int l, int i);
            void update_greens(GreenFunc& gtt, double delta, int i);
            double update_time_slice(std::vector<GF>& greens, int l);
    };

} // namespace model
#endif