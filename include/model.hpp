#ifndef MODEL_HPP
#define MODEL_HPP

#include <armadillo>
#include <cmath>
#include <cassert>
#include "linalg.hpp"
#include "lattice.hpp"
#include "utility.hpp"

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
            
            // Lattice info
            int ns_;     // number of lattice sites
            int nt_;       // number of time slices
            
            // Matrices
            Matrix expK_; 
            Matrix invexpK_; 
            IMatrix fields_;  

            void init_expK(const lattice::Lattice& lat);
            void init_fields();
            void compute_alpha();

        public:
            HubbardAttractiveU(const lattice::Lattice& lat, 
                            double t, double U, double mu, 
                            double dtau, int nt);

            // Getters
            const Matrix& expK() const { return expK_; }
            const IMatrix& fields() const { return fields_; }
            int nt() const { return nt_; }
            int ns() const { return ns_; }

            // Functions that will be used in the simulation
            Matrix calc_B(int t);
            Matrix calc_Bup(int t);
            Matrix calc_Bdn(int t);

            Matrix calc_invB(int t);
            Matrix calc_invBup(int t);
            Matrix calc_invBdn(int t);

            double acceptance_ratio(GreenFunc& Gtt, double delta, int i);
            void update_fields(int l, int i);
            void update_greens(GreenFunc& gtt, double delta, int i);
            double update_time_slice(GreenFunc& Gttup, GreenFunc& Gttdn_, int l);
    };

} // namespace model
#endif