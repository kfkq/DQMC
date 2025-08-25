/*
/   This module provides an implementation of attractive Hubbard model.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <model_base.h>
#include <lattice.h>
#include <field.h>

class HubbardAttractiveU : public ModelBase {
    private:
        // --- Physical and Simulation Parameters ---
        double t_;      // Hopping parameter
        double mu_;     // Chemical potential
        double g_; // sqrt(dtau * U / 2)
        double alpha_; // the bosonic factor

        double delta_;

        // --- Core Components ---
        HSField field_;

        // --- Pre-calculated Matrices ---
        arma::mat expK_;
        arma::mat invexpK_;

    public:
        // Constructor
        HubbardAttractiveU(const utility::parameters& params, const Lattice& lat, utility::random& rng);

        // --- Implementations of the ModelBase virtual functions ---
        int nv() const override;
        int n_size() const override;
        int n_timesteps() const override;
        int n_flavors() const override;

        void set_fields(const arma::imat& new_fields) override;
        void set_field_value(int time_slice, int field_idx, int new_field) override; 
        int propose_field(int time_slice, int field_idx) override;

        const arma::mat& get_expK() const override;
        const arma::mat& get_invexpK() const override;
        arma::vec get_expV(int time_slice, int flavor) override;
        arma::vec get_invexpV(int time_slice, int flavor) override;

        double det_ratio(arma::mat& G00, double delta, int site_idx);

        double local_update_ratio(
            std::vector<GF>& GF,
            int time_slice,
            int site_idx,
            int new_field_value
        ) override;

        void update_greens_local(std::vector<GF>& GF, int time_slice, int site_idx) override;

        double calculate_global_action(const std::vector<GF>& greens) override;
    };