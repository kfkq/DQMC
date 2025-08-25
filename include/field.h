/*
/   This module encapsulates the Hubbard-Stratonovich (HS) field
/   The HS Field is constructed using Gauss-Hermite Quadrature
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <utility.h> 
#include <armadillo>
#include <cmath>

class HSField {
    private:
        // HS parameters
        arma::vec gamma_;
        arma::vec eta_;

        arma::imat proposal_;

        // The discrete field variables {s_i(l)}
        arma::imat fields_;

        // A reference to a random number generator
        utility::random& rng_;

    public:
        // Constructor for the attractive Hubbard model's GHQ decomposition.
        HSField(int nt, int nv, utility::random& rng);

        // --- Getters ---
        double get_gamma(int f) const { return gamma_(f); }
        double get_eta(int f) const { return eta_(f); };
        int get_field_value(int time_slice, int field_idx) const { return fields_(time_slice, field_idx); }
        const arma::imat& get_fields() const { return fields_; }
        int get_nv() const { return fields_.n_cols; }

        // --- Setters ---
        void set_fields(const arma::imat& new_fields);
        void set_field_value(int time_slice, int field_idx, int new_value);

        // --- Field-specific Random Number Generation ---
        // Proposes a new field value that is guaranteed to be different from the old one.
        int propose_new_field(int old_field_value) const;
};