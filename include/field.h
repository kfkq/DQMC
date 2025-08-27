/*
/   This module encapsulates the Hubbard-Stratonovich (HS) field
/   The HS Field is constructed using Gauss-Hermite Quadrature
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <utility.h> 
#include <armadillo>

class GHQField {
    private:
        // HS parameters
        arma::vec gamma_;
        arma::vec eta_;

        arma::imat proposal_;

        // The discrete field variables {s_i(l)}
        arma::imat fields_;
    public:
        // Constructor
        GHQField() = default;
        GHQField(int nt, int nv, utility::random rng)
        {    
            // Gamma and Eta are properties of the GHQ decomposition itself,
            gamma_.set_size(4);
            eta_.set_size(4);
            
            const double s6 = std::sqrt(6.0);

            // Mapping discrete field values {0,1,2,3} to GHQ parameters
            gamma_(0) = 1.0 - s6 / 3.0;
            gamma_(1) = 1.0 + s6 / 3.0;
            gamma_(2) = 1.0 + s6 / 3.0;
            gamma_(3) = 1.0 - s6 / 3.0;

            eta_(0) = -std::sqrt(2.0 * (3.0 + s6));
            eta_(1) = -std::sqrt(2.0 * (3.0 - s6));
            eta_(2) =  std::sqrt(2.0 * (3.0 - s6));
            eta_(3) =  std::sqrt(2.0 * (3.0 + s6));

            proposal_ = {{1, 2, 3},
                        {0, 2, 3},
                        {0, 1, 3},
                        {0, 1, 2}};

            // --- Randomly initialize the field configuration ---
            // The number of columns is now nv.
            fields_.set_size(nt, nv);
            
            std::uniform_int_distribution<int> dist(0, 3);
            for (arma::uword i = 0; i < fields_.n_elem; ++i) {
                fields_(i) = dist(rng.get_generator());
            }
        }

        // --- Getters ---
        double gamma(int f) const { return gamma_(f); }
        double eta(int f) const { return eta_(f); };
        int single_val(int time_slice, int field_idx) const { return fields_(time_slice, field_idx); }
        const arma::imat& fields() const { return fields_; }
        int nv() const { return fields_.n_cols; }

        // --- Setters ---
        void set_single_field(int l, int i, int new_value) {
            fields_(l, i) = new_value;
        }

        void set_fields(const arma::imat& new_fields) {
            fields_ = new_fields;
        }

        int propose_new_field(int old_field, utility::random rng) const {
            // This implementation is for a 4-state discrete field.
            // It proposes one of the other 3 states with equal probability.
            std::uniform_int_distribution<int> dist(0, 2);
            int propose_field = dist(rng.get_generator());

            return proposal_(old_field, propose_field);
        }
};