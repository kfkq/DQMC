/*
/   This file defines the abstract base class for all physical models.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>
#include <stackngf.h>

// Forward-declare Green's function struct
struct GF;

class ModelBase {
    public:
        virtual ~ModelBase() = default;

        // --- Basic Properties ---
        virtual int nv() const = 0;
        virtual int n_size() const = 0;
        virtual int n_timesteps() const = 0;
        virtual int n_flavors() const = 0;

        // --- Fields functions ---
        virtual void set_fields(const arma::imat& new_fields) = 0;
        virtual void set_field_value(int time_slice, int field_idx, int new_field) = 0;
        virtual int propose_field(int time_slice, int field_idx) = 0;

        // --- Hamiltonian Components ---
        virtual const arma::mat& get_expK() const = 0;
        virtual const arma::mat& get_invexpK() const = 0;
        virtual arma::vec get_expV(int time_slice, int flavor) = 0;
        virtual arma::vec get_invexpV(int time_slice, int flavor) = 0;

        // --- Local Update Operations ---
        // update the configuration and Green's function using local update
        virtual double local_update_ratio(
            std::vector<GF>& GF,
            int time_slice,
            int site_idx,
            int new_field_value
        ) = 0;

        virtual void update_greens_local(std::vector<GF>& GF, int time_slice, int field_idx) = 0;

        // --- Global update ---
        virtual double calculate_global_action(const std::vector<GF>& greens) = 0;
    };