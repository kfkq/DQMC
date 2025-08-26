/*
/   This module defines the geometry of the simulation lattice,
/   including real-space vectors, reciprocal space vectors,
/   and helper functions.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>
#include <utility.h>

class Lattice {
private:
    std::array<double,2> a1_, a2_;
    std::vector<std::array<double,2>> orbs_;
    int L1_, L2_, n_orb_;
    std::array<double,2> b1_, b2_;
    std::vector<std::array<double,2>> k_points_;

public:
    /* ---------- factory helpers ---------- */
    Lattice(const utility::parameters& params,
            const std::array<double,2>& a1,
            const std::array<double,2>& a2,
            const std::vector<std::array<double,2>>& orbs)
        : a1_(a1), a2_(a2), orbs_(orbs),
          L1_(params.getInt("Lattice", "L1")), 
          L2_(params.getInt("Lattice", "L2")), 
          n_orb_(static_cast<int>(orbs.size()))
    {
        if (L1_<=0 || L2_<=0 || n_orb_==0) throw std::invalid_argument("Bad lattice dims");

        // compute reduced reciprocal vectors b1, b2
        double det = a1_[0]*a2_[1] - a1_[1]*a2_[0];
        if (std::abs(det) < 1e-12) throw std::invalid_argument("Singular lattice");
        b1_ = {  2*M_PI*a2_[1]/det/L1_, -2*M_PI*a2_[0]/det/L1_ };
        b2_ = { -2*M_PI*a1_[1]/det/L2_,  2*M_PI*a1_[0]/det/L2_ };
        // build k-points shifted to (-π, π] : range −L/2+1 … L/2
        k_points_.reserve(L1_*L2_);
        for (int n = 0; n < L1_; ++n) {
            int qx = n - (L1_)/2 + 1;           // gives −4 … 5 for L=10
            for (int m = 0; m < L2_; ++m) {
                int qy = m - (L2_)/2 + 1;
                k_points_.push_back({qx*b1_[0] + qy*b2_[0],
                                     qx*b1_[1] + qy*b2_[1]});
            }
        }
    }

    /* ---------- basic info ---------- */
    int n_cells() const noexcept { return L1_ * L2_; }
    int n_sites() const noexcept { return L1_ * L2_ * n_orb_; }
    const int& Lx() const noexcept { return L1_; }
    const int& Ly() const noexcept { return L2_; }
    const int& n_orb() const noexcept { return n_orb_; }

    const std::array<double,2>& a1() const noexcept { return a1_; }                                                                                                                                                    
    const std::array<double,2>& a2() const noexcept { return a2_; }     

    /* ---------- coordinate helpers ---------- */
    std::vector<double> site_position(int idx) const {
        if (idx < 0 || idx >= n_cells()) throw std::out_of_range("site index");
        const int cell = idx / n_orb_;
        const int orb  = idx % n_orb_;
        const int ux = cell % L1_;
        const int uy = cell / L1_;
        return {
            ux*a1_[0] + uy*a2_[0] + orbs_[orb][0],
            ux*a1_[1] + uy*a2_[1] + orbs_[orb][1]
        };
    }

    /* ---------- reciprocal lattice helpers ---------- */
    const std::vector<std::array<double,2>>& k_points() const noexcept { return k_points_; }
    const std::array<double,2>& b1() const noexcept { return b1_; }
    const std::array<double,2>& b2() const noexcept { return b2_; }
    
    /* ---------- convenience helpers matching main.cpp ---------- */
    std::array<int,2> site_to_unitcellpos(int idx) const {
        const int cell = idx / n_orb_;
        return {cell % L1_, cell / L1_};
    }

    int cell_to_site(int cell, int orb) const {
        if (orb >= n_orb_) {
            std::cerr << "Warning: orb index " << orb 
                      << " >= n_orb_  : " << n_orb_ << std::endl;
        }
        return cell * n_orb_ + orb;
    }

    std::vector<double> distance_between_site(int i,int j) const {
        std::vector<double> pi = site_position(i);
        std::vector<double> pj = site_position(j);
        return {pj[0] - pi[0], pj[1] - pi[1]};
    }

    int site_neighbors(int idx, std::array<int,2> delta, int orb) const {
        const int cell = idx / n_orb_;
        const int ux = cell % L1_;
        const int uy = cell / L1_;
        int tx = ((ux+delta[0])%L1_+L1_)%L1_;
        int ty = ((uy+delta[1])%L2_+L2_)%L2_;
        return (ty*L1_+tx)*n_orb_+orb;
    }
};
