/*
/   This module defines the geometry of the simulation lattice,
/   including real-space vectors, reciprocal space vectors,
/   and helper functions.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <vector>
#include <array>

class Lattice {
private:
    // Real space properties
    std::array<double, 2> a1_, a2_;
    std::vector<std::array<double, 2>> orbs_;
    int Lx_, Ly_, n_orb_;

    // Reciprocal space properties
    std::array<double, 2> b1_, b2_;
    std::vector<std::array<double, 2>> k_points_;   

public:
    // --- Constructor ---
    Lattice(const std::array<double, 2>& a1,
            const std::array<double, 2>& a2,
            const std::vector<std::array<double, 2>>& orbs,
            int Lx, int Ly);
    
    // --- Basic Info ---
    int n_sites() const noexcept;
    int n_cells() const noexcept;
    int Lx() const noexcept;
    int Ly() const noexcept;
    int n_orb() const noexcept;

    const std::array<double, 2>& a1() const noexcept;
    const std::array<double, 2>& a2() const noexcept;

    // --- Coordinate Helpers ---
    std::vector<double> site_position(int site_idx) const;
    std::vector<double> distance_between_sites(int site_i, int site_j) const;
    std::array<int, 2> site_to_cell_coords(int site_idx) const;
    int cell_coords_to_site(int ux, int uy, int orb) const;
    int site_to_neighbor(int site_idx, std::array<int, 2> delta, int target_orb) const;

    // --- Reciprocal Lattice Helpers ---
    const std::vector<std::array<double, 2>>& k_points() const noexcept;
    const std::array<double, 2>& b1() const noexcept;
    const std::array<double, 2>& b2() const noexcept;
};