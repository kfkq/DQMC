#include <lattice.h>
#include <stdexcept>
#include <cmath>

// ----------------------------------------------------------------------------
// Lattice Class Implementation
// ----------------------------------------------------------------------------

// Constructor
Lattice::Lattice(const std::array<double, 2>& a1,
                 const std::array<double, 2>& a2,
                 const std::vector<std::array<double, 2>>& orbs,
                 int Lx, int Ly)
    : a1_(a1), a2_(a2), orbs_(orbs),
      Lx_(Lx), Ly_(Ly), n_orb_(static_cast<int>(orbs.size()))
{
    if (Lx <= 0 || Ly <= 0 || n_orb_ == 0) {
        throw std::invalid_argument("Lattice dimensions and n_orb must be positive.");
    }

    // Compute reciprocal lattice vectors b1, b2
    double det = a1_[0] * a2_[1] - a1_[1] * a2_[0];
    if (std::abs(det) < 1e-12) {
        throw std::invalid_argument("Lattice vectors are collinear (singular matrix).");
    }
    b1_ = { 2 * M_PI * a2_[1] / det, -2 * M_PI * a2_[0] / det };
    b2_ = { -2 * M_PI * a1_[1] / det,  2 * M_PI * a1_[0] / det };

    // Build the list of k-points in the first Brillouin zone
    k_points_.reserve(Lx_ * Ly_);
    for (int n = 0; n < Lx_; ++n) {
        for (int m = 0; m < Ly_; ++m) {
            // Standard discretization of the BZ for a finite lattice
            double kx = (static_cast<double>(n) / Lx_) * b1_[0] + (static_cast<double>(m) / Ly_) * b2_[0];
            double ky = (static_cast<double>(n) / Lx_) * b1_[1] + (static_cast<double>(m) / Ly_) * b2_[1];
            k_points_.push_back({kx, ky});
        }
    }
}

// --- Basic Info ---
int Lattice::n_sites() const noexcept { return Lx_ * Ly_ * n_orb_; }
int Lattice::n_cells() const noexcept { return Lx_ * Ly_; }
int Lattice::Lx() const noexcept { return Lx_; }
int Lattice::Ly() const noexcept { return Ly_; }
int Lattice::n_orb() const noexcept { return n_orb_; }
const std::array<double, 2>& Lattice::a1() const noexcept { return a1_; }
const std::array<double, 2>& Lattice::a2() const noexcept { return a2_; }

// --- Coordinate Helpers ---
std::vector<double> Lattice::site_position(int site_idx) const {
    if (site_idx < 0 || site_idx >= n_sites()) {
        throw std::out_of_range("Site index is out of range.");
    }
    const int cell_idx = site_idx / n_orb_;
    const int orb_idx  = site_idx % n_orb_;
    const int ux = cell_idx % Lx_;
    const int uy = cell_idx / Lx_;
    return {
        ux * a1_[0] + uy * a2_[0] + orbs_[orb_idx][0],
        ux * a1_[1] + uy * a2_[1] + orbs_[orb_idx][1]
    };
}

std::vector<double> Lattice::distance_between_sites(int site_i, int site_j) const {
    std::vector<double> pos_i = site_position(site_i);
    std::vector<double> pos_j = site_position(site_j);
    return {pos_j[0] - pos_i[0], pos_j[1] - pos_i[1]};
}

std::array<int, 2> Lattice::site_to_cell_coords(int site_idx) const {
    const int cell_idx = site_idx / n_orb_;
    return {cell_idx % Lx_, cell_idx / Lx_};
}

int Lattice::cell_coords_to_site(int ux, int uy, int orb) const {
    if (orb >= n_orb_) {
        // Using std::cerr is okay for a warning, but throwing an exception is better for errors.
        throw std::out_of_range("Orbital index is out of range.");
    }
    return (uy * Lx_ + ux) * n_orb_ + orb;
}

int Lattice::site_to_neighbor(int site_idx, std::array<int, 2> delta, int target_orb) const {
    const int cell_idx = site_idx / n_orb_;
    const int ux = cell_idx % Lx_;
    const int uy = cell_idx / Lx_;
    
    // Apply periodic boundary conditions
    int neighbor_ux = (ux + delta[0] % Lx_ + Lx_) % Lx_;
    int neighbor_uy = (uy + delta[1] % Ly_ + Ly_) % Ly_;
    
    return cell_coords_to_site(neighbor_ux, neighbor_uy, target_orb);
}

// --- Reciprocal Lattice Helpers ---
const std::vector<std::array<double, 2>>& Lattice::k_points() const noexcept { return k_points_; }
const std::array<double, 2>& Lattice::b1() const noexcept { return b1_; }
const std::array<double, 2>& Lattice::b2() const noexcept { return b2_; }