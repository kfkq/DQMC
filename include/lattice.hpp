
#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <armadillo>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

class Lattice {
private:
    std::array<double,2> a1_, a2_;
    std::vector<std::array<double,2>> orbs_;
    int Lx_, Ly_, n_orb_;
    std::array<double,2> b1_, b2_;
    std::vector<std::array<double,2>> k_points_;

    Lattice(const std::array<double,2>& a1,
            const std::array<double,2>& a2,
            const std::vector<std::array<double,2>>& orbs,
            int Lx, int Ly)
        : a1_(a1), a2_(a2), orbs_(orbs),
          Lx_(Lx), Ly_(Ly), n_orb_(static_cast<int>(orbs.size()))
    {
        if (Lx<=0 || Ly<=0 || n_orb_==0) throw std::invalid_argument("Bad lattice dims");
        // compute reduced reciprocal vectors b1, b2
        double det = a1_[0]*a2_[1] - a1_[1]*a2_[0];
        if (std::abs(det) < 1e-12) throw std::invalid_argument("Singular lattice");
        b1_ = {  2*M_PI*a2_[1]/det/Lx_, -2*M_PI*a2_[0]/det/Lx_ };
        b2_ = { -2*M_PI*a1_[1]/det/Ly_,  2*M_PI*a1_[0]/det/Ly_ };
        // build k-points shifted to (-π, π] : range −L/2+1 … L/2
        k_points_.reserve(Lx_*Ly_);
        for (int n = 0; n < Lx_; ++n) {
            int qx = n - (Lx_)/2 + 1;           // gives −4 … 5 for L=10
            for (int m = 0; m < Ly_; ++m) {
                int qy = m - (Ly_)/2 + 1;
                k_points_.push_back({qx*b1_[0] + qy*b2_[0],
                                     qx*b1_[1] + qy*b2_[1]});
            }
        }
    }

public:
    /* ---------- factory helpers ---------- */
    static Lattice create_lattice(const std::array<double,2>& a1,
                                  const std::array<double,2>& a2,
                                  const std::vector<std::array<double,2>>& orbs,
                                  int Lx, int Ly)
    { return Lattice{a1,a2,orbs,Lx,Ly}; }

    /* ---------- basic info ---------- */
    int size() const noexcept { return Lx_ * Ly_ * n_orb_; }

    /* ---------- coordinate helpers ---------- */
    std::vector<double> site_position(int idx) const {
        if (idx < 0 || idx >= size()) throw std::out_of_range("site index");
        const int cell = idx / n_orb_;
        const int orb  = idx % n_orb_;
        const int ux = cell % Lx_;
        const int uy = cell / Lx_;
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
        return {cell % Lx_, cell / Lx_};
    }

    std::vector<int> unitcellpos_to_sites(std::array<int,2> xy) const {
        int cell = ((xy[1]%Ly_)+Ly_)%Ly_*Lx_ + ((xy[0]%Lx_)+Lx_)%Lx_;
        std::vector<int> res(n_orb_);
        for (int o=0; o<n_orb_; ++o) res[o] = cell*n_orb_ + o;
        return res;
    }

    std::vector<double> distance_between_site(int i,int j) const {
        std::vector<double> pi = site_position(i);
        std::vector<double> pj = site_position(j);
        return {pj[0] - pi[0], pj[1] - pi[1]};
    }

    std::vector<int> site_neighbors(int idx, std::array<int,2> delta) const {
        const int cell = idx / n_orb_;
        const int ux = cell % Lx_;
        const int uy = cell / Lx_;
        int tx = ((ux+delta[0])%Lx_+Lx_)%Lx_;
        int ty = ((uy+delta[1])%Ly_+Ly_)%Ly_;
        std::vector<int> res(n_orb_);
        for (int o=0; o<n_orb_; ++o) res[o] = (ty*Lx_+tx)*n_orb_+o;
        return res;
    }
};

#endif // LATTICE_HPP
