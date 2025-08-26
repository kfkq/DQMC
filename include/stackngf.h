/*
/   This file defines the core data structures used by the DQMC,
/   such as the Green's function container (GF) and the stack of
/   LDR-decomposed matrices (LDRStack).
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <stablelinalg.h>
#include <armadillo>

// --- Green's Function Data Structure ---
struct GF {
    // Equal-time Green's function G(τ, τ) for each time slice τ.
    // Gtt[0] is G(0,0), Gtt[1] is G(dτ, dτ), ..., Gtt[nt+1] is G(β,β).
    std::vector<arma::mat> Gtt;

    // Unequal-time Green's functions, populated during measurement sweeps.
    std::vector<arma::mat> Gt0;  // G(τ, 0)
    std::vector<arma::mat> G0t;  // G(0, τ)

    // The log-determinant of the corresponding fermion flavor matrix logdet[M = (I + B)].
    double log_det_M;

    // Default constructor
    GF() : log_det_M(0.0) {}
};

// --- LDR Matrix Stack ---
// A container for a series of LDR-decomposed matrices, used for
// numerically stable propagation through imaginary time.
class LDRStack {
private:
    size_t n_stack_ = 0;
    std::vector<stablelinalg::LDR> stack_;

public:
    LDRStack() = default;
    explicit LDRStack(size_t n_stack) : n_stack_(n_stack), stack_(n_stack) {}
    
    // Move semantics are enabled for efficiency
    LDRStack(LDRStack&& other) noexcept
        : n_stack_(other.n_stack_), stack_(std::move(other.stack_)) {}
    
    LDRStack& operator=(LDRStack&& other) noexcept {
        if (this != &other) {
            n_stack_ = other.n_stack_;
            stack_ = std::move(other.stack_);
        }
        return *this;
    }
    
    // Copying is disallowed to prevent expensive deep copies
    LDRStack(const LDRStack&) = delete;
    LDRStack& operator=(const LDRStack&) = delete;

    // --- Element Access ---
    stablelinalg::LDR& operator[](size_t idx) { 
        if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
        return stack_[idx]; 
    }
    const stablelinalg::LDR& operator[](size_t idx) const { 
        if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
        return stack_[idx]; 
    }
    
    // --- Capacity ---
    constexpr size_t size() const noexcept { return n_stack_; }
    
    // --- Modification ---
    void set(size_t idx, const stablelinalg::LDR& ldr) {
        if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
        stack_[idx] = ldr;
    }
};