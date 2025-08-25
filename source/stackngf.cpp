#include <stackngf.h>

// ----------------------------------------------------------------------------
// LDRStack Class Implementation
// ----------------------------------------------------------------------------

LDRStack::LDRStack(size_t n_stack) 
    : n_stack_(n_stack), stack_(n_stack) {}

LDRStack::LDRStack(LDRStack&& other) noexcept
    : n_stack_(other.n_stack_), stack_(std::move(other.stack_)) {}

LDRStack& LDRStack::operator=(LDRStack&& other) noexcept {
    if (this != &other) {
        n_stack_ = other.n_stack_;
        stack_ = std::move(other.stack_);
    }
    return *this;
}

stablelinalg::LDR& LDRStack::operator[](size_t idx) { 
    if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
    return stack_[idx]; 
}

const stablelinalg::LDR& LDRStack::operator[](size_t idx) const { 
    if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
    return stack_[idx]; 
}

void LDRStack::set(size_t idx, const stablelinalg::LDR& ldr) {
    if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
    stack_[idx] = ldr;
}