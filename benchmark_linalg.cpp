#include "include/linalg.hpp"                                                                                                                                                                
#include <iostream>                                                                                                                                                                          
#include <chrono>                                                                                                                                                                            
                                                                                                                                                                                             
// Helper function to create a random LDR object for testing                                                                                                                                 
linalg::LDR create_random_ldr(int n) {                                                                                                                                                       
    arma::mat M = arma::randu<arma::mat>(n, n);                                                                                                                                              
    return linalg::LDR::from_qr(M);                                                                                                                                                          
}                                                                                                                                                                                            
                                                                                                                                                                                             
int main() {                                                                                                                                                                                 
    const int n_sites = 144;                                                                                                                                                                  
    const int n_iterations = 100;                                                                                                                                                            
                                                                                                                                                                                             
    // --- Benchmark inv_eye_plus_ldr ---                                                                                                                                                    
    std::cout << "Benchmarking inv_eye_plus_ldr..." << std::endl;                                                                                                                            
                                                                                                                                                                                             
    linalg::LDR ldr1 = create_random_ldr(n_sites);                                                                                                                                           
                                                                                                                                                                                             
    auto start1 = std::chrono::high_resolution_clock::now();                                                                                                                                 
    for (int i = 0; i < n_iterations; ++i) {                                                                                                                                                 
        volatile linalg::Matrix g = linalg::LDR::inv_eye_plus_ldr(ldr1);                                                                                                                     
    }                                                                                                                                                                                        
    auto end1 = std::chrono::high_resolution_clock::now();                                                                                                                                   
    std::chrono::duration<double> diff1 = end1 - start1;                                                                                                                                     
    std::cout << "Total time for " << n_iterations << " iterations: " << diff1.count() << " s" << std::endl;                                                                                 
    std::cout << "Average time: " << diff1.count() / n_iterations * 1000 << " ms" << std::endl;                                                                                              
    std::cout << "----------------------------------------" << std::endl;                                                                                                                    
                                                                                                                                                                                             
                                                                                                                                                                                             
    // --- Benchmark inv_eye_plus_ldr_mul_ldr ---                                                                                                                                            
    std::cout << "Benchmarking inv_eye_plus_ldr_mul_ldr..." << std::endl;                                                                                                                    
                                                                                                                                                                                             
    linalg::LDR ldr2 = create_random_ldr(n_sites);                                                                                                                                           
    linalg::LDR ldr3 = create_random_ldr(n_sites);                                                                                                                                           
                                                                                                                                                                                             
    auto start2 = std::chrono::high_resolution_clock::now();                                                                                                                                 
    for (int i = 0; i < n_iterations; ++i) {                                                                                                                                                 
        volatile linalg::Matrix g = linalg::LDR::inv_eye_plus_ldr_mul_ldr(ldr2, ldr3);                                                                                                       
    }                                                                                                                                                                                        
    auto end2 = std::chrono::high_resolution_clock::now();                                                                                                                                   
    std::chrono::duration<double> diff2 = end2 - start2;                                                                                                                                     
    std::cout << "Total time for " << n_iterations << " iterations: " << diff2.count() << " s" << std::endl;                                                                                 
    std::cout << "Average time: " << diff2.count() / n_iterations * 1000 << " ms" << std::endl;                                                                                              
                                                                                                                                                                                             
    return 0;                                                                                                                                                                                
}  