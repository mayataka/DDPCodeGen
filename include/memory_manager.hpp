#ifndef CDDP_MEMORY_MANAGER_H_
#define CDDP_MEMORY_MANAGER_H_


namespace cddp {
namespace memorymanager {

// Allocates memory for a vector whose dimension is dim and set all components 
// zero. Then returns the pointer to the vector.
double* NewVector(const int dim);

// Free memory of a vector. 
void DeleteVector(double* vec);

// Allocates memory for a matrix whose dimensions are given by dim_row and 
// dim_column and set all components zero. Then returns the pointer to the 
// matrix.
double** NewMatrix(const int dim_row, const int dim_column);

//  Free memory of a matrix.
void DeleteMatrix(double** mat);

} // namespace memorymanager
} // namespace cddp 


#endif // CDDP_MEMORY_MANAGER_H_