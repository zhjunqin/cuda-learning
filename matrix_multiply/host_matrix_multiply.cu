#include "Matrix.h"
#include "Common.h"

#define N 1024

void matrixMultiply(Matrix<float> m_a, Matrix<float> m_b, Matrix<float> m_c) {
    for(size_t i = 0; i < N; i++) {
	for(size_t j = 0; j < N; j++) {
            float temp = 0;
	    for(size_t k=0; k < N; k++) {
                temp += m_a.get(i, k) * m_b.get(k, j);
	    }
	    m_c.set(i, j, temp);
	}
    }
}


int main() {
   Matrix<float> m_a, m_b, m_c;
   m_a.init(N, N);
   m_a.host_alloc();
   m_b.init(N, N);
   m_b.host_alloc();
   m_c.init(N, N);
   m_c.host_alloc();

   m_a.randomInit();
   m_b.randomInit();
   
   double start = seconds();
   matrixMultiply(m_a, m_b, m_c); 
   double total = seconds() - start;
   cout << "Host Matrix multiply 1024*1024 time(second): " << total << std::endl;

   m_a.remove();
   m_b.remove();
   m_c.remove();
   return 0;
}
    

