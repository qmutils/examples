#include "qmutils/expression.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/sparse_matrix.h"
#include "qmutils/term.h"

class HubbardChain1D {
 public:
  HubbardChain1D(size_t sites, float t, float U)
      : m_sites(sites), m_t(t), m_U(U) {
    construct_hamiltonian();
  }

  const qmutils::Expression& hamiltonian() const { return m_hamiltonian; }

  size_t sites() const { return m_sites; }
  float t() const { return m_t; }
  float U() const { return m_U; }

 private:
  size_t m_sites;
  float m_t;  // hopping parameter
  float m_U;  // on-site interaction strength
  qmutils::Expression m_hamiltonian;

  void construct_hamiltonian();
};

void HubbardChain1D::construct_hamiltonian() {
  // Hopping terms
  for (size_t i = 0; i < m_sites; ++i) {
    size_t j = (i + 1) % m_sites;  // Periodic boundary conditions
    m_hamiltonian +=
        qmutils::Expression::hopping(i, j, qmutils::Operator::Spin::Up);
    m_hamiltonian +=
        qmutils::Expression::hopping(i, j, qmutils::Operator::Spin::Down);
  }

  // On-site interaction terms
  for (size_t i = 0; i < m_sites; ++i) {
    m_hamiltonian += qmutils::Term::density(qmutils::Operator::Spin::Up, i) *
                     qmutils::Term::density(qmutils::Operator::Spin::Down, i) *
                     m_U;
  }
}

int main() {
  const size_t sites = 4;
  const size_t particles = 3;

  HubbardChain1D chain(sites, 1.0f, 4.0f);
  qmutils::Basis basis(sites, particles);

  auto mat = qmutils::compute_matrix_elements<qmutils::SpMat_cf>(
      basis, chain.hamiltonian());

  for (size_t i = 0; i < basis.size(); ++i) {
    for (size_t j = 0; j < basis.size(); ++j) {
      std::cout << "H(" << i << "," << j << ") = " << mat(i, j) << std::endl;
    }
  }

  return 0;
}
