#include "qmutils/operator.h"

int main() {
  auto a = qmutils::Operator::Fermion::creation(qmutils::Operator::Spin::Up, 0);
  return 0;
}
