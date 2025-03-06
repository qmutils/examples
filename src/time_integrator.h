#pragma once
#include <armadillo>
#include <complex>
#include <iostream>

class TimeIntegratorBase {
 protected:
  const float initial_time_;
  const float final_time_;
  const size_t num_time_steps_;
  float time_step_;
  size_t counter_;

  TimeIntegratorBase(float initial_time, float final_time,
                     size_t num_time_steps)
      : initial_time_(initial_time),
        final_time_(final_time),
        num_time_steps_(num_time_steps),
        counter_(0) {
    if (num_time_steps_ == 0 || final_time_ <= initial_time_) {
      throw std::invalid_argument("Invalid time parameters");
    }
    time_step_ =
        (final_time_ - initial_time_) / static_cast<float>(num_time_steps_);
  }

 public:
  float time() const {
    return initial_time_ + static_cast<float>(counter_) * time_step_;
  }
};

template <typename ArmaMatrix>
class TimeIntegrator;

template <>
class TimeIntegrator<arma::sp_cx_fmat> : public TimeIntegratorBase {
 public:
  TimeIntegrator(float initial_time, float final_time, size_t num_time_steps,
                 const arma::sp_cx_fmat& h_matrix)
      : TimeIntegratorBase(initial_time, final_time, num_time_steps),
        hamiltonian_matrix_(h_matrix) {
    identity_matrix_ = arma::speye<arma::sp_cx_fmat>(
        hamiltonian_matrix_.n_rows, hamiltonian_matrix_.n_cols);

    update();
  }

  void update() {
    auto scaled_h =
        std::complex<float>(0, time_step_ * 0.5f) * hamiltonian_matrix_;

    crank_minus_ = identity_matrix_ - scaled_h;
    crank_plus_ = identity_matrix_ + scaled_h;

    if (!factoriser_.factorise(crank_minus_)) {
      std::runtime_error("Matrix factorisation failed");
    }
  }

  bool step(arma::cx_fvec& state_vector) {
    if (counter_ > num_time_steps_) {
      return false;
    }

    auto right_hand_side = crank_plus_ * state_vector;

    if (!factoriser_.solve(result_, right_hand_side)) {
      throw std::runtime_error("Failed to solve linear system");
    }

    state_vector = result_;
    ++counter_;
    return true;
  }

 private:
  arma::sp_cx_fmat hamiltonian_matrix_;
  arma::sp_cx_fmat identity_matrix_;
  arma::sp_cx_fmat crank_minus_;
  arma::sp_cx_fmat crank_plus_;
  arma::spsolve_factoriser factoriser_;
  arma::cx_fvec result_{};
};

template <>
class TimeIntegrator<arma::cx_fmat> : public TimeIntegratorBase {
 public:
  TimeIntegrator(float initial_time, float final_time, size_t num_time_steps,
                 const arma::cx_fmat& h_matrix)
      : TimeIntegratorBase(initial_time, final_time, num_time_steps),
        hamiltonian_matrix_(h_matrix) {
    identity_matrix_ = arma::eye<arma::cx_fmat>(hamiltonian_matrix_.n_rows,
                                                hamiltonian_matrix_.n_cols);

    update();
  }

  void update() {
    auto scaled_h =
        std::complex<float>(0, time_step_ * 0.5f) * hamiltonian_matrix_;

    crank_minus_ = identity_matrix_ - scaled_h;
    crank_plus_ = identity_matrix_ + scaled_h;
  }

  bool step(arma::cx_fvec& state_vector) {
    if (counter_ > num_time_steps_) {
      return false;
    }

    arma::cx_fvec right_hand_side = crank_plus_ * state_vector;
    arma::cx_fvec result;

    if (!arma::solve(result, crank_minus_, right_hand_side)) {
      throw std::runtime_error("Failed to solve linear system");
    }

    state_vector = result;
    ++counter_;
    return true;
  }

 private:
  arma::cx_fmat hamiltonian_matrix_;
  arma::cx_fmat identity_matrix_;
  arma::cx_fmat crank_minus_;
  arma::cx_fmat crank_plus_;
};

TimeIntegrator(float, float, size_t,
               const arma::sp_cx_fmat&) -> TimeIntegrator<arma::sp_cx_fmat>;

TimeIntegrator(float, float, size_t,
               const arma::cx_fmat&) -> TimeIntegrator<arma::cx_fmat>;
