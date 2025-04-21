// branch/sandeep/sandeep.h
#ifndef BRANCH_SANDEEP_H
#define BRANCH_SANDEEP_H

#include <array>
#include <bitset>
#include <deque>
#include <unordered_map>
#include <cmath>
#include "ooo_cpu.h"
#include "modules.h"
#include "msl/fwcounter.h"

namespace champsim {
namespace configured {

class sandeep : public champsim::modules::branch_predictor {
public:
  template <std::size_t HISTLEN, std::size_t BITS>
  class internal_perceptron {
    using counter_type = champsim::msl::sfwcounter<BITS>;
    counter_type bias{0};
    std::array<counter_type, HISTLEN> weights = {};

  public:
    typename counter_type::value_type predict(std::bitset<HISTLEN> history);
    void update(bool result, std::bitset<HISTLEN> history, int taken_count, int not_taken_count, int output);
    void set_bias_from_counters(int taken_count, int not_taken_count, double B);
  };

  static constexpr std::size_t PERCEPTRON_HISTORY = 24;
  static constexpr std::size_t PERCEPTRON_BITS = 8;
  static constexpr std::size_t NUM_PERCEPTRONS = 163;
  static constexpr std::size_t NUM_UPDATE_ENTRIES = 100;

  struct perceptron_state {
    champsim::address ip{};
    bool prediction = false;
    long long int output = 0;
    std::bitset<PERCEPTRON_HISTORY> history = 0;
  };

  std::array<internal_perceptron<PERCEPTRON_HISTORY, PERCEPTRON_BITS>, NUM_PERCEPTRONS> perceptrons;
  std::deque<perceptron_state> perceptron_state_buf;
  std::bitset<PERCEPTRON_HISTORY> spec_global_history;
  std::bitset<PERCEPTRON_HISTORY> global_history;

  std::unordered_map<std::size_t, std::pair<int, int>> branch_counters;

  explicit sandeep(O3_CPU* core) : branch_predictor(core) {}

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

template <std::size_t HISTLEN, std::size_t BITS>
auto sandeep::internal_perceptron<HISTLEN, BITS>::predict(std::bitset<HISTLEN> history) -> typename counter_type::value_type {
  auto output = bias.value();
  for (std::size_t i = 0; i < history.size(); i++) {
    output += history[i] ? weights[i].value() : -weights[i].value();
  }
  return output;
}

template <std::size_t HISTLEN, std::size_t BITS>
void sandeep::internal_perceptron<HISTLEN, BITS>::update(bool result, std::bitset<HISTLEN> history, int taken_count, int not_taken_count, int output) {
  const int THETA = std::lround(1.93 * HISTLEN + 14);
  if (std::abs(output) > THETA && result == (output >= 0)) return;

  bias += result ? 1 : -1;
  auto upd_mask = result ? history : ~history;
  for (std::size_t i = 0; i < upd_mask.size(); i++) {
    weights[i] += upd_mask[i] ? 1 : -1;
  }
  double B = std::min(2.0, std::max(0.5, std::abs(output) / 20.0)); // Dynamic B scaling
  set_bias_from_counters(taken_count, not_taken_count, B);
}

template <std::size_t HISTLEN, std::size_t BITS>
void sandeep::internal_perceptron<HISTLEN, BITS>::set_bias_from_counters(int taken_count, int not_taken_count, double B) {
  int net_bias = static_cast<int>(std::round(B * (taken_count - not_taken_count)));
  bias = counter_type(net_bias);
}

extern template class sandeep::internal_perceptron<sandeep::PERCEPTRON_HISTORY, sandeep::PERCEPTRON_BITS>;

} // namespace configured
} // namespace champsim

#endif

