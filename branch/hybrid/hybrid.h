// hybrid.h
#ifndef BRANCH_HYBRID_H
#define BRANCH_HYBRID_H

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

class hybrid : public champsim::modules::branch_predictor {
public:
  template <std::size_t HISTLEN, std::size_t BITS>
  class internal_perceptron {
    using counter_type = champsim::msl::sfwcounter<BITS>;

    counter_type bias{0};
    std::array<counter_type, HISTLEN> weights = {};

  public:
    typename counter_type::value_type predict(std::bitset<HISTLEN> history);
    void update(bool result, std::bitset<HISTLEN> history, int taken_count, int not_taken_count, double B);
    void set_bias_from_counters(int taken_count, int not_taken_count, double B = 1.0);
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

  explicit hybrid(O3_CPU* core) : champsim::modules::branch_predictor(core) {}

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

template <std::size_t HISTLEN, std::size_t BITS>
auto hybrid::internal_perceptron<HISTLEN, BITS>::predict(std::bitset<HISTLEN> history) -> typename counter_type::value_type {
  auto output = bias.value();
  for (std::size_t i = 0; i < history.size(); ++i) {
    output += history[i] ? weights[i].value() : -weights[i].value();
  }
  return output;
}

template <std::size_t HISTLEN, std::size_t BITS>
void hybrid::internal_perceptron<HISTLEN, BITS>::update(bool result, std::bitset<HISTLEN> history, int taken_count, int not_taken_count, double B) {
  double eta = B / std::sqrt(taken_count + not_taken_count + 1); // scaled adaptive learning rate
  bias += static_cast<int>(eta * (result ? 1 : -1));

  auto upd_mask = result ? history : ~history;
  for (std::size_t i = 0; i < upd_mask.size(); ++i) {
    weights[i] += static_cast<int>(eta * (upd_mask[i] ? 1 : -1));
  }
}


template <std::size_t HISTLEN, std::size_t BITS>
void hybrid::internal_perceptron<HISTLEN, BITS>::set_bias_from_counters(int taken_count, int not_taken_count, double B) {
  int net_bias = static_cast<int>(std::round(B * (taken_count - not_taken_count)));
  bias = counter_type(net_bias);
}

extern template class hybrid::internal_perceptron<hybrid::PERCEPTRON_HISTORY, hybrid::PERCEPTRON_BITS>;

} // namespace configured
} // namespace champsim

#endif

