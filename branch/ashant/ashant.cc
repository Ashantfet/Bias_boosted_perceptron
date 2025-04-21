#include "ashant.h"
#include <cmath>
template class champsim::configured::ashant::internal_perceptron<
    champsim::configured::ashant::PERCEPTRON_HISTORY,
    champsim::configured::ashant::PERCEPTRON_BITS>;
bool champsim::configured::ashant::predict_branch(champsim::address ip)
{
  const auto index = ip.to<uint64_t>() % NUM_PERCEPTRONS;
  const auto output = this->perceptrons[index].predict(this->spec_global_history);

  bool prediction = (output >= 0);

  this->perceptron_state_buf.push_back({ip, prediction, output, this->spec_global_history});
  if (this->perceptron_state_buf.size() > NUM_UPDATE_ENTRIES)
    this->perceptron_state_buf.pop_front();

  this->spec_global_history <<= 1;
  this->spec_global_history.set(0, prediction);
  return prediction;
}

void champsim::configured::ashant::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  auto state = std::find_if(this->perceptron_state_buf.begin(), this->perceptron_state_buf.end(),
                            [ip](auto x) { return x.ip == ip; });

  if (state == this->perceptron_state_buf.end())
    return;

  auto [_ip, prediction, output, history] = *state;
  this->perceptron_state_buf.erase(state);

  this->global_history <<= 1;
  this->global_history.set(0, taken);

  if (prediction != taken)
    this->spec_global_history = this->global_history;

  const auto THETA = std::lround(1.93 * PERCEPTRON_HISTORY + 14);
  const auto index = ip.to<uint64_t>() % NUM_PERCEPTRONS;

  // Update T/N counters
  auto& [T, N] = this->branch_counters[index];
  if (taken)
    ++T;
  else
    ++N;

  if ((std::abs(output) <= THETA) || (prediction != taken)) {
    this->perceptrons[index].update(taken, history, T, N);

    if (this->global_history.count() >= PERCEPTRON_HISTORY) {
      this->perceptrons[index].set_bias_from_counters(T, N, 0.05); // B = 1.0 default

    }
  }
}
