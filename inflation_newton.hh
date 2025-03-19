#ifndef INFLATION_NEWTON_HH
#define INFLATION_NEWTON_HH

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <memory>
#include <functional>

using CallbackFunction = std::function<bool(size_t)>;

template<class ISheet>
std::unique_ptr<NewtonOptimizer> get_inflation_optimizer(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(), CallbackFunction = nullptr, Real hessianShift = 0.0, Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6);

template<class ISheet>
ConvergenceReport inflation_newton(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction = nullptr, Real hessianShift = 0.0, Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6);

#endif /* end of include guard: INFLATION_NEWTON_HH */
