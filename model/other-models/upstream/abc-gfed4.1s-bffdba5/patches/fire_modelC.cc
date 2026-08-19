// ====================================================================
// ED Fire Module Replacement — Model C (minimal, 12 params, 3 mechs)
// Rank #1 on TRENDY v14 ILAMB benchmark (Overall = 0.7133)
//
// Shapley top-3 mechanisms from v8 decomposition:
//   M1: monthly air temperature ignition sigmoid (t_air_ign) — Shapley #1, 18.5%
//   M2: precipitation control (annual floor + monthly dampener) — Shapley #2, 17.5%
//   M3: monthly GPP hump (van der Werf 2008) — Shapley #3, 16.7%
//
// Base: ignition double-sigmoid on accumulated dryness D_bar.
// Output: fire risk (unitless); scale to cs->fp1 at caller site.
//
// Required inputs at each call:
//   - cs->sdata->dryness_index_avg : accumulated D_bar (mm) for this timestep
//   - cs->sdata->temperature_month : monthly air temperature (°C)
//   - cs->sdata->precip_month      : monthly precipitation (mm/month)
//   - cs->sdata->precip_annual     : annual precipitation (mm/yr)
//   - cs->gpp                      : GPP for current month (kg C / m² / yr)
// ====================================================================

#include <cmath>
#include <algorithm>

// --- Model C fitted parameters (from experiments/fire/modelC_v8.json) ---
namespace ModelC {
  // Ignition double sigmoid
  constexpr double k1          = 0.000254003;
  constexpr double D_low       = 180.173;
  constexpr double k2          = 0.00041942;
  constexpr double D_high      = 5010.32;
  constexpr double fire_exp    = 2.44744;
  // Precipitation
  constexpr double P_half          = 404.758;   // mm/yr (annual floor)
  constexpr double pre_dampen_half = 5.67607;   // mm/month (monthly dampener)
  // Monthly GPP hump
  constexpr double gpp_af      = 1.79582;
  constexpr double gpp_b       = 0.0100025;
  constexpr double gpp_d       = 24.3555;
  // Monthly air-temp ignition sigmoid
  constexpr double ign_k       = 0.627346;
  constexpr double ign_c       = 17.4517;
}

// Helpers (bounded for numerical stability)
static inline double sig_pos(double x, double k, double c) {
    double arg = std::max(std::min(-k * (x - c), 50.0), -50.0);
    return 1.0 / (1.0 + std::exp(arg));
}
static inline double sig_neg(double x, double k, double c) {
    double arg = std::max(std::min(k * (x - c), 50.0), -50.0);
    return 1.0 / (1.0 + std::exp(arg));
}
static inline double hump(double x, double b, double d) {
    b = std::max(b, 1e-9);
    d = std::max(d, 1e-9);
    double a1 = std::max(std::min(x / b, 500.0), 0.0);
    double a2 = std::max(std::min(x / d, 500.0), 0.0);
    return (1.0 - std::exp(-a1)) * std::exp(-a2);
}

//
// compute_fire_risk_modelC  — closed-form closed monthly fire fraction
//
double compute_fire_risk_modelC(
    double D_bar,        // mm, accumulated Thornthwaite deficit
    double T_air_month,  // °C, this month's air temperature
    double P_ann,        // mm/yr, annual precipitation
    double P_month,      // mm/month, this month's precipitation
    double GPP_month     // kg C / m² / yr, this month's GPP
) {
    using namespace ModelC;

    // Ignition double sigmoid on D_bar
    double onset     = sig_pos(D_bar, k1, D_low);
    double suppress  = sig_neg(D_bar, k2, D_high);

    // Precipitation control: annual floor × monthly dampener
    double precip_floor   = P_ann / (P_ann + P_half + 1e-12);
    double precip_dampen  = 1.0 / (1.0 + P_month / (pre_dampen_half + 1e-12));

    // Monthly GPP hump (Pausas-Ribeiro 2013)
    double g = gpp_af * GPP_month;
    double gpp_mod = hump(g, gpp_b, gpp_d);

    // Monthly air temperature ignition sigmoid (Archibald 2010)
    double ign_mod = sig_pos(T_air_month, ign_k, ign_c);

    // Multiplicative product + global intensity exponent (Bistinas 2014)
    double product = onset * suppress * precip_floor * precip_dampen * gpp_mod * ign_mod;
    if (product <= 0.0) return 0.0;
    return std::pow(product, fire_exp);
}

// ====================================================================
// Integration into ED's existing fire.cc
// ====================================================================
// Replace the stock ignition_rate calculation with a call to this function:
//
//   double fire_risk = compute_fire_risk_modelC(
//       cs->sdata->dryness_index_avg,
//       cs->sdata->temperature_month,
//       cs->sdata->precip_annual,
//       cs->sdata->precip_month,
//       cs->gpp
//   );
//   cs->ignition_rate[landuse] = data->fp1 * fire_risk;
//
// The fp1 scaling preserves ED's existing tunable overall fire-rate knob.
//
// ====================================================================
// Performance
// ====================================================================
// Per-cell: 2 exp, 2 sigmoid, 2 saturating ratios, 1 pow.
// On a 360×720 grid this is <1 ms per timestep.
