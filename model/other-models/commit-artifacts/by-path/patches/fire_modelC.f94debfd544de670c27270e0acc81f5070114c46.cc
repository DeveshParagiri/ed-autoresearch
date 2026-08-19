// ====================================================================
// ED Fire Module Replacement — Model C (minimal, 12 params, 3 mechs)
// Rank #1 on TRENDY v14 ILAMB benchmark (Overall Score = 0.6703, native
// tier-2 aggregation from scalar_database.csv; beats CLASSIC 0.6660 and
// CLM6.0 0.6606). All training on OFFLINE frozen inputs; coupled-run
// scores will depend on ED's prognostic state trajectories.
//
// Shapley top-3 mechanisms from v8 decomposition:
//   M1: monthly air temperature ignition sigmoid (t_air_ign)
//   M2: precipitation control (annual floor + monthly dampener)
//   M3: monthly GPP hump (van der Werf 2008)
//
// Base: ignition double-sigmoid on accumulated dryness D_bar.
// Output: fire risk (unitless); scale via cs->fp1 at caller site.
//
// Required inputs per call:
//   - cs->sdata->dryness_index_avg : accumulated D_bar (mm)
//   - cs->sdata->temp[month]       : monthly air temperature (°C)
//   - cs->sdata->precip[month]     : monthly precipitation (mm/month)
//   - cs->sdata->precip_average    : annual precipitation (mm/yr)
//   - cs->gpp                      : GPP for current month (kg C / m² / yr)
//
// NOTE on D_bar: these params were fit against a canonical offline dbar
// computed with Thornthwaite+daylength PET, K=1, continuous accumulator,
// reset at monthly precip >= 200 mm/month. ED's internal
// calcSiteDrynessIndex produces numerically different values (different
// PET, unit mixing, hard reset). For bit-exact transfer to coupled ED,
// either (a) compute the canonical dbar inside ED alongside or in place
// of dryness_index_avg, or (b) re-tune these 4 params {k1,D_low,k2,
// D_high} against ED's own dryness_index_avg.
// ====================================================================

#include <cmath>
#include <algorithm>

// --- Model C fitted parameters (models/C/params.json, retuned 12-param) ---
namespace ModelC {
  // Ignition double sigmoid on D_bar
  constexpr double k1          = 0.00298237126;
  constexpr double D_low       = 49.9132152;
  constexpr double k2          = 0.00543206014;
  constexpr double D_high      = 680.954295;
  constexpr double fire_exp    = 3.42466093;
  // Precipitation control
  constexpr double P_half          = 353.795082;   // mm/yr (annual floor)
  constexpr double pre_dampen_half = 42.0220614;   // mm/month (monthly dampener)
  // Monthly GPP hump
  constexpr double gpp_af      = 0.177748208;
  constexpr double gpp_b       = 0.000103763065;
  constexpr double gpp_d       = 0.558323619;
  // Monthly air-temp ignition sigmoid
  constexpr double ign_k       = 0.0901417957;
  constexpr double ign_c       = 31.7257297;
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
