// ====================================================================
// ED Fire Module Replacement — Model A (full 8-mech, 27 params)
// Rank #4 on TRENDY v14 ILAMB benchmark (Overall = 0.6989)
//
// Full mechanism set; reference implementation for Shapley analysis.
// Models B and C drop mechanisms from this per Shapley ranking.
//
// See models/A/formula.md for per-mechanism citations and reasoning.
// ====================================================================

#include <cmath>
#include <algorithm>

namespace ModelA {
  // Ignition double sigmoid
  constexpr double k1        = 0.000171125;
  constexpr double D_low     = 1676.47;
  constexpr double k2        = 0.0491519;
  constexpr double D_high    = 5756.05;
  constexpr double fire_exp  = 1.14233;
  // Fuel hump (AGB + LAI additive)
  constexpr double af        = 0.0553717;
  constexpr double af_lai    = 0.0693086;
  constexpr double fb        = 0.00057502;
  constexpr double fd        = 0.790748;
  // Deep soil temperature
  constexpr double sc2       = 20.5557;
  constexpr double ss2       = 1.55977;
  // T_deep warming-rate sigmoid (d T_deep / dt)
  constexpr double rate_k    = 0.51632;
  constexpr double rate_c    = -4.12199;
  // Precipitation
  constexpr double P_half          = 2164.09;
  constexpr double pre_dampen_half = 88.9106;
  // Height suppression
  constexpr double h_k       = 0.197162;
  constexpr double h_crit    = 3.14506;
  // Monthly GPP hump
  constexpr double gpp_af    = 1.1591;
  constexpr double gpp_b     = 0.0248485;
  constexpr double gpp_d     = 25.8982;
  // GPP anomaly
  constexpr double anom_k       = 15.3348;
  constexpr double anom_c       = 0.993397;
  constexpr double fuel_anom_k  = 0.0157778;
  // Monthly surface soil temp sigmoid
  constexpr double ts_k      = 0.0206993;
  constexpr double ts_c      = 25.8967;
  // Monthly air-temp ignition sigmoid
  constexpr double ign_k     = 0.0118777;
  constexpr double ign_c     = -2.18421;
}

static inline double sig_pos(double x, double k, double c) {
    double arg = std::max(std::min(-k * (x - c), 50.0), -50.0);
    return 1.0 / (1.0 + std::exp(arg));
}
static inline double sig_neg(double x, double k, double c) {
    double arg = std::max(std::min(k * (x - c), 50.0), -50.0);
    return 1.0 / (1.0 + std::exp(arg));
}
static inline double hump(double x, double b, double d) {
    b = std::max(b, 1e-9); d = std::max(d, 1e-9);
    double a1 = std::max(std::min(x / b, 500.0), 0.0);
    double a2 = std::max(std::min(x / d, 500.0), 0.0);
    return (1.0 - std::exp(-a1)) * std::exp(-a2);
}

double compute_fire_risk_modelA(
    double D_bar,         // mm
    double AGB,           // kg C / m² (cLeaf + cWood)
    double LAI_annual,    // m²/m² (annual mean)
    double T_deep,        // °C (deep soil temp)
    double dT_deep_dt,    // °C / month (month-over-month change)
    double T_surf_month,  // °C
    double T_air_month,   // °C
    double P_ann,         // mm/yr
    double P_month,       // mm/month
    double H_natr,        // m (canopy height natural veg)
    double GPP_month,     // kg C / m² / yr
    double GPP_anomaly    // kg C / m² / yr (monthly - cell mean)
) {
    using namespace ModelA;

    // Ignition
    double onset    = sig_pos(D_bar, k1, D_low);
    double suppress = sig_neg(D_bar, k2, D_high);

    // M1: Fuel hump (AGB + LAI)
    double a = af * AGB + af_lai * LAI_annual;
    double fuel_mod = hump(a, fb, fd);

    // M2: Soil temp (deep sigmoid + warming rate)
    double st_deep = sig_pos(T_deep, ss2, sc2);
    double st_rate = sig_pos(dT_deep_dt, rate_k, rate_c);

    // M3: Precipitation
    double precip_floor  = P_ann / (P_ann + P_half + 1e-12);
    double precip_dampen = 1.0 / (1.0 + P_month / (pre_dampen_half + 1e-12));

    // M4: Canopy height suppression
    double h_arg = std::max(std::min(h_k * (H_natr - h_crit), 50.0), -50.0);
    double h_mod = 1.0 / (1.0 + std::exp(h_arg));

    // M5: Monthly GPP hump
    double g = gpp_af * GPP_month;
    double gpp_mod = hump(g, gpp_b, gpp_d);

    // M6: GPP anomaly
    double anom_supp = sig_neg(GPP_anomaly, anom_k, anom_c);
    double neg_anom  = std::max(-GPP_anomaly, 0.0);
    double anom_boost = 1.0 - std::exp(-neg_anom / (fuel_anom_k + 1e-12));

    // M7: Surface soil temp gate
    double ts_mod = sig_pos(T_surf_month, ts_k, ts_c);

    // M8: Monthly air temp ignition sigmoid
    double ign_mod = sig_pos(T_air_month, ign_k, ign_c);

    double product = onset * suppress * fuel_mod
                   * st_deep * st_rate
                   * precip_floor * precip_dampen
                   * h_mod
                   * gpp_mod * anom_supp * anom_boost
                   * ts_mod * ign_mod;
    if (product <= 0.0) return 0.0;
    return std::pow(product, fire_exp);
}
