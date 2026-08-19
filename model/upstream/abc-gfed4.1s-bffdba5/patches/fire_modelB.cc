// ====================================================================
// ED Fire Module Replacement — Model B (Shapley top-5, 17 params)
// Rank #7 on TRENDY v14 ILAMB benchmark (Overall = 0.6943)
//
// Mechanisms: ignition + precip + gpp_monthly + gpp_anom + t_surf + t_air_ign.
// See models/B/formula.md for details.
// ====================================================================

#include <cmath>
#include <algorithm>

namespace ModelB {
  // Ignition double sigmoid
  constexpr double k1        = 0.00014617;
  constexpr double D_low     = 772.006;
  constexpr double k2        = 1.45135e-05;
  constexpr double D_high    = 41258.1;
  constexpr double fire_exp  = 2.74852;
  // Precipitation
  constexpr double P_half          = 153.115;
  constexpr double pre_dampen_half = 10.6367;
  // Monthly GPP hump
  constexpr double gpp_af    = 0.821456;
  constexpr double gpp_b     = 0.103773;
  constexpr double gpp_d     = 3.32092;
  // GPP anomaly
  constexpr double anom_k       = 0.167347;
  constexpr double anom_c       = -0.257934;
  constexpr double fuel_anom_k  = 0.0146931;
  // Monthly air-temp ignition sigmoid
  constexpr double ign_k     = 0.0146306;
  constexpr double ign_c     = -0.718822;
  // Monthly surface soil temp sigmoid
  constexpr double ts_k      = 1.09777;
  constexpr double ts_c      = 17.4963;
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

double compute_fire_risk_modelB(
    double D_bar,        // mm
    double T_air_month,  // °C
    double T_surf_month, // °C
    double P_ann,        // mm/yr
    double P_month,      // mm/month
    double GPP_month,    // kg C / m² / yr
    double GPP_anomaly   // kg C / m² / yr  (monthly - cell climatology mean)
) {
    using namespace ModelB;

    double onset     = sig_pos(D_bar, k1, D_low);
    double suppress  = sig_neg(D_bar, k2, D_high);
    double precip_floor   = P_ann / (P_ann + P_half + 1e-12);
    double precip_dampen  = 1.0 / (1.0 + P_month / (pre_dampen_half + 1e-12));

    double g = gpp_af * GPP_month;
    double gpp_mod = hump(g, gpp_b, gpp_d);

    // GPP anomaly: suppress above-mean, boost fuel when below-mean
    double anom_supp = sig_neg(GPP_anomaly, anom_k, anom_c);
    double neg_anom = std::max(-GPP_anomaly, 0.0);
    double anom_boost = 1.0 - std::exp(-neg_anom / (fuel_anom_k + 1e-12));

    double ts_mod  = sig_pos(T_surf_month, ts_k, ts_c);
    double ign_mod = sig_pos(T_air_month, ign_k, ign_c);

    double product = onset * suppress * precip_floor * precip_dampen
                   * gpp_mod * anom_supp * anom_boost * ts_mod * ign_mod;
    if (product <= 0.0) return 0.0;
    return std::pow(product, fire_exp);
}
