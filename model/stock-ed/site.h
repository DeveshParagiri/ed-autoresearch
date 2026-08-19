#ifndef EDM_SITE_H
#define EDM_SITE_H

#include "edmodels.h"

struct SiteData;
struct patch;
struct cohort;

////////////////////////////////////////
//    Typedef: site
////////////////////////////////////////
struct site {

   SiteData* sdata; ///< structure containing site-related data
   UserData* data;
   
   // pointers
   site* next_site; ///< pointer to next site
   patch* youngest_patch[N_LANDUSE_TYPES];
   patch* oldest_patch[N_LANDUSE_TYPES];    
   patch* new_patch[N_LANDUSE_TYPES];

   // flags
   unsigned int skip_site; ///< Stop modelling site (in community_dynamics). Memory not cleared
   unsigned int finished;  ///< if=1, then site is equilibrated
   unsigned int forest_harvest_flag;
   unsigned int maintain_pasture_flag;
   
   int is_tropical_site;
   int climate_zone;
   int is_frozen_cold_decid;
   int is_frozen_evergreen_short;
   int is_frozen_early_succ;
   int is_frozen_mid_succ;
   int is_frozen_late_succ;

   // above ground properties
   double site_total_ag_biomass;             ///< total above gnd biomass (kgC/m2)
   double site_total_biomass;                ///< total plant biomass (kgC)
   double site_total_soil_c;                 ///< total soil carbon at site pools
   double site_total_c;                      ///< site total carbon
   double total_ag_biomass[N_LANDUSE_TYPES]; ///< total above gnd biomass (kgC/m2)
   double total_biomass[N_LANDUSE_TYPES];    ///< total plant biomass (kgC)
   double total_soil_c[N_LANDUSE_TYPES];
   double total_c[N_LANDUSE_TYPES];          ///< total carbon at site pools
   double last_site_total_c;
   double last_site_total_c_last_month;
   double last_total_c[N_LANDUSE_TYPES];
   double last_total_c_last_month[N_LANDUSE_TYPES];

   double site_dndt;
   double dndt[N_LANDUSE_TYPES];

#ifdef ED
   double site_total_spp_biomass[NSPECIES]; ///< spp biomass (kgC/m2)
   double site_total_spp_babove[NSPECIES];  ///<  spp above-gnd biomass (kgC/m2)
   double total_spp_biomass[NSPECIES][N_LANDUSE_TYPES]; ///< spp biomass (kgC/m2)
   double total_spp_babove[NSPECIES][N_LANDUSE_TYPES];  ///<  spp above-gnd biomass (kgC/m2)

   double site_avg_height;
   double site_loreys_height;
   double lu_avg_height[N_LANDUSE_TYPES];
   double lu_loreys_height[N_LANDUSE_TYPES];
   double site_basal_area;               ///< site-level total stem basal area (cm2/m2)
   double basal_area[N_LANDUSE_TYPES];   ///< site-level total stem basal area (cm2/m2)
   double site_basal_area_spp[NSPECIES]; ///< site-level total stem basal area by spp (cm2/m2)
   double basal_area_spp[NSPECIES][N_LANDUSE_TYPES]; ///< site level total stem basal
                                                     ///< area for each spp (cm2/m2)
   double lai[N_LANDUSE_TYPES];                      ///< site-level leaf area index
   double lai_profile[N_LANDUSE_TYPES][N_LAI];       ///LAI profile
#endif
   double site_lai;
   double site_aa_lai;             ///< annual average LAI
   double aa_lai[N_LANDUSE_TYPES]; ///< annual average LAI by land use
    
   double site_lai_profile[N_LAI];   ///LAI profile
   double site_aa_lai_profile[N_LAI];
   double aa_lai_profile[N_LAI];
  
   // cfluxes
   double site_npp;                ///< total npp (kgC/m2/yr)
   double site_npp_avg;
   double site_aa_npp;             ///< annual average npp
   double npp[N_LANDUSE_TYPES];    ///< total npp (kgC/m2/yr)
   double npp_avg[N_LANDUSE_TYPES];    ///< total npp (kgC/m2/yr)
   double aa_npp[N_LANDUSE_TYPES]; ///< annual average npp
   double site_nep;                ///< total nep (kgC/m2/yr)
   double site_aa_nep;             ///< annual average nep
   double nep[N_LANDUSE_TYPES];    ///< total nep (kgC/m2/yr)
   double aa_nep[N_LANDUSE_TYPES]; ///< annual average nep
   double site_nep2;               ///< nep2 = total_c(t) - total_c(t-1)
   double site_nep3;              ///same as site_nep2 but update monthly
   double site_nep3_product;     ///include production emission
   double nep2[N_LANDUSE_TYPES];
   double nep3[N_LANDUSE_TYPES];
   double site_rh;                 ///< carbon loss to atmosphere
   double site_rh_avg;                 ///< carbon loss to atmosphere
   double site_aa_rh;              ///< annual average carbon loss to atmosphere
   double rh[N_LANDUSE_TYPES];     ///< carbon loss to atmosphere
   double rh_avg[N_LANDUSE_TYPES];     ///< carbon loss to atmosphere
   double aa_rh[N_LANDUSE_TYPES];  ///< annual average carbon loss to atmosphere
#ifdef ED
   double site_gpp;
   double site_gpp_avg;
   double site_fs_open;
   double site_aa_gpp;
   double gpp[N_LANDUSE_TYPES];
   double gpp_avg[N_LANDUSE_TYPES];
   double fs_open[N_LANDUSE_TYPES];       
   double aa_gpp[N_LANDUSE_TYPES];
   double site_npp2;                ///< integrated npp over the timestep
   double site_aa_npp2;             ///< annual average npp
   double npp2[N_LANDUSE_TYPES];    ///< integrated npp over the timestep
   double aa_npp2[N_LANDUSE_TYPES]; ///< annual average npp
#endif

   double site_litter;         
   double litter[N_LANDUSE_TYPES];  ///< annual average carbon loss to atmosphere
   double hurricane_litter;
   double site_fire_emission;          ///(kgC/m2/yr)
   double fire_emission[N_LANDUSE_TYPES];   /// (kgC/m2/yr), Carbon loss to atmosphere from fire
#ifdef LANDUSE
   double site_forest_harvest;   /// (kgC/m2/yr), carbon from wood harvest and clearing forest for cropland, pasture and others
   double site_product_emission;  /// (kgC/m2/yr), carbon emitted from 1-year, 10-year, 100-year wood product pools
   double site_yr1_decay_product_pool;
   double site_yr10_decay_product_pool;
   double site_yr100_decay_product_pool;
   double last_site_yr1_decay_product_pool;
   double last_site_yr10_decay_product_pool;
   double last_site_yr100_decay_product_pool;
   double site_pasture_harvest;
   double site_crop_harvest;
   double forest_harvest[N_LANDUSE_TYPES];  /// (kgC/m2/yr), harvested carbon from forest
   double pasture_harvest[N_LANDUSE_TYPES]; /// (kgC/m2/yr), harvested carbon from pasture
   double crop_harvest[N_LANDUSE_TYPES];  /// (kgC/m2/yr), harvested carbon from cropland
   double product_emission[N_LANDUSE_TYPES];   //// (kgC/m2/yr), emission from 1-year, 10-year, 100-year wood product pool
   double yr1_decay_product_pool[N_LANDUSE_TYPES];  /// (kgC/m2/yr), product pool of harvested wood with 1-year decay rate
   double yr10_decay_product_pool[N_LANDUSE_TYPES];  /// (kgC/m2/yr), product pool of harvested wood with 10-year decay rate
   double yr100_decay_product_pool[N_LANDUSE_TYPES];  /// (kgC/m2/yr), product pool of harvested wood with 100-year decay rate
   double site_repro_pool[N_LANDUSE_TYPES];
   double site_total_repro_pool;
   double last_site_total_repro_pool;
#endif


   // water
   double months_below_rain_crit[N_LANDUSE_TYPES]; ///<used in new fire model
#ifdef ED
   double water[N_LANDUSE_TYPES];              ///< site level water fields
   double theta[N_LANDUSE_TYPES];              ///< site level water fields
   double total_water_uptake[N_LANDUSE_TYPES];      
   double total_water_demand[N_LANDUSE_TYPES];
   double perc[N_LANDUSE_TYPES];
   double soil_evap[N_LANDUSE_TYPES]; 
   double site_total_water;                   ///< site level water fields
   double site_total_theta;                   ///< site level water fields
   double site_total_water_uptake;      
   double site_total_water_demand;
   double site_total_perc;
   double site_total_soil_evap;
   
#if SNOWPACK_SCHEME == 1
   double snowpack[N_LANDUSE_TYPES];              ///< snowpack, mm
   double site_total_snowpack;
#endif
    
#endif
  
   // soil pools
   double fast_soil_C[N_LANDUSE_TYPES];        ///< total Fast Soil Carbon (kgC)
   double site_fast_soil_C;                    ///< total Fast Soil Carbon (kgC)
   double structural_soil_C[N_LANDUSE_TYPES];  ///< tot Struct. soil Carbon (kgC)
   double site_structural_soil_C;              ///< tot Struct. soil Carbon (kgC)
#ifdef ED
   double passive_soil_C[N_LANDUSE_TYPES];     ///< total Passive Soil Carbon (kgC)
   double mineralized_soil_N[N_LANDUSE_TYPES]; ///< tot min. soil nitrogen (kgN)
   double fast_soil_N[N_LANDUSE_TYPES];        ///< total fast soil N (KgN)
   double slow_soil_C[N_LANDUSE_TYPES];        ///< total Slow Soil Carbon (kgC)
   double structural_soil_L[N_LANDUSE_TYPES];  ///< total Str Soil Lignin (kg)
   double site_mineralized_soil_N;             ///< tot min. soil nitrogen (kgN)
   double site_fast_soil_N;                    ///< total fast soil N (KgN)
   double site_slow_soil_C;                    ///< total Slow Soil Carbon (kgC)
   double site_passive_soil_C;                 ///< total Passive Soil Carbon (kgC)
   double site_structural_soil_L;              ///< total Str Soil Lignin (kg)
   double site_total_soil_N;                   ///< total soil N
#endif

   // disturbance
   double area_burned;                         ///< total area burned
   double slash;                               ///< biomass of slash kg/m2

#if LANDUSE  
   double area_harvested[2][N_SBH_TYPES];
   double biomass_harvested[2][N_SBH_TYPES];
   double biomass_harvested_unmet[2][N_SBH_TYPES];
#endif
  
   double disturbance_rate[NUM_TRACKS][N_LANDUSE_TYPES];
   double lambda1[N_SUB][N_LANDUSE_TYPES];           ///< site avgd fire rate during the year
   double total_disturbance_rate[N_LANDUSE_TYPES];   ///< site total dist rate
   int fire_flag[N_LANDUSE_TYPES];                   ///< fire occurenace flag
   double fuel[N_LANDUSE_TYPES];                     ///< site level fuel
   double ignition_rate[N_LANDUSE_TYPES];            ///< site level ignition rate
   double site_disturbance_rate[NUM_TRACKS];         ///< site level dist rates by track
   double site_total_disturbance_rate;               ///< site total dist rate
   double mean_AGE_sec;

#ifdef ED
   // profiles of various quantities
   double spp_density_profile[NSPECIES][N_DBH_BINS]; 
   double spp_basal_area_profile[NSPECIES][N_DBH_BINS]; 
   double spp_agb_profile[NSPECIES][N_DBH_BINS];
   double density_profile[N_DBH_BINS]; 
   double basal_area_profile[N_DBH_BINS]; 
   double agb_profile[N_DBH_BINS]; 
  
   // misc
   double *total_spp_biomass_compare; ///< biomass of each species at site (kgC/m2)
                                      ///< updated only every compare freq for use
                                      ///< to check if site is finished equilibrating
   double Rl[5];                      ///< potential leaf resp at 5 levels in canopy, for debugging
#endif

   double time_finished;              ///< the year when the site finished being integrated
  
   double area_fraction[N_LANDUSE_TYPES]; ///< land area in each land use type
   int function_calls;

   void Update_FTS(unsigned int);
   double An[2][101];
   double An_shut[2][101];
   double E[2][101];
   double E_shut[2][101];
   double tf[2];
    
   double Leaf_An_pot[NSPECIES];
   double Leaf_An_shut[NSPECIES];
   double Leaf_E_pot[NSPECIES];
   double Leaf_E_shut[NSPECIES];
   
   double ave_temp[24];
   double dyl_factor[N_CLIMATE];
};


// Function Prototypes
// control functions
void community_dynamics (unsigned int t, double t1, double t2, 
                        site** first_site, UserData* data);
void init_sites (site** firsts, UserData* data);
void update_site (site** siteptr,  UserData* data);
int cm_sodeint (patch** patchptr, int timestep, double x1, double x2, UserData* data);
#endif // EDM_SITE_H_ 
