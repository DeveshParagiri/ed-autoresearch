#ifndef EDM_PATCH_H
#define EDM_PATCH_H

#include "edmodels.h"

////////////////////////////////////////
//    Typedef: patch
////////////////////////////////////////
struct patch {
   double age;           ///< Patch age in years
   double area;          ///< Patch area in m^2
   char old_address[20];
   patch* older;         ///< Pointer to next older patch
   patch* younger;       ///< Pointer to next shorter patch
   site* siteptr;        ///< Pointer to the site that the patch is in
#ifdef ED
   cohort* tallest;                    ///< Pointer to patch's tallest cohort
   cohort* shortest;                   ///< Pointer to patch's shortest cohort
   double total_plant_nitrogen_uptake; ///< @TODO total N uptake by plants (kgN/yr) 
                                       ///< dimensions here don't match up. Not sure 
                                       ///< what wanted (formula has extra m^-2 dimension)
   double water;                       ///< Total plant available water in mm
   double theta;                       ///< Relative water availabiility
   double total_water_uptake;          ///< Total plant water uptake in mm kg/yr
   double total_water_demand;          ///< Total plant water uptake in mm kg/yr
   double perc;                        ///< Rate of percolation mm/yr
   double soil_evap;                   ///< Rate of evapotation mm/yr
    
#if SNOWPACK_SCHEME == 1
   double snowpack;
   double snow_melt;
#endif

   double total_spp_biomass[NSPECIES]; ///< Biomass of each spp in patch (kgC/m2)
   double total_spp_babove[NSPECIES];  ///< Agb of each spp in patch (kgC/m2)
   double basal_area;                  ///< Total stem basal area in patch (cm2/m2)
   double basal_area_spp[NSPECIES];    ///< cm^2/m^2
   double gpp;                         ///< kgC/(yr*m^2)
   double fs_open;
   double aa_gpp;                      ///< kgC/(timestep*m^2)
   double aa_npp2;                     ///< Annual average npp kgC/(timestep*m^2)
#endif
   double aa_lai;           ///< Annual average LAI m^2/m^2
   double aa_lai_profile[N_LAI];
   double total_ag_biomass; ///< Total above gnd biomass in patch (kgC/m2)
   double total_biomass;    ///< Total plant biomass in patch (kgC/m2)
   double npp;              ///< npp kgC/(yr m^2)
   double nep;              ///< nep kgC/(yr m^2)
   double aa_npp;           ///< Annual average npp kgC/(yr m^2)
   double aa_nep;           ///< Annual average nep kgC/(yr m^2)
   double total_soil_c;     ///< Total carbon in soil pools
   double total_c;          ///< Total c
   double rh;               ///< Soil carbon lost to atmosphere kgC/(yr m^2)
   double aa_rh;            ///< Annual average rh
   double litter;           ///< Patch level litter flux KgC/m2/yr
   double npp_avg;
   double gpp_avg;
   double rh_avg;
   double fire_c_loss;   /// Save c loss in Litter() due to fire
   double fire_emission;
#if LANDUSE
   double product_emission;
   double crop_harvested_c;
   double past_harvested_c;
   double forest_harvested_c;
   double yr1_decay_product_pool;      ///< product pool of harvested wood with 1-year decay rate;
   double yr10_decay_product_pool;      ///< product pool of harvested wood with 10-year decay rate;
   double yr100_decay_product_pool;      ///< product pool of harvested wood with 100-year decay rate;
#endif
   
   // disturbance values
   double fuel;
   double ignition_rate;
   double lambda1[N_SUB];     ///< Fire disturbance rate during the year
   double disturbance_rate[NUM_TRACKS];
   double total_disturbance_rate;
   double dndt;
   int treefall_as_dndt_flag; ///< flag to do treefalls as dndt
   int fire_as_dndt_flag;
   double fire_dndt_factor;
    
   unsigned int track;        ///< Disturbance track id. track = 1 if fire was
                              ///< last disturbance, equals 0 otherwise, i.e.
                              ///< if treefall was last disturbance
 
   int fuse_flag; 

   // 1-Box Lignin, 4-Box C, 2-BOX N Century
   // soil carbon and nitrogen pools
   double A;                 ///< decomposition rate scalar    dimensionless
   double fast_soil_C;       ///< Fast soil carbon pool (kg C) kgC/m^2
   double structural_soil_C; ///< Structural soil pool (kg C)  kgC/m^2
#ifdef ED
   double slow_soil_C;       ///< Slow soil carbon pool (kg C) kgC/m^2
   double passive_soil_C;    ///< Passive soil Carbon (kg C) kgC/m^2
   double structural_soil_L; ///< structural soil lignin content (kg) kg/m^2

   double mineralized_soil_N;///< mineralized soil nitrogen kgN/m^2
   double fast_soil_N;       ///< Fast soil nitrogen pool (kg N) kgN/m^2
   double fstd;              ///< fractional rate of structural C decomp

   double lai;               ///< leaf area index of patch dimensionless
   double lai_profile[N_LAI];

   // profiles of various quantities
   double spp_density_profile[NSPECIES][N_DBH_BINS];
   double spp_basal_area_profile[NSPECIES][N_DBH_BINS];
   double spp_agb_profile[NSPECIES][N_DBH_BINS];

   double dwdt; ///< rate of change of available soil water
   double dfsc;   ///< rate of change of fast_soil_C
   double dstsc;  ///< rate of change of structural_soil_C
   double dssc;   ///< rate of change of slow_soil_C
   double dpsc;   ///< rate of change of passive_soil_C
   double dstsl;  ///< rate of change of structural_soil_L
   double dfsn;   ///< rate of change of fast_soil_N
   double dmsn;   ///< rate of change of mineralized_soil_N
   //Copy of derivatives for rk2 step
   double dwdt1;  ///< rate of change of available soil water
   double dfsc1;  ///< rate of change of fast_soil_C
   double dstsc1; ///< rate of change of structural_soil_C
   double dssc1;  ///< rate of change of slow_soil_C
   double dpsc1;  ///< rate of change of passive_soil_C
   double dstsl1; ///< rate of change of structural_soil_L
   double dfsn1;  ///< rate of change of fast_soil_N
   double dmsn1;  ///< rate of change of mineralized_soil_N

   double npp2;   ///< integrated npp over the timestep
   
   double repro[NSPECIES]; ///< array containing seedlings born since last cohort introduced   

   // monthly pool loadings
   double ssc_in; ///< kgC/((m^2) yr) 
   double ssl_in; ///< kg/((m^2) yr) 
   double fsc_in; ///< kgC/((m^2) yr) 
   double fsn_in; ///< kgN/((m^2) yr) 
 
   // current element numbers in integration array 
   int fsc_e; ///< patch fast soil carbon                      
   int fsn_e; ///< patch fast soil N                           

   int okint;
   
   // For rk2 solver
   double old_water;
   double old_fast_soil_C;
   double old_structural_soil_C;
   double old_slow_soil_C;
   double old_mineralized_soil_N;
   double old_fast_soil_N;
   double old_passive_soil_C;
   double old_structural_soil_L; 
#endif // ED

   // landuse
   int landuse; ///< land use index                              

   double *phistory;
   
   // Belowgrnd.c
#if defined ED
   double Dwdt (double time, UserData* data);
   double Soil_Canopy_evap(UserData* data);
    
   void Update_Water(double time, UserData* data, double deltat);
   void Dsdt (unsigned int time_period, double time, UserData* data);
#elif defined MIAMI_LU
      void Dsdt(UserData* data);
#endif 
   // Century: function prototypes of functions called in Dsdt
   double A_function (unsigned int time_period, UserData* data);
   
   // In growth.c
   void Litter(double time, UserData* data);

   // Hydrology.c
   double radiative_flux(UserData* data);
   
   // odeint.c
   void Water_and_Nitrogen_Uptake (unsigned int time_period, double time, UserData* data);
   
   // Ollie's functions for split-step integrator
   bool compare_derivatives(double dt);
   int check_for_negatives(double dt);
   void save_old();
   void load_old();
   void copy_derivatives();
   void load_derivatives();
   int check_quantities();
};



void init_patches (site** siteptr, UserData* data);
#ifdef ED
void create_patch(site** siteptr, patch** pnewp, int landuse,
                  int track, double age, double area, double water, double fsc, 
                  double stsc, double stsl, double ssc, double psc, double msn, 
                  double fsn, UserData* data);

#if LANDUSE
void create_patch_LU (site** siteptr, patch** pnewp, int landuse,
                      int track, double age, double area, double water,
                      double fsc, double stsc, double stsl, double ssc, double psc,
                      double msn, double fsn, double yr1_pool, double yr10_pool, double yr100_pool, UserData* data);
#endif

#elif defined MIAMI_LU
void create_patch (site** siteptr, patch** pnewp, int landuse,
                   int track, double age, double area, double fsc, 
                   double stsc, double tb, UserData* data);
#endif
void update_patch (patch** current_patch, UserData* data);
void patch_dynamics (unsigned int t, patch** patchptr, 
                     FILE* outfile, UserData* data);
void terminate_patches (patch** patchptr, UserData* data);
void terminate_patch (patch** patchptr, UserData* data);
void fuse_patches (unsigned int t, patch** patchptr, UserData* data);
void fuse_2_patches (patch** patchptr1, patch** patchptr2, 
                     short update_ptrs, UserData* data);
void calculate_patch_disturbance_rates (unsigned int t, 
                                        patch** current_patch, 
                                        UserData* data);
void light_levels(patch** patchptr, UserData* data);
void species_patch_size_profile (patch** pcurrentp,
                                 unsigned int nbins, UserData* data);

#if SOILGRIDS_SCHEME
float MvG_func(float water, float k_sat ,float soil_depth, float theta_r, float theta_s, float L, float m, float pipcip);
#endif

#endif // EDM_PATCH_H 
