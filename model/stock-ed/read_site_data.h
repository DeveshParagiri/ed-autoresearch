#ifndef EDM_READ_SITE_DATA_H_
#define EDM_READ_SITE_DATA_H_

#include "edmodels.h"

class cohort;

////////////////////////////////////////
//    SiteData contains site-related 
//    environmental data
////////////////////////////////////////
struct SiteData {
   size_t x_;
   size_t y_;

   size_t globX_;
   size_t globY_;

   double lat_;
   double lon_;
   
   char name_[STR_LEN];

   bool soi;

   double grid_cell_area;            ///< land area of grid cell  (m^2)
   double grid_cell_area_total;      ///< total area of grid cell (m^2)

#ifdef ED
   double temp[N_CLIMATE];           ///< monthly mean temperature (deg C)
   double precip[N_CLIMATE];         ///< monthly rate of rainfall (mm/yr)
   double soil_temp[N_CLIMATE];      ///< monthly mean soil temp (deg C)
   double soil_temp1[N_CLIMATE];      ///< monthly mean soil temp (deg C) at 1st layer in MERRA2
   double soil_temp2[N_CLIMATE];      ///< monthly mean soil temp (deg C) at 2nd layer in MERRA2
   double soil_temp3[N_CLIMATE];      ///< monthly mean soil temp (deg C) at 3rd layer in MERRA2
   double soil_temp4[N_CLIMATE];      ///< monthly mean soil temp (deg C) at 4th layer in MERRA2
   double soil_temp5[N_CLIMATE];      ///< monthly mean soil temp (deg C) at 5th layer in MERRA2
   double pet[N_CLIMATE];            ///< potential evapotranspiration (mm/yr)
    double dyl_factor[N_CLIMATE];   
    
#endif // ED
   double precip_average;            ///< avg. yrly precip (mm/yr) 
   double temp_average;              ///< average annual temperature (deg C)
   double soil_temp_average;         ///< annual average soil temp (deg C)
   double pet_average;

   double soil_depth;                ///< depth of soil (mm)
   double soil_evap_conductivity;    ///< soil conductivity for evap water loss

   double dryness_index[N_CLIMATE];  ///< used in fire model */
   double dryness_index_avg;         ///< used in fire model 

   double loss_fraction[NUM_TRACKS]; ///< loss fractions following disturbance

#ifdef MIAMI_LU
   double miami_npp;                 ///< annual net primary production (Kg / (m^2 yr)) 
#endif

#ifdef ED
   double N_conc_in_rain;            ///< concentration of N in rain water 
   double L_top;                     ///< Light at the top of the canopy 
   double Rn_top;                    ///< Net Radiative flux at top of canopy

   double theta_max;                 ///< volume of water in saturated soil (mm/mm^3)
   double k_sat;                     ///< conductivity of saturated soil (yr^-1) 
   double tau;                       ///< conductivity exponent
  
#if SOILGRIDS_SCHEME
    double MvG_soil_depth;
    double MvG_k_sat;
    double MvG_theta_saturated;
    double MvG_theta_residual;
    double MvG_L;
    double MvG_m;
#endif
    
   // potential photosynthesis and evap


    /// The following varaibles are used for photosynthesis.cc that couple Farqhuar with stomactal conductance and leaf energy balance model.
    int C4,iter1, iter2,iter_Ci;
    double Vcm25,Jm25,Vpm25,TPU25,Rd25,Theta,EaVc,Eaj,Eap,Ear,EaVp,Hv,Hj,Hr,Hp,Sv,Sj,Sr,Sp,rJ2V,Q10R,g0,g1,stomaRatio,LfWidth,LfAngFact;
    double  PhotoFluxDensity,  //!< Photosynthetic Flux Density (umol photons m-2 s-1
    R_abs, //!< Absorbed incident radiation (watts m-2)
    Tair,  //!< Air temperature at 2m, (C)
    CO2,   //!< CO2 concentration (umol mol-1 air)
    RH,   //!<  Relative Humidity (%, i.e., 80)
    wind, //!<  Windspeed at 2 meters (km s-1)
    width, //!< Leaf width (m)
    Press;  //!<  Air pressure (kPa)
    
    
    double errTolerance; /*!< Error tolerance for iterations */
    double eqlTolerance; /*!< Equality tolerance */
    
    double AssimilationNet,    //!< Net photosynthesis (umol CO2 m-2 s-1)
    AssimilationGross, //!< Gross photosynthesis (umol CO2 m-2 s-1) (Adjusted for respiration)
    Transpiration,     //!< Transpiration mol H2O m-2 s-1
    Tleaf,  //!< Leaf temperature C
    Ci,     //!< Internal CO2 concentration umol mol-1
    StomatalConductance,     //!< Stomatal conductance umol m-2 s-1
    BoundaryLayerConductance,    //!< Boundary layer conductance umol m-2 s-1
    DarkRespiration,    //!< Plant respiration    umol m-2 s-1
    VPD,    //!< Vapor Pressure Density, kPa */
    Ci_Ca,  //!< Ratio of internal to external CO2, unitless
    iter_total;
    ///
    
    bool SetMECHdefault(UserData& data);
    bool farquhar (double Vmax,double CA, double ta, double ts,double ea, double q, double shade, int C4, double outputs[6]);
    bool farquhar_collatz (double Vmax,double CA, double ta, double ea, double q, double shade, int C4, double outputs[5]);
    bool compute_mech(int pt, int spp,double Vm0, int time_period, int light_index, UserData* data);
    
    
    //The following functions are used for photosynthesis.cc that couple Farqhuar with stomactal conductance and leaf energy balance model.
    void Initilize(int pt,int spp,double Tg, UserData* data);
    double minh(double fn1,double fn2,double theta2);
    double Square(double a);
    double Min(double a, double b, double c);
    double Es(double Temperature);
    double Slope(double Temperature);
    double QuadSolnUpper (double a, double b, double c );
    double QuadSolnLower (double a, double b, double c );
    double SearchCi(double CO2i);
    double EvalCi(double Ci);
    double CalcStomatalConductance();
    double CalcTurbulentVaporConductance(void);
    void PhotosynthesisC3(double Ci);
    void PhotosynthesisC4(double Ci);
    void EnergyBalance();
    void GasEx(void);
    void Farquhar_couple(int pt, int spp,UserData* data,double Tair, double Tsoil,double ea, double shortwaveRad, double Tg, double Ca, double wind, double Press, double shade, double Vm25, double outputs[6]);


    double light_levels[NSPECIES][N_LIGHT];
    double An[NSPECIES][N_CLIMATE][N_LIGHT];  ///< pot. net photosynthesis (gC/(m2 mo))
    double E[NSPECIES][N_CLIMATE][N_LIGHT];   ///< potential transpitation (gW/(m2 mo))
    double Anb[NSPECIES][N_CLIMATE][N_LIGHT]; ///< pot. psyn when shut (g/(m2 mo))
    double Eb[NSPECIES][N_CLIMATE][N_LIGHT];  ///< pot. transp when shut (gW/(m2 mo))
    double tf_air[NSPECIES][N_CLIMATE];           ///< value of temp function (used in resp calc)
    double tf_soil[NSPECIES][N_CLIMATE];           ///< value of temp function (used in resp calc)


   // N.A. stuff 
   double first_frost;
   double last_frost;
   double growing_season_length;
#endif // ED 

#if LANDUSE
   double beta[N_LANDUSE_TYPES][N_LANDUSE_TYPES-1][N_LANDUSE_YEARS];

   double vbh[N_VBH_TYPES][N_LANDUSE_YEARS];
   double sbh[N_SBH_TYPES][N_LANDUSE_YEARS];
#endif // LANDUSE 
  
   double avg_hurricane_disturbance_rate;
   double *hurricane_disturbance_rate;
    
#if 0 // Disabled temporarily until fixed
   double ph[N_LANDUSE_YEARS]; ///< prob_harvest(A) (1/yr) 
#endif
    
#if 0 // Disabled temporarily until fixed (if worth it)
   // business as usual 
   double beta_vs_bau;         ///< transistion rates (fr_total_area/yr)
   double beta_vp_bau;            
   double beta_vc_bau;
   double beta_sp_bau;
   double beta_sc_bau;
   double beta_cs_bau;
   double beta_ps_bau;
   double beta_gc_bau;
   double beta_gp_bau;
   double beta_pg_bau;
   double beta_cg_bau;
   double beta_cp_bau;
   double beta_pc_bau;
#endif

   SiteData (size_t y, size_t x, UserData& data);
   bool readSiteData (UserData& data);

 private:

   bool readEnvironmentalData (UserData& data);
#ifdef ED
   bool readMechanismLUT (UserData& data);
   void calcSiteDrynessIndex (UserData& data);
   double calcPETMonthly (size_t month);
#endif
   void calcPETAverage ();
};


bool is_soi(double lat,double lon, UserData& data); ///< flag sites of interest
size_t read_input_data_layers (UserData* data);
int read_hurricane_disturbance (site** siteptr, UserData* data);
char lu2charname2 (int lu);
bool loabGlobalLUData (UserData* data);
bool loadCropCalendar (UserData* data);
bool setCropCalendar (site** firsts, UserData* data);
bool loadGlobalEnvironmentData(UserData* data);
bool loadGlobalMechanismLUT(UserData* data);
bool loadPREMECH (UserData* data);
bool load_GFED(UserData* data);
void freeGlobalEnvironmentData(UserData* data);
bool freeGlobalMechanismLUT(UserData* data);

#endif // EDM_READ_SITE_DATA_H_ 
