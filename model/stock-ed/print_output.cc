#include <cstring>
#include <cstdio>

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "read_site_data.h"
#include "outputter.h"
#include "math.h"
#ifdef ED
#include "cohort.h"
#endif
#if LANDUSE
#include "landuse.h"
#endif

#include "print_output.h"

float getFunctionCalls(site* cs)     { return cs->function_calls;}
float getDisturbanceRate (site* cs)  { return cs->site_total_disturbance_rate; }
float getSiteAGB (site* cs)          { return cs->site_total_ag_biomass; }
float getSiteBiomass (site* cs)      { return cs->site_total_biomass; }
#ifdef ED
float getSPP0Biomass (site* cs)      { return cs->site_total_spp_biomass[0]; }
float getSPP1Biomass (site* cs)      { return cs->site_total_spp_biomass[1]; }
float getSPP2Biomass (site* cs)      { return cs->site_total_spp_biomass[2]; }
float getSPP3Biomass (site* cs)      { return cs->site_total_spp_biomass[3]; }
float getSPP4Biomass (site* cs)      { return cs->site_total_spp_biomass[4]; }
float getSPP5Biomass (site* cs)      { return cs->site_total_spp_biomass[5]; }
float getSPP6Biomass (site* cs)      { return cs->site_total_spp_biomass[6]; }
float getSPP7Biomass (site* cs)      { return cs->site_total_spp_biomass[7]; }
float getSPP8Biomass (site* cs)      { return cs->site_total_spp_biomass[8]; }
float getSPP9Biomass (site* cs)      { return cs->site_total_spp_biomass[9]; }

float getLeafAnPot_spp1 (site* cs)      { return cs->Leaf_An_pot[0]; }
float getLeafAnPot_spp2 (site* cs)      { return cs->Leaf_An_pot[1]; }
float getLeafAnPot_spp3 (site* cs)      { return cs->Leaf_An_pot[2]; }
float getLeafAnPot_spp4 (site* cs)      { return cs->Leaf_An_pot[3]; }
float getLeafAnPot_spp5 (site* cs)      { return cs->Leaf_An_pot[4]; }
float getLeafAnPot_spp6 (site* cs)      { return cs->Leaf_An_pot[5]; }
float getLeafAnPot_spp7 (site* cs)      { return cs->Leaf_An_pot[6]; }
float getLeafAnShut_spp1 (site* cs)      { return cs->Leaf_An_shut[0]; }
float getLeafAnShut_spp2 (site* cs)      { return cs->Leaf_An_shut[1]; }
float getLeafAnShut_spp3 (site* cs)      { return cs->Leaf_An_shut[2]; }
float getLeafAnShut_spp4 (site* cs)      { return cs->Leaf_An_shut[3]; }
float getLeafAnShut_spp5 (site* cs)      { return cs->Leaf_An_shut[4]; }
float getLeafAnShut_spp6 (site* cs)      { return cs->Leaf_An_shut[5]; }
float getLeafAnShut_spp7 (site* cs)      { return cs->Leaf_An_shut[6]; }
float getLeafEpot_spp1 (site* cs)      { return cs->Leaf_E_pot[0]; }
float getLeafEpot_spp2 (site* cs)      { return cs->Leaf_E_pot[1]; }
float getLeafEpot_spp3 (site* cs)      { return cs->Leaf_E_pot[2]; }
float getLeafEpot_spp4 (site* cs)      { return cs->Leaf_E_pot[3]; }
float getLeafEpot_spp5 (site* cs)      { return cs->Leaf_E_pot[4]; }
float getLeafEpot_spp6 (site* cs)      { return cs->Leaf_E_pot[5]; }
float getLeafEpot_spp7 (site* cs)      { return cs->Leaf_E_pot[6]; }
float getLeafEShut_spp1 (site* cs)      { return cs->Leaf_E_shut[0]; }
float getLeafEShut_spp2 (site* cs)      { return cs->Leaf_E_shut[1]; }
float getLeafEShut_spp3 (site* cs)      { return cs->Leaf_E_shut[2]; }
float getLeafEShut_spp4 (site* cs)      { return cs->Leaf_E_shut[3]; }
float getLeafEShut_spp5 (site* cs)      { return cs->Leaf_E_shut[4]; }
float getLeafEShut_spp6 (site* cs)      { return cs->Leaf_E_shut[5]; }
float getLeafEShut_spp7 (site* cs)      { return cs->Leaf_E_shut[6]; }
// Add 0.0000001 to site_total_biomass in denominator to prevent divide by zero errors
float getPercSPP0    (site* cs)      { return cs->site_total_spp_biomass[0]/(cs->site_total_biomass+0.0000001); }
float getPercSPP1    (site* cs)      { return cs->site_total_spp_biomass[1]/(cs->site_total_biomass+0.0000001); }
float getPercSPP2    (site* cs)      { return cs->site_total_spp_biomass[2]/(cs->site_total_biomass+0.0000001); }
float getPercSPP3    (site* cs)      { return cs->site_total_spp_biomass[3]/(cs->site_total_biomass+0.0000001); }
float getPercSPP4    (site* cs)      { return cs->site_total_spp_biomass[4]/(cs->site_total_biomass+0.0000001); }
float getPercSPP5    (site* cs)      { return cs->site_total_spp_biomass[5]/(cs->site_total_biomass+0.0000001); }
float getPercSPP6    (site* cs)      { return cs->site_total_spp_biomass[6]/(cs->site_total_biomass+0.0000001); }
float getPercSPP7    (site* cs)      { return cs->site_total_spp_biomass[7]/(cs->site_total_biomass+0.0000001); }
float getPercSPP8    (site* cs)      { return cs->site_total_spp_biomass[8]/(cs->site_total_biomass+0.0000001); }
float getPercSPP9    (site* cs)      { return cs->site_total_spp_biomass[9]/(cs->site_total_biomass+0.0000001); }

float getGPP (site* cs)              { return cs->site_gpp; }
float getFS_OPEN (site* cs)          { return cs->site_fs_open; }
float getGPPAA (site* cs)            { return cs->site_aa_gpp;}
#endif
float getLAIAA (site* cs)            { return cs->site_aa_lai;}
float getLAIAA0 (site* cs)           { return cs->site_aa_lai_profile[0];}
float getLAIAA1 (site* cs)           { return cs->site_aa_lai_profile[1];}
float getLAIAA2 (site* cs)           { return cs->site_aa_lai_profile[2];}
float getLAIAA3 (site* cs)           { return cs->site_aa_lai_profile[3];}
float getLAIAA4 (site* cs)           { return cs->site_aa_lai_profile[4];}
float getLAIAA5 (site* cs)           { return cs->site_aa_lai_profile[5];}
float getNPP (site* cs)              { return cs->site_npp; }
float getRH (site* cs)               { return cs->site_rh; }
float getGPP_AVG (site* cs)          { return cs->site_gpp_avg; }
float getNPP_AVG (site* cs)          { return cs->site_npp_avg; }
float getRH_AVG (site* cs)           { return cs->site_rh_avg; }
float getFIRE_EMISSION (site* cs)    { return cs->site_fire_emission;}
#if LANDUSE
float getNEP3_PRODUCT (site* cs)       { return cs->site_nep3_product; }
float getPRODUCT_EMISSION (site* cs)   { return cs->site_product_emission;}
float getYR1_PRODUCT_POOL (site* cs)   { return cs->site_yr1_decay_product_pool;}
float getYR10_PRODUCT_POOL (site* cs)   { return cs->site_yr10_decay_product_pool;}
float getYR100_PRODUCT_POOL (site* cs)   { return cs->site_yr100_decay_product_pool;}
float getFOREST_HARVEST (site* cs)   { return cs->site_forest_harvest;}
float getPASTURE_HARVEST (site* cs)  {return cs->site_pasture_harvest;};
float getCROP_HARVEST (site* cs)     {return cs->site_crop_harvest;};
float getREPRO_POOL (site* cs)     {return cs->site_total_repro_pool;};
#endif

float getNEP (site* cs)              { return cs->site_nep; }
float getNPPAA (site* cs)            { return cs->site_aa_npp;}
float getNEPAA (site* cs)            { return cs->site_aa_nep;}
float getAreaBurned (site* cs)       {
   return cs->area_burned / cs->data->area * cs->sdata->grid_cell_area * KM2_PER_M2;
}
float getDrynessIndex (site* cs)     { return cs->sdata->dryness_index[cs->data->time_period]; }
float getSiteSoilC (site* cs)        { return cs->site_total_soil_c; }
#ifdef ED
float getNPP2 (site* cs)             { return cs->site_npp2; }
float getNEP2 (site* cs)             { return cs->site_nep2; }
float getNEP3 (site* cs)             { return cs->site_nep3; }
float getSiteSoilN (site* cs)        { return cs->site_total_soil_N; }
float getSiteMineralizedN (site* cs) { return cs->site_mineralized_soil_N; }
float getLAI (site* cs)              { return cs->site_lai; }
float getLAI0 (site* cs)             { return cs->site_lai_profile[0]; }
float getLAI1 (site* cs)             { return cs->site_lai_profile[1]; }
float getLAI2 (site* cs)             { return cs->site_lai_profile[2]; }
float getLAI3 (site* cs)             { return cs->site_lai_profile[3]; }
float getLAI4 (site* cs)             { return cs->site_lai_profile[4]; }
float getLAI5 (site* cs)             { return cs->site_lai_profile[5]; }
float getBasalArea (site* cs)        { return cs->site_basal_area; }
float getWater (site* cs)            { return cs->site_total_water; }
float getPercolation (site* cs)      { return cs->site_total_perc; }
float getEvaporation (site* cs)      { return cs->site_total_soil_evap; }

#if SNOWPACK_SCHEME == 1
    float getSnowpack (site* cs)      { return cs->site_total_snowpack; }
#endif

float getTranspiration (site* cs)    { return cs->site_total_water_uptake; }
float getTheta (site* cs)            { return cs->site_total_theta; }
float getMeanHeight (site* cs)       { return cs->site_avg_height; }
float getLoreysHeight (site* cs)     { return cs->site_loreys_height; }
#if WT_Abg_PROFILE
float getAGB0 (site* cs)             { return cs->agb_profile[0]; }
float getAGB1 (site* cs)             { return cs->agb_profile[1]; }
float getAGB2 (site* cs)             { return cs->agb_profile[2]; }
float getAGB3 (site* cs)             { return cs->agb_profile[3]; }
float getAGB4 (site* cs)             { return cs->agb_profile[4]; }
float getAGB5 (site* cs)             { return cs->agb_profile[5]; }
float getAGB6 (site* cs)             { return cs->agb_profile[6]; }
float getAGB7 (site* cs)             { return cs->agb_profile[7]; }
float getAGB8 (site* cs)             { return cs->agb_profile[8]; }
float getAGB9 (site* cs)             { return cs->agb_profile[9]; }
float getAGB10 (site* cs)             { return cs->agb_profile[10]; }
float getAGB11 (site* cs)             { return cs->agb_profile[11]; }
float getAGB12 (site* cs)             { return cs->agb_profile[12]; }
float getAGB13 (site* cs)             { return cs->agb_profile[13]; }
float getAGB14 (site* cs)             { return cs->agb_profile[14]; }
float getAGB15 (site* cs)             { return cs->agb_profile[15]; }
float getAGB16 (site* cs)             { return cs->agb_profile[16]; }
float getAGB17 (site* cs)             { return cs->agb_profile[17]; }
float getAGB18 (site* cs)             { return cs->agb_profile[18]; }
float getAGB19 (site* cs)             { return cs->agb_profile[19]; }
#endif

////////////////////////////////////////////////////////////////////////////////
//! getMaxHeight
//! @TODO: this should be calculated elsewhere
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
float getMaxHeight (site* cs)        {
   float maxh = 0.0;
   patch *cp = cs->youngest_patch[LU_NTRL];
   while (cp != NULL) {
      if (cp->tallest != NULL) {
         if (cp->tallest->hite > maxh)
            maxh = cp->tallest->hite;
      }
      cp = cp->older;
   }
   return maxh;
}

////////////////////////////////////////////////////////////////////////////////
//! getNCohorts
//! @TODO: this should be calculated elsewhere
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
int getNCohorts (site* cs)           {
   int ncohorts = 0;
   patch *cp = cs->oldest_patch[LU_NTRL];
   while (cp != NULL) {
      cohort *cc = cp->shortest;
      while (cc != NULL) {
         ncohorts++;
         cc = cc->taller;
      }
      cp = cp->younger;
   }
   if (ncohorts > 1000) printf("NCOHORTS: %d %s\n", ncohorts, cs->sdata->name_);
   return ncohorts;
}
#endif // ED

////////////////////////////////////////////////////////////////////////////////
//! getNPatches
//! @TODO: this should be calculated elsewhere
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
int getNPatches (site* cs)           {
   int npatches = 0;
   patch *cp = cs->oldest_patch[LU_NTRL];
   while (cp != NULL) {
      npatches++;
      cp = cp->younger;
   }
   return npatches;
}

#if DOES_COMPILE
double getHurricaneLitter (site* cs)  { return cs->hurricane_litter; }
#endif

#if LANDUSE
////////////////////////////////////////////////////////////////////////////////
//! getAreaHarvested
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
float getAreaHarvested (site* cs)    {
   float area_harvested = 0;
   for (int i=0; i<2; i++)
      for (int j=0; j<N_SBH_TYPES; j++)
         area_harvested += cs->area_harvested[i][j];
   return area_harvested;
}
float getSiteLUBiomass (site* cs, size_t lu) { return cs->total_biomass[lu]; }
float getSiteLU_NPP (site* cs, size_t lu)    { return cs->npp[lu]; }
float getSiteLU_NPP_AA (site* cs, size_t lu) { return cs->aa_npp[lu]; }
float getSiteLUSoilC (site* cs, size_t lu)   { return cs->total_soil_c[lu]; }
#ifdef ED
float getSiteLU_GPP (site* cs, size_t lu)    { return cs->gpp[lu]; }
float getSiteLU_FS_OPEN (site* cs, size_t lu)       {return cs->fs_open[lu];}
float getSiteLU_GPP_AA (site* cs, size_t lu) { return cs->aa_gpp[lu]; }
#endif
float getLUFrac (site* cs, size_t lu)        { return cs->area_fraction[lu]; }
#endif

// These are printed once at beginning of run
#ifdef MIAMI_LU
float getMiamiNPP (site* cs)         { return cs->sdata->miami_npp; }
#endif

////////////////////////////////////////////////////////////////////////////////
//! registerOutputVars
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void registerOutputVars(Outputter *o) {
   o->registerVar("Function_calls", &getFunctionCalls, "", -9999.0F, ncFloat);
   o->registerVar("disturbance_rate", &getDisturbanceRate, "", -9999.0F, ncFloat);
   o->registerVar("agb", &getSiteAGB, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass", &getSiteBiomass, "kg/m2", -9999.0F, ncFloat);
#ifdef ED
   o->registerVar("biomass_spp0", &getSPP0Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp1", &getSPP1Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp2", &getSPP2Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp3", &getSPP3Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp4", &getSPP4Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp5", &getSPP5Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp6", &getSPP6Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp7", &getSPP7Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp8", &getSPP8Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("biomass_spp9", &getSPP9Biomass, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp0", &getPercSPP0, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp1", &getPercSPP1, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp2", &getPercSPP2, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp3", &getPercSPP3, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp4", &getPercSPP4, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp5", &getPercSPP5, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp6", &getPercSPP6, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp7", &getPercSPP7, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp8", &getPercSPP8, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("perc_spp9", &getPercSPP9, "kg/m2", -9999.0F, ncFloat);
#endif   
   o->registerVar("aa_LAI", &getLAIAA, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("area_burned", &getAreaBurned, "km2", -9999.0F, ncFloat);
   o->registerVar("dryness_index", &getDrynessIndex, "", -9999.0F, ncFloat);
   o->registerVar("soil_C", &getSiteSoilC, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("NPP", &getNPP, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("NPP_AVG", &getNPP_AVG, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("Rh", &getRH, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("Rh_AVG", &getRH_AVG, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("NEP", &getNEP, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("aa_NPP", &getNPPAA, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("aa_NEP", &getNEPAA, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("fire_emission", &getFIRE_EMISSION, "kg/m2/yr", -9999.0F, ncFloat);
#if LANDUSE
   o->registerVar("NBP_product", &getNEP3_PRODUCT, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("product_emission", &getPRODUCT_EMISSION, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("year1_product_pool", &getYR1_PRODUCT_POOL, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("year10_product_pool", &getYR10_PRODUCT_POOL, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("year100_product_pool", &getYR100_PRODUCT_POOL, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("forest_harvest", &getFOREST_HARVEST, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("pasture_harvest", &getPASTURE_HARVEST, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("crop_harvest", &getCROP_HARVEST, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("repro_pool", &getREPRO_POOL, "kg/m2/yr", -9999.0F, ncFloat);
#endif
    
#ifdef ED
   o->registerVar("GPP", &getGPP, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("GPP_AVG", &getGPP_AVG, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("aa_GPP", &getGPPAA, "kg/m2/yr", -9999.0F, ncFloat);
   o->registerVar("NPP2", &getNPP2, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("NEP2", &getNEP2, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("NEP3", &getNEP3, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("soil_N", &getSiteSoilN, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("mineralized_soil_N", &getSiteMineralizedN, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("LAI", &getLAI, "", -9999.0F, ncFloat);
   o->registerVar("LAI0", &getLAI0, "", -9999.0F, ncFloat);
   o->registerVar("LAI1", &getLAI1, "", -9999.0F, ncFloat);
   o->registerVar("LAI2", &getLAI2, "", -9999.0F, ncFloat);
   o->registerVar("LAI3", &getLAI3, "", -9999.0F, ncFloat);
   o->registerVar("LAI4", &getLAI4, "", -9999.0F, ncFloat);
   o->registerVar("LAI5", &getLAI5, "", -9999.0F, ncFloat);
   o->registerVar("max_height", &getMaxHeight, "m", -9999.0F, ncFloat);
   o->registerVar("basal_area", &getBasalArea, "", -9999.0F, ncFloat);
   o->registerVar("water", &getWater, "", -9999.0F, ncFloat);
   o->registerVar("theta", &getTheta, "", -9999.0F, ncFloat);
   o->registerVar("percolation", &getPercolation, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("evaporation", &getEvaporation, "kg/m2", -9999.0F, ncFloat);
   o->registerVar("transpiration", &getTranspiration, "kg/m2", -9999.0F, ncFloat);
#if SNOWPACK_SCHEME == 1
   o->registerVar("snowpack", &getSnowpack, "", -9999.0F, ncFloat);
#endif
   o->registerVar("n_cohorts", &getNCohorts, "", -99, ncInt);
   o->registerVar("mean_height", &getMeanHeight, "m", -9999.0F, ncFloat);
   o->registerVar("Loreys_height", &getLoreysHeight, "m", -9999.0F, ncFloat);
    
#if WT_Abg_PROFILE
   o->registerVar("AGB0", &getAGB0, "", -9999.0F, ncFloat);
   o->registerVar("AGB1", &getAGB1, "", -9999.0F, ncFloat);
   o->registerVar("AGB2", &getAGB2, "", -9999.0F, ncFloat);
   o->registerVar("AGB3", &getAGB3, "", -9999.0F, ncFloat);
   o->registerVar("AGB4", &getAGB4, "", -9999.0F, ncFloat);
   o->registerVar("AGB5", &getAGB5, "", -9999.0F, ncFloat);
   o->registerVar("AGB6", &getAGB6, "", -9999.0F, ncFloat);
   o->registerVar("AGB7", &getAGB7, "", -9999.0F, ncFloat);
   o->registerVar("AGB8", &getAGB8, "", -9999.0F, ncFloat);
   o->registerVar("AGB9", &getAGB9, "", -9999.0F, ncFloat);
   o->registerVar("AGB10", &getAGB10, "", -9999.0F, ncFloat);
   o->registerVar("AGB11", &getAGB11, "", -9999.0F, ncFloat);
   o->registerVar("AGB12", &getAGB12, "", -9999.0F, ncFloat);
   o->registerVar("AGB13", &getAGB13, "", -9999.0F, ncFloat);
   o->registerVar("AGB14", &getAGB14, "", -9999.0F, ncFloat);
   o->registerVar("AGB15", &getAGB15, "", -9999.0F, ncFloat);
   o->registerVar("AGB16", &getAGB16, "", -9999.0F, ncFloat);
   o->registerVar("AGB17", &getAGB17, "", -9999.0F, ncFloat);
   o->registerVar("AGB18", &getAGB18, "", -9999.0F, ncFloat);
   o->registerVar("AGB19", &getAGB19, "", -9999.0F, ncFloat);
#endif
#endif
   o->registerVar("n_patches", &getNPatches, "", -99, ncInt);
#if DOES_COMPILE
   o->registerVar("hurricane_litter", &getHurricaneLitter, "kg/m2", -9999.0F, ncFloat);
#endif
#if LANDUSE
   o->registerVar("area_harvested", &getAreaHarvested, "", -9999.0F, ncFloat);
   o->registerLUVar("biomass", &getSiteLUBiomass, "kg/m2", -9999.0F, ncFloat);
   o->registerLUVar("NPP", &getSiteLU_NPP, "kg/m2", -9999.0F, ncFloat);
   o->registerLUVar("aa_NPP", &getSiteLU_NPP_AA, "kg/m2", -9999.0F, ncFloat);
#ifdef ED
   o->registerLUVar("GPP", &getSiteLU_GPP, "kg/m2", -9999.0F, ncFloat);
   o->registerLUVar("aa_GPP", &getSiteLU_GPP_AA, "kg/m2", -9999.0F, ncFloat);
#endif
   o->registerLUVar("frac", &getLUFrac, "", -9999.0F, ncFloat);
   o->registerLUVar("soil_C", &getSiteLUSoilC, "", -9999.0F, ncFloat);
#endif

}


FILE* open_landuse_file (unsigned int t, int lu, const char* fname, int bypatch,
                         site** siteprt, UserData* data);

void lu_shortname (int lu, char* name);
void lu_longname (int lu, char* name);

////////////////////////////////////////////////////////////////////////////////
//! lu_shortname
//! name is returned in char* name parameter
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void lu_shortname (int lu, char* name) {
   switch (lu) {
      case LU_NTRL :
         strcpy(name, "natr");
         break;
      case LU_SCND :
         strcpy(name, "scnd");
         break;
      case LU_CROP :
         strcpy(name, "crop");
         break;
      case LU_PAST :
         strcpy(name, "past");
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
//! lu_longname
//! name is returned in char* name parameter
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void lu_longname (int lu, char* name) {
   switch (lu) {
      case LU_NTRL :
         strcpy(name, "natural");
         break;
      case LU_SCND :
         strcpy(name, "secondary");
         break;
      case LU_CROP :
         strcpy(name, "crop");
         break;
      case LU_PAST :
         strcpy(name, "pasture");
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
//! print_initial
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_initial (site* first_site, UserData* data) {
   print_site_chars(&first_site, data);

   if(!data->is_site) {

#ifdef MIAMI_LU
      //data->outputter->outputSingle ("miami_npp", &getMiamiNPP
      //                               "kg/m2", -999, ncFloat, first_site);
#endif
   } /* REGION */
}

////////////////////////////////////////////////////////////////////////////////
//! print_region_files
//! REGIONAL PRINTING ROUTINES
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_region_files (unsigned int t, site** firsts, UserData* data) {

   print_domain_stats(t, firsts, data);

   if(!data->is_site) {
      /* if(((t >= n*PRINTFREQ) && (t < n*PRINTFREQ + N_CLIMATE ))){*/
      if (t % PRINTFREQ == 0) {
         data->outputter->outputAll(*firsts);
      } 
   } /*REGION*/  
}


////////////////////////////////////////////////////////////////////////////////
//! print_soi_files
//! PRINT SITE OF INTEREST (SOI) FILES
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_soi_files (unsigned int t, site** current_site,
                      UserData* data) {

   site* cs;
#ifdef ED
   size_t n;
#endif

   cs = *current_site;

#if defined ED
   n = t / ((int)(PRINTFREQ));
   if ( (t >= n * PRINTFREQ) && (t < n * PRINTFREQ + N_CLIMATE) ) {

#elif defined MIAMI_LU
   if (t % PRINTFREQ == 0) {
#endif

      if (cs->sdata->soi) {
         print_biomass(t, current_site, data);
         print_patches(t, current_site, data);
         /*print_diagnostics(t, current_site, data); */
         print_cfluxes(t, current_site, data);  
         print_soil_pools(t, current_site, data);
         print_area_burned(t, current_site, data);
#ifdef ED
         print_water(t, current_site, data);
         print_light_levels(current_site, t ,data);     
         print_cohorts(t, current_site, data);
         print_nitrogen_budget(t, current_site, data);
#endif
#if LANDUSE
         print_landuse(t, current_site, data);
         print_harvest(t, current_site, data);
#endif  
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
//! open_landuse_file
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
FILE* open_landuse_file (unsigned int t, int lu, const char* fname, int bypatch,
                         site** siteptr, UserData* data) {
   site* cs;
   char luname[STR_LEN];
   char filename[STR_LEN];
   char prefix[STR_LEN];
   FILE* outfile;
  
   cs = *siteptr;

   if (bypatch)
      strcpy(prefix, "patch.");
   else
      strcpy(prefix, "");
   lu_longname(lu, luname);
   if (bypatch)
      sprintf(filename, "%s.%s.%s.patch.%s", data->base_filename, cs->sdata->name_, luname, fname);
   else
      sprintf(filename, "%s.%s.%s.%s", data->base_filename, cs->sdata->name_, luname, fname);

   if (bypatch && ! data->long_patch_file) {
      outfile = fopen(filename, "w");
   } else {
      if (t == 0)
         outfile = fopen(filename, "w");
      else
         outfile = fopen(filename, "a");
   }

   return outfile;
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! print_light_levels
//! print light levels within patches
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_light_levels (site** siteptr, unsigned int time,
                         UserData* data){

   site* cs;
   patch *cp;
   cohort *cc;
   FILE *outfile;
   int lu;

   cs = *siteptr;

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {

      outfile = open_landuse_file(time, lu, "light", 1, siteptr, data);
 
      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
         cc = cp->shortest;
         while (cc != NULL) {
            fprintf(outfile,
                    "%s time %f pid %p track %u age %3.2f area %f spp %u b %f n %f hite %f light %f\n",
                    cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->age, cp->area,
                    cc->species, cc->b, cc->nindivs, cc->hite, cc->lite);
            cc = cc->taller;
         }
         cp=cp->older;
      }
      fclose(outfile);
   }
}

////////////////////////////////////////////////////////////////////////////////
//! print_water
//! print, to file, distribution of water at sites
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_water (unsigned int time, site** siteptr, UserData* data) {

   site* cs;
   patch *cp;
   FILE *outfile; 
   char filename[STR_LEN];
   int lu;

   cs = *siteptr;

   /**** ALL LANDS ***/
   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);
   strcat(filename,".water");

   if(time==0)
      outfile=fopen(filename,"w");
   else
      outfile=fopen(filename,"a");
   
   fprintf(outfile,
           "%s t= %f theta= %f w= %f rain= %f uptake= %f demand= %f perc= %f soil_evap= %f pet= %f\n",
           cs->sdata->name_, time*TIMESTEP, cs->site_total_theta, cs->site_total_water,
           cs->sdata->precip[data->time_period], cs->site_total_water_uptake,
           cs->site_total_water_demand, cs->site_total_perc,
           cs->site_total_soil_evap, cs->sdata->pet[data->time_period]);

   fclose(outfile);   

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {

      outfile = open_landuse_file(time, lu, "water", 0, siteptr, data);
 
      /* print to files */
      fprintf(outfile,
              "%s t= %f theta= %f w= %f rain= %f uptake= %f demand= %f perc= %f soil_evap= %f pet= %f\n",
              cs->sdata->name_, time*TIMESTEP, cs->theta[lu], cs->water[lu],
              cs->sdata->precip[data->time_period], cs->total_water_uptake[lu],
              cs->total_water_demand[lu], cs->perc[lu], cs->soil_evap[lu],
              cs->sdata->pet[data->time_period]);
 
      fclose(outfile);
 
      outfile = open_landuse_file(time, lu, "water", 1, siteptr, data);
      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
          fprintf(outfile,
                  "%s t= %f pid= %p track= %u age= %f theta= %f w= %f rain= %f snowpack= %f snow_melt= %f uptake= %f demand= %f perc= %f lu= %d\n",
                  cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->age,
                  cp->theta, cp->water, cs->sdata->precip[data->time_period],cp->snowpack,cp->snow_melt,
                  cp->total_water_uptake / cp->area,
                  cp->total_water_demand / cp->area,
                  cp->perc, cp->landuse);

         cp = cp->older;
      }
      fclose(outfile);
   }
}
#endif /* ED */

////////////////////////////////////////////////////////////////////////////////
//! print_cfluxes
//! new print, npp, nep at patch and site levels etc..
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_cfluxes (unsigned int time, site** siteptr, UserData* data) {
   patch *cp;
   site* cs;
   FILE *outfile;
   char luname[STR_LEN];
   char filename[STR_LEN];
   int lu;

   cs = *siteptr;

   /*MONTHLY CFLUXES*/
   /* site file */
   strcpy(filename, data->base_filename);
   strcat(filename, ".");
   strcat(filename, cs->sdata->name_);
   strcat(filename, ".cfluxes");

   if (time == 0)
      outfile = fopen(filename,"w");
   else
      outfile = fopen(filename,"a");
 
#ifdef ED
    
#ifdef LANDUSE
    fprintf(outfile,
            "%s time %f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f dndt %f nep3 %f nep3_product %f gpp_avg %f npp_avg %.20f rh_avg %.20f fire_emission %.15f forest_harvest %.15f pasture_harvest %.15f crop_harvest %.15f 1yr_pool %.15f 10_yr_pool %.15f 100_yr_pool %.15f product_emission %.15f\n",
            cs->sdata->name_, time*TIMESTEP, cs->site_npp, cs->site_rh, cs->site_nep,
            cs->site_npp2, cs->site_gpp, cs->site_dndt,cs->site_nep3,cs->site_nep3_product,cs->site_gpp_avg,cs->site_npp_avg,cs->site_rh_avg,cs->site_fire_emission,cs->site_forest_harvest,cs->site_pasture_harvest,cs->site_crop_harvest,cs->site_yr1_decay_product_pool,cs->site_yr10_decay_product_pool,cs->site_yr100_decay_product_pool,cs->site_product_emission);
#else
    fprintf(outfile,
            "%s time %f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f dndt %f nep3 %f gpp_avg %f npp_avg %.20f rh_avg %.20f fire_emission %.15f\n",
            cs->sdata->name_, time*TIMESTEP, cs->site_npp, cs->site_rh, cs->site_nep,
            cs->site_npp2, cs->site_gpp, cs->site_dndt,cs->site_nep3,cs->site_gpp_avg,cs->site_npp_avg,cs->site_rh_avg,cs->site_fire_emission);
#endif
    

#elif defined MIAMI_LU
   fprintf(outfile,
           "%s time %f npp %8.6f rh %8.6f nep %8.6f dndt %f\n",
           cs->sdata->name_, time*TIMESTEP, cs->site_npp, cs->site_rh, cs->site_nep,
           cs->site_dndt);
#endif 
   fclose(outfile);

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      /* total biomass by landuse */
      outfile = open_landuse_file(time, lu, "cfluxes", 0, siteptr, data);

      /* print to files */
#if defined ED
      fprintf(outfile, "%s time %f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f dndt %f nep3 %f gpp_avg %8.6f npp_avg %8.6f rh_avg %8.6f\n",
              cs->sdata->name_, time * TIMESTEP,
              cs->npp[lu], cs->rh[lu] ,cs->nep[lu],
              cs->npp2[lu], cs->gpp[lu], cs->dndt[lu],cs->nep3[lu],cs->gpp_avg[lu],cs->npp_avg[lu],cs->rh_avg[lu]);
#elif defined MIAMI_LU
      fprintf(outfile, "%s time %f npp %8.6f rh %8.6f nep %8.6f dndt %f\n",
          cs->sdata->name_, time * TIMESTEP,
          cs->npp[lu], cs->rh[lu], cs->nep[lu],
          cs->dndt[lu]);
#endif

      fclose(outfile);

      outfile = open_landuse_file(time, lu, "cfluxes", 1, siteptr, data);

      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
#if defined ED
         fprintf(outfile,
                 "%s time %f pid %p track %u area %f age %5.2f npp %8.6f rh %8.6f  nep %8.6f npp2 %8.6f gpp %8.6f npp_avg %8.6f rh_avg %8.6f\n",
                 cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->area, cp->age,
                 cp->npp, cp->rh, cp->nep, cp->npp2, cp->gpp,cp->npp_avg,cp->rh_avg);
#elif defined MIAMI_LU
         fprintf(outfile,
                 "%s time %f pid %p track %u area %f age %f npp %8.6f rh %8.6f nep %8.6f\n" ,
                 cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->area, cp->age,
                 cp->npp, cp->rh, cp->nep); 
#endif
         cp = cp->older;    
      }  /* end loop over patches */
 
      fclose(outfile);
   }
 
   /***********************/
   /*AVERAGE ANNUAL FLUXES*/
   /***********************/

   /*only print the annual average once a year at the end of the year*/
   if (data->time_period == (N_CLIMATE-1)) {

      strcpy(filename, data->base_filename);
      strcat(filename, ".");
      strcat(filename, cs->sdata->name_);
      strcat(filename, ".aa.cfluxes");
   
      if (time == N_CLIMATE-1)
         outfile = fopen(filename, "w");
      else
         outfile = fopen(filename, "a");

      /* print site level annual averages to file */
#if defined ED
      fprintf(outfile,
              "%s time %6.2f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f nep2 %8.6f\n",
              cs->sdata->name_, time*TIMESTEP, cs->site_aa_npp, cs->site_aa_rh, cs->site_aa_nep,
              cs->site_aa_npp2, cs->site_aa_gpp, cs->site_nep2);
#elif defined MIAMI_LU
      fprintf(outfile,
              "%s time %f npp %8.6f rh %8.6f nep %8.6f nep2 %8.6f\n",
              cs->sdata->name_, time*TIMESTEP, cs->site_aa_npp, cs->site_aa_rh,
              cs->site_aa_nep, cs->site_nep2);
#endif
      fclose(outfile);

      for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
         /* total biomass by landuse */
         lu_longname(lu, luname);
         strcpy(filename, data->base_filename);
         strcat(filename, ".");
         strcat(filename, cs->sdata->name_);
         strcat(filename, ".aa.cfluxes.");
         strcat(filename, luname);
        
         if (time == N_CLIMATE - 1)
            outfile = fopen(filename, "w");
         else
            outfile = fopen(filename, "a");
 
#if defined ED
         fprintf(outfile,
                 "%s time %6.2f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f nep2 %8.6f\n",
                 cs->sdata->name_, time*TIMESTEP, cs->aa_npp[lu],  cs->aa_rh[lu], cs->aa_nep[lu],
                 cs->aa_npp2[lu], cs->aa_gpp[lu], cs->nep2[lu]);
#elif defined MIAMI_LU
         fprintf(outfile,
                 "%s time %f npp %8.6f rh %8.6f nep %8.6f nep2 %8.6f\n",
                 cs->sdata->name_, time*TIMESTEP, cs->aa_npp[lu], cs->aa_rh[lu],
                 cs->aa_nep[lu], cs->nep2[lu]);
#endif

         fclose(outfile);

         strcpy(filename, data->base_filename);
         strcat(filename, ".");
         strcat(filename, cs->sdata->name_);
         strcat(filename, ".patch.aa.cfluxes.");
         strcat(filename, luname);
 
         if(data->long_patch_file) {
            if (time == N_CLIMATE - 1)
               outfile = fopen(filename, "w");
            else
               outfile = fopen(filename, "a");
         } else {
            outfile = fopen(filename, "w");
         }
     
         cp = cs->youngest_patch[lu];
         while (cp != NULL) {
#if defined ED
            fprintf(outfile,
                    "%s time %6.2f pid %p track %u area %f age %5.2f npp %8.6f rh %8.6f nep %8.6f npp2 %8.6f gpp %8.6f\n" ,
                    cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->area, cp->age,
                    cp->aa_npp, cp->aa_rh, cp->aa_nep, cp->aa_npp2, cp->aa_gpp);
#elif defined MIAMI_LU
            fprintf(outfile,
                    "%s time %f pid %p track %u area %f age %f npp %8.6f rh %8.6f nep %8.6f\n" ,
                    cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->area, cp->age,
                    cp->aa_npp, cp->aa_rh, cp->aa_nep); 
#endif
            cp = cp->older;    
         }  /* end loop over patches */
 
         fclose(outfile);
      } /* end loop over landuse */
   } /* end if on time period */
}

////////////////////////////////////////////////////////////////////////////////
//! print_soil_pools
//! print Soil Carbon pools
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_soil_pools (unsigned int time, site** siteptr,
                       UserData* data) {

   /*SC pools summed over all patches*/
   patch *cp;
   site* cs;
   FILE *outfile;
   char filename[STR_LEN];
   int lu;

   cs =* siteptr;
 
   /************/
   /*SITE TOTAL*/
   /************/

   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);
   strcat(filename,".sc");
   
   if (time == 0)
      outfile = fopen(filename, "w");
   else
      outfile = fopen(filename, "a");

   /* print to files */
#if defined ED
   fprintf(outfile,
           "%s t= %f dndt %8.6f total_soil_c %8.6f litter %8.6f rh %8.6f fsc %8.6f stsc %8.6f stsl %8.6f ssc %8.6f psc %8.6f fsn  %8.6f msn  %8.6f ",
           cs->sdata->name_, time*TIMESTEP, cs->site_dndt, cs->site_total_soil_c, cs->site_litter, cs->site_rh, cs->site_fast_soil_C,
           cs->site_structural_soil_C, cs->site_structural_soil_L, cs->site_slow_soil_C,
           cs->site_passive_soil_C, cs->site_fast_soil_N, cs->site_mineralized_soil_N);
#elif defined MIAMI_LU
   fprintf(outfile,
           "%s t= %f total_soil_c %f fsc %f stsc %f ",
           cs->sdata->name_, time*TIMESTEP, cs->site_total_soil_c,
           cs->site_fast_soil_C, cs->site_structural_soil_C);
#endif
   if(data->do_hurricane) {
      fprintf(outfile, "hurr_litter %8.6f\n", cs->hurricane_litter);
   } else {
      fprintf(outfile, "\n");
   }
 
   fclose(outfile);
 
   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      outfile = open_landuse_file(time, lu, "sc", 0, siteptr, data);

#if defined ED
      fprintf(outfile,
              "%s t= %f total_soil_c %8.6f fsc %8.6f stsc %8.6f stsl %8.6f ssc %8.6f psc %8.6f fsn  %8.6f msn  %8.6f \n",
              cs->sdata->name_, time*TIMESTEP, cs->total_soil_c[lu], cs->fast_soil_C[lu],
              cs->structural_soil_C[lu], cs->structural_soil_L[lu], cs->slow_soil_C[lu],
              cs->passive_soil_C[lu], cs->fast_soil_N[lu], cs->mineralized_soil_N[lu]);
#elif defined MIAMI_LU
      fprintf(outfile,
              "%s t= %f total_soil_c %f fsc %f stsc %f\n",
              cs->sdata->name_, time*TIMESTEP, cs->total_soil_c[lu],
              cs->fast_soil_C[lu], cs->structural_soil_C[lu]);
#endif

      fclose(outfile);

      outfile = open_landuse_file(time, lu, "sc", 1, siteptr, data);

      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
#if defined ED
         fprintf(outfile,
                 "%s t= %f pid %p landuse %d area %f age %5.2f soil_total_c %8.6f A %6.3f fstd %6.3f fsc %8.6f stsc %8.6f stsl %8.6f ssc %8.6f psc %8.6f fsn  %8.6f msn  %8.6f \n",
                 cs->sdata->name_, time*TIMESTEP, cp, cp->landuse, cp->area, cp->age,
                 cp->total_soil_c, cp->A, cp->fstd, cp->fast_soil_C, cp->structural_soil_C,
                 cp->structural_soil_L, cp->slow_soil_C, cp->passive_soil_C,
                 cp->fast_soil_N, cp->mineralized_soil_N); 
#elif defined MIAMI_LU
         fprintf(outfile,
                 "t= %f pid %p landuse %d area %f age %f total_soil_c %f fsc %f stsc %f\n",
                 time*TIMESTEP, cp, cp->landuse, cp->area, cp->age,
                 cp->total_soil_c, cp->fast_soil_C, cp->structural_soil_C);
#endif
       
         cp = cp->older;    
      }  /* end loop over patches */
      fclose(outfile);
   } /* end loop over landuse */
}

////////////////////////////////////////////////////////////////////////////////
//! print_biomass
//! print plant density and biomass by spp. summed over patches
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_biomass (unsigned int time, site** siteptr, UserData* data) {
   site* cs;
   patch *cp;
   FILE *outfile;
   char filename[STR_LEN];
   int lu;
#ifdef ED
   unsigned int i;
   double tallest;
#endif
 
   cs = *siteptr;

   /* biomass for all lands */
   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);
   strcat(filename,".biomass");

   if (time == 0)
      outfile = fopen(filename,"w");
   else
      outfile = fopen(filename,"a");

   /* print to files */
   cp = *cs->oldest_patch;
#if defined ED
   fprintf(outfile, "%s t= %5.5f total_c %.20f tb %8.5f tagb %8.5f ba %8.5f lai %8.5f havg %8.5f 1yr_pool %.15f 10_yr_pool %.15f 100_yr_pool %.15f repro_pool %.15f ",
           cs->sdata->name_,
           time * TIMESTEP,
           cs->site_total_c,
           cs->site_total_biomass,
           cs->site_total_ag_biomass,
           cs->site_basal_area,
           cs->site_lai,
           cs->site_avg_height,
           cs->site_yr1_decay_product_pool,
           cs->site_yr10_decay_product_pool,
           cs->site_yr100_decay_product_pool,
           cs->site_total_repro_pool);
  
   for (i=0; i<NSPECIES; i++) {
      fprintf(outfile, "spp %d b %8.6f agb %8.6f ba %8.6f ",
              i,
              cs->site_total_spp_biomass[i],
              cs->site_total_spp_babove[i],
              cs->site_basal_area_spp[i]);
   }
   fprintf(outfile,"\n");
#elif defined MIAMI_LU
   fprintf(outfile, "%s t= %f total_c %8.5f tb %8.5f tagb %8.5f \n",
           cs->sdata->name_, time*TIMESTEP, cs->site_total_c,
           cs->site_total_biomass, cs->site_total_ag_biomass);
#endif

   fclose(outfile);
 

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      outfile = open_landuse_file(time, lu, "biomass", 0, siteptr, data);

#if defined ED
      fprintf(outfile, "%s t= %5.5f total_c %8.5f tb %8.5f tagb %8.5f ba %8.5f lai %8.5f ",
              cs->sdata->name_,
              time * TIMESTEP,
              cs->total_c[lu],
              cs->total_biomass[lu],
              cs->total_ag_biomass[lu],
              cs->basal_area[lu],
              cs->lai[lu]);
      for (i=0; i<NSPECIES; i++) {
         fprintf(outfile, "spp %d b %8.6f agb %8.6f ba %8.6f ",
                 i,
                 cs->total_spp_biomass[i][lu],
                 cs->total_spp_babove[i][lu],
                 cs->basal_area_spp[i][lu]);
      }
      fprintf(outfile, "\n");
#elif defined MIAMI_LU
      fprintf(outfile, "%s t= %f total_c %8.5f tb %8.5f tagb %8.5f ",
              cs->sdata->name_, time*TIMESTEP, cs->total_c[lu],
              cs->total_biomass[lu], cs->total_ag_biomass[lu]);
#endif

      fclose(outfile);

      outfile = open_landuse_file(time, lu, "biomass", 1, siteptr, data);
 
      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
#if defined ED
         /* insulate against empty patches */
         if (cp->tallest == NULL)
            tallest = 0.0;
         else
            tallest = cp->tallest->hite;
         fprintf(outfile, "%s t= %5.2f pid= %p ar= %e age= %5.2f trk= %u tagb= %7.4f barea= %7.4f hmax= %7.4f lai= %7.4f\t",
                 cs->sdata->name_,
                 time * TIMESTEP,
                 cp,
                 cp->area,
                 cp->age,
                 cp->track,
                 cp->total_ag_biomass,
                 cp->basal_area,
                 tallest,
                 cp->lai);
         for (i=0; i<NSPECIES; i++) {
            fprintf(outfile, "s%d tb= %8.5f ba= %8.5f ",
                    i,
                    cp->total_spp_biomass[i],
                    cp->basal_area_spp[i]);
         }
         fprintf(outfile,"\n");
#elif defined MIAMI_LU
         fprintf(outfile, "%s t= %f pid= %p ar= %e age= %f trk= %u tagb= %7.4f\n",
                 cs->sdata->name_, time*TIMESTEP, cp, cp->area, cp->age, cp->track, cp->total_ag_biomass);
#endif
         cp = cp->older;
      }  /* end loop over patches */
 
      fclose(outfile);
   }
}

#if LANDUSE
////////////////////////////////////////////////////////////////////////////////
//! print_harvest
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
/******************************************************************************/
void print_harvest (unsigned int time, site** siteptr, UserData* data) {
   site* cs;
   FILE *outfile;
   char filename[STR_LEN];
   double area;
  
   cs = *siteptr;
  
   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);

   strcat(filename,".harvest");
 
   if (time == 0)
      outfile=fopen(filename,"w");
   else
      outfile=fopen(filename,"a");

  area = cs->sdata->grid_cell_area * KM2_PER_M2;
  /* TODO: this doesn't work with sbh3 added in LUH glm output -justin */
  fprintf(outfile,
          "%s t= %f site_area(km2) %f aharv_sec(km2) %f aharv_virgin(km2) %f, aharv_sbh2(km2) %f aharv_vbh2(km2) %f bharv_sec(tC) %f bharv_virgin(tC) %f bharv_sec_sbh2(tC) %f bharv_virgin_vbh2(tC) %f\n" ,
          cs->sdata->name_, time * TIMESTEP, area,
          cs->area_harvested[LU_SCND][0] / data->area * area,
          cs->area_harvested[LU_NTRL][0] / data->area * area,
          cs->area_harvested[LU_SCND][1] / data->area * area,
          cs->area_harvested[LU_NTRL][1] / data->area * area,
          cs->biomass_harvested[LU_SCND][0] * cs->sdata->grid_cell_area / data->area * T_PER_KG,
          cs->biomass_harvested[LU_NTRL][0] * cs->sdata->grid_cell_area / data->area * T_PER_KG,
          cs->biomass_harvested[LU_SCND][1] * cs->sdata->grid_cell_area / data->area * T_PER_KG,
          cs->biomass_harvested[LU_NTRL][1] * cs->sdata->grid_cell_area / data->area * T_PER_KG);
  fclose(outfile);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! print_area_burned
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_area_burned (unsigned int time, site** siteptr,
                        UserData* data) {
   site* cs;
   char filename[STR_LEN];
   FILE *outfile;
 
   cs=*siteptr;
 
   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);
   strcat(filename,".area_burned");
 
   if (time == 0)
      outfile = fopen(filename, "w");
   else
      outfile = fopen(filename, "a");
 
   fprintf(outfile, "%s t= %5.2f site_area %f area_burned %f\n",
           cs->sdata->name_,
           time * TIMESTEP,
           cs->sdata->grid_cell_area * KM2_PER_M2,
           cs->area_burned / data->area * cs->sdata->grid_cell_area * KM2_PER_M2);
   
   fclose(outfile);
}


////////////////////////////////////////////////////////////////////////////////
//! print_diagnostics
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_diagnostics (unsigned int time, site** siteptr,
                        UserData* data) {
   site* cs;
   patch *cp;
#ifdef ED
   cohort *cc;
#endif
   FILE *outfile;
   char filename[STR_LEN];
   int npatches, total_patches;
#ifdef ED
   int ncohorts, total_cohorts;
#endif
   int lu;

   cs = *siteptr;

   total_patches = 0;
#ifdef ED
   total_cohorts = 0;
#endif

   strcpy(filename, data->base_filename);
   strcat(filename, ".");
   strcat(filename, cs->sdata->name_);
   strcat(filename, ".diagnostics");
   if (time == 0)
      outfile = fopen(filename, "w");
   else
      outfile = fopen(filename, "a");

   fprintf(outfile, "site %s time %f ", cs->sdata->name_, time * TIMESTEP);

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      npatches = 0;
#ifdef ED
      ncohorts = 0;
#endif

      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
         npatches++;
#ifdef ED
         cc = cp->shortest;
         while (cc != NULL) {
            ncohorts++;
            cc = cc->taller;
         }
#endif
         cp = cp->older;
      }
 
#if defined ED
      fprintf(outfile, "lu %d npatches %d ncohorts %d lambda0 = %f lambda1 = %f ",
              lu,
              npatches,
              ncohorts,
              cs->disturbance_rate[0][0],
              cs->disturbance_rate[1][0]);
#elif defined MIAMI_LU
      fprintf(outfile, "lu %d npatches %d lambda0 = %f lambda1 = %f ",
              lu,
              npatches,
              cs->disturbance_rate[0][0],
              cs->disturbance_rate[1][0]);
#endif
      fclose(outfile);

      total_patches += npatches;
#ifdef ED
      total_cohorts += ncohorts;
#endif
   } /* end loop over land use types */
 
#if defined ED
   fprintf(outfile, "t_patches %d t_cohorts %d\n",
           total_patches, total_cohorts);
#elif defined MIAMI_LU
   fprintf(outfile, "t_patches %d \n",
           total_patches);
#endif
   fclose(outfile);
}

////////////////////////////////////////////////////////////////////////////////
//! print_patches
//! print patch area
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_patches (unsigned int time, site** siteptr, UserData* data) {
   site* cs;
   patch *cp;
#ifdef ED
   cohort *cc;
   int ncohorts;
#endif
   FILE *patchfile;
   char luname[STR_LEN];
   char filename[STR_LEN];
   int lu;
 
   cs = *siteptr;

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      lu_longname(lu, luname);
      strcpy(filename, data->base_filename);
      strcat(filename, ".");
      strcat(filename, cs->sdata->name_);
      strcat(filename, ".patches.");
      strcat(filename, luname);

      if(data->long_patch_file) {
         if (time == 0)
            patchfile = fopen(filename, "w");
         else
            patchfile = fopen(filename, "a");
      } else {
         patchfile = fopen(filename, "w");
      }
 
 
      cp = cs->youngest_patch[lu];
      while (cp != NULL) {
#if defined ED
         ncohorts = 0;
         cc = cp->shortest;
         while(cc != NULL) {
            ncohorts ++;
            cc = cc->taller;
         }
         fprintf(patchfile, "%s t= %f p= %p trk= %u age= %f area= %f l0= %f l1= %f nchrt= %d lai= %f landuse %d\n",
                 cs->sdata->name_,
                 ((double)(time))*TIMESTEP,
                 cp,
                 cp->track,
                 cp->age,
                 cp->area,
                 cp->disturbance_rate[0],
                 cp->disturbance_rate[1],
                 ncohorts,
                 cp->lai,
                 cp->landuse);
#elif defined MIAMI_LU
         fprintf(patchfile,
                 "%s t= %f p= %p trk= %u age= %f area= %f l0= %f l1= %f landuse %d\n",
                 cs->sdata->name_, time*TIMESTEP, cp, cp->track, cp->age, cp->area,
                 cp->disturbance_rate[0], cp->disturbance_rate[1],
                 cp->landuse);
#endif
         cp = cp->older;
      }
      fclose(patchfile);
   }
}


////////////////////////////////////////////////////////////////////////////////
//! print_site_chars
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_site_chars (site** fsite, UserData* data) {
   site* cs;
   FILE *outfile;
   char filename[STR_LEN];
   double min_precip, max_precip, min_soil_temp, max_soil_temp;
   int j;

   strcpy(filename,data->base_filename);
   strcat(filename,".sites");
   outfile=fopen(filename,"w");
 
   fprintf(outfile,"Number_of_sites = %d\n", data->number_of_sites);

   cs = *fsite;
   while (cs != NULL) {
      fprintf(outfile, "%s %p lat= %6.2f lon= %6.2f soi= %1d ",
              cs->sdata->name_, cs, cs->sdata->lat_, cs->sdata->lon_, cs->sdata->soi);
#ifdef ED
      fprintf(outfile, "l_top= %8.4f Rn_top= %8.4f ",
              cs->sdata->L_top, cs->sdata->Rn_top);
       
#if SOILGRIDS_SCHEME
       fprintf(outfile, "depth= %8.3f ksat= %8.4f theta_satu= %6.3f theta_resi= %6.3f L= %6.3f m= %6.3f conduct= %6.3f ",
               cs->sdata->MvG_soil_depth, cs->sdata->MvG_k_sat,cs->sdata->MvG_theta_saturated,cs->sdata->MvG_theta_residual,
               cs->sdata->MvG_L,cs->sdata->MvG_m,cs->sdata->soil_evap_conductivity);
#else
       //code for using MSTMIP parameters
       fprintf(outfile, "depth= %8.3f ksat= %8.4f theta_max= %6.3f conduct= %6.3f ",
               cs->sdata->soil_depth, cs->sdata->k_sat,
               cs->sdata->theta_max, cs->sdata->soil_evap_conductivity);
#endif
       
#elif defined MIAMI_LU
      fprintf(outfile, "depth= %8.3f conduct= %6.3f ",
              cs->sdata->soil_depth, cs->sdata->soil_evap_conductivity);
#endif
      fprintf(outfile, "exogenous dist params lambda1= %4.2f lambda2= %4.2f \n",
              cs->site_disturbance_rate[0],
              cs->site_disturbance_rate[1]);

#ifdef ED
      min_precip = 9999.9;
      max_precip = -9999.9;
      min_soil_temp = 9999.9;
      max_soil_temp = -9999.9;
           
      for (j=0; j<N_CLIMATE; j++) {
         if (min_precip > cs->sdata->precip[j])
            min_precip = cs->sdata->precip[j];
         if (max_precip < cs->sdata->precip[j])
            max_precip = cs->sdata->precip[j];
         if (min_soil_temp > cs->sdata->soil_temp[j])
            min_soil_temp = cs->sdata->soil_temp[j];
         if (max_soil_temp < cs->sdata->soil_temp[j])
            max_soil_temp = cs->sdata->soil_temp[j];
      } /* end loop over climate values */
#endif

#ifdef MIAMI_LU
      // TODO: this is useless
      min_precip = cs->sdata->precip_average;
      max_precip = cs->sdata->precip_average;
      min_soil_temp = cs->sdata->soil_temp_average;
      max_soil_temp = cs->sdata->soil_temp_average;
      fprintf(outfile, "miami_npp= %8.5f \n",
              cs->sdata->miami_npp);
#endif
      fprintf(outfile, "avgp= %f maxp= %f minp= %f avgst= %f maxst= %f minst= %f ",
              cs->sdata->precip_average, max_precip, min_precip,
              cs->sdata->soil_temp_average, max_soil_temp, min_soil_temp);
      fprintf(outfile,"\n");

      cs = cs->next_site;
   } /* end loop over sites */ 
   fclose(outfile);
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! print_patch_size_profile
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_patch_size_profile(unsigned int t, patch **pcp, unsigned int nbins, UserData* data){
   FILE *pprofile,*pprofile2,*pprofile3;
   char filename[STR_LEN], filename2[STR_LEN], filename3[STR_LEN];
   patch *cp;
   unsigned int i,j;
   cp=*pcp;
   /*  printf("pspp: starting printing to file \n"); */

   strcpy(filename,data->base_filename);
   strcpy(filename2,data->base_filename);
   strcpy(filename3,data->base_filename);
   strcat(filename,".");
   strcat(filename2,".");
   strcat(filename3,".");
   strcat(filename,cp->siteptr->sdata->name_);
   strcat(filename2,cp->siteptr->sdata->name_);
   strcat(filename3,cp->siteptr->sdata->name_);

   strcat(filename,".pprof.n");
   strcat(filename2,".pprof.ba");
   strcat(filename3,".pprof.agb");

   if(t==0){
      pprofile=fopen(filename,"w");
      pprofile2=fopen(filename2,"w");
      pprofile3=fopen(filename3,"w");
   }
   else{
      pprofile = fopen(filename,"a");
      pprofile2 = fopen(filename2,"a");
      pprofile3 = fopen(filename3,"a");
   }
     
   for(i=0;i<NSPECIES;i++){
      fprintf(pprofile,"t= %f p= %p track= %u age= %f  ",t*TIMESTEP,cp,cp->track,cp->age);
      fprintf(pprofile2,"t= %f p= %p track= %u age= %f  ",t*TIMESTEP,cp,cp->track,cp->age);
      fprintf(pprofile3,"t= %f p= %p track= %u age= %f  ",t*TIMESTEP,cp,cp->track,cp->age);

      fprintf(pprofile,"spp%2d  ",i);
      fprintf(pprofile2,"spp%2d  ",i);
      fprintf(pprofile3,"spp%2d  ",i);

      for(j=0;j<N_DBH_BINS;j++){
         fprintf(pprofile,"%2d %8.5f  ",j,cp->spp_density_profile[i][j]);
         fprintf(pprofile2,"%2d %8.5f ",j,cp->spp_basal_area_profile[i][j]);
         fprintf(pprofile3,"%2d %8.5f ",j,cp->spp_agb_profile[i][j]);
      }
      fprintf(pprofile,"\n");
      fprintf(pprofile2,"\n");
      fprintf(pprofile3,"\n");
   }
   /*   printf("pspp: finished printing to file \n"); */
   fclose(pprofile);  
   fclose(pprofile2);
   fclose(pprofile3);
}


////////////////////////////////////////////////////////////////////////////////
//! print_site_size_profile
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_site_size_profile(unsigned int t, site** pcs, unsigned int nbins, UserData* data){
   FILE *pssp,*pssp2,*pssp3;
   char filename[STR_LEN], filename2[STR_LEN], filename3[STR_LEN];
   site* cs;
   unsigned int i;
#if 0
   unsigned int j;
#endif
   cs=*pcs;
   strcpy(filename,data->base_filename);
   strcpy(filename2,data->base_filename);
   strcpy(filename3,data->base_filename);
   strcat(filename,".");
   strcat(filename2,".");
   strcat(filename3,".");
   strcat(filename,cs->sdata->name_);
   strcat(filename2,cs->sdata->name_);
   strcat(filename3,cs->sdata->name_);
   strcat(filename,".sprof.n");
   strcat(filename2,".sprof.ba");
   strcat(filename3,".sprof.agb");

   if(t==0){
      pssp=fopen(filename,"w");
      pssp2=fopen(filename2,"w");
      pssp3=fopen(filename3,"w");
   }
   else{
      pssp = fopen(filename,"a");
      pssp2 = fopen(filename2,"a");
      pssp3 = fopen(filename3,"a");
   }

   /* printf("pssp: starting printing to file \n"); */
#if 0     
   for(i=0;i<NSPECIES;i++){
      fprintf(pssp,"t= %f ",t*TIMESTEP);
      fprintf(pssp2,"t= %f ",t*TIMESTEP);
      fprintf(pssp3,"t= %f ",t*TIMESTEP);
      fprintf(pssp,"spp%2d  ",i);
      fprintf(pssp2,"spp%2d  ",i);
      fprintf(pssp3,"spp%2d  ",i);

      for(j=0;j<N_DBH_BINS;j++){
         fprintf(pssp,"%2d %8.5f  ",j,cs->spp_density_profile[i][j]);
         fprintf(pssp2,"%2d %8.5f ",j,cs->spp_basal_area_profile[i][j]);
         fprintf(pssp3,"%2d %8.5f ",j,cs->spp_agb_profile[i][j]);
      }
      fprintf(pssp,"\n");
      fprintf(pssp2,"\n");
      fprintf(pssp3,"\n");
   } /* end loop over species */
#endif
#if 1
   fprintf(pssp,"t= %f ",t*TIMESTEP);
   fprintf(pssp2,"t= %f ",t*TIMESTEP);
   fprintf(pssp3,"t= %f ",t*TIMESTEP);

   for(i=0;i<N_DBH_BINS;i++){
      fprintf(pssp,"%2d %8.5f ",i,cs->density_profile[i]);
      fprintf(pssp2,"%2d %8.5f ",i,cs->basal_area_profile[i]);
      fprintf(pssp3,"%2d %8.5f ",i,cs->agb_profile[i]);
   }
   fprintf(pssp,"\n");
   fprintf(pssp2,"\n");
   fprintf(pssp3,"\n");
#endif

   /*   printf("pssp: finished printing to file \n"); */
   fclose(pssp);  
   fclose(pssp2);
   fclose(pssp3);
   return;
}


////////////////////////////////////////////////////////////////////////////////
//! print_cohorts
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_cohorts (unsigned int time, site** siteptr, UserData* data) {
   patch *cp;
   cohort *cc;
   site* cs;
   FILE *cohortfile;
   char filename[STR_LEN];
   int lu;

   cs = *siteptr;

   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      strcpy(filename, data->base_filename);
      strcat(filename, ".");
      strcat(filename, cs->sdata->name_);
      switch (lu) {
         case LU_NTRL :
            strcat(filename, ".cohorts.natural");
            break;
         case LU_SCND :
            strcat(filename, ".cohorts.secondary");
            break;
         case LU_CROP :
            strcat(filename, ".cohorts.crop");
            break;
         case LU_PAST :
            strcat(filename, ".cohorts.pasture");
            break;
      }

      if(data->long_cohort_file) {
         if (time==0)   
            cohortfile = fopen(filename, "w");
         else
            cohortfile = fopen(filename, "a");
      } else {
         cohortfile = fopen(filename, "w");
      }
      
      cp = cs->oldest_patch[lu];
      while (cp != NULL) {
         cc = (cp->tallest);
         while (cc != NULL) {
            fprintf(cohortfile,
                    "%s t= %f p %p track %u age %f ar %f c %p h %f dbh %f spp %u sta %d n %f ba %f bd %f bl %f blv %f br %f bsw %f gpp %f npp %f resp %f cbr %f Anp %f Ans %f lai %f md %f fso %f r= %f dhdt %f ddbhdt %f dndt %f Ep %f Es %f wu %f fsw %f gr_resp %f npp_avg %f md_avg %f\n",
                    cs->sdata->name_,
                    time * TIMESTEP,
                    cp, 
                    cp->track,
                    cp->age,
                    cp->area,
                    cc,
                    cc->hite,
                    cc->dbh,
                    cc->species,
                    cc->status,
                    cc->nindivs / cp->area,
                    cc->balive,
                    cc->bdead,
                    cc->bl,
                    cc->blv,
                    cc->br,
                    cc->bsw,
                    cc->gpp,
                    cc->npp,
                    cc->resp,
                    cc->cbr[data->time_period],
                    cc->An_pot,
                    cc->An_shut,
                    cc->lai,
                    cc->md,
                    cc->fs_open,
                    cc->p[0] + cc->p[1],
                    cc->dhdt,
                    cc->ddbhdt,                    
                    cc->dndt / cc->nindivs,
                    cc->E_pot,
                    cc->E_shut,
                    cc->water_uptake,
                    cc->fsw,
                    cc->gr_resp,
                    cc->npp_avg,
                    cc->md_avg);
            cc = cc->shorter;
         }
         cp = cp->younger;
      }
      fclose(cohortfile);
   }
}


////////////////////////////////////////////////////////////////////////////////
//! print_nitrogen_budget
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_nitrogen_budget(unsigned int time, site** currents, UserData* data){

   char filename[STR_LEN];
   FILE *nbudgetfile;
   site *siteptr;
   patch *currentp;
   cohort *currentc;
   double n_veg_alive=0.00,n_veg_dead=0.00,n_veg=0.00,n_soil=0.00,n_total=0.00;

   siteptr = *currents;

   /* open print files */
   strcpy(filename,data->base_filename);
   strcat(filename,".");
   strcat(filename,siteptr->sdata->name_);
   strcat(filename,".nbudget");
   if(time==0) nbudgetfile=fopen(filename,"w");
   else nbudgetfile=fopen(filename,"a");

   currentp = siteptr->oldest_patch[LU_NTRL];

   while(currentp != NULL){

      currentc = currentp->tallest;
      while(currentc != NULL){
     
         n_veg_alive += currentc->nindivs*currentc->balive/data->c2n_leaf[currentc->species];
         n_veg_dead  += currentc->nindivs*currentc->bdead/data->c2n_stem;
     
         currentc=currentc->shorter;
      }
   
      n_soil += (currentp->mineralized_soil_N + currentp->fast_soil_N + currentp->slow_soil_C/data->c2n_slow + currentp->structural_soil_C/data->c2n_structural + currentp->passive_soil_C/data->c2n_passive)*currentp->area/data->area;
    
      currentp=currentp->younger;

   }

   n_veg = n_veg_alive + n_veg_dead;
   n_veg /= data->area;
   n_veg_alive /= data->area;
   n_veg_dead /= data->area;

   n_total = n_veg + n_soil;

   fprintf(nbudgetfile,"t= %f n_veg= %f n_soil= %f n_total= %f\n",time*TIMESTEP,n_veg,n_soil,n_total);

   fclose(nbudgetfile);

   return;
}
#endif /* ED */

////////////////////////////////////////////////////////////////////////////////
//! print_domain_stats
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_domain_stats (unsigned int t, site** sitrptr, UserData* data) {
   site* cs;
   FILE *outfile;
#if LANDUSE
   char luname[STR_LEN];
#endif
   char filename[STR_LEN];
   double area, scale_factor;
   int n = 0;
   double total_area_modeled                     = 0.0;
   double total_area_burned                      = 0.0;
   double total_biomass                          = 0.0;
   double total_ag_biomass                       = 0.0;
   double total_soil_sc                          = 0.0;
   double total_nep2                             = 0.0;
#if LANDUSE
   int lu;
   double total_area_lu[N_LANDUSE_TYPES];
   double total_biomass_lu[N_LANDUSE_TYPES];
   double total_agb_lu[N_LANDUSE_TYPES];
   double total_sc_lu[N_LANDUSE_TYPES];
   double total_area_forest                      = 0.0;
   double total_biomass_forest                   = 0.0;
   double total_agb_forest                       = 0.0;
   double total_sc_forest                        = 0.0;
   double total_forest_dndt                      = 0.0;
   double total_biomass_for_sec                  = 0.0;
   double total_agb_forest_sec                   = 0.0;
   double total_sc_forest_sec                    = 0.0;
   double mean_AGE_sec                           = 0.0;
   double total_area_harvested_secondary         = 0.0;
   double total_area_harvested_virgin            = 0.0;
   double total_area_harvested_virgin_vbh2       = 0.0;
   double total_area_harvested_secondary_sbh2    = 0.0;
   double total_biomass_harvested_secondary      = 0.0;
   double total_biomass_harvested_virgin         = 0.0;
   double total_biomass_harvested_virgin_vbh2    = 0.0;
   double total_biomass_harvested_secondary_sbh2 = 0.0;
#if 0
   double total_biomass_harvested_unmet_vbh      = 0.0;
   double total_biomass_harvested_unmet_vbh2     = 0.0;
   double total_biomass_harvested_unmet_sbh      = 0.0;
   double total_biomass_harvested_unmet_sbh2     = 0.0;
#endif
#endif /* LANDUSE */
   double hurr_litter                            = 0.0;

#if LANDUSE
   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      total_area_lu[lu]                          = 0.0;
      total_biomass_lu[lu]                       = 0.0;
      total_agb_lu[lu]                           = 0.0;
      total_sc_lu[lu]                            = 0.0;
   }
#endif

   /* TODO: this doesn't handle sbh3 -justin */
   cs = *sitrptr;
   while (cs != NULL) {
      n++;
   
      area = cs->sdata->grid_cell_area * KM2_PER_M2;
      scale_factor = cs->sdata->grid_cell_area * T_PER_KG * GT_PER_T;

      /*ALL LAND*/
      total_area_modeled += area;
      total_area_burned += cs->area_burned / data->c2b * area;
      total_biomass += cs->site_total_biomass * scale_factor;
      total_ag_biomass += cs->site_total_ag_biomass * scale_factor;
      total_soil_sc += cs->site_total_soil_c * scale_factor;
      total_nep2 += cs->site_nep2 * scale_factor;

#if LANDUSE
      /*FOREST*/ 
      if (cs->forest_harvest_flag == 1) {
         total_area_forest += (cs->area_fraction[LU_NTRL] + cs->area_fraction[LU_SCND]) * area;
         total_biomass_forest += (cs->total_biomass[LU_NTRL] * cs->area_fraction[LU_NTRL]
                                  + cs->total_biomass[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
         total_agb_forest += (cs->total_ag_biomass[LU_NTRL] * cs->area_fraction[LU_NTRL]
                              + cs->total_ag_biomass[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
         total_sc_forest += (cs->total_soil_c[LU_NTRL] * cs->area_fraction[LU_NTRL]
                             + cs->total_soil_c[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
         total_forest_dndt += (cs->dndt[LU_NTRL] * cs->area_fraction[LU_NTRL]
                               + cs->dndt[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;

         total_biomass_for_sec += (cs->total_biomass[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
         total_agb_forest_sec += (cs->total_ag_biomass[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
         total_sc_forest_sec += (cs->total_soil_c[LU_SCND] * cs->area_fraction[LU_SCND]) * scale_factor;
      }
#endif


#if LANDUSE
      for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
         total_area_lu[lu] += cs->area_fraction[lu] * area;
         total_biomass_lu[lu] += cs->total_biomass[lu] * cs->area_fraction[lu] * scale_factor;
         total_agb_lu[lu] += cs->total_ag_biomass[lu] * cs->area_fraction[lu] * scale_factor;
         total_sc_lu[lu] += cs->total_soil_c[lu] * cs->area_fraction[lu] * scale_factor;

      }
      mean_AGE_sec += cs->mean_AGE_sec * cs->area_fraction[LU_SCND] * cs->sdata->grid_cell_area;

      /* TODO: fix to work with vbh/sbh arrays */
      total_area_harvested_secondary += cs->area_harvested[LU_SCND][0] / data->area * area;
      total_area_harvested_virgin += cs->area_harvested[LU_NTRL][0] / data->area * area;
      total_area_harvested_secondary_sbh2 += cs->area_harvested[LU_SCND][1] / data->area * area;
      total_area_harvested_virgin_vbh2 += cs->area_harvested[LU_NTRL][1] / data->area * area;
      total_biomass_harvested_secondary += cs->biomass_harvested[LU_SCND][0]
         / data->area * scale_factor;
      total_biomass_harvested_virgin += cs->biomass_harvested[LU_NTRL][0] / data->area * scale_factor;
      total_biomass_harvested_secondary_sbh2 += cs->biomass_harvested[LU_SCND][1]
         / data->area * scale_factor;
      total_biomass_harvested_virgin_vbh2 += cs->biomass_harvested[LU_NTRL][1]
         / data->area * scale_factor;
#if 0
      /*already scaled to total area in landuse.c*/
      total_biomass_harvested_unmet_vbh += cs->biomass_harvested_unmet[LU_NTRL][0]
         * T_PER_KG * GT_PER_T;
      total_biomass_harvested_unmet_vbh2 += cs->biomass_harvested_unmet[LU_NTRL][1]
         * T_PER_KG * GT_PER_T;
      total_biomass_harvested_unmet_sbh += cs->biomass_harvested_unmet[LU_SCND][0]
         * T_PER_KG * GT_PER_T;
      total_biomass_harvested_unmet_sbh2+=cs->biomass_harvested_unmet[LU_SCND][1]
         * T_PER_KG * GT_PER_T;
#endif
#endif /* LANDUSE */

      if(data->do_hurricane) {
         hurr_litter += cs->hurricane_litter * T_PER_KG * cs->sdata->grid_cell_area * GT_PER_T;
      }
   
      cs = cs->next_site;
   }

#if LANDUSE
   if(total_area_lu[LU_SCND] > 0.0)
      mean_AGE_sec /= total_area_lu[LU_SCND];
   else mean_AGE_sec = 0.00;
#endif

   strcpy(filename, data->base_filename);
   strcat(filename, ".domain_stats");
   if (t == 0)
      outfile = fopen(filename,"w");
   else
      outfile = fopen(filename,"a");

   fprintf(outfile,
           "t %f Nsites %d amod(km2) %f aburned(km2) %f b(Gt) %f agb(Gt) %f tot_sc(Gt) %f total_c(Gt) %f nep2(Gt/y) %f ",
           t * data->deltat,
           n,
           total_area_modeled,
           total_area_burned,
           total_biomass,
           total_ag_biomass,
           total_soil_sc,
           total_biomass + total_soil_sc,
           total_nep2);

#if LANDUSE
   for (lu=0; lu<N_LANDUSE_TYPES; lu++) {
      lu_shortname(lu, luname);
      if (lu == LU_SCND)
         fprintf(outfile, "mA_scnd(yrs) %f ", mean_AGE_sec);
      fprintf(outfile,
              "a_%s(km2) %f  b_%s(Gt) %f agb_%s(Gt) %f sc_%s(Gt) %f ",
              luname, total_area_lu[lu],
              luname, total_biomass_lu[lu],
              luname, total_agb_lu[lu],
              luname, total_sc_lu[lu]);

   }
   fprintf(outfile,
           "a_for(km2) %f b_for(Gt) %f agb_for(Gt) %f sc_for(Gt) %f dndt_for(Gt/y) %f ",
           total_area_forest,
           total_biomass_forest,
           total_agb_forest,
           total_sc_forest,
           total_forest_dndt );
   fprintf(outfile,
           "b_scnd_for(Gt) %f agb_scnd_for(Gt) %f sc_scnd_for(Gt) %f ",
           total_biomass_for_sec,
           total_agb_forest_sec,
           total_sc_forest_sec);
  
   fprintf(outfile,
           " aharv_virgin(km2) %f aharv_sec(km2) %f aharv_virgin_vbh2(km2) %f aharv_sec_sbh2(km2) %f bharv_virgin(Gt) %f bharv_sec(Gt) %f bharv_virgin_vbh2(Gt) %f bharv_sec_sbh2(Gt) %f ",
           total_area_harvested_virgin, total_area_harvested_secondary,
           total_area_harvested_virgin_vbh2, total_area_harvested_secondary_sbh2,
           total_biomass_harvested_virgin, total_biomass_harvested_secondary,
           total_biomass_harvested_virgin_vbh2, total_biomass_harvested_secondary_sbh2);
#if 0
   prinf(outfile,
         "unmet_vbh(Gt) %f unmet_vbh2(Gt) %f unmet_sbh(Gt) %f unmet_sbh2(Gt) %f total_unmet(Gt) %f ",
         total_biomass_harvested_unmet_vbh, total_biomass_harvested_unmet_vbh2,
         total_biomass_harvested_unmet_sbh, total_biomass_harvested_unmet_sbh2,
         total_biomass_harvested_unmet_vbh + total_biomass_harvested_unmet_vbh2
         + total_biomass_harvested_unmet_sbh + total_biomass_harvested_unmet_sbh2);
#endif
#endif /* LANDUSE */

   if(data->do_hurricane) {
      fprintf(outfile, "hurr_litter %f ", hurr_litter);
   }
 
   fprintf(outfile, "\n");
   fclose(outfile);
}

/******************************************************************************/
