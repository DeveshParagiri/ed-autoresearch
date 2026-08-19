/* 7/29/98 - disturbance_rates.c - PRM/GCH */
#include <cstdio>
#include <cmath>

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "fire.h"
#include "read_site_data.h"
#ifdef ED
#include "cohort.h"
#endif

#include "disturbance.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//! calculate_disturbance_rates
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void calculate_disturbance_rates ( unsigned int t, 
                                   patch** current_patch, 
                                   UserData* data ) {
   patch* cp= *current_patch;
   site* cs= cp->siteptr;
   /* CALCULATE DISTURBANCE RATES */

   /* CALCULATE FIRE DISTURBANCE RATES */
   cp->disturbance_rate[1] = fire(t, &cp, data);
  
   /* CALCULATE TREEFALL DISTURBANCE RATES */
    if (cs->climate_zone==1)
    {
      cp->disturbance_rate[0] = data->treefall_max_disturbance_rate_trop;
#ifdef MIAMI_LU
   } else if((cs->sdata->lat_ > -45.0) && (cs->sdata->lat_ < 45.0)) {
      /* Experimental */
      cp->disturbance_rate[0] = data->treefall_max_disturbance_rate_trop 
         - (data->treefall_max_disturbance_rate_trop - data->treefall_max_disturbance_rate_temp ) 
         / 30.0 * (abs(cs->sdata->lat_) - 15.0);
#endif
   } else {
      cp->disturbance_rate[0] = data->treefall_max_disturbance_rate_temp;
   }

   if (data->do_hurricane) {
      cp->disturbance_rate[0] += get_hurricane_disturbance_rate(t, cs, data);
   }

   /* add in mortality due to selective harvesting */
   if ( (cp->landuse == LU_SCND) && (cs->forest_harvest_flag == 1) ) {
      cp->disturbance_rate[0] += data->selective_harvest_rate;
   }

#if defined ED
   /* 9/00 New: do other disturbance as mean field */
   cp->treefall_as_dndt_flag=0;
   cp->fire_as_dndt_flag=0;
   if( cp->disturbance_rate[1] > cp->disturbance_rate[0] ) { /* fire larger */
      cp->disturbance_rate[0] = 0.00;
      cp->fire_dndt_factor = 0.00;
      cp->treefall_as_dndt_flag = 1;
   }
   else { /* treefall larger */
      cp->fire_dndt_factor = cp->disturbance_rate[1];
      cp->disturbance_rate[1] = 0.0;
      cp->fire_as_dndt_flag = 1;
   }
   cp->total_disturbance_rate = 0.0;
   for (size_t i=0; i<NUM_TRACKS; i++) {
      cp->total_disturbance_rate += cp->disturbance_rate[i];
   }
#elif defined MIAMI_LU
   /* other term goes mean field in ED, so here used both */
   cp->total_disturbance_rate = cp->disturbance_rate[1] + cp->disturbance_rate[0];
#endif

}

////////////////////////////////////////////////////////////////////////////////
//! get_hurricane_disturbance_rate
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double get_hurricane_disturbance_rate ( unsigned int t, 
                                        site* current_site,
                                        UserData* data) {
   double dist_rate = 0.0;

   if (data->hurricane_year < 0) {
      dist_rate = current_site->sdata->avg_hurricane_disturbance_rate;
   } else {
      dist_rate = current_site->sdata->hurricane_disturbance_rate[data->hurricane_year];
   }

   if(data->hurricane_ramp) {
      dist_rate *= data->hurricane_ramp_factor[data->year];
   }

   return dist_rate;
}


////////////////////////////////////////////////////////////////////////////////
//! accumulate_litter_from_disturbance
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void accumulate_litter_from_disturbance ( patch** target_patch,
                                          patch** donor_patch, 
                                          double change_in_area, 
                                          int q, UserData* data ) {
  
   patch* dp = *donor_patch;
   patch* tp = *target_patch;

   site* cs = tp->siteptr;
    
    double tmp_biomass = 0.0;

   double fast_litter   = 0.0;
   double struct_litter = 0.0;
#if defined ED
   double fast_litter_n = 0.0;
    
    double fraction_harvest_left_on_site = 0.0;
    double fraction_clearing_left_on_site = 0.0;

   cohort* cc = dp->shortest;
   while (cc != NULL) {
      size_t spp = cc->species;
      /* clear for wood harvesting */
      if ( q == 9 ) { 
         if ( dp->area > 0.000001 ) {
             double fraction_harvest_left_on_prim_site_spp = 0.0;
             double fraction_harvest_left_on_secd_site_spp = 0.0;
             if (cs->climate_zone==1)
             {
                 fraction_harvest_left_on_prim_site_spp = data->fraction_harvest_left_on_prim_site_tro[spp];
                 fraction_harvest_left_on_secd_site_spp = data->fraction_harvest_left_on_secd_site_tro[spp];
             }
            else
            {
                fraction_harvest_left_on_prim_site_spp = data->fraction_harvest_left_on_prim_site_temp[spp];
                fraction_harvest_left_on_secd_site_spp = data->fraction_harvest_left_on_secd_site_temp[spp];
            }

             if (dp->landuse == LU_NTRL)
                 fraction_harvest_left_on_site = fraction_harvest_left_on_prim_site_spp;
             else if (dp->landuse == LU_SCND)
                 fraction_harvest_left_on_site = fraction_harvest_left_on_secd_site_spp;
             else
                 fraction_harvest_left_on_site = 1.0;
             
             
            fast_litter +=  data->fraction_balive_2_fast * fraction_harvest_left_on_site * cc->balive * ( cc->nindivs / dp->area );
            fast_litter_n +=  data->fraction_balive_2_fast * fraction_harvest_left_on_site * (1.0/data->c2n_leaf[spp]) * cc->balive * (cc->nindivs/dp->area);
            struct_litter += fraction_harvest_left_on_site * cc->bdead * (cc->nindivs/dp->area) + (1.0 - data->fraction_balive_2_fast) * fraction_harvest_left_on_site * cc->balive * (cc->nindivs/dp->area);
             
#if LANDUSE
             tp->forest_harvested_c += (1-fraction_harvest_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area*LANDUSE_FREQ;


             double fraction_harvest_to_1yr_pool = 0.0, fraction_harvest_to_10yr_pool = 0.0, fraction_harvest_to_100yr_pool = 0.0;
             if (cs->climate_zone==1)
             {
                 fraction_harvest_to_1yr_pool = data->fraction_harvest_to_1yr_pool_tro[spp];
                 fraction_harvest_to_10yr_pool = data->fraction_harvest_to_10yr_pool_tro[spp];
                 fraction_harvest_to_100yr_pool = data->fraction_harvest_to_100yr_pool_tro[spp];
             }
             else
             {
                 fraction_harvest_to_1yr_pool = data->fraction_harvest_to_1yr_pool_temp[spp];
                 fraction_harvest_to_10yr_pool = data->fraction_harvest_to_10yr_pool_temp[spp];
                 fraction_harvest_to_100yr_pool = data->fraction_harvest_to_100yr_pool_temp[spp];
             }
             
             tp->yr1_decay_product_pool += fraction_harvest_to_1yr_pool*(1-fraction_harvest_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
             tp->yr10_decay_product_pool += fraction_harvest_to_10yr_pool*(1-fraction_harvest_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
             tp->yr100_decay_product_pool += fraction_harvest_to_100yr_pool*(1-fraction_harvest_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
#endif
         }
      }
       /* clearing for cropland and pasture, not wood harvesting  */
      else if (q==8)
      {
          if ( dp->area > 0.000001 )
          {
              if (cs->climate_zone==1)
                  fraction_clearing_left_on_site = data->fraction_clearing_left_on_site_tro[spp];
              else
                  fraction_clearing_left_on_site = data->fraction_clearing_left_on_site_temp[spp];
              
              fast_litter +=  data->fraction_balive_2_fast * fraction_clearing_left_on_site * cc->balive * (cc->nindivs/dp->area);
              fast_litter_n +=  data->fraction_balive_2_fast * fraction_clearing_left_on_site * (1.0/data->c2n_leaf[spp]) * cc->balive * (cc->nindivs/dp->area);
              struct_litter += fraction_clearing_left_on_site * cc->bdead * (cc->nindivs/dp->area) + (1.0 - data->fraction_balive_2_fast) * fraction_clearing_left_on_site * cc->balive * (cc->nindivs / dp->area);
              
#if LANDUSE
              ///Currently, harvest in primay & secondary and LU change will call this function with flag 9.
              ///I will put all harvested c into forest_harvested_c pool no matter which kind of activity occur.
              ///In next update, the c loss due to LU change may be out into LU_emission instead of forest_harvested_c
              tp->forest_harvested_c += (1-fraction_clearing_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area*LANDUSE_FREQ;
              
              double fraction_clearing_to_1yr_pool = 0.0, fraction_clearing_to_10yr_pool = 0.0, fraction_clearing_to_100yr_pool = 0.0;
              if (cs->climate_zone==1)
              {
                  fraction_clearing_to_1yr_pool = data->fraction_clearing_to_1yr_pool_tro[spp];
                  fraction_clearing_to_10yr_pool = data->fraction_clearing_to_10yr_pool_tro[spp];
                  fraction_clearing_to_100yr_pool = data->fraction_clearing_to_100yr_pool_tro[spp];
              }
              else
              {
                  fraction_clearing_to_1yr_pool = data->fraction_clearing_to_1yr_pool_temp[spp];
                  fraction_clearing_to_10yr_pool = data->fraction_clearing_to_10yr_pool_temp[spp];
                  fraction_clearing_to_100yr_pool = data->fraction_clearing_to_100yr_pool_temp[spp];
              }
              
              tp->yr1_decay_product_pool += fraction_clearing_to_1yr_pool*(1-fraction_clearing_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
              tp->yr10_decay_product_pool += fraction_clearing_to_10yr_pool*(1-fraction_clearing_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
              tp->yr100_decay_product_pool += fraction_clearing_to_100yr_pool*(1-fraction_clearing_left_on_site)*(cc->balive+cc->bdead)*cc->nindivs*change_in_area/dp->area;
#endif
          }
      }
#if 0
      /* selective logged forests */
      else if ( (dp->landuse == LU_SCND) && (cs->forest_harvest_flag == 1) && (q != 1) ) {
         if (dp->area > 0.000001) {
            fast_litter += data->fraction_balive_2_fast * 
               ( 1.0 - data->selective_harvest_rate / dp->disturbance_rate[0] *
                 ( 1.0-data->fraction_harvest_left_on_site ) ) *
               ( 1.0 - cs->sdata->loss_fraction[q] ) * cc->balive * 
               ( 1.0 - cc->survivorship_from_disturbance(q, data) ) * 
               ( cc->nindivs / dp->area );
            fast_litter_n +=  data->fraction_balive_2_fast * 
               ( 1.0-data->selective_harvest_rate / dp->disturbance_rate[0] * 
                 ( 1.0-data->fraction_harvest_left_on_site ) ) * 
               ( 1.0 / data->c2n_leaf[spp] ) * 
               ( 1.0 - cs->sdata->loss_fraction[q] ) * cc->balive * 
               ( 1.0 - cc->survivorship_from_disturbance(q, data) ) * 
               ( cc->nindivs / dp->area );
            struct_litter += ( 1.0 - data->selective_harvest_rate / 
                               dp->disturbance_rate[0] * 
                               ( 1.0 - data->fraction_harvest_left_on_site ) ) * 
               ( 1.0 - cs->sdata->loss_fraction[q] ) * cc->bdead * 
               ( 1.0 - cc->survivorship_from_disturbance(q, data) ) * 
               ( cc->nindivs / dp->area ) + 
               ( 1.0 - data->fraction_balive_2_fast ) * 
               ( 1.0 - data->selective_harvest_rate / dp->disturbance_rate[0] * 
                 ( 1.0 - data->fraction_harvest_left_on_site ) ) * 
               ( 1.0 - cs->sdata->loss_fraction[q] ) * cc->balive * 
               ( 1.0 - cc->survivorship_from_disturbance(q, data) ) * 
               ( cc->nindivs / dp->area );   
         }
      } 
#endif
      /* all other cases */
      else {
         if ( dp->area > 0.000001 ) {
            fast_litter += data->fraction_balive_2_fast * (1.0 - cs->sdata->loss_fraction[q]) * cc->balive * (1.0 - cc->survivorship_from_disturbance(q, data)) * (cc->nindivs/dp->area);
            fast_litter_n +=  data->fraction_balive_2_fast * (1.0/data->c2n_leaf[spp] ) * (1.0 - cs->sdata->loss_fraction[q]) * cc->balive * (1.0 - cc->survivorship_from_disturbance(q, data)) * (cc->nindivs/dp->area);
            struct_litter += ( 1.0 - cs->sdata->loss_fraction[q] ) * cc->bdead * (1.0 - cc->survivorship_from_disturbance(q, data)) * (cc->nindivs/dp->area) + (1.0 - data->fraction_balive_2_fast) * (1.0 - cs->sdata->loss_fraction[q]) * cc->balive * (1.0 - cc->survivorship_from_disturbance(q, data)) * (cc->nindivs/dp->area);
             
             /// carbon loss from fire
             tmp_biomass += (cc->balive+cc->bdead)*cc->nindivs/dp->area;
             if (q==1)
             {
                 /// Here the biomass loss from fire is multipled by 12 as fire_emission is considered as flux and all fluxes variables in ED is yearly-based
                 tp->fire_emission += cs->sdata->loss_fraction[q]*cc->balive*(1.0 -cc->survivorship_from_disturbance(q, data))*cc->nindivs*change_in_area/dp->area*N_CLIMATE;
                 tp->fire_emission += cs->sdata->loss_fraction[q]*cc->bdead*(1.0 -cc->survivorship_from_disturbance(q, data))*cc->nindivs*change_in_area/dp->area*N_CLIMATE;
             }
             
            }
         }
      cc=cc->taller;
   }  /* end loop over cohorts */
#elif defined MIAMI_LU  
   if (q==9) { /*clear cut harvest*/
      if(dp->area > 0.000001){
         /*fast_litter =  data->fraction_harvest_left_on_site*dp->total_biomass*0.1;*/
         /*slash fraction of harvest + below ground carbon*/
         struct_litter = data->fraction_harvest_left_on_site * dp->total_ag_biomass 
            + (dp->total_biomass - dp->total_ag_biomass);  
      }
   } else { /*all other cases*/
      if (dp->area > 0.000001) {
         /* fast_litter = (1.0 - cs->sdata->loss_fraction[q])*dp->total_biomass;*/
         struct_litter = (1.0 - cs->sdata->loss_fraction[q]) * dp->total_biomass;  
      }
   }
#endif /* ED V. MIAMI_LU */

   /* add to litter flux total */
   tp->litter =  fast_litter + struct_litter;

   /************************************************************/
   /* LOAD DISTURBANCE LITTER DIRECTLY INTO CARBON AND N POOLS */ 
   /************************************************************/    
  
  /* spread litter over new patch */
   // @TODO, Each variable holds summation instead of running average, we divide by 
   // total patch area in calling function.
   tp->fast_soil_C       += fast_litter * change_in_area; /* lvs rts */
   tp->structural_soil_C += struct_litter * change_in_area; /* stem */
#ifdef ED
   tp->structural_soil_L += (data->l2n_stem / data->c2n_stem ) * struct_litter * change_in_area;
   tp->fast_soil_N       += fast_litter_n * change_in_area;
#endif
}


////////////////////////////////////////////////////////////////////////////////
//! aggregate_in_soil_state_from_disturbance
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void aggregate_in_soil_state_from_disturbance ( patch** target_patch,
                                              patch** donor_patch,
                                              double change_in_area,
                                              UserData* data ) {
  
   patch* dp = *donor_patch;
   patch* tp = *target_patch;

   // @TODO, Each variable holds summation instead of running average, we divide by 
   // total patch area in calling function.
   tp->fast_soil_C        += dp->fast_soil_C * change_in_area ;
   tp->structural_soil_C  += dp->structural_soil_C * change_in_area ;
#ifdef ED
   tp->fast_soil_N        += dp->fast_soil_N * change_in_area ;
   tp->slow_soil_C        += dp->slow_soil_C * change_in_area ;
   tp->passive_soil_C     += dp->passive_soil_C * change_in_area;
   tp->mineralized_soil_N += dp->mineralized_soil_N * change_in_area ;
   tp->structural_soil_L  += dp->structural_soil_L * change_in_area ;
   tp->water              += dp->water * change_in_area ;
   tp->theta              += dp->theta * change_in_area;
   tp->rh                 += dp->rh * change_in_area ;
   tp->soil_evap          += dp->soil_evap * change_in_area ;
    
#if SNOWPACK_SCHEME == 1
    tp->snowpack              += dp->snowpack * change_in_area ;
    tp->snow_melt              += dp->snow_melt * change_in_area ;
#endif
    
    tp->npp_avg         += dp->npp_avg * change_in_area ;
    tp->rh_avg          += dp->rh_avg * change_in_area;
    tp->gpp_avg         += dp->gpp_avg * change_in_area;
    tp->fire_emission   += dp->fire_emission * change_in_area;
#if LANDUSE
    tp->forest_harvested_c += dp->forest_harvested_c * change_in_area;
    tp->product_emission += dp->product_emission * change_in_area;
    tp->yr1_decay_product_pool += dp->yr1_decay_product_pool * change_in_area;
    tp->yr10_decay_product_pool += dp->yr10_decay_product_pool * change_in_area;
    tp->yr100_decay_product_pool += dp->yr100_decay_product_pool * change_in_area;
    tp->crop_harvested_c += dp->crop_harvested_c * change_in_area;
    tp->past_harvested_c += dp->past_harvested_c * change_in_area;
#endif
    
#endif /* ED */
}

/******************************************************************************/
