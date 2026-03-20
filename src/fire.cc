#include <cstdio>
#include <cstring>
#include <cmath>
#include "netcdf.h"
#include "miami.h"
#include "read_site_data.h"

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "read_site_data.h"
#ifdef ED
#include "cohort.h"
#endif


////////////////////////////////////////////////////////////////////////////////
//! fire
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double fire (int t, patch** patchptr, UserData* data) {
   /* for all linked patches */

   if(data->fire_off) {
      return 0.0;
   }

   patch* cp =* patchptr;
   /* no fires on cropland */
   if (cp->landuse == LU_CROP) {
      return 0.0; 
   }

   double fireterm = 0.0;
   if(data->fire_gfed) {
       if (PATCH_FREQ==12)
       {
           for (size_t mon=0;mon<12;mon++)
           {
               fireterm += data->gfed_bf[mon][cp->siteptr->sdata->globY_][cp->siteptr->sdata->globX_];
           }
           fireterm /=12.0;
       }
       else
           fireterm = data->gfed_bf[data->time_period][cp->siteptr->sdata->globY_][cp->siteptr->sdata->globX_];

      return(fireterm);
   }

#if defined ED 
   for ( size_t i=0; i<N_SUB; i++ ) {
      fireterm += cp->lambda1[i] / ( 1.0 * N_SUB );
   }

#elif MIAMI_LU
   site* cs = cp->siteptr;

   cp->fuel = 0.0; 
   cp->ignition_rate = 0.0;
   cp->fuel = cp->total_biomass;

   if (cp->fuel > 0.0) {
      if (cs->sdata->precip_average < 400 + 40 * cs->sdata->temp_average) {
         fireterm = cp->fuel * (400 + 40 * cs->sdata->temp_average - cs->sdata->precip_average);
      }
   }
#endif

   if (fireterm > data->fire_max_disturbance_rate) {
      fireterm = data->fire_max_disturbance_rate; 
   }

#if DOES_COMPILE
   if(data->fire_suppression) {
       double old_fire;
       /* ramp fire down 1870 to 1970 to 2% of original value; ref Houghton */

       /* using values from ed. miami-lu was using 220*N_SUB in place of 170*N_SUB */

       if( (data->fire_suppression_stop && ((t > 170*N_SUB) && (t < 300*N_SUB)) ) 
               || (!data->fire_suppression_stop && (t > 170*N_SUB))) {
  
          old_fire = fireterm;

          /* using values from ed. miami-lu was using (t-220*N_SUB)*1.0/(50*N_SUB) */
          if ( !strcmp(data->region,"NORTH_AMERICA") || !strcmp(data->region,"US")  ) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.02) 
               fireterm = old_fire * 0.02;
          }
          else if (!strcmp(data->region,"SOUTH_AMERICA")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.09) 
               fireterm = old_fire * 0.09;
          }
          else if (!strcmp(data->region,"EUROPE")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.04) 
               fireterm = old_fire * 0.04;
          }
          else if (!strcmp(data->region,"AFRICA")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.43) 
               fireterm = old_fire * 0.43;
          }
          else if (!strcmp(data->region,"NASIA")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.22) 
               fireterm = old_fire * 0.22;
          }
          else if (!strcmp(data->region,"SASIA")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.22) 
               fireterm = old_fire * 0.22;
          }
          else if (!strcmp(data->region,"AUSTRALIA")) {
            fireterm -=  fireterm*(t-170*NSUB)*1.0/(100.0*N_SUB);
            if(fireterm < old_fire * 0.43) 
               fireterm = old_fire * 0.43;
          }
       }
   }/* FIRE_SUPPRESSION */
#endif
   return(fireterm);  
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! update_fuel
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void update_fuel (int t, patch** patchptr, UserData* data) {
   /* for all linked patches */
  
   patch* youngest_patch = *patchptr;
   site* cs = youngest_patch->siteptr;

   size_t landuse = youngest_patch->landuse;

   /* init */
   cs->fuel[landuse] = 0.0;
   cs->ignition_rate[landuse] = 0.0;
  
   patch* cp = youngest_patch;
   while (cp != NULL) {
      /* init patch fuel and ignition rate */
      cp->fuel = 0.0;
      cp->ignition_rate = 0.0;

      cohort* cc = cp->shortest;
      while(cc != NULL) {
         if (cc->hite < data->fire_hite_threshold) {
            cp->fuel += cc->nindivs * cc->babove;
         }
         cc = cc->taller;
      } /* end loop over cohorts */
    
      /* divide fuel by patch area */
      cp->fuel /= cp->area;
    
      /* add patch total to site total */
      cs->fuel[landuse] += cp->fuel * cp->area;
    
      cp = cp->older;
   }  /* end loop over patches */

   /* divide by site area */
   cs->fuel[landuse] /= (cs->area_fraction[landuse] * data->area);

   /* Fire ignition: sigmoid x fuel hump */
   /* Replaces power-law dryness formulation with intermediate productivity */
   /* hypothesis (Pausas & Ribeiro 2013, GEB 22:728-736). */
   /* Sigmoid: smooth ignition threshold as dryness accumulates. */
   /* Fuel hump: fire peaks at intermediate biomass (savanna optimum), */
   /* not in dense forests (humid) or deserts (no fuel). */
   {
       double D_bar = cs->sdata->dryness_index_avg;
       double AGB = cs->fuel[landuse];

       double k = 0.0286;
       double D0 = 128.4;
       double sigmoid = 1.0 / (1.0 + exp(-k * (D_bar - D0)));

       double fuel_build = 0.0755;
       double fuel_decay = 0.323;
       double fuel_hump = (1.0 - exp(-AGB / fuel_build)) * exp(-AGB / fuel_decay);

       cs->ignition_rate[landuse] = data->fp1 * sigmoid * fuel_hump;
   }

   /* calculate fire dist rate and add into array of within yr values */
   cs->lambda1[t%N_SUB][landuse] = 0.0;
   cp = youngest_patch;
   while (cp != NULL) {
      cp->lambda1[t%N_SUB] = cs->ignition_rate[landuse];
      cs->lambda1[t%N_SUB][landuse] += cp->lambda1[t%N_SUB] * cp->area;
      cp=cp->older;
   }  /* end loop over patches */
   cs->lambda1[t%N_SUB][landuse] /= cs->area_fraction[landuse] * data->area;  
}
#endif /* ED */

