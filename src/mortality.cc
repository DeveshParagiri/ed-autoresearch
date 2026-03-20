#include <cmath>
#include <cstring>
#include "edmodels.h"
#include "site.h"
#include "read_site_data.h"
#include "patch.h"
#include "cohort.h"

////////////////////////////////////////////////////////////////////////////////
//! survivorship_from_disturbance
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double cohort::survivorship_from_disturbance ( int q, UserData* data ) {
   /*q is the disturbance track*/

   /*printf("sfd: entering...\n");*/
   double csurv = 0.0;

   /*treefalls*/    
   if (q == 0) {
      if (hite < data->treefall_hite_threshold) {
         csurv = data->treefall_s_ltht[species]; 
         /*csurv = data->treefall_s_gtht[spp] + data->treefall_s_ltht[spp] * 
            ( 1.0 - cc->hite/data->treefall_hite_threshold );*/
      } else  {
         csurv = data->treefall_s_gtht[species];      
      }
   }  

   /*fires*/    
   else if (q == 1) {
      if (data->is_grass[species]) {
          csurv = 1.00;
      } else {
          csurv = 0.3;
      }

   } /* end if on fires */
   return csurv;
}

////////////////////////////////////////////////////////////////////////////////
//! cohort_modifications_from_disturbance
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void cohort_modifications_from_disturbance ( int q, cohort** pcurrentc, patch** newp,
                                             UserData* data ) {   
   cohort* cc = *pcurrentc;

    patch* np = *newp;
   /* q is disturbance track */

   /* grass species lose all leaf biomass and are returned to HMIN in Fires */

   if (q == 1) {
      if (data->is_grass[cc->species]) {

          double old_balive = cc->balive, old_bdead = cc->bdead;
         cc->hite = data->hgt_min[cc->species];
         cc->dbh = cc->Dbh(data);     
         cc->bdead = cc->Bdead(data);
         cc->balive = cc->balive - cc->bl;
         cc->bl = 0.0;
         cc->b = cc->balive + cc->bdead;
          
          np->fire_emission += (old_balive+old_bdead-cc->balive-cc->bdead)*cc->nindivs*LANDUSE_FREQ;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
//! den_dep_death
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double cohort::den_dep_death (UserData* data ) {
   double dmort = 0.0;

   // Mortality is computed differently for temperate vs tropical species
   if (siteptr->climate_zone==1)
   {
      data->m1 = 0.15;
      dmort = data->m1 * ( data->rho_max1 - data->rho[species] );
      dmort += 0.011;
   }
   else
   {
      dmort = data->den_ind_mort[species];
   }
    
   if ( hite < data->treefall_hite_threshold )
   {
      if (siteptr->climate_zone==1)
      {
         dmort += data->treefall_max_disturbance_rate_trop;
      }
      else
      {
         dmort += data->treefall_max_disturbance_rate_temp;
      }
   }
   
   dmort += data->m2 * 1.0 / (1.0 + exp(data->m3 * cbr_bar));

   if (patchptr->treefall_as_dndt_flag == 1)
   {
      if (siteptr->climate_zone==1)
      {
         dmort += data->treefall_max_disturbance_rate_trop;
      }
      else
      {
         dmort += data->treefall_max_disturbance_rate_temp;
      }
   }
    
   /* do mean field fire when treefall is dominant mode */
   if (patchptr->fire_as_dndt_flag == 1)
   {
      dmort += patchptr->fire_dndt_factor * (1.0 - survivorship_from_disturbance(1, data));
   }

   return dmort;
}


/******************************************************************************/
