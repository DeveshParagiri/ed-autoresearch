
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "read_site_data.h"
#include "cohort.h"

////////////////////////////////////////////////////////////////////////////////
//! Allocate_Biomass
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void cohort::Allocate_Biomass (UserData* data){
   hite = Hite(data);

   double salloc = (1.0 + data->qsw[species] * hite + data->q[species]);
    
   if ( status >= 5 ) {
      bl = 0.0;
      blv = (balive) / salloc;
   } else {
      bl = (balive) / salloc;
      blv = 0.0;
   }
   b = bdead + balive;
   br = data->q[species] * (balive) / salloc;
   bsw = data->qsw[species] * hite * (balive) / salloc; /*TODO bsw is area, not weight. Convert OLCR. Change hite to a parameter = mass of 1m^2 of bsw*/
   bs = bsw + bdead;
   bstem = data->agf_bs * bs;
   babove = bstem + bl;
   bbelow = (1.0 - data->agf_bs) * bs + br;
}

////////////////////////////////////////////////////////////////////////////////
//! Growth_Derivatives
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void cohort::Growth_Derivatives (double time, UserData* data ) {

   double dbldbd, dbrdbd, dbswdbd, dhdbd;
   double va, vs, u;
   double gr_fract;

   /***************************/
   /* growth and reproduction */
   /***************************/


   /* calculate gpp and npp and respiration */
   /* plant_water_and_nitrogen_relations(&cc,data); */
   npp_function(data);
  
   /* calculate maintenance demands = tissue decay */
   md = data->alpha[species][2] * bl +
      data->alpha[species][3] * br + data->alpha[species][4] * blv;

   /* calculate projected size of living biomass compartment */  
   double ba_star = Bleaf(data) * (1.0 + data->q[species] + data->qsw[species] * hite); /*TODO as in allcation bsw*/

   /* calculate excess carbon  (npp - md) */
   carbon_balance = npp - md;
   cb[data->time_period] = carbon_balance;
   cb_toc[data->time_period] = npp_max-md;
   
   double cb_act = 0.0;
   double cb_max = 0.0;

   for(int j=0; j<N_CLIMATE; j++){
      cb_act += cb[j];
      cb_max += cb_toc[j];
   }

   /* calc carbon balance ratio */
   if((cb_max)>0.0) {
      cbr_bar = cb_act/cb_max;
   } else {
      cbr_bar = 0.0;
   }

    cbr[data->time_period] = cbr_bar;
   
      /* on allometry */
   if ( (balive >= ba_star) && (carbon_balance > 0.0) ) { 
         /* calc carbon balance ratio */

         /* calculate va and vs */ 
         /* fraction of carbon going into active vs structural carbon */ 
         if ( dbh <= data->max_dbh[species] ) { /* cap on leaf biomass */
            dbldbd = dDbhdBd(data)/dDbhdBl(data); 
            dbrdbd = data->q[species]*dbldbd;
            dhdbd = dHdBd(data);
            dbswdbd = data->qsw[species]*(hite*dbldbd + bl*dhdbd);
             
            u  = 1.0/(dbldbd + dbrdbd + dbswdbd);     /* TODO Units don't match*/
            va = 1.0/(1.0 + u);
            vs = u/(1.0 + u);
            gr_fract = 1.0 - data->r_fract[species];
         } else {
             dbldbd = 0.0; dbrdbd = 0.0; dbswdbd = 0.0;      
             va = 0.0;
                 vs = 1.0;
                 gr_fract = 1.0 - (data->r_fract[species] + data->c_fract[species]);
            }
             }
      /* off allometry */                                  
      else{ 
            dbldbd = 0.0; dbrdbd = 0.0; dbswdbd = 0.0;
            va = 1.0; vs = 0.0;                         
            gr_fract = 1.0;
         }
   
   // repro_ht_thresh is minimum height of canopy below which there is no reproduction
   if(hite <= data->repro_ht_thresh[species] and data->hgt_lim_to_repro) {
      gr_fract = 1.0;
   }

   /* calculate derivatives of living and dead carbon pools */
   dbalivedt= gr_fract * va * carbon_balance;
   dbdeaddt = gr_fract * vs * carbon_balance;
   p[0] = (1.0 - gr_fract) * carbon_balance;
   p[1] = 0.0;

   /* calculate change in diameter and height */
   ddbhdt = dbdeaddt*dDbhdBd(data);
   dhdt = dbdeaddt*dHdBd(data);
    
   /* measured above ground npp = leaf litter + repro/stem litter +      *
    *                             above ground wood production +         *
    *                             leaf production                        */
   npp2 = bl * data->alpha[species][2] + p[0] +
      data->agf_bs * dbdeaddt + 0.5 * dbalivedt;
    
   /*************/
   /* mortality */
   /*************/
   dndt = -1.0 * den_dep_death(data) * (nindivs);
   
   if(!data->patch_dynamics) { /* if no pd add in all disturbance as addnal mortality */
      dndt -= patchptr->total_disturbance_rate * (nindivs);
   }
} 


////////////////////////////////////////////////////////////////////////////////
//! Litter
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void patch::Litter (double time, UserData* data ) {
//   /* calculate litter input rates from decay and mortality */
//
//   /*****************/
//   /* LITTER INPUTS */
//   /*****************/
//   /* Inputs from litter */

    /// If not working, delete the all below, and uncomment the above which is original version
    /// Here, I assume cohort die before accumulation before, that means this dead part won't participate photosynthesis
    /// Therfore, the biomass of dead part should use old_balive and old_bdead rathan than updated one.
    /// We also could assume this dead after carbon accumulation, but more codes are need to be modified.
    double fast_litter = 0.0;
    double fast_litter_n = 0.0;
    double struct_litter = 0.0;
    fire_c_loss = 0.0;
    
    cohort* cc = shortest;
    while ( cc != NULL ) {
        size_t spp = cc->species;

        /* plant litter now revised to subtract mean field fire, which goes to atm */
        double plant_litter = (cc->md * cc->nindivs + cc->old_balive *
                               (-1.0 * cc->dndt) - cc->old_balive *
                               cc->patchptr->siteptr->sdata->loss_fraction[1] *
                               (1.0 - cc->survivorship_from_disturbance(1, data)) *
                               cc->patchptr->fire_dndt_factor *
                               cc->nindivs) / area;
        if (cc->patchptr->fire_dndt_factor>0.0)
        {
            fire_c_loss += cc->siteptr->sdata->loss_fraction[1]*(cc->old_balive+cc->old_bdead)*(1.0-cc->survivorship_from_disturbance(1, data))*cc->patchptr->fire_dndt_factor*cc->nindivs/area;
        }
        fast_litter += data->fraction_balive_2_fast * plant_litter;
        fast_litter_n += data->fraction_balive_2_fast * (1.0 / data->c2n_leaf[spp]) * plant_litter;
        

        double seed_litter = 0.0;
#if LANDUSE
        if(landuse==LU_CROP)
        {
            seed_litter = ((cc->p[0] * data->sd_mort_crop) * cc->nindivs)/area;
        }
        else
        {
            seed_litter = ((cc->p[0] * data->sd_mort) * cc->nindivs)/area;
        }
#else
        seed_litter = ((cc->p[0] * data->sd_mort) * cc->nindivs)/area;
#endif
        
        double seed_litter_n = (1.0/data->c2n_recruit[spp]) * seed_litter;
        
        fast_litter +=  data->fraction_balive_2_fast * seed_litter;
        fast_litter_n +=  data->fraction_balive_2_fast * seed_litter_n;
        
        /* payment to N fixers */
        if ( data->Nfixer[cc->species] == 1 )
            fast_litter += cc->nindivs * cc->payment_to_Nfixers / area;
        

        /* now revised to include fire dndt loading */
        struct_litter += cc->old_bdead * (-1.0 * cc->dndt - cc->siteptr->sdata->loss_fraction[1] * (1.0 - cc->survivorship_from_disturbance(1, data)) * cc->patchptr->fire_dndt_factor * cc->nindivs)/area + (1.0 - data->fraction_balive_2_fast) * (plant_litter + seed_litter);
        cc = cc->taller;
        
        
    }  /* end loop over cohorts */
    
    litter = fast_litter + struct_litter;
    /* Calc Pool Inputs */
    fsc_in = fast_litter;
    ssc_in = struct_litter;     /*from stem*/
    ssl_in = (data->l2n_stem/data->c2n_stem) * struct_litter;
    fsn_in = fast_litter_n;
}


////////////////////////////////////////////////////////////////////////////////
//! nitrogen_demand_function
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double cohort::nitrogen_demand_function (double time, UserData* data ) {
   
   double n_demand=0.0;
   if (time<0.0001 or data->n_competition) {
      Growth_Derivatives(time, data);
   }
   //Note that arguably Growth_derivatives should be called every time to update key values.
   //However this does not do much, esp. if nitrogen_competition is off, and just slows everything down.

   /*n_demand += currentc->md/data->c2n_leaf[currentc->species];*/
   n_demand += dbalivedt / data->c2n_leaf[species];
   n_demand += dbdeaddt / data->c2n_stem;
   n_demand += p[0] / data->c2n_recruit[species];
   n_demand += p[1] / data->c2n_recruit[species];
   n_demand += md / data->c2n_leaf[species];
    
   return (n_demand);
} 

