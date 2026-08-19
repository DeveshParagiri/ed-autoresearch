#include <cstdio>
#include <cstring>
#include <cmath>
#include "edmodels.h"
#include "site.h"
#include "disturbance.h"
#include "read_site_data.h"
#include "mortality.h"
#ifdef ED
#include "cohort.h"
#endif

#include "patch.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//! init_patches
//! Initialize patches within grid cell
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void init_patches (site** siteptr, UserData* data) {
  
   /* function called during model initialization if not a restart        */
   /* This function creates number of patches specified by data->n_init_patches */

   /*printf("Initializing patches... \n");*/
   site* currents = *siteptr;   
   patch* currentp = NULL;

   /***  initialize patch based on input parameters  ***/   
   for (int i=0; i<data->n_init_patches; i++) {
      patch* newp = NULL;

      int track = 0;
      double age   = 0.0;
      double area = data->area * currents->area_fraction[LU_NTRL] / (data->n_init_patches * 1.0);

      /* initialize Soil Carbon Pools arbitary values */
      double fsc  = 0.01; 
      double stsc = 0.01; 
#if defined ED
      double stsl = 0.001; 
      double ssc  = 0.0; 
      double psc  = 0.0; 
      double msn  = 1.0;
      double fsn  = 1.0;
      /* initialize patch water to eqm value in abs of vegetation */
      double water = (currents->sdata->soil_depth * currents->sdata->theta_max) 
                   * pow(currents->sdata->precip_average / currents->sdata->k_sat, 
                         1.0 / (2.0 * currents->sdata->tau + 2.0));
    
#if SOILGRIDS_SCHEME
       water = 0.0;
       float k_sat = currents->sdata->MvG_k_sat;
       float soil_depth = currents->sdata->MvG_soil_depth;
       float theta_s = currents->sdata->MvG_theta_saturated;
       float theta_r = currents->sdata->MvG_theta_residual;
       float L = currents->sdata->MvG_L;
       float m = currents->sdata->MvG_m;
       float precip = currents->sdata->precip_average;
       float x0, x0_a, x0_b;
       float eps = 0.5;
       int n_iter = 0;
       int max_iter = 50;

       x0 = (theta_s+theta_r)/2.0*soil_depth;
       x0_a = (0.0000001*(theta_s-theta_r)+theta_r)*soil_depth;
       x0_b = (0.9999999*(theta_s-theta_r)+theta_r)*soil_depth;
       float f_x0, f_x0_a, f_x0_b;
       while(abs(x0_a-x0_b)>1.0)
       {
           x0 = (x0_a+x0_b)/2.0;
           f_x0 = MvG_func(x0, k_sat, soil_depth, theta_r, theta_s, L, m, precip);
           f_x0_a = MvG_func(x0_a, k_sat, soil_depth, theta_r, theta_s, L, m, precip);
           f_x0_b = MvG_func(x0_b, k_sat, soil_depth, theta_r, theta_s, L, m, precip);

           if(abs(f_x0)<eps)
               break;

           if (f_x0*f_x0_b<0.0)
               x0_a = x0;
           if (f_x0*f_x0_a<0.0)
               x0_b = x0;

           n_iter++;
           if (n_iter>max_iter)
           {
               printf("error in initilize water at %d-%d, params are %f %f %f %f %f %f %f %f\n",currents->sdata->globX_,currents->sdata->globY_,L,m,k_sat,theta_r,theta_s,precip,x0_a,x0_b);
               break;
           }
       }
       water = (x0_a+x0_b)/2.0;
#endif

      create_patch(&currents, &newp, LU_NTRL, track, age, area, 
                   water, fsc, stsc, stsl, ssc, psc, msn, fsn, data);
      init_cohorts(&newp, data);
#elif defined MIAMI_LU
      double tb    = 0.0;
      create_patch(&currents, &newp, LU_NTRL, track, age, area, fsc, stsc, tb, data);
#endif

      if (i == 0) {
         newp->younger = NULL; 
         newp->older   = NULL;
         currents->youngest_patch[LU_NTRL] = newp;
         currents->oldest_patch[LU_NTRL]   = newp;
         currentp = newp;
      } else {
         newp->older = NULL;
         newp->younger = currentp;
         currentp->older = newp;
         currentp = newp;
         currents->oldest_patch[LU_NTRL] = newp;
      }
   }
}

#if SOILGRIDS_SCHEME

float MvG_func(float water, float k_sat, float soil_depth, float theta_r, float theta_s, float L, float m, float pipcip)
{
    double se = 0.0;
    double y = 0.0;
    se = (water/soil_depth-theta_r)/(theta_s-theta_r);
    se = min(1.0,se);
    se = max(0.0,se);
    y = k_sat*pow(se, L)*pow(1.0-pow(1.0-pow(se,1/m),m),2.0) - pipcip;
    return y;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! create_patch
//! Create new patch
//! This function is called during the model run initialization either from init 
//! patches or from read patch distribition if a restart. It no longer inits cohorts.
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
#if defined ED
void create_patch (site** siteptr, patch** pnewp, int landuse,
                   int track, double age, double area, double water, 
                   double fsc, double stsc, double stsl, double ssc, double psc, 
                   double msn, double fsn, UserData* data) {
#elif defined MIAMI_LU
void create_patch (site** siteptr, patch** pnewp, int landuse,
                   int track, double age, double area, double fsc, double stsc, 
                   double tb, UserData* data) {
#endif

   site* current_site = *siteptr; /* assign pointer to site */

   patch* newpatch = (patch*) malloc (sizeof(patch));
   if (newpatch == NULL) {
      fprintf(stderr, "create_patch: out of memory\n");
      exit(1);
   }
   
   /* assign patch attributes */
   newpatch->track              = track;
   newpatch->age                = age;   
   newpatch->area               = area; 
   newpatch->landuse            = landuse;
   newpatch->siteptr            = current_site; /*pointer to parent site*/
   newpatch->fast_soil_C        = fsc;
   newpatch->structural_soil_C  = stsc;
#if defined ED
   newpatch->water              = water;
   newpatch->structural_soil_L  = stsl;
   newpatch->slow_soil_C        = ssc;
   newpatch->passive_soil_C     = psc;
   newpatch->mineralized_soil_N = msn;
   newpatch->fast_soil_N        = fsn;
#elif defined MIAMI_LU
   newpatch->total_biomass      = tb;
   newpatch->total_ag_biomass   = data->agf_biomass * tb;
#endif

   for (size_t i=0; i<N_SUB; i++) {
      newpatch->lambda1[i] = 0.0;
   }

   for (size_t i=0; i<NUM_TRACKS; i++) {
      newpatch->disturbance_rate[i] = 0.0;
   }

   newpatch->older            = NULL;
   newpatch->younger          = NULL;
   newpatch->rh               = 0.0;
   newpatch->nep              = 0.0;
   newpatch->npp              = 0.0;
   newpatch->gpp_avg           = 0.0;
   newpatch->npp_avg           = 0.0;
   newpatch->rh_avg            = 0.0;
   newpatch->fire_emission     = 0.0;
   newpatch->fire_c_loss       = 0.0;
#if LANDUSE
   newpatch->product_emission = 0.0;
   newpatch->forest_harvested_c = 0.0;
   newpatch->yr1_decay_product_pool = 0.0;
   newpatch->yr10_decay_product_pool = 0.0;
   newpatch->yr100_decay_product_pool = 0.0;
   newpatch->past_harvested_c = 0.0;
   newpatch->crop_harvested_c = 0.0;
#endif
    
   newpatch->fire_dndt_factor = 0.0;
   newpatch->A                = 0.0;

#ifdef ED
   newpatch->tallest            = NULL;
   newpatch->shortest           = NULL;
   newpatch->total_ag_biomass   = 0.0;
   newpatch->total_biomass      = 0.0;
   newpatch->basal_area         = 0.0;    
   newpatch->theta              = newpatch->water / (current_site->sdata->soil_depth * current_site->sdata->theta_max);
   newpatch->total_water_uptake = 0.0;
   newpatch->total_water_demand = 0.0;
   newpatch->soil_evap         = 0.0;
  
#if SNOWPACK_SCHEME == 1
   newpatch->snowpack = 0.0;
   newpatch->snow_melt = 0.0;
#endif
    
   newpatch->perc=0.0;

   /* assign elements of integration array */
   newpatch->fsc_e  = 1;
   newpatch->fsn_e  = 5;
   newpatch->fstd   = 0.0;
   for (size_t spp=0; spp<NSPECIES; spp++) {
      newpatch->repro[spp]             = 0.0; /* initialize birth array to zero */
      newpatch->total_spp_biomass[spp] = 0.0;
      newpatch->total_spp_babove[spp]  = 0.0; 
      newpatch->basal_area_spp[spp]    = 0.0; 
   }
#endif /* ED */

   /*calloc array for disturbance A history and set pointers*/
   if (landuse == LU_SCND) { 
      /*allocate memory for array*/
      double* parray = (double *) calloc(data->n_years_to_simulate + 1, sizeof(double));
      /*initialize array*/
      for (size_t j=0; j<data->n_years_to_simulate+1; j++)
         *(parray + j) = 0.0;
      newpatch->phistory = parray;    
   }

   (*pnewp) = newpatch;
}
    
#if LANDUSE
//this function will read product pool, only active for landuse
void create_patch_LU (site** siteptr, patch** pnewp, int landuse,
                   int track, double age, double area, double water,
                   double fsc, double stsc, double stsl, double ssc, double psc,
                   double msn, double fsn, double yr1_pool, double yr10_pool, double yr100_pool, UserData* data)
    {
        site* current_site = *siteptr; /* assign pointer to site */
        
        patch* newpatch = (patch*) malloc (sizeof(patch));
        if (newpatch == NULL) {
            fprintf(stderr, "create_patch: out of memory\n");
            exit(1);
        }
        
        /* assign patch attributes */
        newpatch->track              = track;
        newpatch->age                = age;
        newpatch->area               = area;
        newpatch->landuse            = landuse;
        newpatch->siteptr            = current_site; /*pointer to parent site*/
        newpatch->fast_soil_C        = fsc;
        newpatch->structural_soil_C  = stsc;
#if defined ED
        newpatch->water              = water;
        newpatch->structural_soil_L  = stsl;
        newpatch->slow_soil_C        = ssc;
        newpatch->passive_soil_C     = psc;
        newpatch->mineralized_soil_N = msn;
        newpatch->fast_soil_N        = fsn;
#elif defined MIAMI_LU
        newpatch->total_biomass      = tb;
        newpatch->total_ag_biomass   = data->agf_biomass * tb;
#endif
        
        for (size_t i=0; i<N_SUB; i++) {
            newpatch->lambda1[i] = 0.0;
        }
        
        for (size_t i=0; i<NUM_TRACKS; i++) {
            newpatch->disturbance_rate[i] = 0.0;
        }
        
        newpatch->older            = NULL;
        newpatch->younger          = NULL;
        newpatch->rh               = 0.0;
        newpatch->nep              = 0.0;
        newpatch->npp              = 0.0;

        newpatch->gpp_avg           = 0.0;
        newpatch->npp_avg           = 0.0;
        newpatch->rh_avg            = 0.0;
        newpatch->fire_emission     = 0.0;
        newpatch->fire_c_loss       = 0.0;
#if LANDUSE
        newpatch->product_emission = 0.0;
        newpatch->forest_harvested_c = 0.0;
        newpatch->yr1_decay_product_pool = yr1_pool;
        newpatch->yr10_decay_product_pool = yr10_pool;
        newpatch->yr100_decay_product_pool = yr100_pool;
        newpatch->past_harvested_c = 0.0;
        newpatch->crop_harvested_c = 0.0;
#endif
        
        newpatch->fire_dndt_factor = 0.0;
        newpatch->A                = 0.0;
        
#ifdef ED
        newpatch->tallest            = NULL;
        newpatch->shortest           = NULL;
        newpatch->total_ag_biomass   = 0.0;
        newpatch->total_biomass      = 0.0;
        newpatch->basal_area         = 0.0;
        newpatch->theta              = newpatch->water / (current_site->sdata->soil_depth
                                                          * current_site->sdata->theta_max);
        newpatch->total_water_uptake = 0.0;
        newpatch->total_water_demand = 0.0;
        newpatch->soil_evap         = 0.0;
        
#if SNOWPACK_SCHEME == 1
        newpatch->snowpack = 0.0;
        newpatch->snow_melt = 0.0;
#endif
        newpatch->perc = 0.0;
        
        /* assign elements of integration array */
        newpatch->fsc_e  = 1;
        newpatch->fsn_e  = 5;
        newpatch->fstd   = 0.0;
        for (size_t spp=0; spp<NSPECIES; spp++) {
            newpatch->repro[spp]             = 0.0; /* initialize birth array to zero */
            newpatch->total_spp_biomass[spp] = 0.0;
            newpatch->total_spp_babove[spp]  = 0.0;
            newpatch->basal_area_spp[spp]    = 0.0;
        }
#endif /* ED */
        
        /*calloc array for disturbance A history and set pointers*/
        if (landuse == LU_SCND) {
            /*allocate memory for array*/
            double* parray = (double *) calloc(data->n_years_to_simulate + 1, sizeof(double));
            /*initialize array*/
            for (size_t j=0; j<data->n_years_to_simulate+1; j++)
                *(parray + j) = 0.0;
            newpatch->phistory = parray;
        }
        
        (*pnewp) = newpatch;
    }
    
#endif

////////////////////////////////////////////////////////////////////////////////
//! update_patch
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void update_patch (patch** current_patch, UserData* data) {

   patch* cp = *current_patch;

#if defined ED
   /* biomass */
   light_levels(&cp, data);
   for (size_t i=0; i<NSPECIES; i++) {
      cp->total_spp_biomass[i] = 0.0;
      cp->total_spp_babove[i]  = 0.0;
      cp->basal_area_spp[i]    = 0.0;
   }
   cp->total_ag_biomass = 0.0;
   cp->total_biomass    = 0.0;
   cp->basal_area       = 0.0;            
   cp->lai              = 0.0;
   for (size_t i=0;i<N_LAI;i++)
   {
      cp->lai_profile[i]=0;
   }
   cohort* cc = cp->shortest;
   while (cc != NULL) {
      if ((cc->dbh >= data->min_dbh_class) && (cc->hite > data->min_hgt_class)) {
         cp->total_spp_biomass[cc->species] +=cc->b * cc->nindivs;
         cp->total_spp_babove[cc->species]  +=cc->babove * cc->nindivs;
         cp->total_ag_biomass               += cc->babove * cc->nindivs;

         if(abs(cc->balive+cc->bdead-cc->b)>1e-9)
         {
            cc->b = cc->balive + cc->bdead;
         }
         cp->total_biomass                  += cc->b * cc->nindivs;
         cp->basal_area                     += (M_PI/4.0) * pow(cc->dbh, 2.0) * cc->nindivs;
         cp->basal_area_spp[cc->species]    += (M_PI/4.0) * pow(cc->dbh, 2.0) * cc->nindivs;      
         cp->lai += cc->lai;
         if (cc->hite>LAI_INTERVAL[N_LAI-1])
         {
            cp->lai_profile[N_LAI-1]+=cc->lai;
         }
         else
         {
            for (size_t i=0;i<N_LAI-1;i++)
            {
               if (cc->hite>LAI_INTERVAL[i] && cc->hite<LAI_INTERVAL[i+1])
               {
                  cp->lai_profile[i]+=cc->lai;
                  break;
               }
            }
         }
      } /* end if */
      cc = cc->taller;
   } /* end loop over cohorts */
   /* divide by patch area */
   for (size_t i=0; i<NSPECIES; i++) {
      cp->total_spp_biomass[i] /= cp->area;
      cp->total_spp_babove[i] /= cp->area;
      cp->basal_area_spp[i] /= cp->area;
   }
   cp->total_ag_biomass /= cp->area;
   cp->total_biomass /= cp->area;
   cp->basal_area /= cp->area;
   
   /* soil_pools */
   cp->total_soil_c = cp->fast_soil_C + cp->structural_soil_C + cp->slow_soil_C + cp->passive_soil_C;

#elif defined MIAMI_LU
   /* soil pools */
   cp->total_soil_c = cp->fast_soil_C + cp->structural_soil_C ;
#endif

   /* total carbon */
   cp->total_c = cp->total_biomass + cp->total_soil_c;
    

   /* c fluxes */
   if (data->time_period == 0) { /* reset annual averages */
      cp->aa_lai  = 0.0;
      for (size_t i=0;i<N_LAI;i++)
      {
         cp->aa_lai_profile[i]=0;
      }
      cp->aa_npp  = 0.0; 
      cp->aa_nep  = 0.0; 
      cp->aa_rh   = 0.0;
#ifdef ED
      cp->aa_npp2 = 0.0; 
      cp->aa_gpp  = 0.0;
#endif
   }
   cp->dndt    = 0.0;
#ifdef ED
   cp->npp     = 0.0;
   cp->nep     = 0.0;
   cp->npp2    = 0.0;
   cp->gpp     = 0.0;
   cp->fs_open     = 0.0;

   cc = cp->shortest;
   double tmp_nindiv_number=0;
   while(cc != NULL){
      double hgtmin = data->hgtmin;
      if(data->allometry_type == 1 and !data->is_grass[cc->species] 
              and !data->is_tropical[cc->species]) {
         hgtmin = data->min_hgt[cc->species];
      }
      if(cc->hite > hgtmin){
         cp->npp += cc->npp * cc->nindivs / cp->area;
         cp->nep += cc->npp * cc->nindivs / cp->area;
         cp->npp2 += cc->npp2 * cc->nindivs / cp->area;
         cp->gpp += cc->gpp * cc->nindivs / cp->area;
         cp->dndt += cc->dndt / cp->area;
         cp->fs_open += cc->fs_open*cc->nindivs / cp->area;
         tmp_nindiv_number += cc->nindivs / cp->area;
      }
      cc = cc->taller;
   } /* end loop over cohorts */
    if (tmp_nindiv_number>0)
        cp->fs_open /=tmp_nindiv_number;
    else
        cp->fs_open = 0.0;
    
   cp->nep -= cp->rh;
#elif defined MIAMI_LU
   cp->npp = cp->siteptr->sdata->miami_npp;
   cp->nep = cp->npp - cp->rh;
#endif

   /* acculmulate annual averages */
   cp->aa_lai  += cp->lai * data->deltat;
    for (size_t i=0;i<N_LAI;i++)
    {
        cp->aa_lai_profile[i]+=cp->lai_profile[i]*data->deltat;
    }
   cp->aa_npp  += cp->npp * data->deltat;
   cp->aa_nep  += (cp->npp - cp->rh) * data->deltat;
   cp->aa_rh   += cp->rh * data->deltat;
#ifdef ED
   cp->aa_npp2 += cp->npp2 * data->deltat;
   cp->aa_gpp  += cp->gpp * data->deltat;
#endif 
}

////////////////////////////////////////////////////////////////////////////////
//! patch_dynamics
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void patch_dynamics ( unsigned int t, patch** patchptr, 
                      FILE* outfile, UserData* data ) {
 
   patch* youngest_patch = *patchptr;
   int lu = youngest_patch->landuse; // Store this in case youngest_patch gets deleted in fusion
   site* currents = youngest_patch->siteptr;
   patch *currentp = NULL;

   if(data->patch_dynamics) {
      /*if((t%(PATCH_FREQ+PATCH_SPAWN_MONTH)==  0)&&(t>0)){*/
      if ( (t % PATCH_FREQ == 0) && (t > 0) ) {

         if(data->patch_fusion) {
            if(data->cd_file) {
               fprintf(outfile, "fusing patches... \n");
            }
            fuse_patches(t, &youngest_patch, data);
         }
         /*landuse*/
         youngest_patch = currents->youngest_patch[lu];

         /*patch dynamics*/   
         if(data->cd_file) {
            fprintf(outfile,"patch dynamics...  \n");
         }

         currentp = youngest_patch;
         while (currentp != NULL) {  
            /* MORTALITY, treefalls only */    
            calculate_disturbance_rates(t, &currentp, data);

            /* AGING */
            // TODO: is PATCH_FREQ in configuration?
            currentp->age += data->deltat * PATCH_FREQ;

#ifdef MIAMI_LU
            double area_disturbed_fire = currentp->area * (1.0 - exp(-1.0 * currentp->disturbance_rate[1]
                                                                     * data->deltat * PATCH_FREQ)); 
            currents->area_burned += area_disturbed_fire;
            double area_disturbed_treefall = currentp->area * (1.0 - exp(-1.0 * currentp->disturbance_rate[0]
                                                                         * data->deltat * PATCH_FREQ));
            accumulate_litter_from_disturbance(&currentp, &currentp, area_disturbed_treefall, 0, data);  
            accumulate_litter_from_disturbance(&currentp, &currentp, area_disturbed_fire, 1, data);
            currentp->total_ag_biomass -= currentp->total_ag_biomass 
               * (1.0 - exp(-1.0 * currentp->total_disturbance_rate * data->deltat * PATCH_FREQ));
            currentp->total_biomass -= currentp->total_biomass 
               * (1.0 - exp(-1.0*currentp->total_disturbance_rate * data->deltat * PATCH_FREQ));
         
            /* GROWTH */
            double factor = 1.0;   /*if(currentp->landuse==LU_PAST) factor=1.5; else factor=1.0;*/      
            currentp->total_ag_biomass += factor * data->deltat * data->agf_biomass 
               * currents->sdata->miami_npp * data->wood_allocation;
            currentp->total_biomass += factor * data->deltat 
               * currents->sdata->miami_npp * data->wood_allocation;
            currentp->structural_soil_C += factor * data->deltat
               * currents->sdata->miami_npp * (1.0 - data->wood_allocation)
               * (1.0 - data->fraction_balive_2_fast);
            
            /* Below ground dynamics */
            currentp->Dsdt(data);
#endif /* MIAMI_LU */
            
            currentp = currentp->older;
         }        

#ifdef ED
         for (int q=0; q<NUM_TRACKS; q++) {
            /* NEW PATCH  INITIALIZATION */
            patch* newp = NULL;
               create_patch(&currents, &newp, youngest_patch->landuse, q,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, data);

            currents->new_patch[youngest_patch->landuse] = newp;

            double area_burned = 0.0;
            double change_in_area = 0.0;
            currentp = youngest_patch;
            while (currentp != NULL) {
               double change_in_area_factor = (currentp->disturbance_rate[q])
                  / (currentp->total_disturbance_rate) 
                  * (1.0 - exp(-1.0 * (currentp->total_disturbance_rate) 
                               * data->deltat * PATCH_FREQ));
               change_in_area += currentp->area * change_in_area_factor;

               if (q == 1) {
                  area_burned +=  currentp->area * change_in_area_factor;
                  area_burned += currentp->area 
                     * (1.0 - exp(-1.0 * (currentp->fire_dndt_factor) 
                                  * data->deltat * PATCH_FREQ));
               }  
               currentp = currentp->older;
            }

            /*Is track important enough to worry about*/
            bool use_track_flag = (change_in_area > data->smallest_new_patch_f * data->area);

            if(q==1) {
               currents->area_burned += area_burned;
            }

            if (use_track_flag == 1) { /*go through patches and init newp*/   
               currentp = youngest_patch;
               while (currentp != NULL) {    
                  /*patch dynamics!*/
                  double change_in_area_factor = (currentp->disturbance_rate[q])
                                               / (currentp->total_disturbance_rate) 
                                               * (1.0 - exp(-1.0 * (currentp->total_disturbance_rate) 
                                               * data->deltat * PATCH_FREQ));
  
                  /*compute change in area to donor patch*/
                  change_in_area = currentp->area * change_in_area_factor;

#if 0 // TODO: make sure this is working with coupled year
                  /*increase area of history array elements*/
                  if (newp->landuse == LU_SCND) { 
                     for (size_t i=0; i<=data->year; i++)
                        *(newp->phistory + i) += *(currentp->phistory + i) * change_in_area_factor;
                  }
#endif
                  /*update area of new patch*/
                  newp->area += change_in_area;

                  /*************************************/
                  /*ACCULMULATE LITTER FROM DISTURBANCE*/
                  /*************************************/
                  accumulate_litter_from_disturbance(&newp, &currentp, change_in_area, q, data);
                  if(data->do_hurricane) {
                     if (q == 0) {
                        double hurr_frac = get_hurricane_disturbance_rate(t, currents, data)
                           / currentp->disturbance_rate[0];
                        /* This only works because litter is being reset in      *
                         * accumulate_litter_from_disturbance for each new patch.* 
                         * Will need to fix when that is fixed. JF               */
                        currents->hurricane_litter += hurr_frac * newp->litter * change_in_area / data->area;
                     }
                  } /* HURRICANE */


                  /***************************************************/
                  /* TRACK SURVIVORS FROM DISTURBANCE INTO NEW PATCH */
                  /***************************************************/
                  cohort* currentc = currentp->shortest;
                  while (currentc != NULL) {                    
                     // make new cohort
                     cohort* newcohort = (cohort*) malloc(sizeof(cohort));
                     if (newcohort == NULL) {
                        printf("pd: out of memory\n");
                        while(getchar() != '\n');
                     }
                       
                     // copy cohort
                     copy_cohort(&currentc,&newcohort);
      
                     // surviving indivs
                     newcohort->nindivs = currentc->survivorship_from_disturbance(q, data)
                        * currentc->nindivs * change_in_area / currentp->area;
                      
                    
                     // Weighted average of dndt 
                     newcohort->dndt = newcohort->nindivs*currentc->dndt/currentc->nindivs;
                    
                     // disturbances such as fire can modify cohort properties
                      /// Pass newp into the below function to receive fire emission
                     cohort_modifications_from_disturbance(q, &newcohort,&newp, data);
       
                     // insert
                     insert_cohort(&newcohort, &newp->tallest, &newp->shortest, data);
     
                     newcohort->patchptr = newp;
                        
                     // loss of individuals, as this is a number not a density
                     currentc->nindivs -= currentc->nindivs*change_in_area/currentp->area;   
                     currentc = currentc->taller;    
                  } /* end loop over cohorts */
     
                  /***************************************************************/
                  /*   AGGREGATE IN PATCH SITE VARIABLES (repro Water, N, etc...)  */
                  /***************************************************************/    
                  /*printf("pd: average in patch site variables\n");*/
                  /* repro  averaged in -weighted, repro not a density*/ 
                  for (size_t i=0;i<NSPECIES;i++) {
                     newp->repro[i] += currentp->repro[i]*change_in_area/currentp->area;
                  }
	
                  aggregate_in_soil_state_from_disturbance(&newp,&currentp,change_in_area,data);
	
                  /* update area of donor patch */ 
                  currentp->area -= change_in_area;
     
#if 0 // TODO: make sure this works with coupled year
                  /*decrease areas of array elements*/
                  if (currentp->landuse == LU_SCND) { 
                     for (size_t i=0; i<data->year+1; i++) {
                        *(currentp->phistory +i) -= *(currentp->phistory+i) * change_in_area_factor;
                     }
                  }
#endif
                  currentp = currentp->older;      
               }
               // Divide aggregate variables (soil C, N etc.) by area of new path
	       // to obtain averages
	       newp->fast_soil_C /= newp->area;
	       newp->structural_soil_C /= newp->area;
	       newp->fast_soil_N /= newp->area;
          newp->slow_soil_C /= newp->area;
	       newp->passive_soil_C /= newp->area;
	       newp->mineralized_soil_N /=newp->area;
	       newp->structural_soil_L /= newp->area;
	       newp->water /= newp->area;
	       newp->theta /= newp->area;
            
#if SNOWPACK_SCHEME == 1
            newp->snowpack/=newp->area;
            newp->snow_melt/=newp->area;
#endif
                
            newp->soil_evap /= newp->area;
	         newp->rh /= newp->area;
            newp->gpp_avg /= newp->area;
            newp->npp_avg /= newp->area;
            newp->rh_avg /= newp->area;
            newp->fire_emission /= newp->area;
                ///
#if LANDUSE
            newp->forest_harvested_c /= newp->area;
            newp->yr1_decay_product_pool /= newp->area;
            newp->yr10_decay_product_pool /= newp->area;
            newp->yr100_decay_product_pool /= newp->area;
                
            newp->crop_harvested_c /= newp->area;
            newp->past_harvested_c /= newp->area;
            newp->product_emission /= newp->area;
#endif
               /**************************/
               /***  INSERT NEW PATCH   **/    
               /**************************/  
               /*printf("pd: insert new patch\n");*/

               patch* target = newp;  
               currentp = youngest_patch;
               target -> older = currentp;
               target -> younger = NULL;
               currentp -> younger = target;
               currents->youngest_patch[youngest_patch->landuse] = target;
                
                double tmp_total_tb = 0.0, tmp_total_sc = 0.0, tmp_total_tc = 0.0;
                patch* mlcp =currents->youngest_patch[youngest_patch->landuse];
                while (mlcp!=NULL) {
                    cohort* mlcc = mlcp->shortest;
                    double tmp_tb = 0.0;
                    while (mlcc!=NULL) {
                        tmp_tb += (mlcc->balive+mlcc->bdead)*mlcc->nindivs/mlcp->area;
                        mlcc = mlcc->taller;
                    }
                    tmp_total_tb += tmp_tb*mlcp->area/data->area;
                    tmp_total_sc += (mlcp->fast_soil_C+mlcp->structural_soil_C+mlcp->slow_soil_C+mlcp->passive_soil_C)*mlcp->area/data->area;
                    mlcp = mlcp->older;
                }
                //printf("tmpck tb %.15f sc %.15f tc %.15f\n",tmp_total_tb,tmp_total_sc,tmp_total_tb+tmp_total_sc);
                

               /*terminate cohorts*/
               terminate_cohorts(&target->tallest,&target->shortest,data);
            } else { /*not using track*/  
               if (newp->landuse == LU_SCND)
                  free(newp->phistory);
               free(newp); 
            }
         } /* end loop over tracks */

#if 0 /* AREA CHECK */
         printf("AREA CHECK END OF PD\n");
         currentp = youngest_patch;
         while(currentp != NULL){
            printf("pd: site %s patch %p  age %f area %f landuse %d standptr %p\n",
                   currents->name, currentp, currentp->age, currentp->area,
                   currentp->landuse, currentp->standptr); 
            currentp = currentp->older;
         }
#endif

         currents->fire_flag[youngest_patch->landuse] = 0; /* reset fire flag */
#endif /* ED */

         if(data->patch_termination) {
            /* patch termination */
            if(data->cd_file)
               fprintf(outfile, "terminating patches \n");  

             terminate_patches(&youngest_patch, data);
         }

      }  /* end t%PATCH_FREQ */
   } /* PATCH_DYNAMICS */
}

////////////////////////////////////////////////////////////////////////////////
//! fuse_patches
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void fuse_patches (unsigned int t, patch** patchptr, UserData* data) {
   /*ALGORITHM*/
   /*set all fusion flags to true*/
   /*create size profiles*/
   /*goto every patch*/
   /*find next older patch with same track*/
   /*check fusion criterion*/
   /*if within criterion, fuse, otherwise, skip*/

   patch* youngest_patch = *patchptr;
   
   char filename[256];
   FILE* outfile;
   if(data->fp_file) {
      strcpy(filename,data->base_filename);
      strcat(filename,".fp");
      if (t==0) outfile=fopen(filename, "w");
      else if(t != 0 && data->long_fp_file) outfile=fopen(filename, "a");
      else outfile=fopen(filename, "w");    
   }
   
   /* loop over patches and create species size profiles */
   patch* currentp = youngest_patch;
   while (currentp != NULL) {
#ifdef ED
      species_patch_size_profile(&currentp, N_DBH_BINS, data);
#endif
      currentp->fuse_flag = 1;
      currentp = currentp->older;
   }

   currentp = youngest_patch;
   while ( (currentp != NULL) && (currentp->older != NULL) ) {
    
      /*find a given patches' fusion candidate*/  
      patch* nextp = currentp->older;
      bool stop = false;
      patch* targetp = NULL;
      while ( (nextp != NULL) && (! stop) ) {
         if (currentp->track == nextp->track) {
            stop = true;
            targetp = nextp;
         }
         nextp = nextp->older;
      }

      if (targetp != NULL) {   /*found a fusion candidate*/
         /*fusion criterion*/
         double norm = 0.0;
#if defined ED
         for (size_t i=0; i<NSPECIES; i++) {          /* loop over species  */
            for (size_t j=0; j<N_DBH_BINS; j++){      /* loop over hgt bins */
               if ((currentp->spp_density_profile[i][j] > data->ntol) 
                   || (targetp->spp_density_profile[i][j] > data->ntol)) {
                  norm = fabs(currentp->spp_density_profile[i][j] 
                              - targetp->spp_density_profile[i][j])
                     / (0.5 * (currentp->spp_density_profile[i][j] 
                               + targetp->spp_density_profile[i][j]));
#elif defined MIAMI_LU
         norm = fabs(currentp->total_biomass-targetp->total_biomass) 
            / (0.5 * (currentp->total_biomass + targetp->total_biomass));
#endif
                  if (norm > data->profile_tol || targetp->age > data->max_patch_age) {
                     currentp->fuse_flag = 0;  /*reject*/
                  }
#ifdef ED
               } /* end if on data->ntol */
            }
         } /* end: fusion criterion */
#endif

         /*fusion*/
         if (currentp->fuse_flag == 1) {
             if(data->fp_file)
               fprintf(outfile,
                    "fp: time %f FUSION donorp %p dt %u age %f area %f targetp %p dt %u age %f area %f\n",
                    t * TIMESTEP, currentp, currentp->track, currentp->age, currentp->area,
                    targetp, targetp->track, targetp->age, targetp->area);

             patch* tmpptr = currentp->older;  /*tmpptr is needed bc f2p frees currentp*/
            fuse_2_patches(&currentp, &targetp, 1, data);
            currentp = tmpptr;
            if(data->fp_file)
               fprintf(outfile,
                    "fp: time %f FUSION RESULT p %p dt %u age %f area %f\n",
                    t * TIMESTEP, targetp, targetp->track, targetp->age, targetp->area);      
         } else {
            currentp = currentp->older;
         }
      } /* end if target patch != NULL */
      else{
         currentp = currentp->older;
      }
   }

   if(data->fp_file) {
      fprintf(outfile,"fp: exiting fuse patches \n"); 
      fclose(outfile);
   }
}

////////////////////////////////////////////////////////////////////////////////
//! fuse_2_patches
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void fuse_2_patches (patch** patchptr1, patch** patchptr2,
                     short update_ptrs, UserData* data) {

   /* This function fuses the two patches specified in the argument.
      It fuses the first patch in the argument (the "donor") into the second
      patch in the argument (the "recipient"), and frees the memory 
      associated with the first patch */

   patch* dp = *patchptr1;
   patch* rp = *patchptr2;
    
   double new_area = rp->area + dp->area;

   /* area weighted average of ages */
   rp->age = (dp->age * dp->area + rp->age * rp->area) / new_area;

   /*area weighted average of soil Carbon and N*/
   rp->fast_soil_C = (dp->fast_soil_C * dp->area 
                      + rp->fast_soil_C * rp->area) / new_area;
   rp->structural_soil_C = (dp->structural_soil_C * dp->area 
                            + rp->structural_soil_C * rp->area) / new_area;
#if defined ED
   rp->slow_soil_C = (dp->slow_soil_C * dp->area 
                      + rp->slow_soil_C * rp->area) / new_area;
   rp->passive_soil_C = (dp->passive_soil_C * dp->area 
                         + rp->passive_soil_C * rp->area) / new_area;
   rp->mineralized_soil_N = (dp->mineralized_soil_N * dp->area 
                             + rp->mineralized_soil_N * rp->area) / new_area;
   rp->fast_soil_N = (dp->fast_soil_N * dp->area 
                      + rp->fast_soil_N * rp->area) / new_area;
   rp->structural_soil_L = (dp->structural_soil_L * dp->area 
                            + rp->structural_soil_L * rp->area) / new_area;

   /*area weighted average of water*/
   rp->water = (dp->water * dp->area 
                + rp->water * rp->area) / new_area;
   rp->theta = (dp->theta * dp->area 
                + rp->theta * rp->area) / new_area;
    
    rp->gpp_avg = (dp->gpp_avg*dp->area+rp->gpp_avg*rp->area)/new_area;
    rp->npp_avg = (dp->npp_avg*dp->area+rp->npp_avg*rp->area)/new_area;
    rp->rh_avg = (dp->rh_avg*dp->area+rp->rh_avg*rp->area)/new_area;
    rp->fire_emission = (dp->fire_emission*dp->area+rp->fire_emission*rp->area)/new_area;
#if LANDUSE
    rp->product_emission = (dp->product_emission*dp->area+rp->product_emission*rp->area)/new_area;
    rp->forest_harvested_c = (dp->forest_harvested_c*dp->area+rp->forest_harvested_c*rp->area)/new_area;
    rp->yr1_decay_product_pool = (dp->yr1_decay_product_pool*dp->area+rp->yr1_decay_product_pool*rp->area)/new_area;
    rp->yr10_decay_product_pool = (dp->yr10_decay_product_pool*dp->area+rp->yr10_decay_product_pool*rp->area)/new_area;
    rp->yr100_decay_product_pool = (dp->yr100_decay_product_pool*dp->area+rp->yr100_decay_product_pool*rp->area)/new_area;
    rp->past_harvested_c = (dp->past_harvested_c*dp->area+rp->past_harvested_c*rp->area)/new_area;
    rp->crop_harvested_c = (dp->crop_harvested_c*dp->area+rp->crop_harvested_c*rp->area)/new_area;
#endif
  
   /* average of repro buffers */
   for (size_t i=0; i<NSPECIES; i++)
      rp->repro[i] = (rp->repro[i] * rp->area 
                      + dp->repro[i] * dp->area) / new_area;
  
#elif defined MIAMI_LU
   /* area weighted average of biomass */
   rp->total_biomass = (dp->total_biomass * dp->area 
                        + rp->total_biomass * rp->area) / new_area;
   rp->total_ag_biomass = (dp->total_ag_biomass * dp->area
                           + rp->total_ag_biomass * rp->area) / new_area;  
#endif

   rp->area = new_area;

   rp->fire_dndt_factor = dp->fire_dndt_factor;
    
#if LANDUSE
#if 0 // TODO: make this works with coupled year
   /*sum of areas in landuse history array*/
   if (rp->landuse == LU_SCND) {
      for (size_t i=0; i<data->n_years_to_simulate; i++) {
         *(rp->phistory + i) += *(dp->phistory + i);
      }
   }
#endif
#endif

#ifdef ED
   /*insert cohorts to new patch*/
   cohort* cc = dp->shortest;
   cohort* nc = NULL;
   if (cc != NULL) {
      nc = cc->taller;
   }
   while (dp->shortest != NULL) {
      insert_cohort(&cc, &rp->tallest, &rp->shortest, data);
      cc->patchptr = rp;
      cc = nc;
      dp->shortest = cc;
      if (cc != NULL) {
         nc = cc->taller;
      }
   }
  
   /* update size profile within patch */
   species_patch_size_profile(&rp, N_DBH_BINS, data);
#endif /* ED */

   /* update ptrs -- if not set, ptrs should be updated elsewhere */
   if (update_ptrs == 1) {
      /* is donor patch youngest? */
      if (dp != dp->siteptr->youngest_patch[dp->landuse]) 
         dp->younger->older = dp->older;
      else 
         dp->siteptr->youngest_patch[dp->landuse] = dp->older;
      /* is donor patch oldest? */
      if (dp != dp->siteptr->oldest_patch[dp->landuse])  
         dp->older->younger = dp->younger;
      else 
         dp->siteptr->oldest_patch[dp->landuse] = dp->younger;
   }

#ifdef ED
   /*free cohorts in patch being terminated*/
   cc = dp->shortest;
   bool stop = (cc == NULL);
   while (! stop) {
      if (cc->taller == NULL) {
         free(cc);
         stop = true;
      } else {
         cc = cc->taller;
         free(cc->shorter);
      }
   }
#endif /* ED */

#if LANDUSE
   /*free the array if present as well*/
   if (dp->landuse == LU_SCND)
      free(dp->phistory);
#endif

   free(dp);
}

////////////////////////////////////////////////////////////////////////////////
//! terminate_patches
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void terminate_patches (patch** patchptr, UserData* data) {

    double epsilon = 0.0;
    double scale_area= 1.0, scale_nIndiv=1.0, scale_soil_C=1.0, scale_GPP=1.0, scale_NPP=1.0, scale_Rh=1.0, scale_fire_emission = 1.0;
    double scale_fst_harv = 1.0, scale_past_harv = 1.0, scale_crop_harv = 1.0, scale_product_emission = 1.0;
    double scale_yr1_pool = 1.0, scale_yr10_pool = 1.0, scale_yr100_pool = 1.0;
    
    double area_terminated_patches = 0.0, Biomass_terminated_patches = 0.0, soil_C_terminated_patches = 0.0, GPP_terminated_patches = 0.0, NPP_terminated_patches = 0.0, Rh_terminated_patches = 0.0, fire_emission_terminated_patches = 0.0;
    double crop_harv_terminated_patches = 0.0, fst_harv_terminated_patches = 0.0, past_harv_terminated_patches = 0.0, product_emission_terminated_patches = 0.0;
    double yr1_product_pool_terminated_patches = 0.0, yr10_product_pool_terminated_patches = 0.0, yr100_product_pool_terminated_patches = 0.0;
    
    double area_all_patches = 0.0, Biomass_all_patches = 0.0, soil_C_all_patches = 0.0, GPP_all_patches = 0.0, NPP_all_patches = 0.0, Rh_all_patches = 0.0, fire_emission_all_patches = 0.0;
    double crop_harv_all_patches = 0.0, fst_harv_all_patches = 0.0, past_harv_all_patches = 0.0, product_emission_all_patches = 0.0;
    double yr1_product_pool_all_patches = 0.0, yr10_product_pool_all_patches = 0.0, yr100_product_pool_all_patches = 0.0;
    double fast_litter = 0.0, fast_litter_n = 0.0, struct_litter = 0.0;
    
    cohort* cc = NULL;
    patch* cp = (*patchptr)->siteptr->youngest_patch[(*patchptr)->landuse];
    site* currents = (*patchptr)->siteptr;
    int landuse = (*patchptr)->landuse;
    while (cp != NULL) {
      patch* np = cp->older;
       cc = cp->shortest;
       double tmp_biomass = 0.0;
       while (cc!=NULL) {
           tmp_biomass += cc->b*cc->nindivs;
           fast_litter += data->fraction_balive_2_fast *cc->balive*cc->nindivs;
           fast_litter_n +=  data->fraction_balive_2_fast *(1.0/data->c2n_leaf[cc->species])*cc->balive*cc->nindivs;
           struct_litter += cc->bdead*cc->nindivs +(1.0-data->fraction_balive_2_fast)*cc->balive*cc->nindivs;
           cc = cc->taller;
       }
       Biomass_all_patches += tmp_biomass;
       soil_C_all_patches += (cp->fast_soil_C+cp->slow_soil_C+cp->structural_soil_C+cp->passive_soil_C)*cp->area;
       GPP_all_patches += cp->gpp_avg*cp->area;
       NPP_all_patches += cp->npp_avg*cp->area;
       Rh_all_patches += cp->rh_avg*cp->area;
       fire_emission_all_patches += cp->fire_emission*cp->area;
#if LANDUSE
       product_emission_all_patches += cp->product_emission*cp->area;
       fst_harv_all_patches += cp->forest_harvested_c*cp->area;
       yr1_product_pool_all_patches += cp->yr1_decay_product_pool*cp->area;
       yr10_product_pool_all_patches += cp->yr10_decay_product_pool*cp->area;
       yr100_product_pool_all_patches += cp->yr100_decay_product_pool*cp->area;
       past_harv_all_patches += cp->past_harvested_c*cp->area;
       crop_harv_all_patches += cp->crop_harvested_c*cp->area;
#endif
       area_all_patches += cp->area;
      if (cp->area < data->f_area * data->area) {
           epsilon += cp->area / data->area;
           Biomass_terminated_patches += tmp_biomass;
           soil_C_terminated_patches += (cp->fast_soil_C+cp->slow_soil_C+cp->structural_soil_C+cp->passive_soil_C)*cp->area;
           GPP_terminated_patches += cp->gpp_avg*cp->area;
           NPP_terminated_patches += cp->npp_avg*cp->area;
           Rh_terminated_patches += cp->rh_avg*cp->area;
           fire_emission_terminated_patches += cp->fire_emission*cp->area;
#if LANDUSE
           product_emission_terminated_patches += cp->product_emission*cp->area;
           fst_harv_terminated_patches += cp->forest_harvested_c*cp->area;
           yr1_product_pool_terminated_patches += cp->yr1_decay_product_pool*cp->area;
           yr10_product_pool_terminated_patches += cp->yr10_decay_product_pool*cp->area;
           yr100_product_pool_terminated_patches += cp->yr100_decay_product_pool*cp->area;
           past_harv_terminated_patches += cp->past_harvested_c*cp->area;
           crop_harv_terminated_patches += cp->crop_harvested_c*cp->area;
#endif
           area_terminated_patches += cp->area;
           terminate_patch(&cp,data);
      }
      cp = np;
   }

   /* ADJUST AREAS AND VARS OF OTHER PATCHES TO COMPENSATE */
   if (epsilon > 0.0) {
       scale_area = area_all_patches/(area_all_patches-area_terminated_patches);
       scale_nIndiv = Biomass_all_patches/(Biomass_all_patches-Biomass_terminated_patches);
       scale_GPP =GPP_all_patches/(GPP_all_patches-GPP_terminated_patches)/scale_area;
       scale_NPP = NPP_all_patches/(NPP_all_patches-NPP_terminated_patches)/scale_area;
       scale_Rh = Rh_all_patches/(Rh_all_patches-Rh_terminated_patches)/scale_area;
       scale_soil_C = soil_C_all_patches/(soil_C_all_patches-soil_C_terminated_patches)/scale_area;
       scale_fire_emission = fire_emission_all_patches/(fire_emission_all_patches-fire_emission_terminated_patches)/scale_area;

       /// Check NaN value
       if ((scale_GPP != scale_GPP) or (isinf(scale_GPP)))
           scale_GPP = 1.0;
       if ((scale_NPP != scale_NPP) or (isinf(scale_NPP)))
           scale_NPP = 1.0;
       if ((scale_Rh != scale_Rh) or (isinf(scale_Rh)))
           scale_Rh = 1.0;
       if ((scale_soil_C != scale_soil_C) or (isinf(scale_soil_C)))
           scale_soil_C = 1.0;
       if ((scale_fire_emission!=scale_fire_emission) or (isinf(scale_fire_emission)))
           scale_fire_emission = 1.0;
#if LANDUSE
       scale_product_emission = product_emission_all_patches / (product_emission_all_patches-product_emission_terminated_patches)/scale_area;
       scale_fst_harv = fst_harv_all_patches/(fst_harv_all_patches-fst_harv_terminated_patches)/scale_area;
       scale_yr1_pool = yr1_product_pool_all_patches/(yr1_product_pool_all_patches-yr1_product_pool_terminated_patches)/scale_area;
       scale_yr10_pool = yr10_product_pool_all_patches/(yr10_product_pool_all_patches-yr10_product_pool_terminated_patches)/scale_area;
       scale_yr100_pool = yr100_product_pool_all_patches/(yr100_product_pool_all_patches-yr100_product_pool_terminated_patches)/scale_area;
       scale_past_harv = past_harv_all_patches/(past_harv_all_patches-past_harv_terminated_patches)/scale_area;
       scale_crop_harv = crop_harv_all_patches/(crop_harv_all_patches-crop_harv_terminated_patches)/scale_area;
       
       if ((scale_product_emission!=scale_fst_harv) or (isinf(scale_product_emission)))
           scale_product_emission = 1.0;
       if ((scale_fst_harv!=scale_fst_harv) or (isinf(scale_fst_harv)))
           scale_fst_harv = 1.0;
       if ((scale_yr1_pool!=scale_yr1_pool) or (isinf(scale_yr1_pool)))
           scale_yr1_pool = 1.0;
       if ((scale_yr10_pool!=scale_yr10_pool) or (isinf(scale_yr10_pool)))
           scale_yr10_pool = 1.0;
       if ((scale_yr100_pool!=scale_yr100_pool) or (isinf(scale_yr100_pool)))
           scale_yr100_pool = 1.0;
       if ((scale_past_harv!=scale_past_harv) or (isinf(scale_past_harv)))
           scale_past_harv = 1.0;
       if ((scale_crop_harv!=scale_crop_harv) or (isinf(scale_crop_harv)))
           scale_crop_harv = 1.0;
#endif
      cp = currents->youngest_patch[landuse];

      while (cp != NULL)
      {
         //cp->area *= (1.0 + epsilon);
          cp->area *= scale_area;
#ifdef ED
          cp->fast_soil_C *= scale_soil_C;
          cp->slow_soil_C *= scale_soil_C;
          cp->structural_soil_C *= scale_soil_C;
          cp->passive_soil_C *= scale_soil_C;
          cp->gpp_avg *= scale_GPP;
          cp->npp_avg *= scale_NPP;
          cp->rh_avg *= scale_Rh;
          cp->fire_emission *=scale_fire_emission;
#if LANDUSE
          cp->product_emission *= scale_product_emission;
          cp->forest_harvested_c *=scale_fst_harv;
          cp->yr1_decay_product_pool *=scale_yr1_pool;
          cp->yr10_decay_product_pool *=scale_yr10_pool;
          cp->yr100_decay_product_pool *=scale_yr100_pool;
          cp->past_harvested_c *= scale_past_harv;
          cp->crop_harvested_c *= scale_crop_harv;
#endif
          if (abs(Biomass_all_patches-Biomass_terminated_patches)<1e-9)   // other non-terminated patches are empty
          {
              cp->fast_soil_C += fast_litter/area_all_patches;
              cp->structural_soil_C += struct_litter/area_all_patches;
              cp->structural_soil_L += (data->l2n_stem/data->c2n_stem) * struct_litter/area_all_patches;
              cp->fast_soil_N       += fast_litter_n/area_all_patches;
          }
          else   // if adjusting nIndiv still work
          {
              cohort* cc = cp->tallest;
              while (cc != NULL) {
                  //cc->nindivs *= (1.0 + epsilon);
                  cc->nindivs *= scale_nIndiv;
                  cc = cc->shorter;
              }
          }
         cp = cp->older;
#endif
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
//! terminate_patch
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void terminate_patch (patch** patchptr,UserData* data) {
   patch* cp = *patchptr;

#ifdef ED
   /* free cohorts in patch being terminated */
   cohort* cc = cp->shortest;
   while (cc != NULL) {
      if (cc->taller != NULL) {
         cc = cc->taller;
         free(cc->shorter);
      } else {
         free(cc);
         cc = NULL;
      }
   }
#endif

   if (cp->landuse == LU_SCND)
      free(cp->phistory);

   /* update links */
   if (cp->younger == NULL)
      cp->siteptr->youngest_patch[cp->landuse] = cp->older;
   else
      cp->younger->older = cp->older;
   if (cp->older == NULL)
      cp->siteptr->oldest_patch[cp->landuse] = cp->younger;
   else
      cp->older->younger = cp->younger;

   free(cp);
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! light_levels
//! calculate light levels
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void light_levels (patch** patchptr, UserData* data) {

   patch* cp = *patchptr;  
   site* current_site = cp->siteptr;
  
   cohort* cc = cp->tallest;
   if(cc !=NULL){ /*make sure not empty patch*/
       //checkLAI
       cc->lai =  cc->nindivs*(1.0/cp->area)*(cc->bl*data->specific_leaf_area[cc->species]);
      cc->lite = current_site->sdata->L_top;
      cc->lite *= exp(-data->cohort_shading*(data->L_extinct)*(cc->lai));
      cc = cc->shorter;

      /* lower cohorts */
      while(cc != NULL){      
         cc->lai =  cc->nindivs*(1.0/cp->area)*(cc->bl*data->specific_leaf_area[cc->species]);
         cc->lite = (cc->taller)->lite*exp(-data->cohort_shading*(data->L_extinct)*(cc->taller->lai));
         /* within cohort shading*/
         cc->lite *= exp(-data->cohort_shading*(data->L_extinct)*(cc->lai));
         cc=cc->shorter;
       } /* end cohort loop */
      
       cc = cp->tallest;
       while (cc != NULL) {
          cc->lite = data->canopy_damage * current_site->sdata->L_top 
            + (1.0 - data->canopy_damage) * cc->lite;
          cc = cc->shorter;
       } /* end cohort loop */  
   } /* end if */
}

////////////////////////////////////////////////////////////////////////////////
//! species_patch_size_profile
//! binned patch size profiles
//! PRM: 5/1/99 modified to compare dbh profiles
//! 
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void species_patch_size_profile ( patch** pcurrentp, unsigned int nbins, 
                                  UserData* data ) {

   patch* cp = *pcurrentp;

   double dh = (data->dbhmax / N_DBH_BINS);

   /* initialize bins */
   for (size_t i=0; i<NSPECIES; i++) {  
      for(size_t j=0; j<N_DBH_BINS; j++) { 
         cp->spp_density_profile[i][j] = 0.0;
         cp->spp_basal_area_profile[i][j] = 0.0;
         cp->spp_agb_profile[i][j] = 0.0;
      }
   }

   /* update bins */ 
   cohort* cc = cp->shortest;
   while (cc != NULL) { /* loop over cohorts */
      double min, max;   
      for (size_t j=0; j<N_DBH_BINS-1; j++) { /* loop over bins */
         if (j == 0) {
            min = 0.0;
         } else {
            min = j * dh;
         }
         max = (j + 1) * dh;
         if ( (cc->dbh > min) && (cc->dbh <= max) ) {
            cp->spp_density_profile[cc->species][j] += cc->nindivs / cp->area;
            cp->spp_basal_area_profile[cc->species][j] += (M_PI / 4.0) * 
               pow(cc->dbh, 2.0) * cc->nindivs / cp->area;
            cp->spp_agb_profile[cc->species][j] += cc->babove*cc->nindivs/cp->area;
         } /* end if */
      } /* end loop over dbh bins*/

      /* deal with largest dbh bin */
      size_t j = N_DBH_BINS - 1;
      min = j * dh;
      if ( cc->dbh > min ) {
         cp->spp_density_profile[cc->species][j] += cc->nindivs / cp->area;
         cp->spp_basal_area_profile[cc->species][j] += (M_PI / 4.0) * 
            pow(cc->dbh, 2.0) * cc->nindivs / cp->area;
         cp->spp_agb_profile[cc->species][j] += cc->babove * cc->nindivs / cp->area;
      } /* end if */
     
      cc = cc->taller;
   } /* end loop over cohorts */   
}


////////////////////////////////////////////////////////////////////////////////
//! radiative_flux
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double patch::radiative_flux (UserData* data) {
   /* calculate net radiatiive flux hitting soil surface within patch         */
   /* this version calculates the fraction of incomming rn that hits the gnd  */
   /* i.e. soil evaporation rate is only influcenced by relative plant cover  */

   /* printf("*** Net Radiative Flux: \n"); */
   
   /* compute patch level lai */
   lai = 0.0;
   cohort* cc = tallest;
   while(cc != NULL){
      lai += cc->lai;
      cc = cc->shorter;
   }

   double rnet = siteptr->sdata->Rn_top * exp(-1.0 * data->Rn_extinct * lai);
   /*printf("site %s patch %p lai= %f rnet= %f \n",cs->name,cp,cp->lai,rnet);*/
   
   return rnet;
}
#endif /* ED */

/******************************************************************************/
