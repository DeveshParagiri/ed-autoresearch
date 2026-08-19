/*mechanism.c*/
/*photosynthesis and evapotranspiration file*/

#include "edmodels.h"
#include "site.h"
#include "read_site_data.h"
#include "cohort.h"
#include "math.h"
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
//! plant_respiration
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void cohort::plant_respiration(UserData* data){
   /* calculates respiration of plant compartments (stem and root) */
   /* taken from Foley (Ibis model) 1996 gbc v10 p603-628 */
   
   site* currents = siteptr;

   /*light index*/  
   //size_t light_index = 0;
   //while(currents->sdata->light_levels[pt][light_index] > lite){
   //   light_index++;
   //}

   /* time index*/
   size_t tindex = (data->time_period*12)/N_CLIMATE;


    if (currents->sdata->tf_air[species][tindex]<-1000)
    {
        currents->sdata->compute_mech(pt,species,Vm0,tindex,0,data);
    }
    if (currents->sdata->tf_soil[species][tindex]<-1000)
    {
        currents->sdata->compute_mech(pt,species,Vm0,tindex,0,data);
    }
    double tf_air = currents->sdata->tf_air[species][tindex];
    double tf_soil = currents->sdata->tf_soil[species][tindex];


   /* resp rates read in are  in gC/m2(leaf)/mo */
   /* calc resp rates in kgC per plant per yr */
   /*5/1/00 not not actual leaf resp anymore. Read actual leaf resp used is
     included below, we now read in net which includes leaf resp and do some
     down regulation under limiting conds*/
   //double r_leaf = (0.001*N_CLIMATE)*(-1.0*currents->sdata->Anb[pt][light_index][tindex])*bl*data->specific_leaf_area[species];
   //double r_leaf = (0.001*N_CLIMATE)*(-1.0*currents->An_shut[pt][(int)(lite*100)])*bl*data->specific_leaf_area[species];
   
   double r_vleaf = data->beta[species][4]*blv*tf_air;
   double r_stem = data->beta[species][1]*bsw*tf_air;
   double r_root = data->beta[species][3]*br*tf_soil;

   /*currentc->resp = r_leaf + r_vleaf + r_stem + r_root;*/
   /* 4/24/00 take out leaf resp here, include in gpp below by reading in net
      easier to deal with down regulation all at once*/
   resp = r_vleaf + r_stem + r_root;

}

////////////////////////////////////////////////////////////////////////////////
//! npp_function
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void cohort::npp_function(UserData* data){
  
   /*This function sets npp, and gpp*/

   site* cs = siteptr;
 
   /*driver index index*/
   size_t time_index = data->time_period;
    
   /* RESPIRATION */
   plant_respiration(data);
  
   /*SET INDICIES*/
   size_t light_index = 0;
    while(cs->sdata->light_levels[species][light_index] > lite){
        light_index++;
    }

    
   /*PHOTOSYNTHESIS**************************/
    if (cs->sdata->An[species][time_index][light_index]<-1000)
    {
        cs->sdata->compute_mech(pt,species,Vm0,time_index,light_index,data);
    }
    if (cs->sdata->Anb[species][time_index][N_LIGHT-1]<-1000)
    {
        cs->sdata->compute_mech(pt,species,Vm0,time_index,N_LIGHT-1,data);
    }
    if (cs->sdata->An[species][time_index][0]<-1000)
    {
        cs->sdata->compute_mech(pt,species,Vm0,time_index,0,data);
    }
    An_pot = 0.001*cs->sdata->An[species][time_index][light_index];
    An_max = 0.001*cs->sdata->An[species][time_index][0];
    An_shut = 0.001*cs->sdata->Anb[species][time_index][N_LIGHT-1];
    An_shut_max = 0.001*cs->sdata->Anb[species][time_index][N_LIGHT-1];

   
   /*convert from KgC/m2/mon to kgC/yr*/
   An_pot *= data->specific_leaf_area[species]*bl*12;
   An_max *= data->specific_leaf_area[species]*bl*12;
   /*convert from KgC/m2/mon to kgC/yr*/
   An_shut *= data->specific_leaf_area[species]*bl*12;
   An_shut_max *= data->specific_leaf_area[species]*bl*12;


   /* nitrogen fixation */
   if(data->Nfixer[species] == 1) payment_to_Nfixers = data->fraction_of_GPP_to_Nfixers*fs_open*An_pot;
   else payment_to_Nfixers = 0.000;
     
   /*CONSEQUENCES OF WATER AND NITROGEN LIMITATION*/
   /*grown fraction of month at potential rate (Fm), and 1-Fm at shut down rates*/
 
   /*4/24/00 this is really npp, dont add back resp term*/
   /*This is not really gross, working towards net*/
   /*treat this way bc easier to test downregulation, i.e. resp down 
     regulated as well*/
   /*must also elim leaf resp from below*/
   gpp =  fs_open*(An_pot) + (1-fs_open)*(An_shut);
   gpp_max =  fs_open*(An_max) + (1-fs_open)*(An_shut_max);

   npp_max =  gpp_max - resp;
   npp =  gpp - resp; /*where resp includes leaf resp*/
   /*careful here, that resp includes leaf resp is gpp is really gpp
     and not  gpp-r_l*/
   /*4/24/00 although here it doesnt*/

   if(npp > 0.00){/*only subtract growth resp if growing*/
      gr_resp = -data->growth_resp*npp;
      npp *= (1.0 - data->growth_resp); 
   }
   else
      gr_resp = 0.00;

   if(npp_max > 0.00) /*only subtract growth resp if growing*/
      npp_max *= (1.0-data->growth_resp);
    
}
/**************************************************************************/
