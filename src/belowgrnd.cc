#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "read_site_data.h"

#ifdef ED
#include "cohort.h"
////////////////////////////////////////////////////////////////////////////////
//! Update_Water
//! When using the Dwdt formula for water, it was found that often the water value 
//! oscillated on a substep, sometimes with oscillations growing wlarger and larger 
//! until other errors occurred. To fix this, we now do a standard euler step if it
//! doesn't take us past the equilibrium, otherwise we find the equilibrium 
//! (min 0, max theta_max) and set the new water value to half way to equilibrium
//! (half way chosen to give trees a chance to change uptake in next substep)
//!   
//! Take standard time step, accept if haven't over shot equilibrium (derivative dwdt 
//! still same sign). Note that this function assumes that the function dwdt is smooth 
//! and monotonically increasing w.r.t water. For most reasonable functions for dwdt 
//! this is logical, but changes to dwdt function must be made bearing this in mind.
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void patch::Update_Water(double time, UserData* data, double deltat){
   double starting_dwdt = dwdt; 
   double starting_water = water;
   water += dwdt*deltat;

#if SNOWPACK_SCHEME == 1
    if (siteptr->sdata->temp[data->time_period]<0)
    {
        snowpack += siteptr->sdata->precip[data->time_period]*deltat;
    }
    else
    {
        snowpack -= snow_melt*deltat;
        snowpack = fmax(snowpack, 0.0);
    }
#endif
   if (Dwdt(time, data)*starting_dwdt>0) return;
   
   //Otherwise find approximation to equilibrium. Note this is not true equilibrium as trees have not had a chance to adjust.
   site* currents = siteptr;
   
   double theta_tol = 0.001; //Acceptable error in % saturation (difference off equilibrium)
   double water_tol = theta_tol*currents->sdata->soil_depth * currents->sdata->theta_max;
   water = starting_water; dwdt = starting_dwdt;
   
   double max_guess, min_guess;
   if (dwdt>0){
      min_guess = water; 
      max_guess = water + dwdt * deltat;
   } else {
      min_guess = water - dwdt * deltat; 
      max_guess = water;
   }
    
   //Binomial search
   while (max_guess-min_guess>water_tol){
      water = (max_guess+min_guess)/2;
      if (Dwdt(time, data)>0) min_guess = water;
      else max_guess = water;
   }
    
   //Adjust half way to equilibrium
   water = (water + starting_water) / 2.;
}

////////////////////////////////////////////////////////////////////////////////
//! Dwdt
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double patch::Dwdt (double time, UserData* data){  

   /* This function, and those that it calls, must use *w ! */
   /* this function calculates and updates dwdt a patch */
   //For Update_Water to work this function must be monotonically increasing w.r.t water
   site* currents = siteptr;
   
   // Call Soil_Canopy_evap function to calculate soil evaporation rate
   soil_evap = Soil_Canopy_evap(data);
    
#if SOILGRIDS_SCHEME == 0
    /* calculate water loss per unit area from patch */
   theta = water / (currents->sdata->soil_depth * currents->sdata->theta_max);
   
   if (water > 0.0) {
      perc = currents->sdata->k_sat * pow(theta, 2.0 * currents->sdata->tau + 2.0);
   } else {
      perc = 0.0;
   }
   dwdt = (currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
    
#if 1
   double theta_crit = 0.3;
    
   if (dwdt*data->deltat+water<theta_crit*(currents->sdata->soil_depth * currents->sdata->theta_max)) // If theta is below critical point, adjust soil_evap
   {
      soil_evap = water+currents->sdata->precip[(int) data->time_period]-total_water_uptake/area-theta_crit*(currents->sdata->soil_depth * currents->sdata->theta_max) - perc;
      if (soil_evap<0) soil_evap=0;
      dwdt=(currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
   }
    
   if (dwdt*data->deltat+water < theta_crit*(currents->sdata->soil_depth * currents->sdata->theta_max))  // If theta is below critical point, adjust percolation
   {
      perc=water+currents->sdata->precip[(int) data->time_period]-total_water_uptake/area-theta_crit*(currents->sdata->soil_depth * currents->sdata->theta_max)-soil_evap;
        
      if (perc<0) perc=0;
      dwdt=(currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
   }
#endif
    
#else
   theta = (water - currents->sdata->MvG_soil_depth*currents->sdata->MvG_theta_residual)/(currents->sdata->MvG_soil_depth*(currents->sdata->MvG_theta_saturated-currents->sdata->MvG_theta_residual));
   theta = fmin(1.0,theta);
   theta = fmax(0.0,theta);
   if(theta>0)
   {
      double se = 0.0;
      se = fmin(1.0,theta);
      se = fmax(0.0,se);
      perc = currents->sdata->MvG_k_sat*pow(se, currents->sdata->MvG_L)*pow(1.0-pow(1.0-pow(se,1/currents->sdata->MvG_m),currents->sdata->MvG_m),2.0);
   }
   else
   {
      perc = 0.0;
   }

#if SNOWPACK_SCHEME == 0
   dwdt = (currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
#else
   double precip = currents->sdata->precip[(int) data->time_period];
   if(currents->sdata->temp[data->time_period]<0) // If temperature drop below freezing point, no precipitation enters soil
   {
      dwdt = (0.0 - perc - total_water_uptake/ area) - soil_evap;
      snow_melt = 0.0;
   }
   else
   {
      if(snowpack>0)
         snow_melt = abs(currents->sdata->temp[data->time_period])*100.0;
      else
         snow_melt = 0.0;

      dwdt = (precip + snow_melt - perc - total_water_uptake / area) - soil_evap;
   }
#endif
    
   double theta_crit = 0.3, updated_theta = 0.0;
    
   updated_theta = (dwdt*data->deltat+water - currents->sdata->MvG_soil_depth*currents->sdata->MvG_theta_residual)/(currents->sdata->MvG_soil_depth*(currents->sdata->MvG_theta_saturated-currents->sdata->MvG_theta_residual));
   if (updated_theta < theta_crit)  // if theta drop below critical point, adjust soil evaporation
   {
      double MvG_theta = currents->sdata->MvG_theta_saturated-currents->sdata->MvG_theta_residual;
      double dwdt_crut = (theta_crit*currents->sdata->MvG_soil_depth*MvG_theta + currents->sdata->MvG_soil_depth*currents->sdata->MvG_theta_residual-water)/data->deltat;
      soil_evap = (currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - dwdt_crut;
      
      if (soil_evap<0) soil_evap=0;
      dwdt = (currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
   }
    
   updated_theta = (dwdt*data->deltat+water - currents->sdata->MvG_soil_depth*currents->sdata->MvG_theta_residual)/(currents->sdata->MvG_soil_depth*(currents->sdata->MvG_theta_saturated-currents->sdata->MvG_theta_residual));
   if (updated_theta < theta_crit) // if theta is still below critical point, adjust percolation
   {
      double MvG_theta = currents->sdata->MvG_theta_saturated-currents->sdata->MvG_theta_residual;
      double dwdt_crut = (theta_crit*currents->sdata->MvG_soil_depth*MvG_theta + currents->sdata->MvG_soil_depth*currents->sdata->MvG_theta_residual-water)/data->deltat;
      perc = (currents->sdata->precip[(int) data->time_period] - soil_evap - total_water_uptake / area) - dwdt_crut;
        
      if (perc<0) perc=0;
      dwdt=(currents->sdata->precip[(int) data->time_period] - perc - total_water_uptake / area) - soil_evap;
   }
#endif

   return dwdt;
}


////////////////////////////////////////////////////////////////////////////////
//! Soil_Canopy_evap
//!
//!
//! @param
//! @return
////////////////////////////////////////////////////////////////////////////////
double patch::Soil_Canopy_evap (UserData* data)
{
    ///// Calculate soil evaportation rate using MODIS algorithm

   double Tair_daytime = 0.0, Ea_daytime = 0.0, swdown_daytime = 0.0, daytime_hours = 0.0;
   double Tair_nighttime = 0.0, Ea_nighttime = 0.0, swdown_nighttime = 0.0;
    
   int globY_ = siteptr->sdata->globY_, globX_ = siteptr->sdata->globX_;
   double tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
   for(int hour=data->time_period*24;hour<data->time_period*24+24;hour++)
   {
      tmp1 = data->global_swd[hour][globY_][globX_];
      tmp2 = data->global_tmp[hour][globY_][globX_];
      tmp3 = data->global_hum[hour][globY_][globX_]*1e2; // convert to kPa from mol/mol
      if(tmp1>0.5)
      {
         swdown_daytime += tmp1;
         Tair_daytime += tmp2;
         Ea_daytime += tmp3;
         daytime_hours += 1.0;
      }
      else
      {
         swdown_nighttime += tmp1;
         Tair_nighttime += tmp2;
         Ea_nighttime += tmp3;
      }
   }
   if(daytime_hours>0)
   {
      Tair_daytime /= daytime_hours;
      swdown_daytime /= daytime_hours;
      Ea_daytime /= daytime_hours;
   }
   else
   {
      Tair_daytime = 0.0;
      swdown_daytime = 0.0;
      Ea_daytime = 0.0;
   }
    
   if(24.0-daytime_hours>0)
   {
      Tair_nighttime /= (24.0-daytime_hours);
      swdown_nighttime /= (24.0-daytime_hours);
      Ea_nighttime /= (24.0-daytime_hours);
   }
   else
   {
      Tair_nighttime = 0.0;
      swdown_nighttime = 0.0;
      Ea_nighttime = 0.0;
   }
//    radiative_flux(data);
   double patch_lai = 0.0;
   cohort* cc = shortest;
   while((cc != NULL) && (cc->patchptr !=NULL))
   {
      patch_lai += cc->lai;
      cc = cc->taller;
   }
    
   if(patch_lai<0)
      patch_lai = 0;
   else if (patch_lai>20)
      patch_lai = 20.0;
    
   double albedo = 0.18; // almost same for all bare soil
   double rho = 1.225; // air density, kg/m3
   double Pa = 101.3; // air pressure, kPa
   double Cp = 1.013*1e-3; // Specfic air heat capacity, MJ/kg/C
   double sigma = 5.6697*1e-8; //Stefen Boltzman constant, W/m2/K^4
   double lamda = 2.45; //latent heat of vaporization, MJ/kg
   double gamma = 0.067165; // Psychrometric constant, kPg/C
   double beta = 0.5; // parameter to scale moisture constraint, kPa
   double gl_sh = 0.01;
   double gl_e_wv = 0.01;
   double Fc = 1.0 - exp(-1.0 * data->Rn_extinct * patch_lai);
   double epslon_a_daytime = 1 - 0.26*exp(-0.77*1e-4*pow(Tair_daytime, 2.0));
   double epslon_a_nighttime = 1 - 0.26*exp(-0.77*1e-4*pow(Tair_nighttime, 2.0));
   double epslon_s = 0.97;
   double epslon = 0.97;
    
    
   double Rnet_daytime = (1.0 - albedo)*swdown_daytime + sigma*(epslon_a_daytime - epslon_s)*pow(273.15+Tair_daytime, 4.0);
   double Rnet_nighttime = (1.0 - albedo)*swdown_nighttime + sigma*(epslon_a_nighttime - epslon_s)*pow(273.15+Tair_nighttime, 4.0);
   
   double Gsoil_daytime = 4.73*Tair_daytime - 20.87;
   double Gsoil_nighttime = 4.73*Tair_nighttime - 20.87;
   Gsoil_daytime = fminf(Gsoil_daytime, 0.39*Rnet_daytime);
   Gsoil_nighttime = fminf(Gsoil_nighttime, 0.39*Rnet_nighttime);
    
   double G_daytime = (1.0-Fc)*Gsoil_daytime;
   double G_nighttime = (1.0-Fc)*Gsoil_nighttime;
   
    
   double Asoil_daytime = ((1.0 - Fc)*Rnet_daytime - G_daytime)/1e6; // MJ/m2
   double Asoil_nighttime = ((1.0 - Fc)*Rnet_nighttime - G_nighttime)/1e6; // MJ/m2
   
   double Acanopy_daytime = Fc*Rnet_daytime/1e6;  // MJ/m2
   double Acanopy_nighttime = Fc*Rnet_nighttime/1e6;  // MJ/m2
   
    
   double Es_daytime = 0.611*exp(17.269*Tair_daytime/(Tair_daytime+237.3));  // kPa
   double Es_nighttime = 0.611*exp(17.269*Tair_nighttime/(Tair_nighttime+237.3));  // kPa
   double E_slope_daytime = Es_daytime*(17.269*273.3)/pow(237.3+Tair_daytime, 2.0);
   double E_slope_nighttime = Es_nighttime*(17.269*273.3)/pow(237.3+Tair_nighttime, 2.0);
   double VPD_daytime = Es_daytime - Ea_daytime; // kPa
   double VPD_nighttime = Es_nighttime - Ea_nighttime; // kPa
   double RH_daytime = Ea_daytime/Es_daytime*100.0;  // %
   double RH_nighttime = Ea_nighttime/Es_nighttime*100.0;
    
   if(RH_daytime>95)
   {
      RH_daytime = 95;
      Ea_daytime = RH_daytime/100.0*Es_daytime;
      VPD_daytime = Es_daytime - Ea_daytime;
   }
   if(RH_nighttime>95)
   {
      RH_nighttime = 95;
      Ea_nighttime = RH_nighttime/100.0*Es_nighttime;
      VPD_nighttime = Es_nighttime - Ea_nighttime;
   }
    
   double Fwet_daytime = 0.0, Fwet_nighttime = 0.0;
   if(RH_daytime>90)
      Fwet_daytime = pow(RH_daytime/100.0, 4.0);
   if(RH_nighttime>90)
      Fwet_nighttime = pow(RH_nighttime/100.0, 4.0);
    
    
   double r_totc = 107.0; // s/m
   double r_corr_daytime = 1.0/(101.300/Pa*pow((Tair_daytime+273.15)/293.15,1.75));
   double r_corr_nighttime = 1.0/(101.300/Pa*pow((Tair_nighttime+273.15)/293.15,1.75));
   double r_tot_daytime = r_totc*r_corr_daytime;
   double r_tot_nighttime = r_totc*r_corr_nighttime;
   double r_hs_daytime = r_tot_daytime;
   double r_hs_nighttime = r_tot_nighttime;
   double r_rs_daytime = rho*Cp/(4.0*sigma*pow(Tair_daytime+273.15,3.0));
   double r_rs_nighttime = rho*Cp/(4.0*sigma*pow(Tair_nighttime+273.15,3.0));
   double r_as_daytime = r_hs_daytime*r_rs_daytime/(r_hs_daytime+r_rs_daytime);
   double r_as_nighttime = r_hs_nighttime*r_rs_nighttime/(r_hs_nighttime+r_rs_nighttime);
    
   double rhc_daytime = 1.0/(gl_sh*patch_lai*Fwet_daytime);
   double rhc_nighttime = 1.0/(gl_sh*patch_lai*Fwet_nighttime);
   
   double rrc_daytime = rho*Cp/4.0/sigma/pow(Tair_daytime+273.15,3.0);
   double rrc_nighttime = rho*Cp/4.0/sigma/pow(Tair_nighttime+273.15,3.0);
   double rhrc_daytime = rhc_daytime*rrc_daytime/(rhc_daytime+rrc_daytime);
   double rhrc_nighttime = rhc_nighttime*rrc_nighttime/(rhc_nighttime+rrc_nighttime);
   double rvc_daytime = 1.0/(gl_e_wv*patch_lai*Fwet_daytime);
   double rvc_nighttime = 1.0/(gl_e_wv*patch_lai*Fwet_nighttime);
    
   double E_wet_soil_pot_daytime = (E_slope_daytime*Asoil_daytime+rho*Cp*(1.0-Fc)*VPD_daytime/r_as_daytime)*Fwet_daytime/(E_slope_daytime+gamma*r_tot_daytime/r_as_daytime)/lamda;  // mm s-1
   double E_dry_soil_pot_daytime = (E_slope_daytime*Asoil_daytime+rho*Cp*(1.0-Fc)*VPD_daytime/r_as_daytime)*(1-Fwet_daytime)/(E_slope_daytime+gamma*r_tot_daytime/r_as_daytime)/lamda; // mm s-1
   double moisture_constraint_daytime = pow(RH_daytime/100.0, VPD_daytime/beta);
    
   double E_wet_soil_pot_nighttime = (E_slope_nighttime*Asoil_nighttime+rho*Cp*(1.0-Fc)*VPD_nighttime/r_as_nighttime)*Fwet_nighttime/(E_slope_nighttime+gamma*r_tot_nighttime/r_as_nighttime)/lamda; // mm s-1
   double E_dry_soil_pot_nighttime = (E_slope_nighttime*Asoil_nighttime+rho*Cp*(1.0-Fc)*VPD_nighttime/r_as_nighttime)*(1-Fwet_nighttime)/(E_slope_nighttime+gamma*r_tot_nighttime/r_as_nighttime)/lamda; // mm s-1
   double moisture_constraint_nighttime = pow(RH_nighttime/100.0, VPD_nighttime/beta);
    
   double E_soil_actual_daytime = E_wet_soil_pot_daytime + E_dry_soil_pot_daytime *moisture_constraint_daytime;
   double E_soil_actual_nighttime = E_wet_soil_pot_nighttime + E_dry_soil_pot_nighttime *moisture_constraint_nighttime;
   double E_soil_actual_total = E_soil_actual_daytime*3600.0*daytime_hours + E_soil_actual_nighttime * 3600.0*(24.0-daytime_hours); //mm day-1
   E_soil_actual_total *= 30.5*N_CLIMATE;
    
   double E_wet_canopy_daytime = 0.0, E_wet_canopy_nighttime = 0.0;
   
   if((Fwet_daytime>0) && (patch_lai>0))
      E_wet_canopy_daytime = (E_slope_daytime*Acanopy_daytime*Fc+rho*Cp*VPD_daytime*Fc/rhrc_daytime)*Fwet_daytime/(E_slope_daytime+(Pa*Cp*rvc_daytime)/(lamda*epslon*rhrc_daytime))/lamda;
    
   if((Fwet_nighttime>0) && (patch_lai>0))
      E_wet_canopy_nighttime = (E_slope_nighttime*Acanopy_nighttime*Fc+rho*Cp*VPD_nighttime*Fc/rhrc_nighttime)*Fwet_nighttime/(E_slope_nighttime+(Pa*Cp*rvc_nighttime)/(lamda*epslon*rhrc_nighttime))/lamda;
    
   double E_wet_canopy_total = E_wet_canopy_daytime*3600.0*daytime_hours + E_wet_canopy_nighttime * 3600.0*(24.0-daytime_hours); //mm day-1
   E_wet_canopy_total *= 30.5*N_CLIMATE; // convert to mm yr-1
    
   return (E_soil_actual_total+E_wet_canopy_total);
}

////////////////////////////////////////////////////////////////////////////////
//! Dsdt
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void patch::Dsdt (unsigned int time_period, double time, UserData* data) {

   /*** This function must set time derivatives for each soil pool ***/
   /* Simplified 4-Box Century, based on Parton et al 1987 and 1993  */
   /* 4 carbon boxes + 2 nitrogen boxes                              */
   /* There is no: leaching, other n inputs, or effect of soil type  */
   double fast_C_loss=0.00,  fast_N_loss=0.00;
   double structural_C_loss=0.00;
   double slow_C_input=0.00,  slow_C_loss=0.00, slow_N_loss=0.00;
   double mineralized_N_input=0.00, mineralized_N_loss=0.00;
   double N_immobilization_demand=0.00;
   double passive_C_input=0.00, passive_C_loss=0.00, passive_N_loss=0.00;
   double structural_L_loss=0.00;

   double Ls; /* the fraction of structural material that is lignin */
   double Lc; /* decomp rate reduction due to lignin */

   /* CENTURY PARAM VALUES */
   /* values are simple averages from lumped Century Pools */
   /* Based on Parton et al 1993 GBC */
   /* 1-structural,2-fast,3=slow,4=passive */
   double K1=4.5, K2=11.0, K3=100.2, K4=0.0;  /* Max decay rate yr^-1;  *
                                               * from Century directly, *
                                               * or averages of lumped  *
                                               * Century pools          */
   /* K3 is high bc we wanted to added back the n immobilization story 
      without tracking the slow pool */
   
   K3 = 0.2;

   /* std values= 1, .3, 1, 0 */
   double r_fsc=1.0, r_stsc=0.3, r_ssc=1.0, r_psc=0.0;
   /* respiration rates of soil pools */
   
   /****************/
   /* OTHER FLUXES */
   /****************/
   /* calculate commonly used terms once */
   A = A_function(time_period, data);

   /* Slow down of decomp due to fraction of structural material that is lignin */
   /* From CENTURY Parton et al 1993 GBC 7(4):785-809 */
   Ls = structural_soil_L / structural_soil_C;

   Lc = exp(-3.0 * Ls);

   if(!data->n_competition) {
      total_plant_nitrogen_uptake = 0.0;
   }

   mineralized_N_loss = total_plant_nitrogen_uptake;

   if(data->n_decomp_limitation) {
      /* N immobilized by the decomp of structural material */
      N_immobilization_demand= A * Lc * K1 * structural_soil_C
         * ((1.0 - r_stsc) * (1.0 / data->c2n_slow) - (1.0 / data->c2n_structural));
      fstd = data->nitrogen2 * mineralized_soil_N
         / (N_immobilization_demand + data->nitrogen2 * mineralized_soil_N);
   } else {
      N_immobilization_demand = 0.0;
      fstd = 1.0;
   }

   mineralized_N_loss +=  N_immobilization_demand*fstd;

   /* STRUCTURAL POOL */
   /* compute associated decomposition of strucural C and lignin */ 
   structural_C_loss = A * Lc * K1 * structural_soil_C * fstd;  
   structural_L_loss = A * Lc * K1 * structural_soil_L * fstd; 

   /* fast pool */
   /* NOTE: one of the fast pool losses in Century includes a soil type 
      dependence thats not included here */ 
   fast_C_loss =  A * K2 * fast_soil_C;
   fast_N_loss =  A * K2 * fast_soil_N;
   mineralized_N_input += fast_N_loss;

   if(data->open_cycles) {
      /* deposition of N in rain */ 
      mineralized_N_input += siteptr->sdata->N_conc_in_rain 
         * siteptr->sdata->precip[data->time_period];

      /* leaching losses of C */
      fast_C_loss += data->NC_perc_coeff * perc * fast_soil_C;
      /* NOTE: CENTURY only has leaching losses from fast C&N and Min N! */ 
      /* leaching losses of N */
      fast_N_loss += data->NC_perc_coeff * perc*fast_soil_N;
      mineralized_N_loss += data->NC_perc_coeff * perc * mineralized_soil_N;
   }


   /* slow pool */
   slow_C_input = (1.0 - r_stsc) * structural_C_loss;
   slow_C_loss =  A * K3 * slow_soil_C;
   slow_N_loss = (1.0 / data->c2n_slow) * slow_C_loss;
   mineralized_N_input += slow_N_loss;
  
   /* passive pool */
   passive_C_input = 0.0;
   passive_C_loss =  A * K4 * passive_soil_C;
   passive_N_loss = (1.0 / data->c2n_passive) * passive_C_loss;
   mineralized_N_input += passive_N_loss;

   /* C POOLS */
   dfsc = fsc_in - fast_C_loss;
   dstsc = ssc_in - structural_C_loss;
   dssc = slow_C_input - slow_C_loss;
   dpsc = passive_C_input - passive_C_loss;
   dstsl = ssl_in - structural_L_loss;

   /* loss of C to atmosphere and leaching */
   rh = r_fsc * fast_C_loss + r_stsc*structural_C_loss
      + r_ssc*slow_C_loss + r_psc*passive_C_loss;
    

   /* NITROGEN POOLS */
   dfsn = fsn_in - fast_N_loss;
   dmsn = mineralized_N_input - mineralized_N_loss;
}
#endif /* ED */


#ifdef MIAMI_LU
////////////////////////////////////////////////////////////////////////////////
//! Dsdt
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void patch::Dsdt (UserData* data) {

   double fast_C_loss = 0.00;
   double structural_C_loss = 0.00;

   int time_period = data->time_period;

   /****************/
   /* OTHER FLUXES */
   /****************/

   /* calculate commonly used terms once */
   A = A_function(time_period, data);

   /*C POOLS*/
   fast_C_loss = fast_soil_C * (1.0 - exp(-data->K1 * A * data->deltat) );
   structural_C_loss = structural_soil_C * (1.0 - exp(-data->K2 * A * data->deltat) );
  
   fast_soil_C -= fast_C_loss;
   structural_soil_C -= structural_C_loss;
  
   /*loss of C to atmosphere and leaching */
   rh = fast_C_loss + structural_C_loss;
}
#endif /* MAIMI_LU */


////////////////////////////////////////////////////////////////////////////////
//! A_function
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double patch::A_function (unsigned int time_period, UserData* data) {
    
   /* The combined reduction in decomposition rate as a funciton of TEMP and MOIST */
   /* Based on CENTURY Parton et al 1993 GBC 7(4):785-809 */
   /* and Ben's copy of century code */

   /************/
   /* WARNINGS */
   /************/
   /* NOTE: This DOES include patch scale variability in soil moisture!, 
      a good thing */
  
   /* NOTE: this DOES NOT include patch scale variability in temperature 
      or soil temperature, a bad thing.*/
  
   site* currents = siteptr;

   double soil_temp;
#ifdef ED
   //double rain = currents->sdata->precip[time_period];
   //double pet = currents->sdata->pet[time_period];
   soil_temp = currents->sdata->soil_temp[time_period];
#endif
#ifdef MIAMI_LU
   soil_temp = currents->sdata->temp_average;
#endif /* MIAMI_LU */

   /* EFFECT OF TEMPERATURE */
   /* Lloyd-Taylor (1994) Functional Ecology 8:315-323 */
   /* Arrhenius-derived response that flattens near freezing, */
   /* replacing Q10 which overestimates temperature sensitivity at low T. */
   /* Optimized against HWSD soil carbon and Hashimoto Rh observations. */
    double Td = 0;
{
    double R0 = 1.93;         /* base decomposition rate */
    double E0 = 243.5;        /* activation energy parameter (K) */
    double T_freeze = 3.6;    /* effective freezing threshold (C) */

    /* average Lloyd-Taylor response across 3 soil temperature layers */
    double soil_temps[3] = {
        currents->sdata->soil_temp1[time_period],
        currents->sdata->soil_temp2[time_period],
        currents->sdata->soil_temp3[time_period]
    };

    for (int i = 0; i < 3; i++) {
        if (soil_temps[i] > T_freeze) {
            double denom = soil_temps[i] + 46.02;
            if (denom > 0.1)
                Td += R0 * exp(E0 * (1.0/56.02 - 1.0/denom));
        }
    }
    Td /= 3.0;
}
   

   /* EFFECT OF MOISTURE */
   /* Log-parabolic response (Moyano et al. 2013, Biogeosciences 10:3961) */
   /* Smooth Gaussian in log-space with optimum at theta_opt. */
   /* Replaces piecewise linear (Linn & Doran 1984) which had hard breakpoints. */
   /* sigma=3.7 indicates weak moisture modulation at global scale, */
   /* consistent with Bond-Lamberty & Thomson (2010). */
   double Wd;
#if defined ED
{
    double theta_opt = 0.19;   /* optimal soil moisture for decomposition */
    double sigma = 3.70;       /* width of log-parabolic response */
    double theta_safe = (theta > 0.01) ? theta : 0.01;
    double log_ratio = log(theta_safe / theta_opt);
    Wd = exp(-log_ratio * log_ratio / (2.0 * sigma * sigma));
}


   return (Td * Wd);

#elif defined MIAMI_LU
   /* Crude Approximation to centruy function, ignoring stored water term. */
  
   double Rmax = 6.0; /* was 6.0, but caused problems at some sites */  
   double Ropt = 0.9;
   double Rshr = 90.; 
   double Rshl = 1.;
   double Rratio = currents->sdata->precip_average / currents->sdata->pet_average;
   if (Rratio > Rmax) Rratio = Rmax - 0.1; /* GCH Trap */
   double R1 = (Rmax - Rratio) / (Rmax - Ropt);
   double R2 = exp( (Rshr / Rshl) * (1. - pow(R1, Rshl)) );
   Wd = pow(R1, Rshr) * R2;

   double term = Td * Wd;
   if (term < 0.005) term = 0.005; /* GCH Trap */

   return term; /* the combined (multiplicative) effect of temp and water on decom rates */

#endif /* ED vs. MIAMI_LU */
}

