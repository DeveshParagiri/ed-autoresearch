#include <cmath>

#include "edmodels.h"
#include "readconfiguration.h"
#include "netcdf.h"

#define QSW 3900.0 /* sapwood area to leaf area ratio (dimensionless m2 leaf/m2 sapwood) */ 
void init_mech_table (UserData* data);
////////////////////////////////////////////////////////////////////////////////
//! init_data
//! Initialize parameters
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void init_data (const char* cfgFile, UserData* data) {

   read_config_file(cfgFile, data);

   printf("init_data... \n"); 

   // Get the list of PFTs which we will simulate
#ifdef ED
   get_list(data, PFTS, "", "list_pfts", data->list_of_pfts);
   data->nspecies = data->list_of_pfts.size();
   if(data->nspecies != NSPECIES) {
       printf("Number of PFTs in config file does not match that in edmodels.h\n");
       exit(0);
   }
#endif
   initialize_model_params(data);
   data->n_years_to_simulate = (size_t)data->tmax;        
   data->number_of_sites = 0; /* initialization of counter */
   data->deltat = TIMESTEP;

   /* disturbance parameters */
   /* data->treefall_max_disturbance_rate = 0.01; */ /* our figure */
   /* data->treefall_max_disturbance_rate = 0.012; */ /* our US figure */

   data->treefall_max_disturbance_rate_trop = get_val<double>(data, PARAMS, "", "treefall_max_disturbance_rate_trop"); /* 0.014, lugo figure for tropics, MIAMI_LU:0.018 in v11 */
   data->treefall_max_disturbance_rate_temp = get_val<double>(data, PARAMS, "", "treefall_max_disturbance_rate_temp"); /* MIAMI_LU: 0.028 in v11 */    
   data->fraction_balive_2_fast             = get_val<double>(data, PARAMS, "", "fraction_balive_2_fast");    /* 0.75 */   
   data->selective_harvest_rate             = get_val<double>(data, PARAMS, "", "selective_harvest_rate");   
   data->max_patch_age                      = get_val<double>(data, PARAMS, "", "max_patch_age"); /* age at which fusions cease */
   data->smoke_fraction                     = get_val<double>(data, PARAMS, "", "smoke_fraction");
   data->fp1                                = get_val<double>(data, PARAMS, "", "fp1"); /* 10.0, 1.0, 0.1 -- disturb rate per kgC/m2 of fuel */
   data->fire_max_disturbance_rate          = get_val<double>(data, PARAMS, "", "fire_max_disturbance_rate"); 
   data->crop_residue                       = get_val<double>(data, PARAMS, "", "crop_residue");  /* fraction crop harvest left on site    */
   data->grazing_intensity                  = get_val<double>(data, PARAMS, "", "grazing_intensity"); /* fraction plants on pasture grazed     */
   data->grazing_residue                    = get_val<double>(data, PARAMS, "", "grazing_residue");  /* fraction left in soil pools           */
   data->fraction_harvest_left_on_site      = get_val<double>(data, PARAMS, "", "fraction_harvest_left_on_site"); /* MIAMI_LU: was 0.5 */
   data->forest_definition                  = get_val<double>(data, PARAMS, "", "forest_definition");  /* min biomass to consider forest kgC/m2 */ 
   
#if defined ED
   data->fire_hite_threshold                = get_val<double>(data, PARAMS, "", "fire_hite_threshold");  /* height threshold for fuel */
   data->treefall_hite_threshold            = get_val<double>(data, PARAMS, "", "treefall_hite_threshold");
   data->sd_mort                            = get_val<double>(data, PARAMS, "", "sd_mort");
#elif defined MIAMI_LU
   data->agf_biomass                        = get_val<double>(data, PARAMS, "", "agf_biomass");
   data->K1                                 = get_val<double>(data, PARAMS, "", "K1");
   data->K2                                 = get_val<double>(data, PARAMS, "", "K2");   /* 0.75  .02*1.06  0.85 */
   data->wood_allocation                    = get_val<double>(data, PARAMS, "", "wood_allocation");    /* fraction of npp allocated to wood : 0.36*1.4=0.5 */
#endif

#ifdef ED
   data->water1                             = get_val<double>(data, PARAMS, "", "water1");  /*80*/   /* access to water */
   /* nitogen model parameters */
   data->nitrogen1                          = get_val<double>(data, PARAMS, "", "nitrogen1");  /* plant nitrogen shutdown     */
   data->nitrogen2                          = get_val<double>(data, PARAMS, "", "nitrogen2");  /* microbial nitrogen shutdown */
   data->NC_perc_coeff                      = get_val<double>(data, PARAMS, "", "NC_perc_coeff"); /* need to parameterize */
   /* PRELIM: If a typical soil N value is 0.1 kg/m2 
      (approx from Buschbacher et al 88 for Paragominas) and leaching and 
      runoff are say 10% of a mean rainfall of 1000 mm, then a reasonable
      coefficient value would be on the order of 10^-6, to get a 
      hydrologic loss on the order of 10^-3 to 10^-5 kgN/m2/y- the range
      of values reported in Vitousek et al 1986. See also Bruijnzeel */
   data->fraction_of_GPP_to_Nfixers           = get_val<double>(data, PARAMS, "", "fraction_of_GPP_to_Nfixers"); /* value from A. Fitter 1997. In       *
                                                                                                        * Crawley, M. ed. Plant Ecology. p.67 */
   /* these are properties of species (cohorts), best place for 'em? */
   /* 3BOX CENTURY */
   data->l2n_stem                             = get_val<double>(data, PARAMS, "", "l2n_stem");   /* values??? made-up */
   data->l2n_root                             = get_val<double>(data, PARAMS, "", "l2n_root");
   data->l2n_leaf                             = get_val<double>(data, PARAMS, "", "l2n_leaf");
   data->c2n_structural                       = get_val<double>(data, PARAMS, "", "c2n_structural"); /* from Century */
   data->c2n_slow                             = get_val<double>(data, PARAMS, "", "c2n_slow");       /* approx from Century */
   data->c2n_passive                          = get_val<double>(data, PARAMS, "", "c2n_passive");    /* approx from Century */
   data->c2n_root                             = get_val<double>(data, PARAMS, "", "c2n_root");
   data->c2n_stem                             = get_val<double>(data, PARAMS, "", "c2n_stem");
      
   /**************************************/
   /* BIODIVERSITY AXIS PARAMETER VALUES */
   /**************************************/    
   /* slope of density-independent mortality rate with wood density */
   data->m1                = get_val<double>(data, PARAMS, "", "m1"); /* 0.15 */
   data->m2                = get_val<double>(data, PARAMS, "", "m2");
   data->m3                = get_val<double>(data, PARAMS, "", "m3"); /* 10, carbon balance mortality parameter */
   data->L_extinct         = get_val<double>(data, PARAMS, "", "L_extinct");  /* light extinction  coefficient as func of LAI  */
   data->Rn_extinct        = get_val<double>(data, PARAMS, "", "Rn_extinct"); /* Net Rad. flux extinction coeff as func of LAI */
   data->cohort_shading    = get_val<double>(data, PARAMS, "", "cohort_shading");     /* degree of within cohort shading */
    
#if LANDUSE
    // From Hansis et al., BLUE model. Supplemental document Page 8. decay_rate = -1.0/(0.534*lamda), lamda is 1, 10, 100 respectively.
    data->yr1_decay_rate                = get_val<double>(data, PARAMS, "", "yr1_decay_rate");
    data->yr10_decay_rate                = get_val<double>(data, PARAMS, "", "yr10_decay_rate");
    data->yr100_decay_rate                = get_val<double>(data, PARAMS, "", "yr100_decay_rate");
    data->sd_mort_crop                            = get_val<double>(data, PARAMS, "", "sd_mort_crop");
    data->crop_mini_nindiv                        = get_val<double>(data, PARAMS, "", "crop_mini_nindiv");
#endif
       
   for(size_t i=0; i<NSPECIES; i++) {
       // Name of PFT which we are initializing
       const char *pft                        = data->list_of_pfts.at(i).c_str();
       
       // Is the PFT a grass?
       if(!strcmp(get_val<const char*>(data, PFTS, pft, "is_grass"), "True")) {
           data->is_grass[i]                  = 1;
       }
       else {
           data->is_grass[i]                  = 0;   
       }
       
       // Is the PFT tropical?
       if(!strcmp(get_val<const char*>(data, PFTS, pft, "is_tropical"), "True")) {
           data->is_tropical[i]               = 1;
       }
       else {
           data->is_tropical[i]               = 0; 
       }
       
       // Is the PFT drought deciduous?
       if(!strcmp(get_val<const char*>(data, PFTS, pft, "is_drought_deciduous"), "True")) {
           data->is_drought_deciduous[i]      = 1;
       }
       else {
           data->is_drought_deciduous[i]      = 0; 
       }
       
       // Is the PFT cold deciduous?
       if(!strcmp(get_val<const char*>(data, PFTS, pft, "is_cold_deciduous"), "True")) {
           data->is_cold_deciduous[i]         = 1;
       }
       else {
           data->is_cold_deciduous[i]         = 0; 
       }
       
       //test_clean
       data->allometry_types[i] = get_val<int>(data, PFTS, pft, "allometry_type");
       
       // Allometry based on Albani et al. GCB 2006
       data->ref_hgt[i] = get_val<double>(data, PFTS, pft, "ref_hgt");
       data->min_hgt[i] = get_val<double>(data, PFTS, pft, "min_hgt");       
         
       data->b1Bl[i] = get_val<double>(data, PFTS, pft, "b1_leaf");
       data->b2Bl[i] = get_val<double>(data, PFTS, pft, "b2_leaf");
       data->b1Bs[i] = get_val<double>(data, PFTS, pft, "b1_structural");
       data->b2Bs[i] = get_val<double>(data, PFTS, pft, "b2_structural");
       data->b1Ht[i] = get_val<double>(data, PFTS, pft, "b1_height");
       data->b2Ht[i] = get_val<double>(data, PFTS, pft, "b2_height");
       
       data->Vm0_max[i] = get_val<double>(data, PFTS, pft, "Vm0_max");
       /* assign inflexion in dbh-h curve  */ 
       /* 40m  => 84.16cm     35m => 68.31cm     30m => 53.68cm */
       /* 25m  => 40.4cm      20m => 28.5cm      15m => 18.1cm */
       /* 12m  => 12.83cm     7m => 5.53cm       4m => 2.30cm */
       /* 1.5m => 0.5cm       0.75m => 0.17cm  */ 
       data->max_dbh[i]                       = get_val<double>(data, PFTS, pft, "max_dbh");

       /* assign physiological types of species c3=0,c4=1  */
       data->pt[i]                            = get_val<int>(data, PFTS, pft, "pt");
       
       /* assign phenology, evergreen =0,"drought-deciduous"=1,"cold-deciduous"=2 */ 
       data->phenology[i]                     = get_val<int>(data, PFTS, pft, "phenology");
       
       /* assign wood density */
       data->rho[i]                           = get_val<double>(data, PFTS, pft, "rho");
       data->initial_density[i]               = get_val<double>(data, PFTS, pft, "initial_density");
       
       /* assign decay rates of carbon pools yr^-1 */
       data->alpha[i][0]                      = get_val<double>(data, PFTS, pft, "alpha_repro");          /* repro          */
       data->alpha[i][1]                      = get_val<double>(data, PFTS, pft, "alpha_sapwood");        /* sapwood        */
       data->alpha[i][2]                      = get_val<double>(data, PFTS, pft, "alpha_leaf");           /* leaf           */
       data->alpha[i][3]                      = get_val<double>(data, PFTS, pft, "alpha_root");           /* root           */
       data->alpha[i][4]                      = get_val<double>(data, PFTS, pft, "alpha_virtual_leaves"); /* virtual leaves */
       data->alpha[i][5]                      = get_val<double>(data, PFTS, pft, "alpha_structural");     /* structural     */
       
       /* assign leaf life spans in months */
       /* This is the reference biodiversity axis parameter */
       data->title[i]                         = get_val<const char*>(data, PFTS, pft, "title");
       
       if (!strcmp(data->title[i],"cold_decid"))
           data->leaf_life_span[i]             = 6.0;
       else if (!strcmp(data->title[i],"c3_grass"))
           data->leaf_life_span[i]             = 6.0;
       else
          data->leaf_life_span[i]             = 12.0/data->alpha[i][2];

      data->treefall_s_gtht[i]                = get_val<double>(data, PFTS, pft, "treefall_s_gtht");
      data->treefall_s_ltht[i]                = get_val<double>(data, PFTS, pft, "treefall_s_ltht");
      data->seed_rain[i]                      = get_val<double>(data, PFTS, pft, "seed_rain");       /* external recruitment */

      /* assign resp rates of carbon pools */
      /* taken from Foley (Ibis model) 1996 gbc v10 p603-628 */
      data->beta[i][0]                        = get_val<double>(data, PFTS, pft, "beta_repro");          /* repro          */
      data->beta[i][1]                        = get_val<double>(data, PFTS, pft, "beta_sapwood");        /* sapwood        */
      data->beta[i][2]                        = get_val<double>(data, PFTS, pft, "beta_leaf");           /* leaf           */
      data->beta[i][3]                        = get_val<double>(data, PFTS, pft, "beta_root");           /* root           */
      data->beta[i][4]                        = get_val<double>(data, PFTS, pft, "beta_virtual_leaves"); /* virtual leaves */

      /* fraction of dispersal that is global scaled to disturbance scale */
      data->m[i]                              = get_val<double>(data, PFTS, pft, "m"); 
      /* Minimum height threshold for reproduction */
      data->repro_ht_thresh[i]                = get_val<double>(data, PFTS, pft, "repro_ht_thresh"); 
      /* assign N fixer flag for plants with symbiotic N fixers */
      data->Nfixer[i]                         = get_val<int>(data, PFTS, pft, "Nfixer");
      /* fine root biomass leaf biomass ratio */ 
      data->q[i]                              = get_val<double>(data, PFTS, pft, "q");
      /* assign allocation fractions  */ 
      data->r_fract[i]                        = get_val<double>(data, PFTS, pft, "r_fract");    
      data->c_fract[i]                        = get_val<double>(data, PFTS, pft, "c_fract");    
      data->hgt_min[i]                        = get_val<double>(data, PFTS, pft, "hgt_min");
      data->bs_min[i]                         = get_val<double>(data, PFTS, pft, "bs_min");
      data->bl_min[i]                         = get_val<double>(data, PFTS, pft, "bl_min"); 
      data->den_ind_mort[i]                   = get_val<double>(data, PFTS, pft, "den_ind_mort");
      if(data->is_cold_deciduous[i]) {
         data->den_ind_mort_s_hemi[i]                   = get_val<double>(data, PFTS, pft, "den_ind_mort_s_hemi"); 
      }
      /* compute leaf c2n ratio and sla, Reich et al.*/
      /* leaf n to biomass (mg N/g Biomass) */
      /*c2n*/
      data->c2n_leaf[i]                       = pow(10.0, (1.65 - 0.34 * log10(data->leaf_life_span[i])));
      /* convert to c2n (g C/g N) */
      data->c2n_leaf[i]                       = 1000.0 / data->c2n_leaf[i] * (1.0 / data->c2b);
       
      /* 1/2 is carbon/biomass ratio (we need this) */
      /* SPECIFIC LEAF AREA */
      /* calculate specific leaf area (cm2/g(biomass)) */
      /* Global Raich et al 94 PNAS pp 13730-13734 */
      data->specific_leaf_area[i]             = pow(10.0, (2.4 - 0.46 * log10(data->leaf_life_span[i])));
      /* Amazonian Raich et al 92 Oecologia vol 86 p 16-24 */
      /*data->specific_leaf_area[i]           = pow(10.0, (2.37 - 0.33 * log10(data->leaf_life_span[i])));*/
      /* convert to (m2/kg(carbon) */
      data->specific_leaf_area[i]             = data->c2b * data->specific_leaf_area[i] * 1000.0 / 10000.0;
      printf("spp %s  sla = %f c2n= %f \n", pft, data->specific_leaf_area[i],data->c2n_leaf[i] );
      /* ratio of sapwood area (m2) leaf biomass (kg) *
       * QSW = sapwood area /leaf area ratio          */ 
      data->qsw[i]                            = (1.0 / QSW) * data->specific_leaf_area[i];
      /* data->qsw[i]= 0.05; */
      printf("spp %d  qsw = %f \n", i, data->qsw[i]);
       
#if LANDUSE
       //Loading two different parameters set for early-, mid- and late- succ in tropical and temperate region
       // But just one set for grass and conifer

       // If PFT is early-, mid- and late succ
       if(!strcmp(data->title[i],"early_succ_trop") || !strcmp(data->title[i],"mid_succ_trop") || !strcmp(data->title[i],"late_succ_trop") || !strcmp(data->title[i],"early_succ_temp") || !strcmp(data->title[i],"mid_succ_temp") || !strcmp(data->title[i],"late_succ_temp"))
       {
           //Read parameters for tropical reigon
           data->fraction_harvest_left_on_prim_site_tro[i]          =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_prim_site_tro");
           data->fraction_harvest_left_on_secd_site_tro[i]          =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_secd_site_tro");
           data->fraction_harvest_to_1yr_pool_tro[i]                =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_1yr_pool_tro");
           data->fraction_harvest_to_10yr_pool_tro[i]               =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_10yr_pool_tro");
           data->fraction_harvest_to_100yr_pool_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_100yr_pool_tro");
           
           data->fraction_clearing_left_on_site_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_left_on_site_tro");
           data->fraction_clearing_to_1yr_pool_tro[i]               =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_1yr_pool_tro");
           data->fraction_clearing_to_10yr_pool_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_10yr_pool_tro");
           data->fraction_clearing_to_100yr_pool_tro[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_100yr_pool_tro");
           
           //Read parameters for temperate region
           data->fraction_harvest_left_on_prim_site_temp[i]         =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_prim_site_temp");
           data->fraction_harvest_left_on_secd_site_temp[i]         =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_secd_site_temp");
           data->fraction_harvest_to_1yr_pool_temp[i]               =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_1yr_pool_temp");
           data->fraction_harvest_to_10yr_pool_temp[i]              =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_10yr_pool_temp");
           data->fraction_harvest_to_100yr_pool_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_100yr_pool_temp");
           
           data->fraction_clearing_left_on_site_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_left_on_site_temp");
           data->fraction_clearing_to_1yr_pool_temp[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_1yr_pool_temp");
           data->fraction_clearing_to_10yr_pool_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_10yr_pool_temp");
           data->fraction_clearing_to_100yr_pool_temp[i]            =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_100yr_pool_temp");
       }
       else
       {
           data->fraction_harvest_left_on_prim_site_tro[i]          =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_prim_site");
           data->fraction_harvest_left_on_secd_site_tro[i]          =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_secd_site");
           data->fraction_harvest_to_1yr_pool_tro[i]                =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_1yr_pool");
           data->fraction_harvest_to_10yr_pool_tro[i]               =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_10yr_pool");
           data->fraction_harvest_to_100yr_pool_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_100yr_pool");
           
           data->fraction_clearing_left_on_site_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_left_on_site");
           data->fraction_clearing_to_1yr_pool_tro[i]               =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_1yr_pool");
           data->fraction_clearing_to_10yr_pool_tro[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_10yr_pool");
           data->fraction_clearing_to_100yr_pool_tro[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_100yr_pool");
           
           data->fraction_harvest_left_on_prim_site_temp[i]         =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_prim_site");
           data->fraction_harvest_left_on_secd_site_temp[i]         =   get_val<double>(data, PFTS, pft, "fraction_harvest_left_on_secd_site");
           data->fraction_harvest_to_1yr_pool_temp[i]               =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_1yr_pool");
           data->fraction_harvest_to_10yr_pool_temp[i]              =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_10yr_pool");
           data->fraction_harvest_to_100yr_pool_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_harvest_to_100yr_pool");

           data->fraction_clearing_left_on_site_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_left_on_site");
           data->fraction_clearing_to_1yr_pool_temp[i]              =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_1yr_pool");
           data->fraction_clearing_to_10yr_pool_temp[i]             =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_10yr_pool");
           data->fraction_clearing_to_100yr_pool_temp[i]            =   get_val<double>(data, PFTS, pft, "fraction_clearing_to_100yr_pool");
       }
#endif
   } /* end loop over species */ 
     
   /* ratio of above gnd stem to total stem (stem plus structural roots) */
   data->agf_bs                               = get_val<double>(data, PARAMS, "", "agf_bs");
   data->rho_max1                             = get_val<double>(data, PARAMS, "", "rho_max1");
   data->rho_max2                             = get_val<double>(data, PARAMS, "", "rho_max2"); /* eliminate mortality from this source for NA spp */
#endif
   /* ncid file handles... init to zero, will be set when opened */
   data->climate_file_ncid                    = 0; 
   data->soil_file_ncid                       = 0;
   data->lu_file_ncid                         = 0; 
#if FTS
   init_mech_table(data);
#endif
   }


#if FTS
////////////////////////////////////////////////////////////////////////////////
//! init_mech_table
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void init_mech_table (UserData* data) {
   int rv, ncid, varid;
   if ((rv = nc_open(data->C3_FILE, NC_NOWRITE, &ncid)))
      NCERR(data->C3_FILE, rv);
   
   if ((rv = nc_inq_varid(ncid, "An", &varid))) NCERR("An", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->An[0][0][0][0]))) NCERR("An", rv);
   if ((rv = nc_inq_varid(ncid, "Anb", &varid))) NCERR("Anb", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->Anb[0][0][0][0]))) NCERR("Anb", rv);
   if ((rv = nc_inq_varid(ncid, "E", &varid))) NCERR("E", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->E[0][0][0][0]))) NCERR("E", rv);
   if ((rv = nc_inq_varid(ncid, "Eb", &varid))) NCERR("Eb", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->Eb[0][0][0][0]))) NCERR("Eb", rv);
   
   if ((rv = nc_open(data->C4_FILE, NC_NOWRITE, &ncid)))
      NCERR(data->C4_FILE, rv);
   
   if ((rv = nc_inq_varid(ncid, "An", &varid))) NCERR("An", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->An[1][0][0][0]))) NCERR("An", rv);
   if ((rv = nc_inq_varid(ncid, "Anb", &varid))) NCERR("Anb", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->Anb[1][0][0][0]))) NCERR("Anb", rv);
   if ((rv = nc_inq_varid(ncid, "E", &varid))) NCERR("E", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->E[1][0][0][0]))) NCERR("E", rv);
   if ((rv = nc_inq_varid(ncid, "Eb", &varid))) NCERR("Eb", rv);
   if ((rv = nc_get_var_double(ncid, varid, &data->Eb[1][0][0][0]))) NCERR("Eb", rv);
}
#endif
/******************************************************************************/
/******************************** END OF FILE *********************************/
/******************************************************************************/
