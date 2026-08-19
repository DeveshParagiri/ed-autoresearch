#include <cstring>
#include <sys/stat.h>
#include "edmodels.h"
#include "outputter.h"
#include "readconfiguration.h"

using namespace libconfig;

////////////////////////////////////////////////////////////////////////////////
//! fileExists
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
//! get_section_name
//! returns section.value, if no section is given, returns just the name of the param 
//! in the configuration file
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
char* get_section_name(const char* section, const char* param) {
  char *tmp = (char *)malloc(STR_LEN * sizeof(char));  
  if (tmp == NULL) {
      fprintf(stderr, "get_section_name: out of memory - can't allocate tmp\n");
      exit(0);
  }
  tmp[0] = '\0';
  
  if(strlen(section)>0) {
      strcat(tmp, section);
      strcat(tmp,".");
  }  
  strcat(tmp, param);    
   
  return tmp;
}

int read_config_file(const char* cfgFile, UserData* data) {
  char* section_name;    
  const char* file_name_default;
  const char* file_name_alternate;

  // Read all the configuration files mentioned in MODEL_CONFIG_FILE 
  // If there is an error, report it and exit.
  try
  {
    
    data->model_cfg = new Config();
    if (cfgFile == NULL) {
       data->model_cfg->readFile(MODEL_CONFIG_FILE);
    } else {
       data->model_cfg->readFile(cfgFile);
    }

    // Which model are we running? ED or MIAMI_LU?
#ifdef ED
    data->model_name = "ed";
#else    
    data->model_name = "mlu";
#endif

    // Initialize other configuration file objects
    data->io_cfg_default       = new Config();
    data->io_cfg_alternate     = new Config();
    data->params_cfg_default   = new Config();
    data->params_cfg_alternate = new Config();
    data->pfts_cfg_default     = new Config();
    data->pfts_cfg_alternate   = new Config();

    // Depending on which model we are running, read the remaining configuration files
    /* PARAMS */
    section_name = get_section_name(data->model_name,"params_default");
    data->model_cfg->lookupValue(section_name, file_name_default);
    free(section_name);
    section_name = get_section_name(data->model_name,"params_alternate");
    data->model_cfg->lookupValue(section_name, file_name_alternate);
    free(section_name);
    
    if (fileExists(file_name_default)) {
       data->params_cfg_default->readFile(file_name_default);
    } else {
       data->params_cfg_default->readFile(file_name_alternate);
    }
    if(fileExists(file_name_alternate)) {
       data->params_cfg_alternate->readFile(file_name_alternate);
    } else {
       data->params_cfg_alternate->readFile(file_name_default);
    }
    
    /* IO */
    section_name = get_section_name(data->model_name,"io_default");
    data->model_cfg->lookupValue(section_name, file_name_default);
    free(section_name);
    section_name = get_section_name(data->model_name,"io_alternate");
    data->model_cfg->lookupValue(section_name, file_name_alternate);
    free(section_name);
    if (fileExists(file_name_default)) {
       data->io_cfg_default->readFile(file_name_default);
    } else {
       data->io_cfg_default->readFile(file_name_alternate);
    }
    if(fileExists(file_name_alternate)) {
       data->io_cfg_alternate->readFile(file_name_alternate);
    } else {
       data->io_cfg_alternate->readFile(file_name_default);
    }
    
    /* PFT'S */
    if(!strcmp(data->model_name,"ed")) {
        section_name = get_section_name(data->model_name,"pfts_default");
        data->model_cfg->lookupValue(section_name, file_name_default);
        free(section_name);
        section_name = get_section_name(data->model_name,"pfts_alternate");
        data->model_cfg->lookupValue(section_name, file_name_alternate);
        free(section_name);
        if (fileExists(file_name_default)) {
           data->pfts_cfg_default->readFile(file_name_default);
        } else {
           data->pfts_cfg_default->readFile(file_name_alternate);
        }
        if(fileExists(file_name_alternate)) {
           data->pfts_cfg_alternate->readFile(file_name_alternate);
        } else {
           data->pfts_cfg_alternate->readFile(file_name_default);
        }
    }
  }
  catch(const FileIOException &fioex)
  {
    printf("I/O error while reading configuration file\n");
    exit(0);
  }
  catch(const ParseException &pex)
  {
    printf("Parse error at %s : %d\n",pex.getFile(), pex.getLine());
    printf("Error: %s\n", pex.getError());
    exit(0);
  }

return(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! free_user_data
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void free_user_data(UserData* data) {
    free(data->lats);
    free(data->lons);

    delete(data->outputter);
    delete(data->model_cfg);
    delete(data->io_cfg_default);
    delete(data->io_cfg_alternate);
    delete(data->params_cfg_default);
    delete(data->params_cfg_alternate);
    delete(data->pfts_cfg_default);
    delete(data->pfts_cfg_alternate);
    
    free(data);
}

////////////////////////////////////////////////////////////////////////////////
//! initialize_model_params
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void initialize_model_params(UserData* data) {
    data->c2b                      = get_val<double>(data, PARAMS, "", "c2b");           /* carbon to biomass conversion */
    data->allometry_type           = get_val<int>(data, PARAMS, "", "allometry_type");           /* which set of allometric equations to use */

    /*  REGION  */
    data->is_site        = get_val<int>(data, MODEL_IO, "", "is_site");
    data->region         = get_val<const char*>(data, MODEL_IO, "", "region");
    data->latmin         = get_val<double>(data, MODEL_IO, data->region, "LATMIN");
    data->latmax         = get_val<double>(data, MODEL_IO, data->region, "LATMAX");
    data->lonmin         = get_val<double>(data, MODEL_IO, data->region, "LONMIN");
    data->lonmax         = get_val<double>(data, MODEL_IO, data->region, "LONMAX");  

    /* LANDUSE */
    data->landuse_bau    = get_val<int>(data, PARAMS, "", "landuse_bau");    
    data->landuse_stop   = get_val<int>(data, PARAMS, "", "landuse_stop");
    
    data->is_external_crop_calendar = get_val<int>(data, PARAMS, "", "is_external_crop_calendar");

    /* FIRE  */
    data->fire_suppression_stop = get_val<int>(data, PARAMS, "", "fire_suppression_stop"); 
    data->fire_suppression      = get_val<int>(data, PARAMS, "", "fire_suppression"); 
    data->fire_off              = get_val<int>(data, PARAMS, "", "fire_off");         
    data->fire_gfed              = get_val<int>(data, PARAMS, "", "fire_gfed");

    /* INPUT DATA PATHS*/
    data->output_base_path  = get_val<const char*>(data, MODEL_IO, "", "output_base_path"); 
    data->which_mech_to_use = get_val<const char*>(data, MODEL_IO, "", "which_mech_to_use");
    data->gridspec          = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "gridspec");
    data->climate_file      = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "climate_file");
    data->climate_file_avg      = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "climate_file_avg");
    data->soil_file         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "soil_file");
#if SOILGRIDS_SCHEME
    data->soilgrids_file = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "soilgrids_file");
#endif
    data->lu_file           = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "lu_file");
    data->lu_init_c_file    = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "lu_init_c_file");
    data->crop_calendar_file= get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "crop_calendar_file");

    if(data->gfed_bf)
    {
        data->gfedbf_file       = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "gfedbf_file");
        data->gfedbf_file_avg       = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "gfedbf_file_avg");
    }
#if FTS
    data->QAIR_FILE         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "QAIR_FILE");
    data->TAIR_FILE         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "TAIR_FILE");
    data->SW_FILE           = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "SW_FILE");
    data->C3_FILE           = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "C3_FILE");
    data->C4_FILE           = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "C4_FILE");
#endif
    
    data->PREMECH         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "PREMECH");
    data->PREMECH_avg         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "PREMECH_avg");
    data->PREMECH_CO2_avg         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "PREMECH_CO2_avg");
    data->PREMECH_CO2         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "PREMECH_CO2");
    
#if FORECAST
    data->forecast_climate_file         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "forecast_climate_file");
    data->forecast_PREMECH         = get_val<const char*>(data, MODEL_IO, data->which_mech_to_use, "forecast_PREMECH");
#endif

    data->single_year       = get_val<int>(data, MODEL_IO, data->which_mech_to_use, "single_year");   
    data->do_yearly_mech    = get_val<int>(data, MODEL_IO, data->which_mech_to_use, "do_yearly_mech");
    data->m_int             = get_val<int>(data, MODEL_IO, data->which_mech_to_use, "m_int");   
    data->m_string          = get_val<int>(data, MODEL_IO, data->which_mech_to_use, "m_string"); 


    /* MULTIPLE Vm0s*/
    data->num_Vm0           = get_val<int>(data, PARAMS, "", "num_Vm0s");
    /* HURRICANES */
    data->hurricane            = get_val<int>(data, PARAMS, "", "hurricane");
    data->hurricane_file       = get_val<const char*>(data, PARAMS, "", "hurricane_file");
    data->do_hurricane         = get_val<int>(data, PARAMS, "", "do_hurricane");
    data->hurricanetology      = get_val<int>(data, PARAMS, "", "hurricanetology");
    data->hurricane_ramp       = get_val<int>(data, PARAMS, "", "hurricane_ramp");
    data->hurricane_ramp_exp   = get_val<const char*>(data, PARAMS, "", "hurricane_ramp_exp"); /*  "A2" A1B" "A2" "B1" */
    data->hurricane_ramp_years = get_val<int>(data, PARAMS, "", "hurricane_ramp_years");
    data->n_hurricane_years    = get_val<int>(data, PARAMS, "", "n_hurricane_years"); 
    data->hurricane_start_year = get_val<int>(data, PARAMS, "", "hurricane_start_year");

    /*************************************/
    /***    INTEGRATION                ***/
    /*************************************/
    data->tmax                   = get_val<double>(data, PARAMS, "", "tmax"); /*number of years to simulated */
#ifdef ED
    data->stiff_light            = get_val<int>(data, PARAMS, "", "stiff_light"); /* 1= yes to stiff integration of light levels */
    data->substeps               = get_val<int>(data, PARAMS, "", "substeps"); 
#endif
    data->patch_dynamics         = get_val<int>(data, PARAMS, "", "patch_dynamics");  /* patch dynamics flag, 1=yes to patch dynamics */
   
    data->restart                  = get_val<int>(data, PARAMS, "", "restart");
    data->old_restart_write        = get_val<int>(data, PARAMS, "", "old_restart_write");
    data->old_restart_read         = get_val<int>(data, PARAMS, "", "old_restart_read");
    data->old_restart_exp_name     = get_val<const char*>(data, PARAMS, "", "old_restart_exp_name"); 

    data->new_restart_write        = get_val<int>(data, PARAMS, "", "new_restart_write");
    data->new_restart_read         = get_val<int>(data, PARAMS, "", "new_restart_read");
    data->restart_dir              = get_val<const char*>(data, PARAMS, "", "restart_dir");  
    
#ifdef ED
    /**************************************/
    /***    BIOLOGY/BIOGEOCHEMISTRY     ***/
    /**************************************/
    data->canopy_damage            = get_val<double>(data, PARAMS, "", "canopy_damage"); /* fraction of canopy damaged by small scale dist */
    data->open_cycles              = get_val<int>(data, PARAMS, "", "open_cycles");     /* open biogeochemical cycles, 1=yes,0=no */

    data->water_competition        = get_val<int>(data, PARAMS, "", "water_competition");      /* competition for water flag, 1=yes */
    data->n_competition            = get_val<int>(data, PARAMS, "", "n_competition");          /* competition for N flag, 1=yes */
    data->n_decomp_limitation      = get_val<int>(data, PARAMS, "", "n_decomp_limitation");    /* N control of structuralc decomp, 1=yes */
    data->internal_recruitment     = get_val<int>(data, PARAMS, "", "internal_recruitment");   /* internal recruitment flag 1=yes */
    data->external_recruitment     = get_val<int>(data, PARAMS, "", "external_recruitment");   /* external fixed recruitment flag 1=yes */
    data->hgtmin                   = get_val<double>(data, PARAMS, "", "hgtmin");        /* lowest height value assigned to data->hgt_min */
#endif
    /* set in pde to reasonable value, say 10000.0, for *
     * numerics, actual site area is read in, is huge,  *
     * and can be used for area dependent calcs later   */
    data->area                     = get_val<double>(data, PARAMS, "", "area");
    
    data->growth_resp              = get_val<double>(data, PARAMS, "", "growth_resp");   /* 0.333 fraction of npp lost as growth respiration */
    data->mass_of_water            = get_val<double>(data, PARAMS, "", "mass_of_water"); /* (kg/mm/m2), roughly- fix! */

    /**************************************/
    /***    FUSION/FISSION              ***/
    /**************************************/
    data->patch_termination           = get_val<int>(data, PARAMS, "", "patch_termination"); // terminate small patches and adj area 
    data->patch_fusion                = get_val<int>(data, PARAMS, "", "patch_fusion");  // patch fusion flag, 1=yes to patch fusion
#ifdef ED
    data->cohort_fusion               = get_val<int>(data, PARAMS, "", "cohort_fusion"); // cohort fusion flag, 1=yes to cohort fusion
    data->cohort_fission              = get_val<int>(data, PARAMS, "", "cohort_fission"); // cohort fission flag, 1=yes to cohort fission
    data->cohort_termination          = get_val<int>(data, PARAMS, "", "cohort_termination"); // terminate small cohorts 
#endif
    data->f_area                      = get_val<double>(data, PARAMS, "", "f_area"); // 0.01 min area of patch as a fraction of total area */
    data->profile_tol                 = get_val<double>(data, PARAMS, "", "profile_tol"); /* 0.2 fractional tolerance for patch spp hgt profiles */
#ifdef ED
    data->btol                        = get_val<double>(data, PARAMS, "", "btol");     /* 0.0001 termination tol. for cohort biomass (kg m^-2) */
    data->ntol                        = get_val<double>(data, PARAMS, "", "ntol");      /* min plant density for hgt bin to be used in height profile comparisons */
    data->fusetol                     = get_val<double>(data, PARAMS, "", "fusetol");     /* min fractional difference in dbh bet cohorts */
    data->lai_tol                     = get_val<double>(data, PARAMS, "", "lai_tol");     /* maximum LAI allowed for a cohort */
    data->dbhmax                      = get_val<double>(data, PARAMS, "", "dbhmax");    /* max dbh value used in hgt profile comparison */
#endif
    data->smallest_new_patch_f        = get_val<double>(data, PARAMS, "", "smallest_new_patch_f");   /* 0.00001 as fraction of data->area*/
    data->min_change_in_area          = get_val<double>(data, PARAMS, "", "min_change_in_area");
    data->min_change_in_area_patch    = get_val<double>(data, PARAMS, "", "min_change_in_area_patch");
    data->min_area_fraction           = get_val<double>(data, PARAMS, "", "min_area_fraction");
    data->min_patch_area              = get_val<double>(data, PARAMS, "", "min_patch_area");
    data->min_landuse_area_fraction   = get_val<double>(data, PARAMS, "", "min_landuse_area_fraction");
    data->min_landuse_change_fraction = get_val<double>(data, PARAMS, "", "min_landuse_change_fraction");
    
    /**************************************/
    /***    MISC                        ***/
    /**************************************/
    data->do_downreg              = get_val<int>(data, PARAMS, "", "do_downreg");
    data->light_reg              =  get_val<int>(data, PARAMS, "", "light_reg");
    data->additional_mort         = get_val<int>(data, PARAMS, "", "additional_mort");
    data->mort_s_hemi             = get_val<int>(data, PARAMS, "", "mort_s_hemi");    
    data->hgt_lim_to_repro        = get_val<int>(data, PARAMS, "", "hgt_lim_to_repro");
    data->tropic_n_limit          = get_val<double>(data, PARAMS, "", "tropic_n_limit");  
    data->tropic_s_limit          = get_val<double>(data, PARAMS, "", "tropic_s_limit");      
    data->n_init_patches          = get_val<int>(data, PARAMS, "", "n_init_patches");  
    data->rarify_factor           = get_val<int>(data, PARAMS, "", "rarify_factor");        /* to coarsen regional run, skip every # of sites */    
#ifdef ED
    data->cohort_shading          = get_val<double>(data, PARAMS, "", "cohort_shading");     /* degree of within cohort shading */ 
    data->self_shading            = get_val<double>(data, PARAMS, "", "self_shading");       /* 0.5, degree of self shading */ 
#endif
    data->cell_area               = get_val<double>(data, PARAMS, "", "cell_area");
    
   /**************************************/
   /***    PRINTING                    ***/
   /**************************************/
   
   /* 1 = yes, do you want all of those output    *
    * files printed?                              *
    * 1 to print full time dependent files,       *
    * 0 for small file for most recent time       *
    *   period only,                              *
    * warning: long file could get huge           */
   data->print_output_files = get_val<int>(data, MODEL_IO, "", "print_output_files");     
   data->print_system_state = get_val<int>(data, MODEL_IO, "", "print_system_state");   /* flag to print system state files */
   data->print_ss_freq      = get_val<int>(data, MODEL_IO, "", "print_ss_freq");     /* in NSUB units */
   /* these two (cd_file, fp_file) have to be off if using gcd !!!! */
   data->cd_file            = get_val<int>(data, MODEL_IO, "", "cd_file");              
   data->fp_file            = get_val<int>(data, MODEL_IO, "", "fp_file");        
   data->long_patch_file    = get_val<int>(data, MODEL_IO, "", "long_patch_file");
   data->long_cd_file       = get_val<int>(data, MODEL_IO, "", "long_cd_file");
   data->long_fp_file       = get_val<int>(data, MODEL_IO, "", "long_fp_file");
   data->long_cohort_file   = get_val<int>(data, MODEL_IO, "", "long_cohort_file");      
   /**************************************/
   /***    ACCOUNTING                  ***/
   /**************************************/
   /* minimum sizes used in calc of patch site chars */
   data->min_dbh_class      = get_val<double>(data, PARAMS, "", "min_dbh_class"); /* min dbh (cm) class for dbh calc */
   data->min_hgt_class      = get_val<double>(data, PARAMS, "", "min_hgt_class"); /* min h(m) class used for dbh calc */   
}
