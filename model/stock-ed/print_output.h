#ifndef EDM_PRINT_OUTPUT_H_
#define EDM_PRINT_OUTPUT_H_

#include "outputter.h"

struct patch;

// print region routines
void registerOutputVars (Outputter* o);
void print_initial (site* first_site, UserData* data);
void print_region_files (unsigned int t, site** firsts, UserData* data);

// print site routines
void print_soi_files (unsigned int t, site** current_site, UserData* data);
void print_cfluxes (unsigned int time, site** siteptr, UserData* data);
void print_soil_pools (unsigned int time, site** siteptr, UserData* data);
void print_biomass (unsigned int time, site** siteptr, UserData* data);
void print_harvest (unsigned int time, site** siteptr, UserData* data);
void print_area_burned (unsigned int time,site** siteptr,UserData* data);
void print_diagnostics (unsigned int time, site** siteptr,UserData* data);
void print_patches (unsigned int time, site** siteptr, UserData* data);
void print_site_chars (site** fsite, UserData* data);

// ED-specific printing
void print_light_levels (site** siteptr, unsigned int time,
                         UserData* data);
void print_water (unsigned int time, site** siteptr, UserData* data); 
void print_patch_size_profile (unsigned int t, patch** pcurrentp, 
                               unsigned int nbins, UserData* data);
void print_site_size_profile (unsigned int t, site** pcurrents, 
                              unsigned int nbins, UserData* data);
void print_cohorts (unsigned int time, site** siteptr, UserData* data);
void print_nitrogen_budget (unsigned int time, site** currents, 
                            UserData* data);

// print summary routines
void print_domain_stats (unsigned int t, site** sitrptr, UserData* data);


#endif // EDM_PRINT_OUTPUT_H_ 
