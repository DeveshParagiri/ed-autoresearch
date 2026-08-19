#ifndef EDM_LANDUSE_H_
#define EDM_LANDUSE_H_

#include "edmodels.h"

/* function prototypes */
void read_initial_landuse_fractions (UserData* data);
void read_initial_landuse_c (UserData* data);
int read_transition_rates (site** first_site, UserData* data);

void init_landuse_patches (site** siteptr, UserData* data);
void landuse_dynamics (unsigned int t, site** siteptr, UserData* data);
void update_landuse (site* siteptr, UserData& data);
void update_mean_age_secondary_lands (site** siteptr, UserData* data);

void print_landuse (unsigned int time, site** siteptr, UserData* data);
void print_region_landuse (unsigned int t, site** siteptr, UserData* data);

#endif // EDM_LANDUSE_H_ 
