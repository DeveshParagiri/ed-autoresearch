#ifndef EDM_DISTURBANCE_H_
#define EDM_DISTURBANCE_H_

#include "edmodels.h"

struct patch;

void calculate_disturbance_rates ( unsigned int t, 
                                   patch** current_patch, 
                                   UserData* data );

void accumulate_litter_from_disturbance ( patch** target_patch,
                                          patch** donor_patch, 
                                          double change_in_area,
                                          int q,
                                          UserData* data );

void aggregate_in_soil_state_from_disturbance ( patch** target_patch,
                                              patch** donor_patch,
                                              double change_in_area,
                                              UserData* data );

double get_hurricane_disturbance_rate ( unsigned int t, 
                                        site* current_site,
                                        UserData* data );

#endif // EDM_DISTURBANCE_H_ 
