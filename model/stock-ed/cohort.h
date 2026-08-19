#ifndef EDM_COHORT_H_
#define EDM_COHORT_H_

#include "edmodels.h"

class patch;

////////////////////////////////////////
//    Typedef: cohort
////////////////////////////////////////
struct cohort{
   int species;      ///< species number                   
   int pt;           ///< physiology 1=c3 2=c4             
   double nindivs;   ///< number of individuals in cohort  
   double dbh;       ///< dbh in cm                        
   double hite;      ///< height in meters                 
   double b;         ///< total biomass per indiv kgC per indiv 
   double babove;    ///< total above ground biomass kgC per indiv 
   double bbelow;    ///< total below ground biomass kgC per indiv 
   double balive;    ///< total living biomass per indiv kgC per indiv 
   double bdead;     ///< dead biomass per indiv kgC per indiv 
   double bsw;       ///< sapwood in stem and roots UNKNOWN. Looks like area per indiv, used like kg per indiv
   double bl;        ///< leaf biomass per indiv kgC per indiv 
   double blv;       ///< leaf biomass in storage "virtual leaves" kgC per indiv     
   double bs;        ///< structural biomass per indiv stem + structual roots kgC per indiv 
   double bstem;     ///< stem biomass per indiv kgC per indiv 
   double br;        ///< fine root biomass per indiv kgC per indiv 
   double leaf_area; ///< leaf area of plant m^2 per indiv
   double lai;       ///< leaf area index of plant dimensionless 
   double lite;      ///< light level for cohort  percent 
   double gpp;       ///< net primary PER PLANT! Kg/plant over timestep kgC/yr per indiv 
   double npp;       ///< net primary PER PLANT! Kg/plant over timestep kgC/yr per indiv 
   double npp2;      ///< net primary PER PLANT! Kg/plant over timestep 
   double gpp_max;   ///< kgC/yr per indiv 
   double npp_max;   ///< kgC/yr per indiv 
   double resp;      ///< plant respiration  kgC/yr per indiv 
   double md;        ///< plant tissue maintenence kg/plant over timestep kgC/yr per indiv
   double npp_avg;
   double gpp_avg;
   double md_avg;
   int plantint_time;
   int harvesting_time;
   
   // For rk2 integrator UNITS - see above
   double old_nindivs;                
   double old_dbh;
   double old_balive;
   double old_bdead;

   double An_max;             ///< kgC/yr per indiv 
   double An_pot;             ///< kgC/yr per indiv    
   double An_shut;            ///< kgC/yr per indiv 
   double An_shut_max;        ///< kgC/yr per indiv 
   double fs_open;            ///< fraction of month with stomates open (dimensionless)    
   double fs_open_max;
   double fsn;                ///< degree of n limitation (dimensionless) 
   double fsw;                ///< degree of water limitation (dimensionless) 
   double carbon_balance;     ///< carbon balance kgC/yr per indiv 
   double cb[N_CLIMATE];
   double cb_toc[N_CLIMATE];
   double cbr[N_CLIMATE];     ///< carbon balance ratio Dimensionless
   double cbr_bar;            ///< running average carbon balance ratio  
   double payment_to_Nfixers; ///< carbon flux to symbionts kgC/yr 
   double p[2];               ///< reproduction seed and clonal kgC/yr per indiv
   double p_avg[2];
   int status;                ///< growth status of plant 

   double Vm0;                ///< vm for that cohort, called in mechanism code 

   double gr_resp;            ///< kgC/yr per indiv 

   // water fields 
   double water_uptake;       ///< kgW/yr per indiv
   double E_pot;              ///< kgW/yr per indiv    
   double E_shut;             ///< kgW/yr per indiv 

   // variables needed for integration 
   double dndt;               ///< time derivative of cohort size yr^-1 
   double dhdt;               ///< time derivative of height m/yr 
   double ddbhdt;             ///< time derivative of dbh cm/yr 
   double dbalivedt;          ///< time derivative of total living biomass kgC/yr per indiv 
   double dbdeaddt;           ///< time derivative of dead biomass kgC/yr per indiv 
   // Copy of neccessary derivatives for rk2 step 1 UNITS - see above
   double dndt1;
   double ddbhdt1;
   double dbalivedt1;
   double dbdeaddt1;

   // nitrogen fields 
   double nitrogen_uptake;     ///< kgN/(yr*m^2) per indiv averaged over whole patch
   double N_uptake_pot;        ///< kgN/(yr*m^2) per indiv averaged over whole patch
   double N_uptake_shut;       ///< kgN/(yr*m^2) per indiv averaged over whole patch
  
   // linked list fields 
   cohort *taller;             ///< pointer to next tallest cohort     
   cohort *shorter;            ///< pointer to next shorter cohort     
   patch *patchptr;            ///< pointer to patch that cohort is in 
   site *siteptr;              ///< pointer to site that cohort is in 
   
   double Dbh(UserData *data);    
   
   // In allometry.c
   double Hite(UserData* data); 
   double Bleaf(UserData* data);
   double Bdead(UserData* data);
   double dHdBd(UserData* data);
   double dDbhdBd(UserData* data);
   double dDbhdBl(UserData* data);
   
   // In growth.c
   double nitrogen_demand_function(double time, UserData* data);
   void Growth_Derivatives(double time, UserData* data);
   void Allocate_Biomass(UserData* data);
   
   // In mortality.c
   double den_dep_death(UserData* data);
   double survivorship_from_disturbance(int q, UserData* data);
   
   // In mechanism.c 
   void npp_function(UserData* data);
   void plant_respiration(UserData* data);
   
   // In cohort.cc 
   void get_cohort_vm0(UserData *data);
};


////////////////////////////////////////
//    Function Prototypes
////////////////////////////////////////
double radians(double degree);
double degrees(double radians);
double get_day_length(double lat, double month, int get_max);
void cohort_dynamics(unsigned int t, double t1, double t2,
                     patch** patchptr, FILE* outfile, UserData *data);
void init_cohorts(patch** patchptr, UserData* data);
void create_cohort(unsigned int spp, double nindivs, double hite, double dbh, 
                   double balive, double bdead, patch** patchptr, 
                   UserData* data);

void create_cohort_test_mor(unsigned int spp, double nindivs, double hite, double dbh,
                            double balive, double bdead, double bl, double br, double blv,
                            double bsw, patch** patchptr, double cb[N_CLIMATE], double cb_toc[N_CLIMATE],UserData* data);

cohort* next_taller(cohort* current, double* stp);
void terminate_cohorts(cohort** ptallest, cohort** pshortest,
                       UserData* data);
void spawn_cohorts(unsigned int t, patch** patchptr, UserData* data);
void split_cohorts(patch **patchptr, UserData* data);
void copy_cohort(cohort** currentc, cohort** copyc);
void fuse_cohorts(patch** patchptr, UserData* data);
void sort_cohorts(patch** patchptr, UserData* data);
void insert_cohort(cohort** pcurrentc, cohort** ptallest, 
                   cohort** pshortest, UserData* data);
void reproduction(unsigned int t, patch** patchptr, UserData* data);

#endif // EDM_COHORT_H_ 
