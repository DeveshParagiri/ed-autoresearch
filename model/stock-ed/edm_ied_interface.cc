#include <cstdio>
#include "netcdf.h"

// TODO: this makes things messy
#include "../iGLM/glm_coupler.h"

#include "edmodels.h"
#include "site.h"
#include "patch.h"
#include "cohort.h"
#include "read_site_data.h"
#include "outputter.h"
#include "restart.h"

#include "edm_ied_interface.h"



void delete_world(site* world);


EDMiEDInterface::EDMiEDInterface () {
}


void EDMiEDInterface::initialize (char* aExpName, new_data* aGLMData, int aStartYear) {
   // TODO: should have the ability to pass config file name through 
   edmControl = ed_initialize(aExpName, NULL);
   edmControl->glm_data = aGLMData;
   edmControl->start_year = aStartYear;
   potentialBiomass = (double**)malloc_2d(NY, NX, sizeof(double));
   readRegionAEZFile();
   readDiscountedCarbonFile();
}


void EDMiEDInterface::doYear (int aYear) {
   ed_step(aYear, *edmControl);
}


void EDMiEDInterface::finalize (int aYear) {
   saveRestartState(aYear);
   ed_finalize(*edmControl);
}

void EDMiEDInterface::saveRestartState (int aYear) {
   edmControl->restartWriter->storeStates (edmControl->first_site, aYear);
}

// TODO: make sure we are indexing the same way
double EDMiEDInterface::getRegAEZDiscountedBiomassDensity(int aReg, int aAEZ, int aCrop) {
   return regaezDiscountedBiomassDensity[aReg][aAEZ][aCrop];
}


double EDMiEDInterface::getRegAEZSoilCarbonDensity(int aReg, int aAEZ, int aCrop) {
   return regaezSoilCarbonDensity[aReg][aAEZ][aCrop];
}


// TODO: what is the sign convention
double EDMiEDInterface::getGlobalNetFlux() {
   double newTotalC = 0.0;
   site* cs = edmControl->first_site;
   while (cs != NULL) {
      newTotalC += cs->site_total_c * cs->sdata->grid_cell_area * T_PER_KG * GT_PER_T;
      cs = cs->next_site;
   }

   double flux = edmControl->lastTotalC - newTotalC;
   edmControl->lastTotalC = newTotalC;
   return flux * 1000; // Convert GT C to MT C
}


double** EDMiEDInterface::getGriddedPotentialBiomass() {
   for (size_t y=0; y<edmControl->n_lat; y++) 
      for (size_t x=0; x<edmControl->n_lon; x++) 
         potentialBiomass[y][x] = 0.0;
   
   site* cs = edmControl->first_site;
   while (cs != NULL) {
      potentialBiomass[cs->y][cs->x] = cs->total_ag_biomass[LU_NTRL];
      cs = cs->next_site;
   }
   return potentialBiomass;
}


void EDMiEDInterface::backupWorld() {
   if (edmControl->sitelist_copy != NULL) {
      delete_world(edmControl->sitelist_copy);
   }
   edmControl->sitelist_copy = copyWorld(edmControl->first_site);
   edmControl->backupTotalC = edmControl->lastTotalC;
   lastRecNo = edmControl->outputter->recNo;
}


void EDMiEDInterface::restoreWorld() {
   delete_world(edmControl->first_site);
   edmControl->first_site = copyWorld(edmControl->sitelist_copy);

#if TBB
   site *cs = edmControl->first_site;
   for (size_t i=0; i<edmControl->number_of_sites; i++) {
      edmControl->site_arr[i] = cs;
      cs = cs->next_site;
   }
#endif
   edmControl->lastTotalC = edmControl->backupTotalC;
   edmControl->outputter->recNo = lastRecNo;
}


void EDMiEDInterface::readDiscountedCarbonFile ( ) {
   FILE* inf;
#ifdef MIAMI_LU
   inf = fopen("/lustre/data/fisk/dsct_c_distexp_nf_double.txt","r");
#endif
   // TODO: file for ED

   // get rid of header
   char buffer[256];
   fgets(buffer, 256, inf);

   for (int i=0; i<N_GCAM_REG; i++) {
      for (int j=0; j<N_GCAM_AEZ; j++) {
         for (int k=0; k<N_GCAM_CROP; k++) {
            regaezDiscountedBiomassDensity[i][j][k] = 0.0;
            regaezSoilCarbonDensity[i][j][k] = 0.0;
         }
      }
   }
   int reg, aez, crop;
   double dbc, sc;
   while (fscanf(inf, "%d %d %d %lf %lf\n", &reg, &aez, &crop, &dbc, &sc) != EOF) {
      regaezDiscountedBiomassDensity[reg-1][aez-1][crop] = dbc;
      regaezSoilCarbonDensity[reg-1][aez-1][crop] = sc;
   }
   fclose(inf);
}

            
void EDMiEDInterface::readRegionAEZFile ( ) {
   
   gcamRegMap = (int**)malloc_2d(edmControl->n_lat, edmControl->n_lon, sizeof(int));
   gcamAEZMap = (int**)malloc_2d(edmControl->n_lat, edmControl->n_lon, sizeof(int));
   
   size_t index[2], count[2];
   index[0] = edmControl->start_lat;
   index[1] = edmControl->start_lon;
   count[0] = edmControl->n_lat;
   count[1] = edmControl->n_lon;
   
   int rv, ncid, varid;
   if ((rv = nc_open(REGAEZFILE, NC_NOWRITE, &ncid)))
      NCERR(REGAEZFILE, rv);
   
   if ((rv = nc_inq_varid(ncid, "region", &varid)))
      NCERR("region", rv);
   if ((rv = nc_get_vara_int(ncid, varid, index, count, &gcamRegMap[0][0])))
      NCERR("region", rv);
   
   if ((rv = nc_inq_varid(ncid, "aez", &varid)))
      NCERR("aez", rv);
   if ((rv = nc_get_vara_int(ncid, varid, index, count, &gcamAEZMap[0][0])))
      NCERR("aez", rv);

   for (size_t i=0; i<edmControl->n_lat; i++) {
      for (size_t j=0; j<edmControl->n_lon; j++) {
         gcamAEZMap[i][j] %= 100;
      }
   }
}
   


site* EDMiEDInterface::copyWorld(site* world) {
   site *new_world = NULL;
   site *ls = NULL;
   site *cs = world;
   site *ns;
   while (cs != NULL) {
      //ns = new site(*cs);
      ns = (site*)malloc(sizeof(site));
      *ns = *cs;
      if (ls != NULL) {
         ls->next_site = ns;
      } else {   
         new_world = ns;
      }
 
      for (int lu=0; lu<N_LANDUSE_TYPES; lu++) {
         ns->youngest_patch[lu] = NULL;
         patch *np = NULL;
         patch *lp = NULL;
         patch *cp = cs->youngest_patch[lu];
         while (cp != NULL) {
            //np = new patch(*cp);
            np = (patch*)malloc(sizeof(patch));
            *np = *cp;
            if (lu == LU_SCND){
               np->phistory = (double*)calloc(edmControl->n_years_to_simulate+1, sizeof(double));
#if 0
               for (int i=0; i<edmControl->n_years_to_simulate+1; i++)
                  np->phistory[i] = cp->phistory[i];
#endif
            }
            np->older = NULL;
            np->siteptr = ns;
            if (lp != NULL) {
               lp->older = np;
               np->younger = lp;
            } else {
               np->younger = NULL;
               ns->youngest_patch[lu] = np;
            }

#ifdef ED
            np->tallest = NULL;
            np->shortest = NULL;
            cohort *nc = NULL;
            cohort *lc = NULL;
            cohort *cc = cp->shortest;
            while (cc != NULL) {
               //nc = new cohort(cc);
               nc = (cohort*)malloc(sizeof(cohort));
               *nc = *cc;
               nc->taller = NULL;
               nc->patchptr = np;
               nc->siteptr = ns;
               if (lc != NULL) {
                  lc->taller = nc;
                  nc->shorter = lc;
               } else {
                  nc->shorter = NULL;
                  np->shortest = nc;
               }
               cc = cc->taller;
            }
            np->tallest = nc;
#endif
            cp = cp->older;
         }
         ns->oldest_patch[lu] = np;
      }
      ls = ns;
      cs = cs->next_site;
   }
   ns->next_site = NULL;

   return new_world;
}


void delete_world(site* world) {
   site *cs = world;
   while (cs != NULL) {
      for (int lu=0; lu<N_LANDUSE_TYPES; lu++) {
         patch *cp = cs->youngest_patch[lu];
         while (cp != NULL) {
#ifdef ED
            cohort *cc = cp->shortest;
            while (cc != NULL) {
               cohort *tc = cc;
               cc = cc->taller;
               free(tc);
            }
#endif
            if (lu == LU_SCND)
               free(cp->phistory);
            patch *tp = cp;
            cp = tp->older;
            free(tp);
         }
      }
      site *ts = cs;
      cs = ts->next_site;
      free(ts);
   }
}

