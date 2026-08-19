#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

#include "edmodels.h"
#include "site.h"
#include "read_site_data.h"
#include "patch.h"
#ifdef ED
#include "cohort.h"
#endif

#include "restart.h"

using namespace std;

bool Restart::useTransactions_ = false;


////////////////////////////////////////////////////////////////////////////////
//! Restart
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
Restart::Restart (string dbDir) : dbEnv_(0)
{
   siteSequence_ = 1;
   patchSequence_ = 1;

   u_int32_t envFlags = DB_CREATE | DB_INIT_LOG | DB_INIT_MPOOL;
   u_int32_t dbFlags = DB_CREATE;

   if (useTransactions_) {
      envFlags = envFlags | DB_INIT_TXN | DB_INIT_LOCK;
      dbFlags = dbFlags | DB_AUTO_COMMIT;
   }
   dbEnv_.open(dbDir.c_str(), envFlags, 0);
   dbTxn_ = NULL;

   string dbFile = dbDir + "/restart.db";
   // open databases
   siteDB_ = new Db (&dbEnv_, 0);
   siteDB_->open(NULL, dbFile.c_str(), "site", DB_BTREE, dbFlags, 0);

   patchDB_ = new Db (&dbEnv_, 0);
   patchDB_->set_flags(DB_DUP);
   patchDB_->open(NULL, dbFile.c_str(), "patch", DB_BTREE, dbFlags, 0);

   patchHistoryDB_ = new Db (&dbEnv_, 0);
   patchHistoryDB_->set_flags(DB_DUP);
   patchHistoryDB_->open(NULL, dbFile.c_str(), "patchHistory", DB_BTREE, dbFlags, 0);

#ifdef ED
   cohortDB_ = new Db (&dbEnv_, 0);
   cohortDB_->set_flags(DB_DUP);
   cohortDB_->open(NULL, dbFile.c_str(), "cohort", DB_BTREE, dbFlags, 0);
#endif
}

////////////////////////////////////////////////////////////////////////////////
//! ~Restart
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
Restart::~Restart () {
   if (dbTxn_ != NULL) {
      dbTxn_->abort();
   }
#ifdef ED
   cohortDB_->close(0);
#endif
   patchHistoryDB_->close(0);
   patchDB_->close(0);
   siteDB_->close(0);
   
   dbEnv_.close(0);
}


////////////////////////////////////////////////////////////////////////////////
//! storeStates
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Restart::storeStates (site* first_site, int year) {
   u_int32_t c;

   if (useTransactions_) {
      dbEnv_.txn_begin(NULL, &dbTxn_, 0);
   }
   try {
#ifdef ED
      cohortDB_->truncate(dbTxn_, &c, 0);
#endif
      patchHistoryDB_->truncate(dbTxn_, &c, 0);
      patchDB_->truncate(dbTxn_, &c, 0);
      patchSequence_ = 0;
      siteDB_->truncate(dbTxn_, &c, 0);
      siteSequence_ = 0;
      for(site* s=first_site; s!=NULL; s=s->next_site) {
         storeState(*s, year);
      }
      if (useTransactions_) {
         dbTxn_->commit(0);
      }
      dbTxn_ = NULL;
   } catch (DbException &e) {
      cerr << "Error in Restart DB transaction: " << e.what() << endl;
      if (useTransactions_) {
         dbTxn_->abort();
      }
      exit (1);
   }
   siteDB_->sync(0);
}


////////////////////////////////////////////////////////////////////////////////
//! storeState
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Restart::storeState (site& s, int year) {

   // TODO: make a real sequence for site id
   SiteRestart sr;
   sr.id_ = siteSequence_++;
   sr.year_ = year;

   Dbt sk (s.sdata->name_, (u_int32_t)strlen(s.sdata->name_)+1);
   Dbt sd (&sr, sizeof(SiteRestart));
   if (siteDB_->put(dbTxn_, &sk, &sd, 0) != 0) {
      // TODO: should this rollback and try again?
      cerr << "Failed to insert site restart" << endl;
      exit (1);
   }

   for (int lu=0; lu<N_LANDUSE_TYPES; lu++) {
      for (patch* p=s.youngest_patch[lu]; p!=NULL; p=p->older) {
         PatchRestart pr;
         pr.id_ = patchSequence_++;
         pr.landUse_ = lu;
         pr.disturbanceTrack_ = p->track;
         pr.age_ = p->age;
         pr.area_ = p->area;
         pr.fastSoilCarbon_ = p->fast_soil_C;
         pr.structuralSoilCarbon_ = p->structural_soil_C;
#ifdef ED
         pr.slowSoilCarbon_ = p->slow_soil_C;
         pr.passiveSoilCarbon_ = p->passive_soil_C;
         pr.structuralSoilLignin_ = p->structural_soil_L;
         pr.fastSoilNitrogen_ = p->fast_soil_N;
         pr.mineralizedSoilNitrogen_ = p->mineralized_soil_N;
         pr.water_ = p->water;
#elif defined MIAMI_LU
         pr.totalBiomass_ = p->total_biomass;
#endif
         Dbt pk (&sr.id_, sizeof(sr.id_));
         Dbt pd (&pr, sizeof(PatchRestart));
         if (patchDB_->put(dbTxn_, &pk, &pd, 0) != 0) {
            // TODO: should this rollback and try again?
            cerr << "Failed to insert patch restart" << endl;
            exit (1);
         }
         if (lu == LU_SCND) {
            for (int hYear=0; hYear<=year; hYear++) {
               PatchHistoryRestart phr;
               phr.year_ = hYear;
               phr.value_ = p->phistory[hYear];
               Dbt phk (&pr.id_, sizeof(pr.id_));
               Dbt phd (&phr, sizeof(PatchHistoryRestart));
               if (patchHistoryDB_->put(dbTxn_, &phk, &phd, 0) != 0) {
                  // TODO: should this rollback and try again?
                  cerr << "Failed to insert patch history restart" << endl;
                  exit (1);
               }
            }
         }
#if ED
         for (cohort* c=p->shortest; c!=NULL; c=c->taller) {
            CohortRestart cr;
            cr.pft_ = c->species;
            cr.dbh_ = c->dbh;
            cr.height_ = c->hite;
            cr.nIndivs_ = c->nindivs / p->area;
            cr.bDead_ = c->bdead;
            cr.bAlive_ = c->balive;
            Dbt ck (&pr.id_, sizeof(pr.id_));
            Dbt cd (&cr, sizeof(CohortRestart));
            if (cohortDB_->put(dbTxn_, &ck, &cd, 0) != 0) {
               // TODO: should this rollback and try again?
               cerr << "Failed to insert cohort restart" << endl;
               exit (1);
            }
         }
#endif // ED
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
//! readPatchDistribution
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Restart::readPatchDistribution (site* s, UserData& data) {
   // TODO: probably more efficient to select year once elsewhere. 

   SiteRestart sr;
   Dbt sk (s->sdata->name_, (u_int32_t)strlen(s->sdata->name_)+1);
   Dbt sd;
   sd.set_data(&sr);
   sd.set_ulen(sizeof(SiteRestart));
   sd.set_flags(DB_DBT_USERMEM);

   // TODO: check for failures
   if (siteDB_->get(NULL, &sk, &sd, 0) != 0) {
      cerr << "failed to find site record for " << s->sdata->name_ << endl;
   } 

   int last_lu = -1;
   patch* lastPatch = NULL;

   Dbc *patchCursor;
   patchDB_->cursor (NULL, &patchCursor, 0);

   Dbt pk(&sr.id_, sizeof(sr.id_));
   PatchRestart pr;
   Dbt pd;
   pd.set_data(&pr);
   pd.set_ulen (sizeof(PatchRestart));
   pd.set_flags (DB_DBT_USERMEM);

   int c = 0;
   int rv = patchCursor->get (&pk, &pd, DB_SET);
   while (rv != DB_NOTFOUND) {
      patch* newp = NULL;
#ifdef ED
      create_patch(&s, &newp, pr.landUse_, pr.disturbanceTrack_, 
                   pr.age_, pr.area_, pr.water_, 
                   pr.fastSoilCarbon_, pr.structuralSoilCarbon_, 
                   pr.structuralSoilLignin_, pr.slowSoilCarbon_, 
                   pr.passiveSoilCarbon_, pr.mineralizedSoilNitrogen_, 
                   pr.fastSoilNitrogen_, &data);
      readCohortDistribution(newp, pr.id_, data);
#elif defined MIAMI_LU
      create_patch(&s, &newp, pr.landUse_, pr.disturbanceTrack_, 
                   pr.age_, pr.area_,                       
                   pr.fastSoilCarbon_, pr.structuralSoilCarbon_, 
                   pr.totalBiomass_, &data);
#endif
      int lu = pr.landUse_;
      if (lu == LU_SCND) {
         readPatchHistory (*newp, pr.id_);
      }
      if (lu != last_lu) {
         newp->younger = NULL; 
         newp->older = NULL;
         s->youngest_patch[lu] = newp;
         s->oldest_patch[lu] = newp;
         lastPatch = newp;
         last_lu = lu;
      } else {
         newp->younger = lastPatch;
         newp->older = NULL;
         lastPatch->older = newp;
         lastPatch = newp;
         s->oldest_patch[lu] = newp;
      }
      c++;
      rv = patchCursor->get (&pk, &pd, DB_NEXT_DUP);
   } 
   patchCursor->close();

   //cout << "num patches: " << c << endl;
   if (c == 0) {
      cerr << "FAILED TO FIND ANY PATCHES! " << s->sdata->name_ << endl;
   }
   if (last_lu > LU_NTRL) {
      data.start_time = (int) floor(sr.year_) * N_CLIMATE + 1;
   } else {
      data.start_time = 0;
   }
//    //test_multi_restart
//   data.start_time = (int) floor(sr.year_) * N_CLIMATE + 1;
   
}

////////////////////////////////////////////////////////////////////////////////
//! readPatchHistory
//! @TODO: use bdb sequency data type for IDs
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Restart::readPatchHistory (patch& p, unsigned long patchID) {

   Dbt phk(&patchID, sizeof(patchID));
   PatchHistoryRestart phr;
   Dbt phd;
   phd.set_data(&phr);
   phd.set_ulen (sizeof(PatchHistoryRestart));
   phd.set_flags (DB_DBT_USERMEM);

   Dbc *patchHistoryCursor;
   patchHistoryDB_->cursor (NULL, &patchHistoryCursor, 0);

   int rv = patchHistoryCursor->get (&phk, &phd, DB_SET);
   while (rv != DB_NOTFOUND) {
      p.phistory[phr.year_] = phr.value_;
      rv = patchHistoryCursor->get (&phk, &phd, DB_NEXT_DUP);
   }
   patchHistoryCursor->close();
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! readCohortDistribution
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Restart::readCohortDistribution ( patch* p, unsigned long patchID,
                                       UserData& data ) {
   p->tallest  = NULL;
   p->shortest = NULL;

   Dbt ck(&patchID, sizeof(patchID));
   CohortRestart cr;
   Dbt cd;
   cd.set_data(&cr);
   cd.set_ulen (sizeof(CohortRestart));
   cd.set_flags (DB_DBT_USERMEM);

   Dbc *cohortCursor;
   cohortDB_->cursor (NULL, &cohortCursor, 0);

   int rv = cohortCursor->get (&ck, &cd, DB_SET);
   while (rv != DB_NOTFOUND) {
      create_cohort(cr.pft_, cr.nIndivs_ * p->area, cr.height_, 
                    cr.dbh_, cr.bAlive_, cr.bDead_, &p, &data);
      rv = cohortCursor->get (&ck, &cd, DB_NEXT_DUP);
   }
   cohortCursor->close();
}
#endif // ED



/******************************************************************************/
// Old style restart functions

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
//! print_system_state
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_system_state (unsigned int time, site** siteptr,
                         UserData* data) {

   site* cs = *siteptr;
  
   char basename[STR_LEN];
   sprintf(basename, "%s/RESTART/%s.%s", 
           data->outdir, data->expname, cs->sdata->name_);   

   /* patch distribution file */
   char pfilename[STR_LEN];
   FILE *pfile;
   sprintf(pfilename, "%s.pss", basename);
   pfile = fopen(pfilename,"w");

#if defined ED
//   fprintf(pfile,"time patch trk age area water fsc stsc stsl ssc psc msn fsn lu ");
    //test_restart
    // above line does not output product pools
#if LANDUSE
    fprintf(pfile,"time patch trk age area water fsc stsc stsl ssc psc msn fsn lu 1yr_pool 10yr_pool 100yr_pool ");
    
    //test_crop, should remove the code block below
    for(size_t spp=0;spp<NSPECIES;spp++)
    {
        fprintf(pfile,"repro_spp%d ",spp);
    }
#else
    fprintf(pfile,"time patch trk age area water fsc stsc stsl ssc psc msn fsn lu ");
#endif
    
#elif defined MIAMI_LU
   fprintf(pfile,"time patch trk age area fsc stsc lu tb ");
#endif

#if LANDUSE
   for (size_t i=0; i<=data->year; i++) {
      fprintf(pfile, "%2ld ",i);
   }
#endif
   fprintf(pfile,"\n");

#ifdef ED
   /* cohort distribution file */
   char cfilename[STR_LEN];
   sprintf(cfilename, "%s.css", basename);
   FILE* cfile = fopen(cfilename, "w");
//   fprintf(cfile, "time patch cohort dbh hite spp nindivs bdead balive \n");
    //test_mor
    fprintf(cfile, "time patch cohort dbh hite spp nindivs bdead balive bl br blv bsw\n");
#endif /* ED */

   for (int lu=0; lu<N_LANDUSE_TYPES; lu++) {
      patch* cp = cs->oldest_patch[lu];
      while (cp != NULL) {
#if defined ED
         /* print patch distribution */
#if LANDUSE
          fprintf(pfile, "%6.3f %p %d %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %1d %8.8f %8.8f %8.8f ",
                  time * TIMESTEP,
                  cp,
                  cp->track,
                  cp->age,
                  cp->area,
                  cp->water,
                  cp->fast_soil_C,
                  cp->structural_soil_C,
                  cp->structural_soil_L,
                  cp->slow_soil_C,
                  cp->passive_soil_C,
                  cp->mineralized_soil_N,
                  cp->fast_soil_N,
                  cp->landuse,
                  cp->yr1_decay_product_pool,
                  cp->yr10_decay_product_pool,
                  cp->yr100_decay_product_pool);
         
          for(size_t spp=0;spp<NSPECIES;spp++)
          {
              fprintf(pfile,"%4.4f ",cp->repro[spp]);
          }
#else
          fprintf(pfile, "%6.3f %p %d %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f %1d ",
                  time * TIMESTEP,
                  cp,
                  cp->track,
                  cp->age,
                  cp->area,
                  cp->water,
                  cp->fast_soil_C,
                  cp->structural_soil_C,
                  cp->structural_soil_L,
                  cp->slow_soil_C,
                  cp->passive_soil_C,
                  cp->mineralized_soil_N,
                  cp->fast_soil_N,
                  cp->landuse);
#endif
          
          
#if 0
         if (lu == LU_SCND)
            for (int i=0; i<=data->year; i++) 
               fprintf(pfile, "%8.4f ", *(cp->phistory + i));
         fprintf(pfile,"\n");
#endif

         /* print cohort distribution */
         cohort* cc = (cp->tallest);
         while (cc != NULL){
             fprintf(cfile, "%6.3f %p %p %8.8f %8.8f %2d %8.20f %8.8f %8.8f %8.8f %8.8f %8.8f %8.8f ",
                     time * TIMESTEP,
                     cp,
                     cc,
                     cc->dbh,
                     cc->hite,
                     cc->species,
                     cc->nindivs / cp->area,
                     cc->bdead,
                     cc->balive,
                     cc->bl,
                     cc->br,
                     cc->blv,
                     cc->bsw);
             
             for(size_t imon=0;imon<N_CLIMATE;imon++)
             {
                 fprintf(cfile, "%8.8f %8.8f ",cc->cb[imon], cc->cb_toc[imon]);
             }
             fprintf(cfile, "\n");
             
            cc = cc->shorter;
         } /* end loop over cohorts */
#elif defined MIAMI_LU
         fprintf(pfile, "%f %p %d %f %f %f %f %1d %f\n",
                 time*TIMESTEP, cp, cp->track, cp->age, cp->area, 
                 cp->fast_soil_C, cp->structural_soil_C, cp->landuse,
                 cp->total_biomass);
#endif
         cp = cp->younger;
      } /* end loop over patches */
   } /* end loop over landuse types */

   fclose(pfile);
#ifdef ED
   fclose(cfile);
#endif
}

////////////////////////////////////////////////////////////////////////////////
//! print_system_states
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void print_system_states (unsigned int t, site* firstSite, UserData* data) {

   for (site* s=firstSite; s!=NULL; s=s->next_site) {
      print_system_state(t, &s, data);
   }
}

////////////////////////////////////////////////////////////////////////////////
//! read_patch_distribution
//! read in initial patch distribution from file
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
/******************************************************************************/
/***        read in initial patch distribution from file                    ***/
/******************************************************************************/
void read_patch_distribution (site** siteptr, UserData* data) {
   double old_time, start_time = 0;
   char old_address[20];
   int track, lu;
#if defined ED
   double stsl, ssc, psc, msn, fsn;
#elif defined MIAMI_LU
   double tb;
#endif
   double age, area, water, fsc, stsc;
   char dummy[100000];
    
    //test_restart
#if LANDUSE & RESTART_FROM_LU
    double yr1_pool, yr10_pool, yr100_pool;
    
    //test_crop
#if RESTART_FROM_REPRO
    double repro[NSPECIES];
#endif // RESTART_FROM_REPRO
    
#endif

   site* cs = *siteptr; /* assign pointer to site */

   char filename[STR_LEN], pfilename[STR_LEN];
   strcpy(filename,data->output_base_path);
   strcat(filename,data->old_restart_exp_name);
   strcat(filename,"/RESTART/");
   strcat(filename,data->old_restart_exp_name); 
   strcat(filename,".");
   strcat(filename,cs->sdata->name_);
   strcpy(pfilename, filename);
   strcat(pfilename,".pss");

   /* read data from patch file into newpatch structure */
   printf("Reading in patches for %s from %s \n",cs->sdata->name_,pfilename);
   FILE *infile = NULL;
   if(!(infile=fopen(pfilename,"r"))){
      printf("rpd: Can't open file: %p %s \n",infile,pfilename);
      return;
   }

   /*function to read over header line*/
   fgets(dummy,100000,infile);

   patch* cp = NULL;
   int last_lu = -1;
   /* read in data elements */
   int count = 0;
    
#if defined ED
    printf("ED mode\n");
//   while( fscanf( infile, "%lf%s%d%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d",
//                  &old_time, old_address, &track, &age, &area, &water,
//                  &fsc, &stsc, &stsl, &ssc, &psc, &msn, &fsn, &lu ) != EOF ){
    
    //test_restart
#if LANDUSE & RESTART_FROM_LU
    while( fscanf( infile, "%lf%s%d%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d%lf%lf%lf",
                  &old_time, old_address, &track, &age, &area, &water,
                  &fsc, &stsc, &stsl, &ssc, &psc, &msn, &fsn, &lu, &yr1_pool, &yr10_pool, &yr100_pool) != EOF ){
    
//test_crop, shoule remove the blow if block
#if RESTART_FROM_REPRO
        for(size_t spp=0;spp<NSPECIES;spp++)
        {
            fscanf( infile, "%lf",&repro[spp]);
        }
#endif // RESTART_FROM_REPRO
        
#else
   while( fscanf( infile, "%lf%s%d%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d",
                  &old_time, old_address, &track, &age, &area, &water,
                  &fsc, &stsc, &stsl, &ssc, &psc, &msn, &fsn, &lu ) != EOF ){
#endif
    
        
      if (start_time < old_time)
         start_time = old_time;
#elif defined MIAMI_LU
   while ( fscanf(infile, "%lf%s%d%lf%lf%lf%lf%d%lf",
                  &old_time, old_address, &track, &age, 
                  &area, &fsc, &stsc, &lu, &tb) != EOF ) {
#endif
      /*numerical error traps: correct for reading numerical zeros*/
      if (area  < 0.0001) area  = 0.0001; 
      if (water < 0.0001) water = 0.0001; 
      if (fsc   < 0.0001) fsc   = 0.0001; 
      if (stsc  < 0.0001) stsc  = 0.0001;
       
#if LANDUSE & RESTART_FROM_LU
       //test_restart
       if (yr1_pool  < 0.0001) yr1_pool  = 0.0001;
       if (yr10_pool  < 0.0001) yr10_pool  = 0.0001;
       if (yr100_pool  < 0.0001) yr100_pool  = 0.0001;
#endif
       
#if defined ED
      if (stsl  < 0.0001) stsl  = 0.0001; 
      if (ssc   < 0.0001) ssc   = 0.0001; 
      if (psc   < 0.0001) psc   = 0.0001; 
      if (msn   < 0.0001) msn   = 0.0001; 
      if (fsn   < 0.0001) fsn   = 0.0001; 
#elif defined MIAMI_LU
      if (tb   < 0.0001) tb   = 0.0001; 
#endif
    
      printf("Initializing patch %d \n", count);
      patch* newp = NULL;
#if defined ED
      
#if LANDUSE & RESTART_FROM_LU
       create_patch_LU( &cs, &newp, lu, track, age, area, water,fsc, stsc, stsl, ssc, psc, msn, fsn, yr1_pool, yr10_pool, yr100_pool, data);

#if RESTART_FROM_REPRO
       for(size_t spp=0;spp<NSPECIES;spp++)
       {
           newp->repro[spp] = repro[spp];
       }
#endif // RESTART_FROM_REPRO
       
#else
       create_patch( &cs, &newp, lu, track, age, area, water,fsc, stsc, stsl, ssc, psc, msn, fsn, data);
#endif
       
#if 0 //switch to 1 if line 431 is on indicating history are output
      if (lu == LU_SCND) {
         double tmp = 0.0;
         for(size_t i=0; i<start_time+1; i++){
            fscanf(infile,"%lf",&tmp);
            *(newp->phistory+i)=tmp;
            /*printf("tmp %f\t patch %f\n",tmp,*(newp->phistory+i));*/
         }
      }
#endif
       
#elif defined MIAMI_LU
      create_patch(&cs, &newp, lu, track, age, area, fsc, stsc, tb, data);
#endif    
      fscanf(infile,"\n");

      if (lu != last_lu) {
         newp->younger = NULL; 
         newp->older = NULL;
         cs->youngest_patch[lu] = newp;
         cs->oldest_patch[lu] = newp;
         cp = cs->youngest_patch[lu];
      } else {
         newp->younger = NULL;
         newp->older = cp;
         (cp->younger) = newp;
         cp = cp->younger;
         cs->youngest_patch[lu] = newp;
      }
#ifdef ED
      read_cohort_distribution(filename, &cs,&newp,old_address,data);
#endif

      count++;
      last_lu = lu;
   } /* end while !EOF */
  
   fclose(infile);

#if LANDUSE && RESTART_FROM_LU
    data->start_time = (int) floor(start_time) * N_CLIMATE;
#else
    data->start_time = 0;
#endif
}


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! read_cohort_distribution
//! read in initial cohort distribution from file
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void read_cohort_distribution( char* filename, site** siteptr, 
                               patch** patchptr, char* paddress1, 
                               UserData* data ){

   FILE *infile;
   char cfilename[STR_LEN];
   double dbh, hite, nindivs, bdead, balive;
    //test_mor
    double bl,br,blv,bsw;
    
   int  spp;
   char paddress2[20];
   double fdum;
   char cdum[20];
   char dummy[10000];
   int count=0;

   patch* cp = *patchptr; /* assign pointer to patch */

   cp->tallest  = NULL;
   cp->shortest = NULL;

   strcpy(cfilename, filename);
   strcat(cfilename, ".css");
   printf("rcd: reading cohorts for patch %s", paddress1);
   if ( ! (infile = fopen(cfilename, "r")) ) {
      printf("rcd: Can't open file: %p %s \n", infile, cfilename);
      return;
   }
   
   double cb[N_CLIMATE];
   double cb_toc[N_CLIMATE];

   /*function to read over header line*/
   fgets(dummy,10000,infile);

//   /* read in  data elements */
    while( fscanf( infile, "%lf%s%s%lf%lf%d%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf\n",
                  &fdum, paddress2, cdum, &dbh, &hite,
                  &spp, &nindivs, &bdead, &balive,&bl,&br,&blv,&bsw,&cb[0],&cb_toc[0],&cb[1],&cb_toc[1],&cb[2],&cb_toc[2],&cb[3],&cb_toc[3],&cb[4],&cb_toc[4],&cb[5],&cb_toc[5],&cb[6],&cb_toc[6],&cb[7],&cb_toc[7],&cb[8],&cb_toc[8],&cb[9],&cb_toc[9],&cb[10],&cb_toc[10],&cb[11],&cb_toc[11]) != EOF){
        if(dbh     < 0.0001) dbh     = 0.0001;
        if(hite    < data->hgtmin) hite    = data->hgtmin;

        /* compare patch addresses if match create cohort with read-in parameters */
        if ( strcmp(paddress1, paddress2) == 0 )
        {
            create_cohort_test_mor(spp, nindivs * cp->area, hite, dbh, balive, bdead,bl,br,blv,bsw, &cp, cb, cb_toc, data);
            count++;
        }  /* end if */
    } /* end while */

   printf("rcd: cohorts read= %d \n", count);
   fclose(infile);
}
#endif // ED
