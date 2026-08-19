#ifndef EDM_RESTART_H_
#define EDM_RESTART_H_

#include <string>
#include "db_cxx.h"

class UserData;
class site;
class patch;

struct SiteRestart {
   unsigned long id_;
   int year_;
};

struct PatchRestart {
   unsigned long id_;
   int landUse_;
   int disturbanceTrack_;
   double age_;
   double area_;
   double fastSoilCarbon_;
   double structuralSoilCarbon_;
#ifdef ED
   double slowSoilCarbon_;
   double passiveSoilCarbon_;
   double structuralSoilLignin_;
   double fastSoilNitrogen_;
   double mineralizedSoilNitrogen_;
   double water_;
#elif defined MIAMI_LU
   double totalBiomass_;
#endif
};


struct PatchHistoryRestart {
   int year_;
   double value_;
};


struct CohortRestart {
   int pft_;
   double nIndivs_;
   double height_;
   double dbh_;
   double bAlive_;
   double bDead_;
};


class Restart {

 public:
   Restart (std::string dbName);
   ~Restart ();

   void storeStates (site* first_site, int year);
   void readPatchDistribution (site* s, UserData& data);

 protected:
   void storeState (site& s, int year);
   void readPatchHistory (patch& p, unsigned long patchID);
   void readCohortDistribution (patch* p, unsigned long patchID, UserData& data);

   static bool useTransactions_;

   DbEnv dbEnv_;
   DbTxn* dbTxn_;
   Db* siteDB_;
   unsigned long siteSequence_;
   Db* patchDB_;
   unsigned long patchSequence_;
   Db* patchHistoryDB_;
   Db* cohortDB_;
};


// functions for old style restarts
void print_system_states(unsigned int t, site *firstSite, UserData *data);

void read_patch_distribution(site** new_site, UserData* data);

void read_cohort_distribution(char* filename, site** siteptr, patch** patchptr,
                              char* paddress1, UserData* data);


#endif // EDM_RESTART_H_ 
