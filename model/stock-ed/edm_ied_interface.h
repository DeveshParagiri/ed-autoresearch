#ifndef EDM_IED_INTERFACE_H_
#define EDM_IED_INTERFACE_H_
#include "edmodels.h"

// TODO: where are we putting these constants?
#define N_GCAM_REG 14
#define N_GCAM_AEZ 18
#define N_GCAM_CROP 2
// TODO: this needs to go to the configuration file
#define REGAEZFILE "/lustre/data/fisk/region_aez.nc"

// glm's new_data structure
struct new_data;


class EDMiEDInterface {

 public:

   EDMiEDInterface ();

   void initialize ( char* aExpName, new_data* aGLMData, int aStartYear );
   void doYear ( int aYear );
   void finalize ( int aYear );
   void saveRestartState ( int aYear );
   double** getGriddedPotentialBiomass ( );
   double getRegAEZDiscountedBiomassDensity ( int aReg, int aAEZ, int aCrop );
   double getRegAEZSoilCarbonDensity ( int aReg, int aAEZ, int aCrop );
   double getGlobalNetFlux ( );
   void backupWorld ( );
   void restoreWorld ( );

 private:

   void readRegionAEZFile ( );
   void readDiscountedCarbonFile ( );
   site* copyWorld (site* world);

   UserData* edmControl;
   double** potentialBiomass;
   int** gcamRegMap;
   int** gcamAEZMap;
   double regaezDiscountedBiomassDensity[N_GCAM_REG][N_GCAM_AEZ][N_GCAM_CROP];
   double regaezSoilCarbonDensity[N_GCAM_REG][N_GCAM_AEZ][N_GCAM_CROP];
   size_t lastRecNo;
};


#endif

