#include <iostream>
#include <netcdfcpp.h>
#include <string>

#include "edmodels.h"
#include "site.h"
#include "read_site_data.h"

#include "outputter.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//! Outputter
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
Outputter::Outputter (UserData* data) :
      recNo(0),
      data(data)
{
   string filename (data->base_filename);
   filename += ".region.nc";

   NcError err(NcError::verbose_nonfatal);
   outputFile = new NcFile(filename.c_str(), NcFile::Replace);

   // TODO: get rid of these exception throws
   if (!outputFile->is_valid()) throw "Failed to create file";
   NcDim *timeDim, *latDim, *lonDim;
   if (!(timeDim = outputFile->add_dim("time")))
      throw "Failed to create time dim";
   if (!(latDim = outputFile->add_dim("lat", data->n_lat)))
      throw "Failed to create lat dim";
   if (!(lonDim = outputFile->add_dim("lon", data->n_lon)))
      throw "Failed to create lon dim";
 
   // TODO: need to deal with time variable
   NcVar *latVar, *lonVar;
   if (!(latVar = outputFile->add_var("lat", ncFloat, latDim)))
      throw "Failed to create lat var";
   latVar->add_att("units", "degrees_north");
   if (!(lonVar = outputFile->add_var("lon", ncFloat, lonDim)))
       throw "Failed to create lon var";
   lonVar->add_att("units", "degrees_east");
   
   latVar->put(&data->lats[0], data->n_lat);
   lonVar->put(&data->lons[0], data->n_lon);
}

////////////////////////////////////////////////////////////////////////////////
//! getOrCreateVariable
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
template <class T>
NcVar* Outputter::getOrCreateVariable (Var<T> *v, bool isRec) {
   NcError err(NcError::silent_nonfatal);
      
   NcVar *var = outputFile->get_var(v->name.c_str());
   if (var == NULL || !var->is_valid()) {
      if (isRec)
         var = outputFile->add_var(v->name.c_str(), v->dtype, 
                                   outputFile->get_dim("time"),
                                   outputFile->get_dim("lat"),
                                   outputFile->get_dim("lon"));
      else
         var = outputFile->add_var(v->name.c_str(), v->dtype, 
                                   outputFile->get_dim("lat"),
                                   outputFile->get_dim("lon"));
      if (!v->units.empty())
         var->add_att("units", v->units.c_str());
      var->add_att("missing_value", v->fill);
      var->add_att("_fillValue", v->fill);
   } 
   return var;
}

////////////////////////////////////////////////////////////////////////////////
//! getOrCreateLUVariable
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
template <class T>
NcVar* Outputter::getOrCreateLUVariable (LUVar<T> *v, size_t luType, bool isRec) {
   string name = v->name + "_" + this->luShortName(luType);

   NcError err(NcError::silent_nonfatal);

   NcVar *var = outputFile->get_var(name.c_str());
   if (var == NULL || !var->is_valid()) {
      if (isRec)
         var = outputFile->add_var(name.c_str(), v->dtype, 
                                   outputFile->get_dim("time"),
                                   outputFile->get_dim("lat"),
                                   outputFile->get_dim("lon"));
      else
         var = outputFile->add_var(name.c_str(), v->dtype, 
                                   outputFile->get_dim("lat"),
                                   outputFile->get_dim("lon"));

      if (!v->units.empty())
         var->add_att("units", v->units.c_str());
      var->add_att("missing_value", v->fill);
      var->add_att("_fillValue", v->fill);
   } 

   return var;
}

////////////////////////////////////////////////////////////////////////////////
//! outputAll
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void Outputter::outputAll (site* firstsite) {
   vector<VarBase *>::iterator i;

   for (i=registeredVars.begin(); i!=registeredVars.end(); i++) {
      if ((*i)->dtype == ncFloat)
         outputRec<float> (dynamic_cast<Var<float>* >((VarBase*)(*i)), firstsite);
      else if ((*i)->dtype == ncInt)
         outputRec<int> (dynamic_cast<Var<int>* >((VarBase*)(*i)), firstsite);
   }

   for (i=registeredLUVars.begin(); i!=registeredLUVars.end(); i++) {
      if ((*i)->dtype == ncFloat)
         outputLURec<float> (dynamic_cast<LUVar<float>* >((VarBase*)(*i)), firstsite);
      else if ((*i)->dtype == ncInt)
         outputLURec<int> (dynamic_cast<LUVar<int>* >((VarBase*)(*i)), firstsite);
   }
   recNo++;
}

////////////////////////////////////////////////////////////////////////////////
//! outputRec
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
template <class T>
   void Outputter::outputRec (Var<T> *v, site* firstsite, bool isRec) {
   T d[data->n_lat][data->n_lon];

   NcVar *var = getOrCreateVariable(v, isRec);

   for (size_t i=0; i<data->n_lat; i++)
      for (size_t j=0; j<data->n_lon; j++) 
         d[i][j] = v->fill;

   site* cs = firstsite;
   while (cs != NULL) {
      if (!cs->skip_site) d[cs->sdata->y_][cs->sdata->x_] = (*(v->get))(cs);
      cs = cs->next_site;
   }

   if (isRec) {
      if (!var->put_rec(&d[0][0], recNo))
         cout << "could not output " << v->name << " " << recNo << endl;
   } else {
      if (!var->put(&d[0][0], data->n_lat, data->n_lon))
         cout << "count not output " << v->name << endl;
   }

   outputFile->sync();
}

////////////////////////////////////////////////////////////////////////////////
//! outputLURec
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
template <class T>
void Outputter::outputLURec (LUVar<T> *v, site* firstsite, bool isRec) {
   T d[N_LANDUSE_TYPES][data->n_lat][data->n_lon];
   NcVar *var[N_LANDUSE_TYPES];

   for (size_t lu=0; lu<N_LANDUSE_TYPES; lu++) 
      var[lu] = getOrCreateLUVariable(v, lu, isRec);

   for (size_t lu=0; lu<N_LANDUSE_TYPES; lu++)
      for (size_t i=0; i<data->n_lat; i++)
         for (size_t j=0; j<data->n_lon; j++) 
            d[lu][i][j] = v->fill;

   site* cs = firstsite;
   while (cs != NULL) {
      if (!cs->skip_site) 
         for (size_t lu=0; lu<N_LANDUSE_TYPES; lu++)
            d[lu][cs->sdata->y_][cs->sdata->x_] = (*(v->get))(cs, lu);
      cs = cs->next_site;
   }

   if (isRec) {
      for (size_t lu=0; lu<N_LANDUSE_TYPES; lu++)
         if (!var[lu]->put_rec(&d[lu][0][0], recNo))
            cout << "could not output " << v->name << "_" << this->luShortName(lu) << " " << recNo << endl;
   } else {
      for (size_t lu=0; lu<N_LANDUSE_TYPES; lu++)
         if (!var[lu]->put(&d[lu][0][0], data->n_lat, data->n_lon))
            cout << "count not output " << v->name << "_" << this->luShortName(lu) << endl;
   }

   outputFile->sync();
}

////////////////////////////////////////////////////////////////////////////////
//! luShortName
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
string Outputter::luShortName (size_t luType) {
   string name;
   switch (luType) {
      case LU_NTRL : 
         name = "natr";
         break;
      case LU_SCND : 
         name = "scnd";
         break;
      case LU_CROP : 
         name = "crop";
         break;
      case LU_PAST : 
         name = "past";
         break;
   }
   return name;
}
