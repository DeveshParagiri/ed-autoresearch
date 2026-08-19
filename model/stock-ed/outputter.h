#ifndef EDM_OUTPUTTER_H_
#define EDM_OUTPUTTER_H_

#include <vector>
#include "netcdfcpp.h"

// Forward declarations
struct UserData;
struct site;

class Outputter {

   class VarBase {
    public:
      VarBase (std::string name, std::string units, NcType dtype)
         : name(name), units(units), dtype(dtype)
      {
      }
      virtual ~VarBase () {}

      std::string name;
      std::string units;
      NcType dtype;
   };

   template <class T> class Var : public VarBase {
    public: 
      Var (std::string name, T (*get)(site*), std::string units, T fill, NcType dtype) 
         : VarBase(name, units, dtype), get(get), fill(fill)
      {
      }
            
      T (*get)(site*);
      T fill;
   };

   template <class T> class LUVar : public VarBase {
    public: 
      LUVar (std::string name, T (*get)(site*, size_t luType), std::string units, T fill, NcType dtype) 
         : VarBase(name, units, dtype), get(get), fill(fill)
      {
      }
            
      T (*get)(site*, size_t luType);
      T fill;
   };

 public:
   
   Outputter (UserData* data);

   template <class T> void registerVar (const char *name, T (*get)(site*), 
                                        const char *units, T fill, NcType dtype) {
      registeredVars.push_back(new Var<T>(std::string(name), get, units, fill, dtype));
   }

   template <class T> void registerLUVar (const char *name, T (*get)(site*, size_t luType), 
                                          const char *units, T fill, NcType dtype) {
      registeredLUVars.push_back(new LUVar<T>(std::string(name), get, units, fill, dtype));
   }

   void outputAll (site* firstsite);

   template <class T> void outputRec (Var<T> *v, site* firstsite, bool isRec=true);
   template <class T> void outputLURec (LUVar<T> *v, site* firstsite, bool isRec=true);

   template <class T> void outputSingle (const char *name, T (*get)(site*), 
                                         const char *units, T fill, NcType dtype,
                                         site* firstsite) {
      Var<T> v(name, get, units, fill, dtype);
      outputRec(&v, firstsite, false);
   }

   std::string luShortName (size_t luType);
   size_t recNo;

 private:
   template <class T>
      NcVar* getOrCreateVariable (Var<T> *v, bool isRec=true);

   template <class T>
      NcVar* getOrCreateLUVariable (LUVar<T> *v, size_t luType, bool isRec=true);

   UserData *data;
   NcFile *outputFile;
   std::vector<VarBase *> registeredVars;
   std::vector<VarBase *> registeredLUVars;

};


#endif // EDM_OUTPUTTER_H_

