#ifndef EDM_READCONFIGURATION_H
#define  EDM_READCONFIGURATION_H
#include <cstring>

#include "libconfig.h++"
#include "edmodels.h"


void initialize_model_params (UserData* data);
bool fileExists(const std::string& filename);
char* get_section_name (const char* section, const char* value);
int read_config_file (const char* cfgFile, UserData* data);
void free_user_data (UserData* data);

template<typename T>
T get_val(UserData* data, const char* config_filename, const char* section, const char* value);

template<typename T>
void get_list(UserData* data, const char* config_filename, const char* section, const char* value, std::vector<T>  & v);

template<typename T>
void get_list(UserData* data, const char* config_filename, const char* section, const char* value, std::vector<T>  & v) {
   int i = 0;
   char* section_name = get_section_name(section,value);
   
   if(!strcmp(config_filename, PFTS)) {
      try {
         libconfig::Setting &setting = data->pfts_cfg_alternate->lookup(section_name);
       
         if(setting.getLength()) {           
            for(i = 0; i < setting.getLength(); i++) {
               v.push_back(setting[i]);
            }
         } 
         else {
            printf("Parameter: %s is empty\n", value);
         } 
      } catch(const libconfig::SettingNotFoundException &snfex) {
         try {
            libconfig::Setting &default_setting = data->pfts_cfg_default->lookup(section_name);
            if(default_setting.getLength()) {           
               for(i = 0; i < default_setting.getLength(); i++) {
                  v.push_back(default_setting[i]);
               }
            }
            else {
               printf("Parameter: %s is empty\n", value);
            }
         } catch(const libconfig::SettingNotFoundException &snfex) {
            printf("Parameter: %s not present in specified section\n", value);
         }            
      }
   }
   else if(!strcmp(config_filename, PARAMS)) {
      try {
         libconfig::Setting &setting = data->params_cfg_alternate->lookup(section_name);
       
         if(setting.getLength()) {           
            for(i = 0; i < setting.getLength(); i++) {
               v.push_back(setting[i]);
            }
         } 
      } catch(const libconfig::SettingNotFoundException &snfex) {
         try {
            libconfig::Setting &default_setting = data->params_cfg_default->lookup(section_name);

            if(default_setting.getLength()) {           
               for(i = 0; i < default_setting.getLength(); i++) {
                  v.push_back(default_setting[i]);
               }
            } else {
               printf("Parameter: %s is empty\n", value);
            }
         }
         catch(const libconfig::SettingNotFoundException &snfex) {
            printf("Parameter: %s not present in specified section\n", value);
         }         
        }        
   } else if(!strcmp(config_filename, MODEL_IO)) {
      try {
         libconfig::Setting &setting = data->io_cfg_alternate->lookup(section_name);

         if(setting.getLength()) {           
            for(i = 0; i < setting.getLength(); i++) {
               v.push_back(setting[i]);
            }
         }
      } catch(const libconfig::SettingNotFoundException &snfex) {
         try {
            libconfig::Setting &default_setting = data->io_cfg_default->lookup(section_name);

            if(default_setting.getLength()) {           
               for(i = 0; i < default_setting.getLength(); i++) {
                  v.push_back(default_setting[i]);
               }
            } else {
               printf("Parameter: %s is empty\n", value);
            }
         }
         catch(const libconfig::SettingNotFoundException &snfex) {
            printf("Parameter: %s not present in specified section\n", value);
         }
        }         
   } else {
      printf("Unrecognized section %s in configuration file\n", config_filename);
   }
       
   free(section_name);
}


template<typename T>
T get_val(UserData* data, const char* config_filename, const char* section, const char* value) {
   T val;
   int error = 0;
   char* section_name;
   section_name = get_section_name(section,value);

   if(!strcmp(config_filename, "params")) {
      if(!data->params_cfg_alternate->lookupValue(section_name, val)) {
         if(!data->params_cfg_default->lookupValue(section_name, val)) {
            error = 1;
         }
      }
   } else if(!strcmp(config_filename, "io")){
      if(!data->io_cfg_alternate->lookupValue(section_name, val)) {
         if(!data->io_cfg_default->lookupValue(section_name, val)) {
            error = 1;
         }
      }
   } else if(!strcmp(config_filename, "pfts")) {
      //cout << section_name << " " << val << "\n";
      /*if(!data->pfts_cfg_alternate->lookupValue(section_name, val)) {  
            
        if(!data->pfts_cfg_default->lookupValue(section_name, val)) {
        error = 1;
        }
        }*/
         
      data->pfts_cfg_alternate->lookupValue(section_name, val);
   } else {
      printf("Unrecognized section %s in configuration file\n", config_filename);
   }
    
   if(error) {
      printf("Parameter: %s not present in configuration file\n", value);
      exit(-1);
   }        
   free(section_name);
   return val;
}

#endif   // EDM_READCONFIGURATION_H 

