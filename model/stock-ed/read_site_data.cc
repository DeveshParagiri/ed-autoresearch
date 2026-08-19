#include <cstdio>
#include <cmath>
#include <cstring>
#include "netcdf.h"
#include <stddef.h>
#include <stdarg.h>

#include "edmodels.h"
#include "site.h"
#include "miami.h"
#if LANDUSE
#include "landuse.h"
#endif
#if USEMPI
#include "edmpi.h"
#endif
#include "fire.h"
#include "read_site_data.h"


size_t read_gridspec (UserData* data);
void read_sois (UserData* data);
////////////////////////////////////////////////////////////////////////////////
//! read_input_data_layers
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
size_t read_input_data_layers (UserData* data) {

   read_sois(data);

   size_t nPotentialSites = read_gridspec(data);
#if LANDUSE
#ifndef COUPLED
   read_initial_landuse_fractions(data);
#endif
   read_initial_landuse_c(data);
#endif

   printf("gfed bool= %d\n", data->fire_gfed);
   printf("fre_off bool= %d\n", data->fire_off);
   if(!data->fire_off && data->fire_gfed){
      printf("In read_input_data_layer(), fire_gfed bool is true... \n");
      load_GFED(data);
   }	
   
   return nPotentialSites;
}


////////////////////////////////////////////////////////////////////////////////
//! read_sois
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void read_sois (UserData* data) {
   soi *new_soi, *current_soi;
   int i;
   double lat, lon;
   char name[STR_LEN];
   char filename[STR_LEN];
   FILE *infile;

  
   strcpy(filename, "sois.txt");
   infile = fopen(filename,"r");

   printf("read_sois... %s\n", filename);

   if(fscanf(infile,"%*s %d\n",&(data->num_sois)) != 1){
      printf("rmi: Error reading from file: %p\n", infile);
      fclose(infile);
      return;
   } 

   printf("num_sois= %d\n", data->num_sois);

   if(data->num_sois > 0){
      data->first_soi=(soi *) malloc (sizeof(soi));
      fscanf(infile, "%s %lf %lf\n", name, &lat, &lon);
      data->first_soi->lat = lat;
      data->first_soi->lon = lon;
      strcpy(data->first_soi->name, name);
      data->first_soi->next_soi = NULL;

      if(data->num_sois > 1){
         current_soi=data->first_soi;
         for(i=1;i<data->num_sois;i++){
            fscanf(infile, "%s %lf %lf\n", name, &lat, &lon);
            new_soi=(soi*) malloc (sizeof(soi));
            new_soi->lat = lat;
            new_soi->lon = lon;
            strcpy(new_soi->name, name);
            new_soi->next_soi = NULL;
            current_soi->next_soi = new_soi;
            current_soi = new_soi;
         }
      }      
   }
   fclose(infile);
 
   current_soi = data->first_soi;
   while (current_soi != NULL) {
      printf("soi = %s lat= %lf lon= %lf\n",
             current_soi->name, current_soi->lat, current_soi->lon);
      current_soi = current_soi->next_soi;
   }

   printf("read_sois COMPLETE\n");
   return;
}


////////////////////////////////////////////////////////////////////////////////
//! is_soi
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
bool is_soi (double lat, double lon, UserData& data) { /* flag site of interest */

   double dlat = fabs(data.lats[0] - data.lats[1]);
   double dlon = fabs(data.lons[0] - data.lons[1]);
   double lat_min = lat - dlat/2.0;
   double lat_max = lat + dlat/2.0;
   double lon_min = lon - dlon/2.0;
   double lon_max = lon + dlon/2.0;


   bool flag = false;
   soi* cur_soi = data.first_soi;
   while (cur_soi != NULL) {
      if( (cur_soi->lat >= lat_min) && (cur_soi->lat < lat_max)
         && (cur_soi->lon >= lon_min) && (cur_soi->lon < lon_max) )
         flag = true;
      cur_soi = cur_soi->next_soi;
   }
   return flag;   
}


////////////////////////////////////////////////////////////////////////////////
//! get_grid_bounds
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void get_grid_bounds (int ncid, UserData* data) {
   int rv, latid, lonid, varid;
   double *tmp;
   size_t i, start, end, count;
   int in_region_flag;


   if ((rv = nc_inq_dimid(ncid, "lat", &latid)))
      NCERR("lat", rv);
   if ((rv = nc_inq_dimlen(ncid, latid, &count)))
      NCERR("lat", rv);
   tmp = (double *)malloc(count * sizeof(double));
   if ((rv = nc_inq_varid(ncid, "lat", &varid)))
      NCERR("lat", rv);
   if ((rv = nc_get_var_double(ncid, varid, &tmp[0])))
      NCERR("lat", rv);

   in_region_flag = 0;
   for (i=0; i<count; i++) {
      if ((!in_region_flag) && (tmp[i] <= data->latmax) && (tmp[i] >= data->latmin)) {
         start = i; 
         in_region_flag = 1;
      }
      if ((in_region_flag) && ((tmp[i] > data->latmax) || (tmp[i] < data->latmin))) {
         end = i; 
         break;
      } else if (i == count - 1) {
         end = i + 1;
      }
   }
   
   data->start_lat = start;
   data->n_lat = end - start;
   data->lats = (double *)malloc(data->n_lat * sizeof(double));
   for (i=start; i<end; i++) {
      data->lats[i-start] = tmp[i];
   }
   free(tmp);

   if ((rv = nc_inq_dimid(ncid, "lon", &lonid)))
      NCERR("lon", rv);
   if ((rv = nc_inq_dimlen(ncid, lonid, &count)))
      NCERR("lon", rv);
   tmp = (double *)malloc(count * sizeof(double));
   if ((rv = nc_inq_varid(ncid, "lon", &varid)))
      NCERR("lon", rv);
   if ((rv = nc_get_var_double(ncid, varid, &tmp[0])))
      NCERR("lon", rv);
   
   in_region_flag = 0;
   for (i=0; i<count; i++) {
      if ((!in_region_flag) && (tmp[i] <= data->lonmax) && (tmp[i] >= data->lonmin)) {
         start = i; 
         in_region_flag = 1;
      }
      if ((in_region_flag) && ((tmp[i] > data->lonmax) || (tmp[i] < data->lonmin))) {
         end = i; 
         break;
      } else if (i == count - 1) {
         end = i + 1;
      }
   }
   
   data->start_lon = start;
   data->n_lon = end - start;
   data->lons = (double *)malloc(data->n_lon * sizeof(double));
   for (i=start; i<end; i++) {
      data->lons[i-start] = tmp[i];
   }
   free(tmp);
}

////////////////////////////////////////////////////////////////////////////////
//! malloc_2d
//! This is a kludge to dynamically allocate a contiguous 2d array
//! that will still be able to be accessed with arr[i][j]
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void** malloc_2d (size_t nrows, size_t ncols, int elementsize) {
   size_t i;
   void ** ptr;
   if ( (ptr = (void**)malloc(nrows * sizeof(void *))) == NULL ) {
      fprintf(stderr, "malloc_2d: out of memory\n");
      exit(1);
   }
   if ( (ptr[0] = malloc(nrows * ncols * elementsize)) == NULL ) {
      fprintf(stderr, "malloc_2d: out of memory\n");
      exit(1);
   }
   /* this is dangerous. I am assuming that char* has the same size as whatever
      pointer I am using at all times */
   for (i=1; i<nrows; i++) 
      ptr[i] = (char*)ptr[0] + i * ncols * elementsize;
   return ptr;
}


////////////////////////////////////////////////////////////////////////////////
//! read_gridspec
//! This function reads land/sea mask, ice and water fractions, grid
//! cell area, and country codes from the grid spec file 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
size_t read_gridspec (UserData* data) {
   int rv, ncid, varid;
   unsigned int i, j;
   if ((rv = nc_open(data->gridspec, NC_NOWRITE, &ncid)))
      NCERR(data->gridspec, rv);

   size_t index[2], count[2]; 
   get_grid_bounds(ncid, data);

   index[0] = data->start_lat;
   index[1] = data->start_lon;

   count[0] = data->n_lat;
   count[1] = data->n_lon;

   // This is the only layer that is for whole region when using mpi
   // All others are local to processor
   data->wtr_ice_f = (double **)malloc_2d(data->n_lat, data->n_lon, sizeof(double));
   printf("read_water_and_ice_fractions...\n");
   if ((rv = nc_inq_varid(ncid, "wtr_ice_frac", &varid)))
      NCERR("wtr_ice_frac", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index, count, &data->wtr_ice_f[0][0])))
      NCERR("wtr_ice_frac", rv);

   size_t offset = 0;
#ifdef USEMPI
   size_t old_start_lon = data->start_lon;
   narrow_lonbounds_to_proc(*data);
   offset = data->start_lon - old_start_lon;

   index[1] = data->start_lon;
   count[1] = data->n_lon;
#endif

   /* allocate arrays */
   data->map = (site ***)malloc_2d(data->n_lat, data->n_lon, sizeof(site*));
   data->grid_cell_area = (double **)malloc_2d(data->n_lat, data->n_lon, sizeof(double));
   data->grid_cell_area_total = (double **)malloc_2d(data->n_lat, data->n_lon, sizeof(double));

#if 0
   data->mask = (unsigned char **)malloc_2d(data->n_lat, data->n_lon, sizeof(unsigned char));
   data->gcode = (int **)malloc_2d(data->n_lat, data->n_lon, sizeof(int));

   printf("read_mask...\n");
   if ((rv = nc_inq_varid(ncid, "mask", &varid)))
      NCERR("mask", rv);
   if ((rv = nc_get_vara_uchar(ncid, varid, index, count, &(data->mask[0][0]))))
      NCERR("mask", rv);
#endif


   printf("read_grid_cell_area...\n");
   if ((rv = nc_inq_varid(ncid, "grid_cell_area", &varid)))
      NCERR("gridcellarea", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index, count, &data->grid_cell_area_total[0][0])))
      NCERR("gridcellarea", rv);
   size_t nPotentialSites = 0;
   for (i=0; i<data->n_lat; i++) {
      for (j=0; j<data->n_lon; j++) {
         data->grid_cell_area_total[i][j] *= 1000000.0; /*km2 to m2*/
         data->grid_cell_area[i][j] = data->grid_cell_area_total[i][j] 
            * (1.0000 - data->wtr_ice_f[i][j+offset]);
         if (data->grid_cell_area[i][j] < 0) {
            data->grid_cell_area[i][j] = 0.0;
         } else {
            nPotentialSites++;
         }
      }
   }

#if 0
   printf("read_country_codes...\n");
   if ((rv = nc_inq_varid(ncid, "countrycode", &varid))) {
      NCERR("countrycode", rv);
   }
   if ((rv = nc_get_vara_int(ncid, varid, index, count, &data->gcode[0][0]))) {
      NCERR("countrycode", rv);
   }
#endif

   if ((rv = nc_close(ncid))) {
      NCERR(data->gridspec, rv);  
   }
   return nPotentialSites;
}

char lu2charname2 (int lu) {
    char name;
    switch (lu) {
        case LU_NTRL:
            name = 'v';
            break;
        case LU_SCND:
            name = 's';
            break;
        case LU_CROP:
            name = 'c';
            break;
        case LU_PAST:
            name = 'p';
            break;
        default:
            fprintf(stderr, "unkown landuse type: %d\n", lu);
            return 0;
    }
    return name;
}

float* malloc_1d_float (size_t dim1)
{
    float *array_1d = (float *)malloc(dim1*sizeof(float));
    if (array_1d==NULL)
        printf("malloc_1d: out of memory\n");
    return array_1d;
}

float** malloc_2d_float (size_t dim1, size_t dim2)
{
    float *allElements = (float *)malloc(dim1*dim2*sizeof(float));
    if (allElements==NULL)
        printf("malloc_2d-1: out of memory\n");
    float **array_2d = (float **)malloc(dim1 * sizeof(float *));
    if (array_2d==NULL)
        printf("malloc_2d-2: out of memory\n");

    for(size_t i = 0; i < dim1; i++)
    {
        array_2d[i] =allElements + (i*dim2);
    }
    return array_2d;
}

float*** malloc_3d_float (size_t dim1, size_t dim2,size_t dim3)
{
    float *allElements = (float *)malloc(dim1*dim2*dim3*sizeof(float));
    if (allElements==NULL)
        printf("malloc_3d-1: out of memory\n");

    float ***array_3d = (float ***)malloc(dim1 * sizeof(float **));
    if (array_3d==NULL)
        printf("malloc_3d-2: out of memory\n");

    for(size_t i = 0; i < dim1; i++)
    {
        array_3d[i] = (float **)malloc(dim2 * sizeof(float *));
        if (array_3d[i]==NULL)
            printf("malloc_3d-3: out of memory\n");

        for(size_t j = 0; j < dim2; j++)
        {
            array_3d[i][j] = allElements + (i * dim2 * dim3) + (j * dim3);
        }
    }
    return array_3d;
}

float**** malloc_4d_float (size_t dim1, size_t dim2,size_t dim3,size_t dim4)
{
    float *allElements = (float *)malloc(dim1*dim2*dim3*dim4*sizeof(float));
    if (allElements==NULL)
        printf("malloc_4d-1: out of memory\n");
    
    float ****array_4d = (float ****)malloc(dim1 * sizeof(float ***));
    if (array_4d==NULL)
        printf("malloc_4d-2: out of memory\n");
    
    for(size_t i = 0; i < dim1; i++)
    {
        array_4d[i] = (float ***)malloc(dim2 * sizeof(float **));
        if (array_4d[i]==NULL)
            printf("malloc_4d-3: out of memory\n");
        
        for(size_t j = 0; j < dim2; j++)
        {
            array_4d[i][j]=(float **)malloc(dim3 * sizeof(float *));
            if (array_4d[i][j]==NULL)
                printf("malloc_4d-4: out of memory\n");
            
            for (size_t k=0;k<dim3;k++)
            {
                array_4d[i][j][k] = allElements + (i *dim2 * dim3*dim4) + (j * dim3 * dim4)+(k * dim4);
            }
        }
    }
    return array_4d;
}

float***** malloc_5d_float (size_t dim1, size_t dim2,size_t dim3,size_t dim4,size_t dim5)
{
    float *****array_5d =(float *****)malloc(dim1*sizeof(float ****));
    if (array_5d==NULL)
        printf("malloc_5d-1: out of memory\n");
    
    for (size_t i=0;i<dim1;i++)
    {
        array_5d[i]=(float ****)malloc(dim2*sizeof(float ***));
        if (array_5d[i]==NULL)
            printf("malloc_5d-2: out of memory\n");
        
        for (size_t j=0;j<dim2;j++)
        {
            array_5d[i][j]=malloc_3d_float(dim3,dim4,dim5);
        }
    }
    return array_5d;
}

float****** malloc_6d_float (size_t dim1, size_t dim2,size_t dim3,size_t dim4,size_t dim5,size_t dim6)
{
    float ******array_6d =(float ******)malloc(dim1*sizeof(float *****));
    if (array_6d==NULL)
        printf("malloc_6d-1: out of memory\n");
    
    for (size_t i=0;i<dim1;i++)
    {
        array_6d[i]=(float *****)malloc(dim2*sizeof(float ****));
        if ( array_6d[i]==NULL)
            printf("malloc_6d-1: out of memory\n");
        
        for (size_t j=0;j<dim2;j++)
        {
            array_6d[i][j]=malloc_4d_float(dim3,dim4,dim5,dim6);
        }
    }
    return array_6d;
}

void dealloc_2d_float(float** array_2d, size_t dim1, size_t dim2)
{
    free(array_2d);
}

void dealloc_3d_float (float*** array_3d,size_t dim1,size_t dim2,size_t dim3)
{
    free(array_3d[0][0]);
    for (size_t i=0;i<dim1;i++)
    {
        free(array_3d[i]);
    }
    free(array_3d);
}

void dealloc_4d_float (float**** array_4d,size_t dim1,size_t dim2,size_t dim3,size_t dim4)
{
    free(array_4d[0][0][0]);
    for(size_t i = 0; i < dim1; i++)
    {
        for(size_t j = 0; j < dim2; j++)
        {
            free(array_4d[i][j]);
        }
        free(array_4d[i]);
    }
    free(array_4d);
}

void dealloc_5d_float (float***** array_5d,size_t dim1,size_t dim2,size_t dim3,size_t dim4,size_t dim5)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            dealloc_3d_float(array_5d[i][j],dim3,dim4,dim5);
        }
    }
}

void dealloc_6d_float (float****** array_6d,size_t dim1,size_t dim2,size_t dim3,size_t dim4,size_t dim5,size_t dim6)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            dealloc_4d_float(array_6d[i][j],dim3,dim4,dim5,dim6);
        }
    }
}

void reset_1d_float(float* array_1d, size_t dim1)
{
    for (size_t i=0;i<dim1;i++)
    {
        array_1d[i]=0;
    }
}

void reset_2d_float(float** array_2d, size_t dim1, size_t dim2)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            array_2d[i][j]=0;
        }
    }
}

void reset_3d_float(float*** array_3d, size_t dim1, size_t dim2, size_t dim3)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            for (size_t k=0;k<dim3;k++)
            {
                array_3d[i][j][k]=0;
            }
        }
    }
}

void reset_4d_float(float**** array_4d, size_t dim1, size_t dim2, size_t dim3,size_t dim4)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            for (size_t k=0;k<dim3;k++)
            {
                for (size_t l=0;l<dim4;l++)
                {
                    array_4d[i][j][k][l]=0;
                }
            }
        }
    }
}

void reset_5d_float(float***** array_5d, size_t dim1, size_t dim2, size_t dim3,size_t dim4,size_t dim5)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            reset_3d_float(array_5d[i][j],dim3,dim4,dim5);
        }
    }
}

void reset_6d_float(float****** array_6d, size_t dim1, size_t dim2, size_t dim3,size_t dim4,size_t dim5,size_t dim6)
{
    for (size_t i=0;i<dim1;i++)
    {
        for (size_t j=0;j<dim2;j++)
        {
            reset_4d_float(array_6d[i][j],dim3,dim4,dim5,dim6);
        }
    }
}


#if LANDUSE
bool setCropCalendar (site** sitrptr,UserData* data)
{
    site* cs = *sitrptr;
    double temp_crit = 10.0;
    double precip_crit = 100.0; // mm/month
    while (cs != NULL)
    {
        int ilat = cs->sdata->globY_;
        int ilon = cs->sdata->globX_;
        int climate_zone = 9999;
        double annual_minimum_temperature = 100000.0;
        
        //Find temperature of coldest month of the year
        for (size_t mon=0;mon<N_CLIMATE;mon++)
        {
            if(cs->sdata->temp[mon] < annual_minimum_temperature)
                annual_minimum_temperature = cs->sdata->temp[mon];
        }
        // Find if this site is temperature or preciptation limited
        if (annual_minimum_temperature < temp_crit)
            climate_zone = 1;
        else
            climate_zone = 0;
        
        //if temperature limited
        if(climate_zone==1)
        {
            for (size_t mon=0;mon<N_CLIMATE-1;mon++)
            {
                if((cs->sdata->temp[mon] < temp_crit) && (cs->sdata->temp[mon+1] >= temp_crit))
                    data->planting_probability[mon][ilat][ilon] = 1.0;
                else
                    data->planting_probability[mon][ilat][ilon] = 0.0;
                
                if((cs->sdata->temp[mon] > temp_crit) && (cs->sdata->temp[mon+1] <= temp_crit))
                    data->harvest_probability[mon][ilat][ilon] = 1.0;
                else
                    data->harvest_probability[mon][ilat][ilon] = 0.0;
            }
        }
        else
        {
            for (size_t mon=0;mon<N_CLIMATE-1;mon++)
            {
                if((cs->sdata->precip[mon] < precip_crit) && (cs->sdata->precip[mon+1] >= precip_crit))
                    data->planting_probability[mon][ilat][ilon] = 1.0;
                else
                    data->planting_probability[mon][ilat][ilon] = 0.0;
                
                if((cs->sdata->precip[mon] > precip_crit) && (cs->sdata->precip[mon+1] <= precip_crit))
                    data->harvest_probability[mon][ilat][ilon] = 1.0;
                else
                    data->harvest_probability[mon][ilat][ilon] = 0.0;
            }
        }
        cs = cs->next_site;
    }
    
    return 1;
}


bool loadCropCalendar (UserData* data)
{
    int rv, ncid, varid, dlu, tlu, i;
    char dn, tn;
    size_t index[3], count[3];
    double factor;
    
    
    index[0] = 0;
    index[1] = 0;
    index[2] = 0;
    
    count[0] = 12;
    count[1] = 360;
    count[2] = 720;
    
    rv = nc_open(data->crop_calendar_file, NC_NOWRITE, &ncid);
    rv = nc_inq_varid(ncid, "planting_Probability", &varid);
    rv = nc_get_vara_double(ncid, varid, index, count,&data->planting_probability[0][0][0]);
    
    rv = nc_inq_varid(ncid, "harvest_Probability", &varid);
    rv = nc_get_vara_double(ncid, varid, index, count,&data->harvest_probability[0][0][0]);
    
    return 1;
}

bool loabGlobalLUData (UserData* data)
{
    int rv, ncid, varid, dlu, tlu, i;
    char dn, tn;
    char varname[10];
    size_t index[3], count[3];
    double factor;
    
    
    index[0] = 0;
    index[1] = 0;
    index[2] = 0;
    
    count[0] = N_LANDUSE_YEARS;
    count[1] = 360;
    count[2] = 720;
    
    if (data->lu_file_ncid == 0) {
        if ((rv = nc_open(data->lu_file, NC_NOWRITE, &ncid)))
            NCERR(data->lu_file, rv);
        data->lu_file_ncid = ncid;
    } else {
        ncid = data->lu_file_ncid;
    }

    for (dlu=0; dlu<N_LANDUSE_TYPES; dlu++) {
        for (tlu=1; tlu<N_LANDUSE_TYPES; tlu++) {
            /* skip transitions self->self and v->s (v->s dealt with in sbh/vbh) */
            if ((dlu != tlu) && !( (dlu == LU_NTRL) && (tlu == LU_SCND) ) ) {
                if ((dn = lu2charname2(dlu)) && (tn = lu2charname2(tlu))) {
                    sprintf(varname, "gfl%c%c", dn, tn);
                    if ((rv = nc_inq_varid(ncid, varname, &varid)))
                        NCERR(varname, rv);
                    if ((rv = nc_get_vara_float(ncid, varid, index, count,
                                                &(data->gfl[dlu][tlu-1][0][0][0]))))
                        NCERR(varname, rv);
                }
            }
        }
    }
    
    count[0] = N_LANDUSE_YEARS;
    count[1] = 360;
    count[2] = 720;
    for (dlu=0; dlu<N_VBH_TYPES; dlu++) {
        sprintf(varname, "gfvh%d", dlu+1);
        if ((rv = nc_inq_varid(ncid, varname, &varid)))
            NCERR(varname, rv);
        
        if ((rv = nc_get_vara_float(ncid, varid, index, count,
                                    &(data->gfvh[dlu][0][0][0]))))
            NCERR(varname, rv);
    }
    
    for (dlu=0; dlu<N_SBH_TYPES; dlu++) {
        sprintf(varname, "gfsh%d", dlu+1);
        if ((rv = nc_inq_varid(ncid, varname, &varid)))
            NCERR(varname, rv);
        
        if ((rv = nc_get_vara_float(ncid, varid, index, count,
                                    &(data->gfsh[dlu][0][0][0]))))
            NCERR(varname, rv);
    }
    
    rv = nc_close(ncid);
    return 1;
}
#endif

bool loadGlobalEnvironmentData(UserData* data)
{
    int rv, ncid, varid,x,y,z;
    
    size_t index1[2] = { 0, 0};
    size_t index2[3] = { 0,0,0};
    size_t count1[3]  = { 360, 720};
    size_t count2[3]  = { 12,360, 720};
    

    // Added below for yearly climate
    char nc[4] = ".nc"  ;
    char base[256] = "";
    char convert[256];
    char climatename[256];
    if (data->do_yearly_mech) {
        if (data->m_int)
        {
            if (data->mechanism_year>MERRA2_YEAR_START_USE)
            {
                if (data->mechanism_year<MERRA2_START)
                {
                    data->MERRA2_timestamp=(MERRA2_END-MERRA2_START+1)-(MERRA2_START-1-data->mechanism_year)%(MERRA2_END-MERRA2_START+1);
                }
                else
                {
                    data->MERRA2_timestamp=data->mechanism_year-MERRA2_START+1;
                }
                
                printf("mechanism_year %d MERRA2_timestamp %d\n",data->mechanism_year,data->MERRA2_timestamp);
                sprintf(convert, "%s%d", base, data->MERRA2_timestamp+MERRA2_START-1);
                strcpy(climatename, data->climate_file);
                strcat(climatename, convert);
                strcat(climatename, nc);
#if FORECAST
                if(abs(FORECAST_yr-data->mechanism_year)<0.5)
                {
                    printf("Loading forecast meteorology for climate\n");
                    strcpy(convert, "");
                    sprintf(convert, "_%s%d", base, FORECAST_yr);
                    strcpy(climatename, data->forecast_climate_file);
                    strcat(climatename, FORECAST_ENS);
                    strcat(climatename, convert);
                    strcat(climatename, nc);
                }
#endif
            }
            else
            {
                strcpy(climatename, data->climate_file_avg);
            }
            printf("Loading Environmental data from %s\n",climatename);
        }
        
        if(data->m_string) {
            strcpy(climatename,data->climate_file);
            strcat(climatename,data->mech_year_string);
            strcat(climatename, nc);
        }
    } else if(data->single_year) {
        strcpy(climatename, data->climate_file_avg);
    }
#if 0
    printf("climate file used is %s\n",climatename);
#endif
    
    if (data->soil_depth==NULL)
    {
        printf("Start allocate soil_depth\n");
        data->soil_depth=malloc_2d_float(360,720);
    }
    else
    {
        printf("Start reset soil_depth\n");
        reset_2d_float(data->soil_depth,360,720);
    }
    
    if (data->theta_max==NULL)
    {
        data->theta_max=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->theta_max,360,720);
    }
    
    if (data->k_sat==NULL)
    {
        data->k_sat=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->k_sat,360,720);
    }
    
    if (data->tau==NULL)
    {
        data->tau=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->tau,360,720);
    }

#if SOILGRIDS_SCHEME
    if (data->MvG_soil_depth==NULL)
    {
        data->MvG_soil_depth=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_soil_depth,360,720);
    }
    if (data->MvG_k_sat==NULL)
    {
        data->MvG_k_sat=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_k_sat,360,720);
    }
    if (data->MvG_L==NULL)
    {
        data->MvG_L=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_L,360,720);
    }
    if (data->MvG_m==NULL)
    {
        data->MvG_m=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_m,360,720);
    }
    if (data->MvG_theta_saturated==NULL)
    {
        data->MvG_theta_saturated=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_theta_saturated,360,720);
    }
    if (data->MvG_theta_residual==NULL)
    {
        data->MvG_theta_residual=malloc_2d_float(360,720);
    }
    else
    {
        reset_2d_float(data->MvG_theta_residual,360,720);
    }
    int ncid_soilGrids;
    char soil_file_soilGrids[256];
    strcpy(soil_file_soilGrids,data->soilgrids_file);
    printf("SOILGRIDS_SCHEME is ON, loading paramemters from %s\n",soil_file_soilGrids);
    if ((rv = nc_open(soil_file_soilGrids, NC_NOWRITE, &ncid_soilGrids)))
    {
        NCERR(soil_file_soilGrids, rv);
    }
    nc_inq_varid(ncid_soilGrids, "soil_depth", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_soil_depth[0][0]);
    nc_inq_varid(ncid_soilGrids, "k_sat", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_k_sat[0][0]);
    nc_inq_varid(ncid_soilGrids, "L", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_L[0][0]);
    nc_inq_varid(ncid_soilGrids, "m", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_m[0][0]);
    nc_inq_varid(ncid_soilGrids, "residual_theta", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_theta_residual[0][0]);
    nc_inq_varid(ncid_soilGrids, "saturated_theta", &varid);
    nc_get_vara_float(ncid_soilGrids, varid, index1,count1, &data->MvG_theta_saturated[0][0]);
    ncclose(ncid_soilGrids);
#endif
    
    
    if (data->soil_file_ncid == 0) {
        if ((rv = nc_open(data->soil_file, NC_NOWRITE, &ncid))) {
            NCERR(data->soil_file, rv);
        }
        data->soil_file_ncid = ncid;
    } else {
        ncid = data->soil_file_ncid;
    }
    
    if ((rv = nc_inq_varid(ncid, "soil_depth", &varid))) {
        NCERR("soil_depth", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index1,count1, &data->soil_depth[0][0]))) {
        NCERR("soil_depth", rv);
    }
    //soil_depth *= 10.0; // convert from cm to mm
    
    if ((rv = nc_inq_varid(ncid, "soil_theta_max", &varid))) {
        NCERR("theta_max", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index1,count1, &data->theta_max[0][0]))) {
        NCERR("theta_max", rv);
    }
    
    if ((rv = nc_inq_varid(ncid, "soil_k_sat", &varid))) {
        NCERR("k_sat", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index1,count1, &data->k_sat[0][0]))) {
        NCERR("k_sat", rv);
    }
    
    if ((rv = nc_inq_varid(ncid, "soil_tau", &varid))) {
        NCERR("tau", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index1,count1, &data->tau[0][0]))) {
        NCERR("tau", rv);
    }
    ncclose(data->soil_file_ncid);
    data->soil_file_ncid=0;
    
    
    if (data->climate_temp==NULL)
    {
        printf("Start allocate climate_temp\n");
        data->climate_temp=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_temp\n");
        reset_3d_float(data->climate_temp,N_CLIMATE,360,720);
    }
    
    if (data->climate_precip==NULL)
    {
        printf("Start allocate climate_precip\n");
        data->climate_precip=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_precip\n");
        reset_3d_float(data->climate_precip,N_CLIMATE,360,720);
    }
    
    if (data->climate_soil==NULL)
    {
        printf("Start allocate climate_soil\n");
        data->climate_soil=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil\n");
        reset_3d_float(data->climate_soil,N_CLIMATE,360,720);
    }
    
    if (data->climate_file_ncid == 0) {
        if ((rv = nc_open(climatename, NC_NOWRITE, &ncid))) {
            NCERR(climatename, rv);
        }
        data->climate_file_ncid = ncid;
    } else {
        ncid = data->climate_file_ncid;
    }
    
    if ((rv = nc_inq_varid(ncid, "precipitation", &varid))) {
        NCERR("precip", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_precip[0][0][0]))) {
        NCERR("precip", rv);
    }
    
    if ((rv = nc_inq_varid(ncid, "temperature", &varid))) {
        NCERR("temp", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_temp[0][0][0]))) {
        NCERR("temp", rv);
    }
    
    if (data->climate_soil1==NULL)
    {
        printf("Start allocate climate_soil1\n");
        data->climate_soil1=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil1\n");
        reset_3d_float(data->climate_soil1,N_CLIMATE,360,720);
    }
    if ((rv = nc_inq_varid(ncid, "soil_temp1", &varid))) {
        NCERR("soil_temp1", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_soil1[0][0][0]))) {
        NCERR("soil_temp1", rv);
    }

    if (data->climate_soil2==NULL)
    {
        printf("Start allocate climate_soil2\n");
        data->climate_soil2=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil2\n");
        reset_3d_float(data->climate_soil2,N_CLIMATE,360,720);
    }
    if ((rv = nc_inq_varid(ncid, "soil_temp2", &varid))) {
        NCERR("soil_temp2", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_soil2[0][0][0]))) {
        NCERR("soil_temp2", rv);
    }

    if (data->climate_soil3==NULL)
    {
        printf("Start allocate climate_soil3\n");
        data->climate_soil3=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil3\n");
        reset_3d_float(data->climate_soil3,N_CLIMATE,360,720);
    }
    if ((rv = nc_inq_varid(ncid, "soil_temp3", &varid))) {
        NCERR("soil_temp3", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_soil3[0][0][0]))) {
        NCERR("soil_temp3", rv);
    }

    if (data->climate_soil4==NULL)
    {
        printf("Start allocate climate_soil4\n");
        data->climate_soil4=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil4\n");
        reset_3d_float(data->climate_soil4,N_CLIMATE,360,720);
    }
    if ((rv = nc_inq_varid(ncid, "soil_temp4", &varid))) {
        NCERR("soil_temp4", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_soil4[0][0][0]))) {
        NCERR("soil_temp4", rv);
    }

    if (data->climate_soil5==NULL)
    {
        printf("Start allocate climate_soil5\n");
        data->climate_soil5=malloc_3d_float(N_CLIMATE,360,720);
    }
    else
    {
        printf("Start reset climate_soil5\n");
        reset_3d_float(data->climate_soil5,N_CLIMATE,360,720);
    }
    if ((rv = nc_inq_varid(ncid, "soil_temp5", &varid))) {
        NCERR("soil_temp5", rv);
    }
    if ((rv = nc_get_vara_float(ncid, varid, index2, count2, &data->climate_soil5[0][0][0]))) {
        NCERR("soil_temp5", rv);
    }

    for (size_t ilat=0;ilat<360;ilat++)
    {
        for (size_t ilon=0;ilon<720;ilon++)
        {
            if (data->climate_soil1[0][ilat][ilon]>-274)
            {
                for (size_t t=0;t<N_CLIMATE;t++)
                {
                    data->climate_soil[t][ilat][ilon]+=data->climate_soil1[t][ilat][ilon];
                    data->climate_soil[t][ilat][ilon]+=data->climate_soil2[t][ilat][ilon];
                    data->climate_soil[t][ilat][ilon]+=data->climate_soil3[t][ilat][ilon];
                    data->climate_soil[t][ilat][ilon]+=data->climate_soil4[t][ilat][ilon];
                    data->climate_soil[t][ilat][ilon]+=data->climate_soil5[t][ilat][ilon];
                    data->climate_soil[t][ilat][ilon]/=5.0;
                }
            }
            else
            {
                for (size_t t=0;t<N_CLIMATE;t++)
                {
                    data->climate_soil[t][ilat][ilon]=-9999.0;
                }
            }
        }
    }
    
    ncclose(data->climate_file_ncid);
    data->climate_file_ncid=0;

    return 1;
}

bool loadPREMECH (UserData* data)
{
    char premech_clim[256],premech_co2[256], tairname[256], swname[256];
    int rv,ncid,varid,ncid_co2,varid_co2;
    
    size_t index3[3] = { 0, 0, 0 };
    size_t count3[3]  = {288, 360, 720 };
    
    if (data->do_yearly_mech)
    {
        if (data->mechanism_year>MERRA2_YEAR_START_USE)
        {
            if (data->mechanism_year<MERRA2_START)
            {
                data->MERRA2_timestamp=(MERRA2_END-MERRA2_START+1)-(MERRA2_START-1-data->mechanism_year)%(MERRA2_END-MERRA2_START+1);
            }
            else
            {
                data->MERRA2_timestamp=data->mechanism_year-MERRA2_START+1;
            }
            printf("mechanism_year %d MERRA2_timestamp %d\n",data->mechanism_year,data->MERRA2_timestamp);
            strcpy(premech_clim, data->PREMECH);
            sprintf(premech_clim, "%s%d", premech_clim, data->MERRA2_timestamp+MERRA2_START-1);
            strcat(premech_clim, ".nc");
           
#if FORECAST
            if(abs(FORECAST_yr-data->mechanism_year)<0.5)
            {
                printf("Loading forecast meteorology for premech\n");
                strcpy(premech_clim, data->forecast_PREMECH);
                sprintf(premech_clim, "%s%s_%d", premech_clim,FORECAST_ENS, FORECAST_yr);
                strcat(premech_clim, ".nc");
            }
#endif
            
        }
        else
        {
            strcpy(premech_clim, data->PREMECH_avg);
        }
        printf("Loading premech file from %s\n",premech_clim);
        
        if(data->mechanism_year<=2000) //before 2001, using average climatology co2 and scaling factor
        {
            strcpy(premech_co2, data->PREMECH_CO2_avg);
            printf("Loading CO2 premech file from %s\n",premech_co2);
        }
        else if (data->mechanism_year>2000 and data->mechanism_year<2015)//After 2001 but before 2015, use actual data
        {
            strcpy(premech_co2, data->PREMECH_CO2);
            sprintf(premech_co2, "%s%d", premech_co2, data->mechanism_year);
            strcat(premech_co2, ".nc");
            printf("Loading CO2 premech file from %s\n",premech_co2);
        }
        else //After 2014, using 2014
        {
            strcpy(premech_co2, data->PREMECH_CO2);
            sprintf(premech_co2, "%s%d", premech_co2, 2014);
            strcat(premech_co2, ".nc");
            printf("Loading CO2 premech file from %s\n",premech_co2);
        }
    } else if(data->single_year)
    {
        strcpy(premech_clim, data->PREMECH_avg);
        printf("Loading premech file from %s\n",premech_clim);
        
        strcpy(premech_co2, data->PREMECH_CO2_avg);
        printf("Loading CO2 premech file from %s\n",premech_co2);
    }
    
    if ((rv = nc_open(premech_clim, NC_NOWRITE, &ncid))){
        NCERR(premech_clim, rv);
    }
    else{
        data->premech_file_ncid = ncid;
    }
    
    if ((rv = nc_open(premech_co2, NC_NOWRITE, &ncid_co2))){
        NCERR(premech_co2, rv);
    }
    
    if ((rv = nc_inq_varid(ncid, "AirTemp", &varid))) NCERR("AirTemp", rv);
    if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &data->global_tmp[0][0][0]))) NCERR("AirTemp", rv);
    
    if ((rv = nc_inq_varid(ncid, "specific_humidity", &varid))) NCERR("specific_humidity", rv);
    if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &data->global_hum[0][0][0]))) NCERR("specific_humidity", rv);
    
    if ((rv = nc_inq_varid(ncid, "swdown", &varid))) NCERR("swdown", rv);
    if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &data->global_swd[0][0][0]))) NCERR("swdown", rv);
    
    if ((rv = nc_inq_varid(ncid, "wind", &varid))) NCERR("wind", rv);
    if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &data->global_windspeed[0][0][0]))) NCERR("wind", rv);
    
    if ((rv = nc_inq_varid(ncid, "SoilTemp", &varid))) NCERR("SoilTemp", rv);
    if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &data->global_soiltmp[0][0][0]))) NCERR("SoilTemp", rv);
    
    for (size_t mon=0;mon<288;mon++)
    {
        for (size_t lat_n=0;lat_n<360;lat_n++)
        {
            for (size_t lon_n=0;lon_n<720;lon_n++)
            {
                if (data->global_tmp[mon][lat_n][lon_n]>0) data->global_tmp[mon][lat_n][lon_n]-=273.2;
                if (data->global_soiltmp[mon][lat_n][lon_n]>0) data->global_soiltmp[mon][lat_n][lon_n]-=273.2;
                if (data->global_hum[mon][lat_n][lon_n]>0) data->global_hum[mon][lat_n][lon_n]=data->global_hum[mon][lat_n][lon_n]*28.96/18.02;
            }
        }
    }
   
    //load spatial CO2 data
    if ((rv = nc_inq_varid(ncid_co2, "co2_ambient", &varid_co2))) NCERR("co2_ambient", rv);
    if ((rv = nc_get_vara_double(ncid_co2, varid_co2, index3, count3, &data->global_CO2[0][0][0]))) NCERR("co2_ambient", rv);
    rv = nc_close(ncid);
    rv = nc_close(ncid_co2);

    return true;
}

bool load_GFED(UserData* data)
{
    int rv, ncid_gfed, varid;
    size_t k,i,j;
    char gfed[256];
    
    size_t index[3] = { 0, 0, 0 };
    size_t count[3]  = {12, 360, 720 };
    
    if (data->do_yearly_mech)
    {
        if(data->mechanism_year<1996 or data->mechanism_year>2016)
        {
            strcpy(gfed, data->gfedbf_file_avg);
        }
        else if(data->mechanism_year>=1996 && data->mechanism_year<=2016)
        {
            strcpy(gfed, data->gfedbf_file);
            sprintf(gfed, "%s%d", gfed, data->mechanism_year);
            strcat(gfed, ".nc");
        }
    }
    else if(data->single_year)
    {
        strcpy(gfed, data->gfedbf_file_avg);
    }
    
    if ((rv = nc_open(gfed, NC_NOWRITE, &ncid_gfed))) {
        NCERR(gfed, rv);
    }
    
    printf("Loading GFED file from %s\n",gfed);
    
    if ((rv = nc_inq_varid(ncid_gfed, "burnedfraction", &varid))) {
        NCERR("burnedfraction", rv);
    }
    
    if ((rv = nc_get_vara_double(ncid_gfed, varid, index, count, &(data->gfed_bf[0][0][0])))) {
        NCERR("burnedfraction", rv);
    }
    
    /* set all missing values to 0 */

   // TODO: 360, 720 should be replaced by macro variables
    for (k=0;k<N_CLIMATE;k++){
        for (i=0; i<360; i++) {
            for (j=0; j<720; j++) {
                if (data->gfed_bf[k][i][j] < 0.0)
                {
                    data->gfed_bf[k][i][j] = 0.0;
                }
                data->gfed_bf[k][i][j] *= 12.0;
            }
        }
    }
    
    rv = nc_close(ncid_gfed);
    return true;
}

void freeGlobalEnvironmentData(UserData* data)
{
    dealloc_2d_float(data->soil_depth,360,720);
    dealloc_2d_float(data->theta_max,360,720);
    dealloc_2d_float(data->k_sat,360,720);
    dealloc_2d_float(data->tau,360,720);
    dealloc_3d_float(data->climate_temp,N_CLIMATE,360,720);
    dealloc_3d_float(data->climate_precip,N_CLIMATE,360,720);
    dealloc_3d_float(data->climate_soil,N_CLIMATE,360,720);
}

////////////////////////////////////////////////////////////////////////////////
//! SiteData
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
SiteData::SiteData (size_t y, size_t x, UserData& data) {
   y_ = y;
   x_ = x;

   globY_ = y + data.start_lat;
   globX_ = x + data.start_lon;

   lat_ = data.lats[y];
   lon_ = data.lons[x];

   sprintf(name_, "lat%.2flon%.2f", lat_, lon_);

   soi = is_soi(lat_, lon_, data);

   // site area
   grid_cell_area = data.grid_cell_area[y][x];
   grid_cell_area_total = data.grid_cell_area_total[y][x];

   // TODO: should the rest be here? Same for all sites

   // max soil evaporation per mm of soil moisture per yr
   soil_evap_conductivity = 0.0;

   loss_fraction[0] = 0.0; // nothing lost during treefall
   loss_fraction[1] = data.smoke_fraction;

#ifdef ED
   if (data.open_cycles) {
      /* assume a basin wide avg of 1000 mm/yr and a basin wide avg *
       * deposisiton of N of 0.001 kg N /m2 / y, the latter is      *
       * reported in a ref in Vitousek et al 1986                   */
      N_conc_in_rain = 0.0000001; // kg N / m2 * mm  /// old incorrect value 0.0000001
   }
   L_top  = 1.0;  // Light at the top of the canopy
   Rn_top = 1.0;  // Net Radn flx at the top of the canopy
#endif
}


////////////////////////////////////////////////////////////////////////////////
//! readSiteData
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
bool SiteData::readSiteData (UserData& data) {

   if ( ! readEnvironmentalData(data) ) {
      return false;
   }

   calcPETAverage ();

#ifdef ED
   for (size_t i=0; i<12; i++) {
      pet[i] = calcPETMonthly (i);
   }
   calcSiteDrynessIndex(data);
   
    if (!SetMECHdefault(data)) {
        return false;
    }

#elif defined MIAMI_LU
   // calculate miami npp
   miami_npp = miami(precip_average, temp_average);
#endif

   return true;
}


#ifdef MIAMI_LU
////////////////////////////////////////////////////////////////////////////////
//! readEnvironmentalData
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
bool SiteData::readEnvironmentalData (UserData& data) {
   int rv, ncid, varid;

   size_t index[2];
   index[0] = globX_;
   index[1] = globY_;

   if (data.climate_file_ncid == 0) {
      if ((rv = nc_open(data.climate_file, NC_NOWRITE, &ncid))) {
         NCERR(data.climate_file, rv);
      }
      data.climate_file_ncid = ncid;
   } else {
      ncid = data.climate_file_ncid;
   }

   // precip 
   if ((rv = nc_inq_varid(ncid, "annual_precipitation", &varid))) {
      NCERR("precip", rv);
   }
   if ((rv = nc_get_var1_double(ncid, varid, index, &precip_average))) {
      NCERR("precip", rv);
   }
   if (precip_average < 0) {
      //fprintf(stderr, "No precip data found for site lat %f lon %f\n", cs->lat, cs->lon);
      return false;
   } 

   if ((rv = nc_inq_varid(ncid, "annual_temperature", &varid))) {
      NCERR("temp", rv);
   }
   if ((rv = nc_get_var1_double(ncid, varid, index, &temp_average))) {
      NCERR("temp", rv);
   }
   if (temp_average == -9999.0) { // TODO: better test?
      //fprintf(stderr, "No temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
      return false;
   } else {
      temp_average -= 273.2; // convert to celsius
   }

   if ((rv = nc_inq_varid(ncid, "annual_soil_temp", &varid))) {
      // No soil_temp? default to use air temp
      if ((rv = nc_inq_varid(ncid, "annual_temperature", &varid))) {
         NCERR("soil_temp", rv);
      }
   }
   if ((rv = nc_get_var1_double(ncid, varid, index, &soil_temp_average)))
      NCERR("soil_temp", rv);
   if (soil_temp_average == -9999.0) { // TODO: better test?
      //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
      return false;
   } else {
      soil_temp_average -= 273.2; // convert to celsius
   }

   return true;
}
#endif /* MIAMI_LU */


#ifdef ED
////////////////////////////////////////////////////////////////////////////////
//! readEnvironmentalData
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
bool SiteData::readEnvironmentalData (UserData& data)
{
    soil_depth=data.soil_depth[globY_][globX_];
    theta_max=data.theta_max[globY_][globX_];
    k_sat=data.k_sat[globY_][globX_];
    tau=data.tau[globY_][globX_];

#if SOILGRIDS_SCHEME
    MvG_soil_depth = data.MvG_soil_depth[globY_][globX_];
    MvG_k_sat = data.MvG_k_sat[globY_][globX_];
    MvG_L = data.MvG_L[globY_][globX_];
    MvG_m = data.MvG_m[globY_][globX_];
    MvG_theta_saturated = data.MvG_theta_saturated[globY_][globX_];
    MvG_theta_residual = data.MvG_theta_residual[globY_][globX_];
#endif
    
    if ( (soil_depth <= 0) || (theta_max == -9999.0)
        || (k_sat == -9999.0) || (tau == -9999.0) ) {
        //fprintf(stderr, "No soil char data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    precip_average=0.0;
    temp_average=0.0;
    soil_temp_average=0.0;
    
    if (data.climate_precip[0][globY_][globX_] < 0.0) {
        //fprintf(stderr, "No precip data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_temp[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_soil[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    
    if (data.climate_soil1[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_soil2[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_soil3[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_soil4[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    if (data.climate_soil5[0][globY_][globX_] == -9999.0) { // TODO: better test?
        //fprintf(stderr, "No soil temp data found for site lat %f lon %f\n", cs->lat, cs->lon);
        return false;
    }
    
    for (size_t i=0;i<N_CLIMATE;i++)
    {
        precip[i] = data.climate_precip[i*12/N_CLIMATE][globY_][globX_]*12.; // to make units avg mm/yr
        precip_average += precip[i] / N_CLIMATE;
        temp[i] = data.climate_temp[i*12/N_CLIMATE][globY_][globX_] - 273.15;
        temp_average += temp[i] / N_CLIMATE;
        soil_temp[i] = data.climate_soil[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        soil_temp_average += soil_temp[i] / N_CLIMATE;
        
        soil_temp1[i] = data.climate_soil1[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        soil_temp2[i] = data.climate_soil2[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        soil_temp3[i] = data.climate_soil3[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        soil_temp4[i] = data.climate_soil4[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        soil_temp5[i] = data.climate_soil5[i*12/N_CLIMATE][globY_][globX_] - 273.15; // convert Kelvin to C
        
    }
    
    return true;
}


#if FTS
////////////////////////////////////////////////////////////////////////////////
//! readFTSData
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////

bool SiteData::readFTSdata (UserData& data) {
   
   size_t lengthp;
   int t_id, rv, ncid, varid;
   size_t index3[3] = { 0, globY_, globX_ };
   size_t count3[3]  = {N_CLIMATE_INPUT*CLIMATE_INPUT_INTERVALS, 1, 1 };
   char qairname[256], tairname[256], swname[256];
   
   // Read in the appropriate TEMP/HUM/PAR files 
   strcpy(qairname, data.QAIR_FILE);
   if ((rv = nc_open(qairname, NC_NOWRITE, &ncid))){ 
      NCERR(qairname, rv); 
   }
   else{
      data.qair_file_ncid = ncid;
   }
   if ((rv=nc_inq_dimid(ncid, "time", &t_id))) NCERR("time", rv);
   if ((rv=nc_inq_dimlen(ncid, t_id, &lengthp))) NCERR("time_len", rv);
   if (lengthp!=N_CLIMATE_INPUT*CLIMATE_INPUT_INTERVALS) {printf("Input file time length not correct - qair\n"); exit(1);}
   if ((rv = nc_inq_varid(ncid, "qair", &varid))) NCERR("qair", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &Input_Specific_Humidity[0]))) NCERR("qair", rv);
   rv = nc_close(ncid);
   
   strcpy(tairname, data.TAIR_FILE);
   if ((rv = nc_open(tairname, NC_NOWRITE, &ncid))){ 
      NCERR(tairname, rv); 
   }
   else{
      data.tair_file_ncid = ncid;
   }
   if ((rv=nc_inq_dimid(ncid, "time", &t_id))) NCERR("time", rv);
   if ((rv=nc_inq_dimlen(ncid, t_id, &lengthp))) NCERR("time_len", rv);
   if (lengthp!=N_CLIMATE_INPUT*CLIMATE_INPUT_INTERVALS) {printf("Input file time length not correct - qair\n"); exit(1);}
   if ((rv = nc_inq_varid(ncid, "tair", &varid))) NCERR("tair", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &Input_Temperature[0]))) NCERR("tair", rv);
   rv = nc_close(ncid);
   
   strcpy(swname, data.SW_FILE);
   if ((rv = nc_open(swname, NC_NOWRITE, &ncid))){ 
      NCERR(swname, rv); 
   }
   else{
      data.sw_file_ncid = ncid;
   }
   if ((rv=nc_inq_dimid(ncid, "time", &t_id))) NCERR("time", rv);
   if ((rv=nc_inq_dimlen(ncid, t_id, &lengthp))) NCERR("time_len", rv);
   if (lengthp!=N_CLIMATE_INPUT*CLIMATE_INPUT_INTERVALS) {printf("Input file time length not correct - qair\n"); exit(1);}
   if ((rv = nc_inq_varid(ncid, "swdown", &varid))) NCERR("swdown", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index3, count3, &Input_Par[0]))) NCERR("swdown", rv);
   rv = nc_close(ncid);
   
   //Convert inputs to right units
   int i;
   for (i=0;i<N_CLIMATE_INPUT*CLIMATE_INPUT_INTERVALS;i++){
      Input_Temperature[i]-=273.2;
      Input_Par[i]/=2.;
      Input_Specific_Humidity[i]*=28.96 / 18.02;
   }
   
   // Monthly Averaging done the same way as mech processing 
   if (1){ 
      int ndays = 365;
      float dpp = ndays/12.0;
      float dc = 0.;
      //Two grids to keep the leftover fraction of each month
      float averages[12] ={0};
      float averages2[12] = {0};
      int start_day = 0;
      int j,k,s,t;
      t = 0;
      float check = 0;
      //loop through days and apply appropriate fraction for that day/hour
      for (i=0;i<ndays;i++){
         dc +=1 ;
         if (dc >= dpp){
            float frac = 1 - (dc-dpp);
            for (k=0;k<4;k++){
               if (t%2==0){
                  averages[k]+=Input_Specific_Humidity[4*i+k]*frac;
                  averages[4+k]+=Input_Temperature[4*i+k]*frac;
                  averages[8+k]+=Input_Par[4*i+k]*frac;
                  averages2[k]+=Input_Specific_Humidity[4*i+k]*(1-frac);
                  averages2[4+k]+=Input_Temperature[4*i+k]*(1-frac);
                  averages2[8+k]+=Input_Par[4*i+k]*(1-frac);
                  
               }
               else{
                  averages[k]+=Input_Specific_Humidity[4*i+k]*(1-frac);
                  averages[4+k]+=Input_Temperature[4*i+k]*(1-frac);
                  averages[8+k]+=Input_Par[4*i+k]*(1-frac);
                  averages2[k]+=Input_Specific_Humidity[4*i+k]*(frac);
                  averages2[4+k]+=Input_Temperature[4*i+k]*(frac);
                  averages2[8+k]+=Input_Par[4*i+k]*(frac);
               }
            }
            dc = (1-frac);
            for (j=start_day;j<=i;j++){
               for (k=0;k<4;k++){
                  if (t%2==0){
                     Input_Specific_Humidity[4*j+k] = averages[k]/(dpp);
                     Input_Temperature[4*j+k] = averages[4+k]/(dpp);
                     Input_Par[4*j+k] = averages[8+k]/(dpp);
                  }
                  else{
                     Input_Specific_Humidity[4*j+k] = averages2[k]/(dpp);
                     Input_Temperature[4*j+k] = averages2[4+k]/(dpp);
                     Input_Par[4*j+k] = averages2[8+k]/(dpp);
                  }
               }
            }
            for (s=0;s<12;s++){
               if (t%2==0){
                  averages[s]=0;
               }
               else{
                  averages2[s]=0;
               }
            }
            t +=1;
            start_day = i+1;
         }
         else{
            for (k=0;k<4;k++){
               if (t%2 ==0){
                  averages[k]+=Input_Specific_Humidity[4*i+k];
                  averages[4+k]+=Input_Temperature[4*i+k];
                  averages[8+k]+=Input_Par[4*i+k];
               }
               else{
                  averages2[k]+=Input_Specific_Humidity[4*i+k];
                  averages2[4+k]+=Input_Temperature[4*i+k];
                  averages2[8+k]+=Input_Par[4*i+k];
               }
            }
         }
      }
   }
   // End averaging  
   return true;
}
#endif //FTS


bool SiteData::SetMECHdefault(UserData& data)
{
    int QRES=20;
    for (size_t spp=0;spp<NSPECIES;spp++)
    {
        for (size_t x=0;x<N_CLIMATE;x++)
        {
            for (size_t y=0;y<N_LIGHT;y++)
            {
                An[spp][x][y]=-9999.0;
                Anb[spp][x][y]=-9999.0;
                E[spp][x][y]=-9999.0;
                Eb[spp][x][y]=-9999.0;
                light_levels[spp][y]= exp(-1.0 * y / (1.0 * QRES));
            }
            tf_air[spp][x]=-9999.0;
            tf_soil[spp][x]=-9999.0;
            light_levels[spp][N_LIGHT-1]=0;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! calcPETMonthly
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double SiteData::calcPETMonthly (size_t month) {
   /* Based on Thornthwaite 1948, as presented in Bras, R. L. 1990. Hydrology: *
    * An introduction to hydrologic science. Addison-Wesley. NY. p.224.        */

   /* function used in soil biogeochemistry      */
   /* This function returns monthly pet in mm/yr */

   /* adjustment factor for daylength and month length- lat dependent, this   *
    * value is the average of the values given in table 5.9 on page 225 of    *
    * Bras 1990                                                               */

   double b[12];
   if ( (lat_ <= 5.0) && (lat_ >= -5.0) ) {
      b[0]=1.04; b[1]=0.94; b[2]=1.04; b[3]=1.01; b[4]=1.04; b[5]=1.01;
      b[6]=1.04; b[7]=1.04; b[8]=1.01; b[9]=1.04; b[10]=1.01; b[11]=1.04;
   }
   else if ( (lat_ <= 15.0) && (lat_ >= -15.0) ) {
      b[0]=1.00; b[1]=0.91; b[2]=1.03; b[3]=1.03; b[4]=1.08; b[5]=1.06;
      b[6]=1.08; b[7]=1.07; b[8]=1.02; b[9]=1.02; b[10]=0.98; b[11]=0.99;
   }
   else if ( (lat_ <= 25.0) && (lat_ >= -25.0) ) {
      b[0]=0.95; b[1]=0.90; b[2]=1.03; b[3]=1.05; b[4]=1.13; b[5]=1.11;
      b[6]=1.14; b[7]=1.11; b[8]=1.02; b[9]=1.00; b[10]=0.93; b[11]=0.94;
   }
   else if ( (lat_ <= 32.5) && (lat_ >= -32.5) ) {
      b[0]=0.90; b[1]=0.87; b[2]=1.03; b[3]=1.08; b[4]=1.18; b[5]=1.17;
      b[6]=1.20; b[7]=1.14; b[8]=1.03; b[9]=0.98; b[10]=0.89; b[11]=0.88;
   }
   else if ( (lat_ <= 37.5) && (lat_ >= -37.5) ) {
      b[0]=0.87; b[1]=0.85; b[2]=1.03; b[3]=1.09; b[4]=1.21; b[5]=1.21;
      b[6]=1.23; b[7]=1.16; b[8]=1.03; b[9]=0.97; b[10]=0.86; b[11]=0.85;
   }
   else if ( (lat_ <= 42.5) && (lat_ >= -42.5) ) {
      b[0]=0.84; b[1]=0.83; b[2]=1.03; b[3]=1.11; b[4]=1.24; b[5]=1.25;
      b[6]=1.27; b[7]=1.18; b[8]=1.04; b[9]=0.96; b[10]=0.83; b[11]=0.81;
   }
   else if ( (lat_ <= 47.5) && (lat_ >= -47.5) ) {
      b[0]=0.80; b[1]=0.81; b[2]=1.02; b[3]=1.13; b[4]=1.28; b[5]=1.29;
      b[6]=1.31; b[7]=1.21; b[8]=1.04; b[9]=0.94; b[10]=0.79; b[11]=0.75;
   }
   else if( (lat_ <= 52.5) && (lat_ >= -52.5) ) {
      b[0]=0.74; b[1]=0.78; b[2]=1.02; b[3]=1.15; b[4]=1.33; b[5]=1.36;
      b[6]=1.37; b[7]=1.25; b[8]=1.06; b[9]=0.92; b[10]=0.76; b[11]=0.70;
   }
   /*WARNING! These next latitudes are off the table! values are wrong*/
   else if ( (lat_ <= 90.0) && (lat_ >= -90.0) ) {
      b[0]=0.74; b[1]=0.78; b[2]=1.02; b[3]=1.15; b[4]=1.33; b[5]=1.36;
      b[6]=1.37; b[7]=1.25; b[8]=1.06; b[9]=0.92; b[10]=0.76; b[11]=0.70;
   }

   double I = 0.0; /* annual heat index */
   for(size_t i=0; i<12; i++){
      double T = temp[i];
      /*printf("T %f \t pow(T/5.0,1.51) %f\n",T,pow(T/5.0,1.51));*/

      /* KLUDGE prevent PET calc from crashing,        *
       * cant raise negative numner to the 1.51 power! */
      if (T < 0.0001) T = 0.0001;

      I += pow(T / 5.0, 1.51);
      /*printf("T %f \t pow(T/5.0,1.51) %f\n",T,pow(T/5.0,1.51));*/
   }

   double a = 67.5 * pow(10.0, -8.0) * pow(I, 3.0) - 77.1 * pow(10.0, -6.0)
            * pow(I, 2.0) + 0.0179 * I + 0.492;

   double T = temp[month];
   /* KLUDGE prevent PET calc from crashing;        *
    * cant raise negative number to the 1.51 power! */
   if (T < 0.0001) T = 0.0001;

   /* monthly potential evapotranspiration (cm/month) */
   double pet = 1.62 * b[month] * pow(10.0 * T / I, a);
   pet *= 10.0; /*convert cm to mm*/
   pet *= 12.0; /*convert mm/month to mm/yr*/

#if 0
   printf("pet monthly: month %d b= %f\n",month,b[month]);
   for(i=0;i<12;i++) printf("pet monthly: month %d temp %f pet= %f \n",month,T, pet);
#endif

   return(pet);
}


////////////////////////////////////////////////////////////////////////////////
//! calcSiteDrynessIndex
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void SiteData::calcSiteDrynessIndex (UserData& data) {

   size_t month = 0;
   double di = 0.0;
   if (pet[month] / precip[month] > 1.0) {
      di += 30.0 * (pet[month] - precip[month]);
      int stop = 0;
      if (stop == 0) {
         for (size_t month2=11; month2>month; month2--) {
            if ( (pet[month2] / precip[month2] > 1.0) && (stop == 0) ) {
               di += 30.0 * (pet[month2] - precip[month2]);
            } else {
               stop = 1; // wet month breaks consectuive string
            }
         }
      }

      dryness_index[month] = di;
      //printf("month %d di_month %f\n",month,cs->sdata->dryness_index[month]);
   }

   for (size_t month=1; month<12; month++) {
      if (pet[month] / precip[month] > 1.0) {
         di = dryness_index[month-1] + 30.0 * (pet[month] - precip[month]);
      } else {
         di = 0.0;
      }

      dryness_index[month] = di;
      //printf("month %d di_month %f\n",month,cs->sdata->dryness_index[month]);
   }

   // average
   dryness_index_avg = 0.0;
   for (size_t month=0; month<12; month++) {
      dryness_index_avg += dryness_index[month] / 12.0;
   }

   //printf("di avg %f\n",cs->sdata->dryness_index_avg);
}
#endif /* ED */


////////////////////////////////////////////////////////////////////////////////
//! calcPETAverage
//! Based on Thornthwaite 1948, as presented in Bras, R. L. 1990. Hydrology
//! An introduction to hydrologic science. Addison-Wesley. NY. p.224.
//! function used in soil biogeochemistry, this function returns annual average pet in mm/yr
//! this function is the function of the average adjustment factor for daylength 
//! and month length- lat dependent, this value is the average of the values given in 
//! table 5.9 on page 225 of Bras 1990
//! 
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
void SiteData::calcPETAverage () {
   double b = 1.02;

   double T = temp_average;
   if (T < 0.0001) T = 0.0001; // cant take negative number to the 1.51 power!
   double I = 12 * pow(T / 5.0, 1.51); // annual heat index
   double a = 67.5 * pow(10.0, -8.0) * pow(I, 3.0) - 77.1 * pow(10.0, -6.0)
   * pow(I, 2.0) + 0.0179 * I + 0.492;

   // monthly potential evapotranspiration (cm/month)
   // last two terms convert cm to mm and mm/month to mm/yr
   pet_average = 1.62 * b * pow(10.0 * T / I, a) * 10.0 * 12.0;
   //printf("hydrology: pet yrly average: pet= %f \n",pet_average);
}



////////////////////////////////////////////////////////////////////////////////
//! read_hurricane_disturbance
//! 
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
int read_hurricane_disturbance(site** current_site, UserData* data) {
   int rv, ncid, varid;
   size_t index1[2], index2[3], count[3];

   site* cs = *current_site;

   index1[0] = cs->sdata->lat_ - 25.5;
   index1[1] = cs->sdata->lon_ + 101.5;

   index2[0] = 0;
   index2[1] = cs->sdata->lat_ - 25.5;
   index2[2] = cs->sdata->lon_ + 101.5;

   count[0] = data->n_hurricane_years;
   count[1] = 1;
   count[2] = 1;

   cs->sdata->hurricane_disturbance_rate = (double *)malloc(data->n_hurricane_years * sizeof(double));
   if ((rv = nc_open(data->hurricane_file, NC_NOWRITE, &ncid)))
      NCERR(data->hurricane_file, rv);

   if ((rv = nc_inq_varid(ncid, "average", &varid)))
      NCERR("average", rv);
   if ((rv = nc_get_var1_double(ncid, varid, index1, 
                               &(cs->sdata->avg_hurricane_disturbance_rate))))
      NCERR("average", rv);

   if ((rv = nc_inq_varid(ncid, "damage", &varid)))
      NCERR("damage", rv);
   if ((rv = nc_get_vara_double(ncid, varid, index2, count, 
                               &(cs->sdata->hurricane_disturbance_rate[0]))))
      NCERR("damage", rv);

   if (cs->sdata->avg_hurricane_disturbance_rate <= 0.0) {
      //fprintf(stderr, "No hurricane data found for site lat %f lon %f\n", cs->lat, cs->lon);
      return 0;
   } 

#if 0 /*temp set avg to 0 for landuse runs from potential -- needs it for years before hurr rec*/
   cs->sdata->avg_hurricane_disturbance_rate = 0;
#endif
   if ((rv = nc_close(ncid)))
      NCERR(data->hurricane_file, rv);  

   return 1;
}

/******************************************************************************/
