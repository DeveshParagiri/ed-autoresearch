#include <iostream>
#include "mpi.h"

#include "edmodels.h"

#include "edmpi.h"

using namespace std;


void init_mpi (UserData& data) {

   //Initialize the MPI environment
   MPI::Init();
   data.mpi_rank = MPI::COMM_WORLD.Get_rank();
   data.mpi_nproc = MPI::COMM_WORLD.Get_size();

   cout << "This is " << data.mpi_rank << " of " <<  data.mpi_nproc << "\n";
}


void narrow_lonbounds_to_proc (UserData& data) {   
   // count the total number of land cells
   size_t count = 0;
   for (size_t y=0; y<data.n_lat; y++) {
      for (size_t x=0; x<data.n_lon; x++) {
         if (data.wtr_ice_f[y][x] < 1.0) {
            count++;
         }
      }
   }

   size_t n_per_node = count / data.mpi_nproc;

   size_t node = 0;
   size_t start_lon = data.start_lon;
   count = 0;
   for (size_t x=0; x<data.n_lon; x++) {
      if (node == data.mpi_nproc-1) {
         // last node, just take the rest
         data.n_lon = data.n_lon - start_lon;
         break;
      }
      if (count > n_per_node * (node+1)) {
         if (node == data.mpi_rank) {
            // if this is us, set bounds
            data.n_lon = x - start_lon;
            break;
         } else {
            // move on to next processor
            start_lon = x;
            node++;
         }
      } 
      for (size_t y=0; y<data.n_lat; y++) {
         if (data.wtr_ice_f[y][x] < 1.0) {
            count++;
         }
      }
   }
   double* tmp = (double*)malloc(data.n_lon*sizeof(double));
   memcpy(tmp, &data.lons[start_lon - data.start_lon], data.n_lon*sizeof(double));
   free(data.lons);
   data.lons = tmp;
   data.start_lon = start_lon;
}

void mpi_collect_data (UserData& data) {
   MPI::COMM_WORLD.Barrier();
   size_t test_sum;
   MPI::COMM_WORLD.Reduce(&data.mpi_rank, &test_sum, 1, MPI::UNSIGNED_LONG, MPI::SUM, 0);
   if (data.mpi_rank == 0) {
      cout << "MPI::Reduce test: " << test_sum << "\n";
   }
}


#if 0
double* aggregate_var (site *firstsite, UserData *data) {

   if (data->mpi_rank != 0) {
      double tmp[data->n_lat][data->n_lon];
      for (int y=0; y<n_lat; y++)
         for (int x=0; x<n_lon; x++)
            tmp[y][x] = -9.9;
      
      site *cs = firstsite;
      while (cs != NULL) {
         tmp[cs->y][cs->x] = *get(cs);
         cs = cs->next_site;
      }
      
      MPI_Send();
      return NULL;
   } else {
      
      double *data = new float[data->n_lat_reg][data->n_lon_reg];

      MPI_Receive();

      // merge into data
      // add our own data
      return data;
   }
}
#endif


