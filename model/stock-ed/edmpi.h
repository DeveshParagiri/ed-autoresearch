#ifndef EDMPI_H_
#define EDMPI_H_

struct UserData;


void init_mpi (UserData& data);
void narrow_lonbounds_to_proc (UserData& data);
void mpi_collect_data (UserData& data);

#endif

