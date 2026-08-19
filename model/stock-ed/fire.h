#ifndef EDM_FIRE_H_
#define EDM_FIRE_H_

struct patch;
struct UserData;

//void read_gfed_bf  (UserData* data);
double fire (int t, patch** patchptr, UserData* data);

void update_fuel(int t, patch** patchptr, UserData* data);


#endif // EDM_FIRE_H_ 
