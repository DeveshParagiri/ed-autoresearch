/******************************************************************************
 *****                    Program to run Miami Model                      *****
 *****                    G. Hurtt, 970717, 970929                        *****
 ******************************************************************************/
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
//! miami
//! Miami Model (Leith, 1972)
//!
//! @param  
//! @return 
////////////////////////////////////////////////////////////////////////////////
double miami (double precipitation, double temperature) {
   double npp_final, npp_temp, npp_precip;

   npp_precip = 3000.0 * ( 1.0 - exp( -0.000664 * precipitation ) );
   npp_temp   = 3000.0 / ( 1.0 + exp( 1.315 - 0.119 * temperature ) );

   if( npp_temp <=  npp_precip ) npp_final = npp_temp;
   else npp_final = npp_precip;
 
   /* convert grams dry matter to carbon units in KG *  
    * King et al., 1997, Climate Change 35:199-227   */
   npp_final *= 0.45 / 1000.0;
  
   return npp_final;
}

/******************************************************************************/
