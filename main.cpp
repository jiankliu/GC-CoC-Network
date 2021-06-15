
/* 
 This is the model to simulate a neural network with GC and GoC.
 */

#include <omp.h>
//#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
//#include <conio.h>  //  getch();   
#include <cstdlib>  // pause

using namespace std;

#include "param.h"          // file for all parameters 
#include "synapse.h"

//------------Number of ODE for each cell -------------------------------
const int N_EQ_G = 4;  //  6 if we solve nmda
const int N_EQ_M = 2;  //  
const int N_EQ_Go = 4; // 3 if Ex-model
const int N_EQ_U = 5; 

#define N_EQ   ( N_EQ_Go*NGoC + N_EQ_G*NGC + N_EQ_U*NUBC + N_EQ_M*NMF )    //  Complete number of ODE
//+++++++++++++++++++ MAIN PROGRAM +++++++++++++++++++++++++++++++++++++++++++

//----------external variables ---------------------------------------------
int no_GC[NGC][N_EQ_G],   no_UBC[NUBC][N_EQ_U], no_GoC[NGoC][N_EQ_Go],
    no_eMF[NMF][N_EQ_M];  

MatDoub g_eG(NGC,N_eG, 0.),    g_iG(NGC,N_iG, 0.),    g_GoG(NGC,N_GoG, 0.), 
        g_MU(NUBC,N_MU, 0.),   g_UU(NUBC,N_UU, 0.),   g_GoU(NUBC,N_GoU, 0.),
        g_MGo(NGoC,N_MGo, 0.), g_UGo(NGoC,N_UGo, 0.), g_GGo(NGoC,N_GGo, 0.),
		g_GG(NGoC,N_GG, 0.);
MatInt pre_eG(NGC,N_eG, 0),  pre_iG(NGC,N_iG, 0), pre_GoG(NGC,N_GoG, 0), 
       pre_MU(NUBC,N_MU, 0), pre_UU(NUBC,N_UU, 0),pre_GoU(NUBC,N_GoU, 0),
       pre_MGo(NGoC,N_MGo, 0),  pre_UGo(NGoC,N_UGo, 0), pre_GGo(NGoC,N_GGo, 0),
	   pre_GG(NGoC,N_GG, 0); 

// learning parameters
VecDoub d_GG(NGoC, 0.), d_GCGC(NGC, 0.);


//----------external classes (beginning of initialization)------------------
UBCIFB2 UBC[NUBC];
//GCIAF  GC[NGC];  
//GoCIAF GoC[NGoC];  
GCBRUNEL GC[NGC];  
GoCBRUNEL GoC[NGoC];  

//mossy firbers
MFIAF  eMF[NMF];

// postsynapse currernt 
IGo   *syn_GoG[NGC];    // GoC->GC
IGo   *syn_GoU[NUBC];   // GoC->UBC
IMFGC *syn_iG[NGC];     // iMF -> GC
IMFGC *syn_eG[NGC];     // eMF->GC
IMU   *syn_MU[NUBC];    // MF->UBC
IMFGC *syn_MGo[NGoC];   // MF ->GoC
IMFGC *syn_UGo[NGoC];   // iMF ->GoC
IMU   *syn_UU[NUBC];    // UBC->UBC
IMF   *syn_GGo[NGoC];   // GC ->GoC
IGG   *syn_GG[NGoC];    // GoC ->GoC

//   -----external functions----------------------------------------------
void fun(double x, double *y_ini, double *f_ini);
void rkForth(int n, void fun(double, double*, double*),
		double h, double x, double* y, double* f, double* s, double* yk);
void rkSecond(int n, void fun(double, double*, double*),
		double h, double x, double* y, double* f, double* s, double* yk);

void euler(int n, void fun(double, double*, double*),
		double h, double x, double* y, double* f);

////////////////////////////////////////////////////////////////////////////////////
//++++++Main program+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////////////////////////////////////////////////////////////////////////////////////

int main(int argc,char **argv) 
{
	//MPI
	int num_procs;
	double zz,ww;

	// running time counter
	clock_t start,finish;
	double totaltime;
	//start = clock();
	start = omp_get_wtime();

	FILE *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *f9, *f10, *f11, *f12, *f71;

	//---------allocate place for ALL variables and functions-----------
	double y_ini[N_EQ], f_ini[N_EQ];

	//---------allocate place for TWO temporal arrays using by RK.C solver 
	double y1[N_EQ], y2[N_EQ];

	//---------general parameters----------------------------------------------
	double t = 0, h = DT;
	int i, j, k, ii = 0;


	
   //
   //  How many processors are available?
   //
   num_procs = omp_get_num_procs ( );

   cout << "\n";
   cout << "  The number of processors available:\n";
   cout << "  OMP_GET_NUM_PROCS () = " << num_procs << "\n";
   
  int nthreads, thread_id;
#if TEST
  nthreads = 2;
#else
  nthreads = 4;
#endif
  omp_set_num_threads ( nthreads );

// Fork the threads and give them their own copy of the variables 
#pragma omp parallel private(nthreads, thread_id)
  {
    
    //* Get the thread number 
    thread_id = omp_get_thread_num();
    printf("Thread %d says: Hello World\n", thread_id);

    if (thread_id == 0)     // This is only done by the master 
	{
		nthreads = omp_get_num_threads();
		printf("Thread %d reports: the number of threads are %d\n", thread_id, nthreads);
	}
	
  }    // Now all threads join master thread and they are disbanded 


  
  /* Seed the random-number generator with current time so that
    * the numbers will be different every time we run.
    */
   //srand( (unsigned)time( NULL ) );
   srand( SEED );

   //----------arrays initialization----------------------------------------------
#pragma omp parallel private(i,j,k)
{

#pragma omp for nowait
   for(i=0; i<N_EQ; i++){
	   y_ini[i] = 0,  f_ini[i] = 0; 
	   y1[i] = 0,     y2[i] = 0;
   }


   //----------classes initialization (continue)----------------------------
   // depend on the order of defination synapse class
#pragma omp for nowait
   for(i=0; i < NGC; i++) {
	   syn_iG[i] = new IMFGC[N_iG];
	   syn_eG[i] = new IMFGC[N_eG];
	   syn_GoG[i] = new IGo[N_GoG];
   }

#pragma omp for nowait
   for(i=0; i < NUBC; i++) {
	   syn_MU[i] = new IMU[N_MU];
	   syn_GoU[i] = new IGo[N_GoU];
	   syn_UU[i] = new IMU[N_UU];
   }

#pragma omp for nowait
   for(i=0; i < NGoC; i++) {
	   syn_MGo[i] = new IMFGC[N_MGo];
	   syn_UGo[i] = new IMFGC[N_UGo];
	   syn_GGo[i] = new IMF[N_GGo];
	   syn_GG[i]  = new IGG[N_GG];
   }

   //----------creating the integer arrays containing the addresses--------
   //----------of  ALL internal variables for ALL objects RE, TC ----------
   //----------and GB classes (e.g., no_re[i][j][k] is the address --------	
   //----------of the variable y[k] for the object re_cell[i][j]) ---------
   //----------NOTE: this is the relative addresses and you should ---------
   //----------add the REAL address of the first element of the -----------
   //----------original 1D array to use these arrays----------------------- 
#pragma omp for nowait
   for(i=0; i < NGC; i++)
	   for(k=0; k < N_EQ_G; k++)
		   no_GC[i][k] = k + i * N_EQ_G;

#pragma omp for nowait
   for(i=0; i < NUBC; i++)
	   for(k=0; k < N_EQ_U; k++)
		   no_UBC[i][k] = NGC*N_EQ_G + k + i * N_EQ_U;  

#pragma omp for nowait
   for(i=0; i < NMF; i++)
	   for(k=0; k < N_EQ_M; k++)
		   no_eMF[i][k] = NGC*N_EQ_G + NUBC*N_EQ_U + k + i * N_EQ_M;  

#pragma omp for nowait
   for(i=0; i < NGoC; i++)
	   for(k=0; k < N_EQ_Go; k++)
		   no_GoC[i][k] = NGC*N_EQ_G + NUBC*N_EQ_U + NMF*N_EQ_M + k + i * N_EQ_Go;  

#pragma omp barrier

   //---variable initialization (additional for standard constructor)----------
#pragma omp for nowait
   for(i=0; i<NGC; i++) {
	   GC[i].init(y_ini+no_GC[i][0]);
	   GC[i].reset();
   }

#pragma omp for nowait
   for(i=0; i<NUBC; i++) {
	   UBC[i].init(y_ini+no_UBC[i][0]);
	   UBC[i].reset();
   }
	 
#pragma omp for nowait
   for(i=0; i<NMF; i++) {
	   eMF[i].init(y_ini+no_eMF[i][0]);
	   eMF[i].reset();
   }
#pragma omp for nowait
   for(i=0; i<NGoC; i++) {
	   GoC[i].init(y_ini+no_GoC[i][0]);
	   GoC[i].reset();
   }
} //end omp

   //--------------open ALL files-------------------------------------
   if(    ( f1 = fopen("RasterGC.dat", "w") ) == NULL
	   || (f2 = fopen("RasterMF.dat", "w")) == NULL
	   || ( f12 = fopen("RasterGoC.dat", "w") )      ==NULL
	   || ( f3 = fopen("connection.dat", "w") )==NULL
	   || ( f4 = fopen("wcon.dat", "w") )==NULL
	   || ( f5 = fopen("STIM.dat", "w") )==NULL
	   || ( f6 = fopen("PSC.dat", "w") )==NULL
	   || ( f8 = fopen("W.dat", "w") )==NULL
	   || ( f9 = fopen("EPSC.dat", "w") )==NULL
	   || (f10 = fopen("meanrate.dat", "w")) == NULL
	   || (f11 = fopen("rate.dat", "w")) == NULL
	   || (f71 = fopen("LFP.dat", "w")) == NULL
	   || ( f7 = fopen("Voltage.dat", "w") )==NULL){
		   printf("can't open files\n");
		   exit(0);
   }

   //fprintf(f6, "%4d\n",9999 );
  // fprintf(f7, "%4d\n",9999 );
   fprintf(f8, "%4d %4d \n", 9999, 0 );

   //----------Connection matrix-------------------------------
   printf("\n Begin Connect Matrix");

   //-------------- Network topology ------------------------------------
   int pre, exists;

   for(i=0; i<NGC; i++) {
	   // MF -> GC
	   for(j=0; j<N_eG; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			   pre = GetRandom(NMF);
			   //pre = GetRandom(NMF / 2);
			   for (k=0;k<j;k++) if (pre_eG[i][k]==pre) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_eG[i][j]= pre;
	   }
	   // iMF -> G
	   for(j=0; j<N_iG; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			   pre = GetRandom(NUBC);
			   for (k=0;k<j;k++) if (pre_iG[i][k]==pre) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_iG[i][j]= pre;
		   
	   }
	   // GoC -> GC
	   for(j=0; j<N_GoG; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			   pre = GetRandom(NGoC);
			   for (k=0;k<j;k++) if (pre_GoG[i][k]==pre) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_GoG[i][j]= pre;
	   }
   }


   for(i=0; i<NGoC; i++) { 
	   // MF -> GoC 
	   for(j=0; j<N_MGo; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			  pre = GetRandom(NMF);
			   //pre = getrandom(NMF/2,NMF);

			   for (k=0;k<j;k++) if (pre_MGo[i][k]==pre||pre==NMF) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_MGo[i][j]= pre;
		  // printf("\n Pre= %4d ",pre);
		   fprintf(f3, "%5d %5d \n", i,pre);
	   }
	   // UBC -> Go
	   for(j=0; j<N_UGo; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			   pre = GetRandom(NUBC);
			   for (k=0;k<j;k++) if (pre_UGo[i][k]==pre) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_UGo[i][j]= pre;
	   }
	   // GC -> GoC
	   for(j=0; j<N_GGo; j++) {
		   do{
			   exists = 0;		// avoid multiple synapses
			   pre = GetRandom(NGC);
			   for (k=0;k<j;k++) if (pre_GGo[i][k]==pre) exists = 1;	// synapse already exists  
		   }while (exists == 1);
		   pre_GGo[i][j]= pre;
	   }
   }
		   
	   // GoC -> UBC
	   for(i=0; i<NUBC; i++) {
		   for(j=0; j<N_GoU; j++) {
			   do{
				   exists = 0;		// avoid multiple synapses
				   pre = GetRandom(NGoC);
				   for (k=0;k<j;k++) if (pre_GoU[i][k]==pre) exists = 1;	// synapse already exists  
			   }while (exists == 1);
			   pre_GoU[i][j]= pre;
		   }
	   }
	   
	   //UBC-UBC
	  for(i=0; i<NUBC; i++) {
		  for(j=0; j<N_UU; j++) {
			  do{
				  exists = 0;		// avoid multiple synapses
				  pre = GetRandom(NUBC);
				  if (pre==i) exists=1;  	  	     				// no self-synapses 
				  for (k=0;k<j;k++) if (pre_UU[i][k]==pre) exists = 1;	// synapse already exists  
			  }while (exists == 1);
			  pre_UU[i][j]= pre;
			  //fprintf(f3,"%5d \n", pre );	   //save pre cell id
		  }
	  }
	 // fprintf(f3,"%5d \n", 9999 );

	  // MF -> UBC 
	  for(i=0; i<NUBC; i++) 
		  for(j=0; j<N_MU; j++) {
			   do{
				   exists = 0;		// avoid multiple synapses
				   pre = GetRandom(NMF);
				   for (k=0;k<j;k++) if (pre_MU[i][k]==pre) exists = 1;	// synapse already exists  
			   }while (exists == 1);
			   pre_MU[i][j]= pre;
		  }
   

	
	  // 2D space local connected GoC -> GoC gap junction with some prob.
      //start from 1, let Neuron[0] be empty
	  for (i=0;i<NGoC; i++) {
		  for (j=0;j<N_GG;j++) {
			  do{
				  exists = 0;		// avoid multiple synapses				   
				  pre = GetRandom(NGoC);
				  if (pre==i)  exists=1;  	  	      			        	// no self-synapses 
				  for (k=0;k<j;k++) if ((pre_GG[i][k]==pre) ) exists = 1;	// synapse already exists  
			  }while (exists == 1);
			  pre_GG[i][j]= pre;
			  fprintf(f4,"%5d \n", pre );	   //save pre cell id
		  }
	  }
	  fprintf(f4,"%5d \n", 9999 );
	  printf("\n End Connect Matrix");


   //--------the initial conductances----------------------------------- 
   // normal distribution
   double weight,weight1;
   int fir = 1;
   for (weight = 3; weight <= 3; weight = weight + 0.5) {
	   cout << "  Weight  = " << weight << "\n";
	   for (weight1 = 4; weight1 <= 4; weight1 = weight1 + 0.5) {
		   double WiG = weight1;
		   Normaldev grandWiG(WiG, WiG * var, SEED);

		   double WeG = weight1;
		   Normaldev grandWeG(WeG, WeG * var, SEED);

		   double WGoG = weight;
		   Normaldev grandWGoG(WGoG, WGoG * var, SEED);

		   double WMU = wmu;
		   Normaldev grandWMU(WMU, WMU * varUBC, SEED);

		   double WGoU = wgou;
		   Normaldev grandWGoU(WGoU, WGoU * var, SEED);

		   double WUU = wuu;
		   Normaldev grandWUU(WUU, WUU * varUBC, SEED);

		   double WMGo = wmgo;
		   Normaldev grandWMGo(WMGo, WMGo * var, SEED);

		   double WUGo = wugo;
		   Normaldev grandWUGo(WUGo, WUGo * var, SEED);

		   double WGGo = wggo;
		   Normaldev grandWGGo(WGGo, WGGo * var, SEED);

		   double WGG = wgg;
		   Normaldev grandWGG(WGG, WGG * var, SEED);

		   for (i = 0; i < NGC; i++) {
			   for (j = 0; j < N_iG; j++) {
#if TEST
				   g_iG[i][j] = WiG;
#else
				   g_iG[i][j] = grandWiG.dev();
				   if (g_iG[i][j] < 0.)    g_iG[i][j] = GetRandomDoub(2.*WiG);
#endif
			   }

			   for (j = 0; j < N_eG; j++) {
#if TEST
				   g_eG[i][j] = WeG;
#else
				   g_eG[i][j] = grandWeG.dev();
				   if (g_eG[i][j] < 0.)    g_eG[i][j] = GetRandomDoub(2.*WeG);
#endif
			   }
			   for (j = 0; j < N_GoG; j++) {
#if TEST
				   g_eG[i][j] = WeG;
#else		
				   g_GoG[i][j] = grandWGoG.dev();
				   if (g_GoG[i][j] < 0.)    g_GoG[i][j] = GetRandomDoub(2.*WGoG);
#endif
			   }


		   }

		   for (i = 0; i < NUBC; i++) {
			   for (j = 0; j < N_MU; j++) {
				   g_MU[i][j] = WMU;
			   }
			   for (j = 0; j < N_GoU; j++) {
				   g_GoU[i][j] = WGoU;
			   }
			   for (j = 0; j < N_UU; j++) {
				   g_UU[i][j] = WUU;
			   }
		   }

		   Normaldev bigMGo(15, 15.*var, SEED);
		   for (i = 0; i < NGoC; i++) {
			   for (j = 0; j < N_MGo; j++) {
#if TEST
				   g_MGo[i][j] = WMGo;
#else
				   g_MGo[i][j] = grandWMGo.dev();
				   if (g_MGo[i][j] < 0.)    g_MGo[i][j] = GetRandomDoub(2.*WMGo);
				   //		   if (j==0) {
				   //			   //g_MGo[i][j]   = bigMGo.dev()*WMGo;
				   //			   g_MGo[i][j]   = (GetRandomDoub(4.)+2.)*WMGo;
				   ////		   } else if ( 0<j && j<3) {
				   ////			   g_MGo[i][j]   = (GetRandomDoub(13.)+2.)*WMGo;
				   //		   } else {
				   //			   g_MGo[i][j]   = grandWMGo.dev();
				   //			   if ( g_MGo[i][j] < 0. )    g_MGo[i][j] = GetRandomDoub(2.*WMGo);
				   //		   }
#endif
			   }
			   for (j = 0; j < N_GGo; j++) {
				   g_GGo[i][j] = grandWGGo.dev();
				   if (g_GGo[i][j] < 0.)    g_GGo[i][j] = GetRandomDoub(2.*WGGo);
			   }


			   for (j = 0; j < N_UGo; j++) {
				   g_UGo[i][j] = WUGo;
			   }
		   }
		   if (fir == 1) {
			   VecDoub x(NGoC, 0.), y(NGoC, 0.), density(NGoC, 0.0);
			   double dmean = 400 / sqrt((double)NGoC);
			   double rmean = 70.0; // 35; 70;
			   double densitymean = 1.0;
			   //Normaldev grand( 10.0, 10.0*0.1, SEED+i); // radius of each cell's density
			   k = 0;
			   for (i = 0; i < NGoC; i++) {
				   if ((i % (int)sqrt((double)NGoC)) == 0) k = k + 1;
				   x[i] = (fmod((double)i, sqrt((double)NGoC)) + 1.0) * dmean + dmean * (GetRandomDoub(0.5) - 0.25);
				   //  fprintf(f4,"%lf\n", x[i] );
				   y[i] = k * dmean + dmean * (GetRandomDoub(0.5) - 0.25);
				   //fprintf(f4,"%lf\n", y[i] );
				   d_GG[i] = rmean * (1.0 + GetRandomDoub(0.6) - 0.3);
				   //fprintf(f4,"%lf\n", d_GG[i] );
				   density[i] = densitymean * (1.0 + GetRandomDoub(1.0) - 0.5);
				   //fprintf(f4,"%lf\n", density[i] );
			   }
			   //fprintf(f4,"%5d \n", 9999 );

			   double xy = 0.0;
			   double lap = 0.0, dummy1 = 0.0, dummy2 = 0.0;
			   for (i = 0; i < NGoC; i++) {
				   for (j = 0; j < N_GG; j++) {
					   pre = pre_GG[i][j];
					   xy = sqrt(pow(x[i] - x[pre], 2.0) + pow(y[i] - y[pre], 2.0));
					   if (xy > d_GG[i] + d_GG[pre]) { // no intersection
						   g_GG[i][j] = 0.0;
					   }
					   else if (xy < fabs(d_GG[i] - d_GG[pre])) { // one circle is contained within the other.
						   dummy1 = GetRandomDoub(1.0)*100.;
						   dummy2 = -1745. + 1836. / (1 + exp((xy - 267.) / 39.));
						   if (dummy1 <= dummy2) {
							   g_GG[i][j] = WGG * density[pre] * density[i] * (PI * pow(MIN(d_GG[i], d_GG[pre]), 2.0)) / (PI * pow(d_GG[pre], 2.0) + PI * pow(d_GG[i], 2.0));
						   }
						   else {
							   g_GG[i][j] = 0.0;
						   }

						   //normalized with postsynaptic neuron
						   //g_GG[i][j] = WGG * (PI * pow(MIN( d_GG[i], d_GG[pre] ),2.0) ) / ( PI * pow(d_GG[i],2.0) );   
					   }
					   else {
						   dummy1 = 1. / (2. * d_GG[i] * xy) * (pow(d_GG[i], 2.0) - pow(d_GG[pre], 2.0) + pow(xy, 2.0));
						   dummy1 = 2. * acos(dummy1);
						   dummy1 = (1. / 2.)* pow(d_GG[i], 2.0) * (dummy1 - sin(dummy1));

						   dummy2 = 1. / (2. * d_GG[pre] * xy) * (pow(d_GG[pre], 2.0) - pow(d_GG[i], 2.0) + pow(xy, 2.0));
						   dummy2 = 2. * acos(dummy2);
						   dummy2 = (1. / 2.)* pow(d_GG[pre], 2.0) * (dummy2 - sin(dummy2));

						   //normalized with both pre- and post-synaptic neuron to get symmetric conductance
						   lap = (dummy1 + dummy2) / (PI * pow(d_GG[pre], 2.0) + PI * pow(d_GG[i], 2.0));

						   //normalized with postsynaptic neuron
						   //lap = ( dummy1 + dummy2 ) / ( PI * pow(d_GG[i],2.0) );

						   dummy1 = GetRandomDoub(1.0)*100.;
						   dummy2 = -1745. + 1836. / (1 + exp((xy - 267.) / 39.));
						   if (dummy1 <= dummy2) {
							   g_GG[i][j] = WGG * lap * density[pre] * density[i];
						   }
						   else {
							   g_GG[i][j] = 0.0;
						   }

					   }
					  // fprintf(f4, "%lf\n", g_GG[i][j]);
				   }
			   }
			   //fprintf(f4, "%5d \n", 9999);
		   }
		   fir = fir + 1;
		   printf("\n GoC network is done ! ");

		   //**********************************************************

		   //**********************************************************
		   // local connected GC->GoC and GoC->GC.
		   // 2D space
		   VecDoub xGC(NGC, 0.), yGC(NGC, 0.), densityGC(NGC, 0.0);
		   double dmeanGC = 433. / sqrt((double)NGC);
		   double rmeanGC = 2.*dmeanGC;
		   k = 0;
		   for (i = 0; i < NGC; i++) {
			   if ((i % (int)sqrt((double)NGC)) == 0) k = k + 1;
			   xGC[i] = (fmod((double)i, sqrt((double)NGC)) + 1.0) * dmeanGC + dmeanGC * (GetRandomDoub(0.5) - 0.25);
			   fprintf(f4, "%lf\n", xGC[i]);
			   yGC[i] = k * dmeanGC + dmeanGC * (GetRandomDoub(0.5) - 0.25);
			   fprintf(f4, "%lf\n", yGC[i]);
			   d_GCGC[i] = rmeanGC * (1.0 + GetRandomDoub(0.6) - 0.3);
			   fprintf(f4, "%lf\n", d_GCGC[i]);
		   }
		   fprintf(f4, "%5d \n", 9999);

		   int kk = 0;
		   // g_GGo g_PrePost
		//   for (i=0;i<NGoC; i++) {
		//	   k = 0;
		//	   for (j=0;j<NGC;j++) {
		//		   xy = sqrt( pow( x[i] - xGC[j], 2.0 ) + pow(y[i] - yGC[j], 2.0 ) );
		//		   if ( xy <= d_GG[i] + d_GCGC[j] ) { // no intersection
		//			   dummy1 = GetRandomDoub(1.0);
		//			   if (dummy1 <=0.3){
		//				   pre_GGo[i][k]= j;
		//#if TEST
		//				   g_GGo[i][k] = WGGo;
		//#else
		//				   g_GGo[i][k] = grandWGGo.dev();
		//				   if ( g_GGo[i][k] < 0. )    g_GGo[i][k] = GetRandomDoub(2.*WGGo);
		//#endif
		//				   k++;
		//			   }
		//		   }
		//	   }
		//	   kk = MAX(kk,k);
		//	   for (j=0;j<N_GGo;j++) {
		//		   fprintf(f4,"%lf\n", g_GGo[i][j] );
		//	   }
		//   }
		  // fprintf(f4,"%5d \n", 9999 );
		  // printf("\n max No. GC per GoC=%4d", kk);

		   kk = 0;
		   // g_GoG g_PrePost
		//   for (i=0;i<NGC; i++) {
		//	   k = 0;
		//	   for (j=0;j<NGoC;j++) {
		//		   xy = sqrt( pow( xGC[i] - x[j], 2.0 ) + pow(yGC[i] - y[j], 2.0 ) );
		//		   if ( xy <= d_GG[j] + d_GCGC[i] ) { // no intersection
		//			   dummy1 = GetRandomDoub(1.0);
		//			   if (dummy1 <=2) {
		//				   pre_GoG[i][k]= j;
		//#if TEST
		//				   g_GoG[i][k] = WGoG;
		//#else
		//				   g_GoG[i][k] = grandWGoG.dev();
		//				   if ( g_GoG[i][k] < 0. )    g_GoG[i][k] = GetRandomDoub(2.*WGoG);
		//#endif
		//				   k++;
		//			   }
		//		   }
		//	   }
		//	   kk = MAX(kk,k);
		//	   for (j=0;j<N_GoG;j++) {
		//		   fprintf(f4,"%lf\n", g_GoG[i][j] );
		//	   }
		//   }
		   fprintf(f4, "%5d \n", 9999);
		   printf("\n max No. GoC per GC=%4d", kk);
		   printf("\n GC network is done ! ");
		   //**********************************************************

		   // select eMF or iMF for each GC
		   double c1 = 0;
		   for (i = 0; i < NGC; i++) {
			   c1 = 0;
			   for (j = 0; j < c1; j++) {
				   g_eG[i][j] = 0;
			   }
			   for (j = c1; j < N_eG; j++) {
				   g_iG[i][j] = 0;
			   }
		   }

		   // select eMF or iMF for each UBC
		   for (i = 0; i < NUBC; i++) {
			   //if (i>=MFinput) g_MU[i][0]   = 0; 
			   //else g_UU[i][0]   = 0; 
			   c1 = 0;
			   for (j = 0; j < c1; j++) {
				   g_MU[i][j] = 0;
			   }
			   for (j = c1; j < N_MU; j++) {
				   g_UU[i][j] = 0;
			   }
		   }

		   VecInt STIM(NINPUT);
		   if (RANDIN == 1) {
			   //randome inputs
			   for (j = 0; j < NINPUT; j++) {
				   do {
					   exists = 0;		// avoid multiple stimuli cells 
					   pre = GetRandom(NGoC);
					   for (k = 0; k < j; k++) if (STIM[k] == pre) exists = 1;	// cell already exists  
				   } while (exists == 1);
				   STIM[j] = pre;
				   // record stimulated neuon ID
				   //fprintf(f5,"%5d \n", pre );
			   }
		   }

		   int RANDSTIM = 1;
		   //0 - no stim
		   //1 - single stim
		   //2 - local stim
		   //3 - rand subnet stim

		   //if (RANDSTIM == 0) {
		   //		// KA stimulated network
		   //}else if  (RANDSTIM == 1) {
		   //	//single cell stimulation
		   //	g_MGo[70][j] = WMGo; 
		   //}else if (RANDSTIM == 2) {
		   //	for(i=0; i<NGoC; i++) {
		   //		// stimulated local subnetwork
		   //		if ( i> 60 & i< 90 ) { // no GC -> GoC
		   //		//if ( i> 10 & i< 21 ) { // no GC -> GoC
		   //			for(j=0; j<N_MGo; j++) {
		   //				g_MGo[i][j] = WMGo; 
		   //			} 
		   //		}
		   //	}
		   //} else {
		   //	for(k=0; k<NINPUT; k++) {
		   //		// stimulated rand subnetwork
		   //		for(j=0; j<N_MGo; j++) {
		   //			g_MGo[ STIM[k] ][j]   = WMGo; 
		   //		} 
		   //	}
		   //}
		//  printf("\n Stimulus is done ! ");


		  //---changes of the parameters to get variability---------------------------
		   double Starttime = starttime;
		   Normaldev grandstart(Starttime, 3 * Starttime * var, SEED);
		   int cy;
		   double meanfre;
		   //#if MFSTIM
		   for (cy = 25; cy <= 25; cy = cy + 1) {
			   t = 0.; ii = 0;
			   meanfre = 1000 / cy;
			   for (i = 0; i < NMF; i++) {
				   eMF[i].TYPE = 1;  //randomly select TYPE I or II
				   eMF[i].k = 1.0;
				   //MF-UBC
				   //eMF[i].trecAMPA = TRECAMPA; //t1.dev(); 
				   //eMF[i].trecNMDA = TRECNMDA; //t2.dev(); 
				   //eMF[i].para();

				   eMF[i].slow_invl = meanfre;//Poisson mean ISI
				   eMF[i].start = 0; //ST*T+450;  //500;
				  /* if (i < NMF / 2) {
					   eMF[i].start = grandstart.dev() +15.0;
				   }else{
					   eMF[i].start = grandstart.dev();
				   }*/

				   //eMF[i].start = 0;
				   eMF[i].end = TMAX; //ST*T+1500; //2550;
				   eMF[i].fast_invl = 11.1;// len;   //ST*T+1500; //2550;
				   eMF[i].burst_len = 51;// len * 5;    // spike number, 1 for single stimulus 5:2Brurst;8:3Burst
										  //1->28 5->45 10->90 15->130

				   eMF[i].noise = 0;
				   eMF[i].reset_input();
				   eMF[i].lastspk = -9999;
				   eMF[i].q = 0;
				   eMF[i].init(y_ini + no_eMF[i][0]);
			   }
			   //#else
			   //   for(i=0; i<NMF; i++) {
			   //	   eMF[i].TYPE = 1;  //randomly select TYPE I or II
			   //	   eMF[i].k =  1.0;
			   //	   eMF[i].trecAMPA = TRECAMPA; //t1.dev(); 
			   //	   eMF[i].trecNMDA = TRECNMDA; //t2.dev(); 
			   //	   eMF[i].para();
			   //
			   //	   eMF[i].slow_invl = 20;//T;
			   //	   eMF[i].start      = ST*T+450;  //500;
			   //	   eMF[i].end        = TMAX; //ST*T+1500; //2550;
			   //	   eMF[i].fast_invl  = 1;   //ST*T+1500; //2550;
			   //	   eMF[i].burst_len  = 20;    // spike number, 1 for single stimulus 
			   //
			   //	   eMF[i].noise = 0;
			   //	   eMF[i].reset_input();
			   //   }
			   //#endif

			   Normaldev grand2(THR_UBC, fabs(THR_UBC * 0.05), SEED);
			   //   for(i=0; i<NUBC; i++) {
			   //#if TEST
			   //	   UBC[i].Vthr = THR_UBC;
			   //#else
			   //	   UBC[i].Vthr = grand2.dev();
			   //#endif
			   //	   UBC[i].trecAMPA = TRECAMPA; //t1.dev(); 
			   //	   UBC[i].trecNMDA = TRECNMDA; //t2.dev(); 
			   //	   UBC[i].para();
			   //	   UBC[i].GATE_AMPA = 0.0;
			   //	   UBC[i].GATE_M2 = 1.0;
			   //	   UBC[i].GATE_M1 = 1.0;
			   //	   UBC[i].init(y_ini+no_UBC[i][0]);
			   //	   UBC[i].reset();
			   //
			   //	   for(j = 0; j < N_GoU; ++j){
			   //		   syn_GoU[i][j].mix =  1; //UBC: only fast component
			   //	   }
			   //	   for(j = 0; j < N_UU; ++j){
			   //		   syn_UU[i][j].AMPA_NMDA_RATIO = NAR_MU; //slow component for iMF
			   //		   syn_UU[i][j].AMPA_AMPASL_RATIO = NAR_SL; //slow component for iMF
			   //		   syn_UU[i][j].AMPA_MGLUR2_RATIO = NAR_M2U; //slow component for iMF
			   //		   syn_UU[i][j].AMPA_MGLUR1_RATIO = NAR_M1U; //slow component for iMF
			   //	   }
			   //	   for(j = 0; j < N_MU; ++j){
			   //		   syn_MU[i][j].AMPA_NMDA_RATIO = NAR_MU; //slow component for iMF
			   //		   syn_MU[i][j].AMPA_AMPASL_RATIO = NAR_SL; //slow component for iMF
			   //		   syn_MU[i][j].AMPA_MGLUR2_RATIO = NAR_M2U; //slow component for iMF
			   //		   syn_MU[i][j].AMPA_MGLUR1_RATIO = NAR_M1U; //slow component for iMF
			   //	   }
			   //   }
			   //   printf("\n Setting up UBC is done ! ");

			   Normaldev grand1(THR_GC, fabs(THR_GC * 0.05), SEED);
			   //Normaldev NMDA1( NAR_MG, NAR_MG*var, SEED);
			   for (i = 0; i < NGC; i++) {
#if TEST
				   GC[i].Vthr = THR_GC;
#else
				   GC[i].Vthr = grand1.dev();
				   GC[i].lastspk = -9999;
				   GC[i].TimeCounter = -1.0;
				   GC[i].q = 0;
				   GC[i].init(y_ini + no_GC[i][0]);
				   GC[i].postB = 0;
#endif

				   for (j = 0; j < N_eG; ++j) {
					   syn_eG[i][j].AMPA_NMDA_RATIO = NAR_MG;
					   syn_eG[i][j].AMPA_AMPAsl_RATIO = NAR_MG2;
					   syn_eG[i][j].I = 0;
				   }
				   for (j = 0; j < N_iG; ++j) {
					   syn_iG[i][j].AMPA_NMDA_RATIO = NAR_MG;
					   syn_iG[i][j].AMPA_AMPAsl_RATIO = NAR_MG2;
					   syn_iG[i][j].I = 0;
				   }
				   for (j = 0; j < N_GoG; ++j) {
#if TEST
					   syn_GoG[i][j].mix = 1; //mixed fast and slow component
#else
					   syn_GoG[i][j].mix = GetRandomDoub(1); //mixed fast and slow component
					   syn_GoG[i][j].I = 0;
#endif
				   }
			   }
			   printf("\n Setting up GC is done ! ");

			   Normaldev gka(GKA, 0.3*GKA, SEED);
			   Normaldev gL(3.0, 0.05*3.0, SEED);
			   Normaldev tAHP(20.0, 0.1 * 20, SEED);
			   Normaldev delay1(1.0, 0.2, SEED);
			   Normaldev grand3(THR_GoC, fabs(THR_GoC * 0.05), SEED);
			   //Normaldev NMDA2( NAR_MGo, NAR_MGo*var, SEED);
			   //Normaldev NMDA3( NAR_GGo, NAR_MG*var, SEED);
			   for (i = 0; i < NGoC; i++) {
#if TEST
				   GoC[i].delayA = 1.0;
				   GoC[i].delayB = 1.0;
				   GoC[i].para();
				   GoC[i].G_ka = GKA;
				   GoC[i].G_l = 3;
				   GoC[i].Vthr = THR_GoC;
				   GoC[i].tauAHP = 20;
#else
				   GoC[i].delayA = delay1.dev();
				   GoC[i].delayB = delay1.dev();
				   GoC[i].para();
				   GoC[i].G_ka = 0;//gka.dev(); 
				   GoC[i].Cm = 20;// GetRandomDoub(40.) + 20.; //20-60
				   GoC[i].G_l = GoC[i].Cm / GoC[i].taum;
				   GoC[i].tauAHP = 20; //tAHP.dev(); 
				   GoC[i].Vthr = grand3.dev();
				   GoC[i].E_l = -1.0*EL;

				   GoC[i].lastspk = -9999;
				   GoC[i].TimeCounter = -1.0;
				   GoC[i].q = 0;
				   GoC[i].postB = 0;
				   GoC[i].init(y_ini + no_GoC[i][0]);
				   GoC[i].reset();

				   fprintf(f9, "%lf\n", GoC[i].Cm);
#endif
				   for (j = 0; j < N_MGo; ++j) {
					   syn_MGo[i][j].AMPA_AMPAsl_RATIO = NAR_MGo2;
					   syn_MGo[i][j].AMPA_NMDA_RATIO = NAR_MGo;
					   syn_MGo[i][j].I = 0;
				   }
				   for (j = 0; j < N_GGo; ++j) {
					   syn_GGo[i][j].AMPA_NMDA_RATIO = NAR_GGo; //mixed fast and slow component
					   syn_GGo[i][j].I = 0;
				   }
				   for (j = 0; j < N_GG; ++j) {
					   syn_GG[i][j].I = 0;
				   }
			   }
			   printf("\n Setting up GoC is done ! ");
			   //--------------end variability------------------------------------



			   //---------- end: changing variables----------------------
			   int it = 0, trial = 0;
			   double Vstim = 0.0, PSC = 0.0, GCFR = 0.0, GCFRmean = 0.0, GCfired = 0.0, Gcontrol = 0.0, GoCFRmean = 0.0, GoCfired = 0.0, MFFRmean = 0.0, MFfired = 0.0;
			   double PCrate = 0.0, MFrate = 0.0, UBCrate = 0.0, GCrate = 0.0, GoCrate = 0.;
			   double VolGC = 0.0, VolGoC = 0.0;
			   //----------------CALCULATION----------------------------------------
			   printf("\n CALCULATION IN PROGICSS!!!: TMAX= %lf (min)", TMAX / (1000.*60.));

			   for (i = 0; i < NMF; i++) {
				   zz = GetRandomDoub(1.0);

				   if (zz <= 1) {
					   eMF[i].Inoise = -200;

				   }
			   }

			   while (t <= TMAX) {

				   //add burst in Poisson Burst 1:3ms;3:8ms;7:20ms;10;28ms
				   //20-100;1-30. 28-130.10 spike
				   //20-20-1£»20-30-3£»20-70-7
				   /* if (t >=1000 && t <= (1000+51)) {
							for (i = 0; i < NMF; i++) {
								eMF[i].Startburst = 1;
								if (t >= 1000 && t <= 1001) {
									eMF[i].TimeCounter = -3;
								}
							}
				   }
				   else {
						  for (i = 0; i < NMF; i++) {
							eMF[i].Startburst = 0;
					 }
				   }*/
				   //len = len +4;

				   //printf("time= %lf (ms)", t);
#if HOLD
				   y_ini[no_GC[0][0]] = -70;
				   y_ini[no_UBC[1][0]] = -70;
				   y_ini[no_GoC[0][0]] = -70;
				   y_ini[no_PC[0][0]] = -70;
#endif
				   if (t >= ST*SHORTTIME) {
					   Vstim = A*1e-3 * sin(1 * 2 * PI * t / T);
					   //Vstim = MFRATE*1e-3 * ( A* 1./T*1e3 * sin( 2*PI * t/T) );	
				   }

#if MFSTIM
				   //---------- modulate stimulating patterns ----------------------
				   for (i = 0; i < NMF; i++) {
					   eMF[i].mu = eMF[i].k * fabs(Vstim);
				   }
				   // bidirectional varying stimuli
				   if (Vstim >= THR && t >= ST*SHORTTIME) {

					   for (i = 0; i < NMF; i++) {

						   zz = GetRandomDoub(1.0);
						   //printf("%lf\n", zz);
						   ww = DT * (MFRATE*1e-3 + eMF[i].mu);
						   if ((eMF[i].TYPE == 1) && (zz - ww < 0)) {
							   eMF[i].Inoise = -200;
							   //fprintf(f2, "%5f %5d \n", t,i);
						   }
						   else if ((eMF[i].TYPE == 2) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 - eMF[i].mu))) {
							   eMF[i].Inoise = -200;
						   }
						   else {
							   eMF[i].Inoise = GetRandomDoub(1.0) - 0.5;
						   }
					   }

				   }
				   else if (Vstim < THR && t >= ST*SHORTTIME) {
					   for (i = 0; i < NMF; i++) {
						   if ((eMF[i].TYPE == 2) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 + eMF[i].mu))) {
							   eMF[i].Inoise = -200;
						   }
						   else if ((eMF[i].TYPE == 1) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 - eMF[i].mu))) {
							   eMF[i].Inoise = -200;
							   //fprintf(f2, "%5f %5d \n", t, i);
						   }
						   else {
							   eMF[i].Inoise = GetRandomDoub(1.0) - 0.5;
						   }
					   }
				   }
#endif
#if BURST
				   //---------- modulate stimulating patterns ----------------------
				   for (i = 0; i < NMF; i++) {
					   eMF[i].mu = eMF[i].k * fabs(Vstim);
				   }
				   // bidirectional varying stimuli
				   if (Vstim >= THR && t >= ST*SHORTTIME) {

					   for (i = 0; i < NMF; i++) {

						   zz = GetRandomDoub(1.0);
						   //printf("%lf\n", zz);
						   ww = DT * (MFRATE*1e-3 + eMF[i].mu);
						   if ((eMF[i].TYPE == 1) && (zz - ww < 0)) {
							   eMF[i].Inoise = -200;
							   //fprintf(f2, "%5f %5d \n", t,i);
						   }
						   else if ((eMF[i].TYPE == 2) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 - eMF[i].mu))) {
							   eMF[i].Inoise = -200;
						   }
						   else {
							   eMF[i].Inoise = GetRandomDoub(1.0) - 0.5;
						   }
					   }

				   }
				   else if (Vstim < THR && t >= ST*SHORTTIME) {
					   for (i = 0; i < NMF; i++) {
						   if ((eMF[i].TYPE == 2) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 + eMF[i].mu))) {
							   eMF[i].Inoise = -200;
						   }
						   else if ((eMF[i].TYPE == 1) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 - eMF[i].mu))) {
							   eMF[i].Inoise = -200;
							   //fprintf(f2, "%5f %5d \n", t, i);
						   }
						   else {
							   eMF[i].Inoise = GetRandomDoub(1.0) - 0.5;
						   }
					   }
				   }
#endif
#if POISSON
				   //---------- modulate stimulating patterns ----------------------
				   //for (i = 0; i < NMF; i++) {
				   // eMF[i].mu = eMF[i].k * fabs(A*1e-3 / 10);
				   //}
				   //// bidirectional varying stimuli
				   //if (t >= ST*SHORTTIME) {

				   // for (i = 0; i < NMF; i++) {
				   //  //eMF[i].Inoise = -200;

				   //  zz = GetRandomDoub(1.0);
				   //  //printf("%lf\n", zz);
				   //  ww = DT * (MFRATE*1e-3 + eMF[i].mu);
				   //  if ((eMF[i].TYPE == 1) && (zz - ww < 0)) {
				   //   eMF[i].Inoise = -200;
				   //   // fprintf(f2, "%5f %5d \n", t, i);
				   //  }
				   //  else if ((eMF[i].TYPE == 2) && (GetRandomDoub(1.0) < DT * (MFRATE*1e-3 - eMF[i].mu))) {
				   //   eMF[i].Inoise = -200;
				   //  }
				   //  else {
				   //   eMF[i].Inoise = GetRandomDoub(1.0) - 0.5;
				   //  }
				   // }

				   //}




#endif

	   //rkForth(N_EQ, fun, h, t, y_ini, f_ini, y1, y2);
	   //rkSecond(N_EQ, fun, h, t, y_ini, f_ini, y1, y2);
				   euler(N_EQ, fun, h, t, y_ini, f_ini);
				   t += h;
				   ii++;

#if SAVE_VOLTAGE 

				   //------save voltage data -----------------------------------------------------------------
				   if (ii % int(1) == 0) {
					   //if(  ii % int(1/h) == 0 ) {
						   //for(i = 0; i < NGC; i++) {
							   //fprintf(f7,"%lf\n ", y_ini[no_GC[i][0]]);
							   //fprintf(f7,"%lf %lf  %lf \n ",t, y_ini[no_GC[NGC/2][0]], y_ini[no_GC[NGoC / 2][0]]);
						  // }
					   VolGC = 0.0; VolGoC = 0.0;
					   for (i = 0; i < NGC; i++) {
						   VolGC += y_ini[no_GC[i][0]];
					   }
					   VolGC = VolGC / NGC;

					   for (i = 0; i < NGoC; i++) {
						   VolGoC += y_ini[no_GoC[i][0]];
					   }
					   VolGoC = VolGoC / NGoC;
					   fprintf(f71, "%lf %lf  %lf \n ", t, VolGC, VolGoC);
					   fprintf(f7, "%lf %lf  %lf \n ", t, y_ini[no_GC[2][0]], y_ini[no_GoC[1][0]]);
					   //for(i = 0; i < NUBC; i++) { 
						 //  fprintf(f7,"%lf\n",  y_ini[no_UBC[i][0]]);
					   //}

					   //for(i = 0; i < NGoC; i++) { 
						   //fprintf(f7,"%lf\n",  y_ini[no_GoC[i][0]]);
						   //fprintf(f7,"%lf\n",  y_ini[no_GoC[NGoC/2][0]]);
					   //}
				  // fprintf(f7, "%lf %lf %lf %lf %lf\n", t, eMF[10].ampaMG, eMF[10].nmdaMG, eMF[10].AMPAfR, eMF[10].AMPAfu);
				   }
#endif

#if SAVE_EPSC
				   if (ii % int(1 / h) == 0) {

					   PSC = 0.0;
					   //-----------AMPA and NMDA from iMF to GC cells--------------------------------------
					   for (j = 0; j < N_iG; ++j) {
						   PSC += syn_iG[NGC / 2][j].I;
					   }

					   //-----------AMPA and NMDA from eMF to GC cells--------------------------------------
					   for (j = 0; j < N_eG; ++j) {
						   PSC += syn_eG[NGC / 2][j].I;
						   //PSC += syn_eG[NGC/2][j].I1;
					   }
					   //fprintf(f6,"%lf ", PSC);

					   PSC = 0.0;
					   //----------- GABA from GoC to GC cells--------------------------------------
					   for (j = 0; j < N_GoG; ++j) {
						   PSC += syn_GoG[NGC / 2][j].I;
					   }
					   //PSC = TONICI/1000.0*(y_ini[no_GC[NGC/2][0]]+75);
				   //fprintf(f6,"%lf %lf\n ",t, PSC);


					   PSC = 0.0;
					   //-----------AMPA and NMDA from MF to GoC cells--------------------------------------
					   for (j = 0; j < N_MGo; ++j) {
						   PSC += syn_MGo[NGoC / 2][j].I;
					   }
					   //-----------AMPA and NMDA from UBC to GoC cells--------------------------------------
					   for (j = 0; j < N_UGo; ++j) {
						   PSC += syn_UGo[NGoC / 2][j].I;
					   }
					   //fprintf(f6,"%lf ", PSC);

					   PSC = 0.0;
					   //-----------AMPA and NMDA from GC to GoC cells--------------------------------------
					   for (j = 0; j < N_GGo; ++j) {
						   PSC += syn_GGo[0][j].I;
					   }
					   //fprintf(f6,"%lf ", PSC);
					   //fprintf(f6, "%lf %lf\n ", t, PSC);

					   PSC = 0.0;
					   //-----------GAP from GoC to GoC cells--------------------------------------
					   for (j = 0; j < N_GG; ++j) {
						   PSC += syn_GG[NGoC / 2][j].I;
					   }
					   fprintf(f6, "%lf %lf %lf \n ", t, PSC, y_ini[no_GC[NGC / 2][3]]);
					   // fprintf(f6,"%lf ", PSC);		
				   }
#endif
#if SAVE_RATE
				   //if (ii % int(1 / h) == 0) {
					  // PCrate = 0.0, MFrate = 0.0, UBCrate = 0.0, GCrate = 0.0, GoCrate = 0.;
					  // //GC rate in bin=1ms
					  // for (i = 0; i < NGC; i++) {
						 //  for (j = 0; j < GC[i].Ca; j++) {
							//   if (GC[i].spiketimes[j]>(t - 1)) {
							//	   GCrate += 1;
							//   }
						 //  }
					  // }
					  // //MF rate in bin=1ms
					  // for (i = 0; i < NMF; i++) {
						 //  for (j = 0; j < eMF[i].Ca; j++) {
							//   if (eMF[i].spiketimes[j]>(t - 1)) {
							//	   MFrate += 1;
							//   }
						 //  }
					  // }
					  // //GoC rate in bin=1ms
					  // for (i = 0; i < NGoC; i++) {
						 //  for (j = 0; j < GoC[i].Ca; j++) {
							//   if (GoC[i].spiketimes[j]>(t - 1)) {
							//	   GoCrate += 1;
							//   }
						 //  }
					  // }

					  // fprintf(f11, "%lf %lf %lf %lf  \n", t, MFrate, GoCrate, GCrate);

				   //}
#endif
				   //--------  at the end of one trial -----------------------------------
				   if (ii % int(SHORTTIME / h) == 0) {
					   it++;
					   trial = (int)ceil((double)it / (double)NSTIM);

					   printf("\n Trial = %d ", (int)ceil((double)it / (double)NSTIM));
					   //if ( it ==40 ) system("PAUSE"); 		  

					   //fprintf(f7, "%4d\n", 9999);
					   fprintf(f8, "%4d %4d \n", 9999, 0);

					   //------save spike data -----------------------------------------------------------------
					   fprintf(f1, " %4d %4d %4d\n", 9999, it, 1);

					   GCFRmean = 0.0;
					   GCfired = 0.0;
					   GoCFRmean = 0.0;
					   GoCfired = 0.0;
					   MFFRmean = 0.0;
					   MFfired = 0.0;
					   for (i = 0; i < NGC; i++) {
						   for (j = 0; j<GC[i].Ca; j++) {
							   //fprintf(f1, "%4d %4d %lf \n", trial, i, GC[i].spiketimes[j] - (it - 1)*SHORTTIME);
							   fprintf(f1, "%4d %4d %lf \n", trial, i, GC[i].spiketimes[j]);
						   }
						   if (GC[i].Ca > 0.0) {
							   GCfired += 1.0;
							   GCFRmean += GC[i].Ca / (T / 1000);
						   }
					   }
					   if (GCfired > 0.0) {
						   GCFRmean /= NGC;
					   }
					   else {
						   GCFRmean = 0.0;
					   }
					   Gcontrol += ALPHA * (GCFRmean - GCRATE);
					   for (i = 0; i < NGC; i++) {
						   GC[i].Ginh = Gcontrol;
					   }

					   //fprintf(f2,"%4d %lf %lf \n", trial, Gcontrol, GCFRmean);

					 /*  for (i = 0; i < NUBC; i++) {
						   for (j = 0; j < UBC[i].Ca; j++) {
							   fprintf(f1, "%4d %4d %lf \n", trial, NGC + i, UBC[i].spiketimes[j] - (it - 1)*SHORTTIME);
						   }
					   }*/
					   for (i = 0; i < NGoC; i++) {
						   for (j = 0; j < GoC[i].Ca; j++) {
							   //fprintf(f12, "%4d %4d %lf \n", trial, i, GoC[i].spiketimes[j] - (it - 1)*SHORTTIME);
							   fprintf(f12, "%4d %4d %lf \n", trial, i, GoC[i].spiketimes[j]);
						   }
						   if (GoC[i].Ca > 0.0) {
							   GoCfired += 1.0;
							   GoCFRmean += GoC[i].Ca / (T / 1000);
						   }
					   }
					   if (GoCfired > 0.0) {
						   GoCFRmean /= NGoC;
					   }
					   else {
						   GoCFRmean = 0.0;
					   }
					   for (i = 0; i < NMF; i++) {
						   for (j = 0; j < eMF[i].Ca; j++) {
							   //fprintf(f2, "%4d %4d %lf \n", trial, i, eMF[i].spiketimes[j] - (it - 1)*SHORTTIME);
							   fprintf(f2, "%4d %4d %lf \n", trial, i, eMF[i].spiketimes[j]);
						   }
						   if (eMF[i].Ca > 0.0) {
							   MFfired += 1.0;
							   MFFRmean += eMF[i].Ca / (T / 1000);
						   }
					   }
					   if (MFfired > 0.0) {
						   MFFRmean /= NMF;
					   }
					   else {
						   MFFRmean = 0.0;
					   }
					   printf("Trial=%4d, MFrate=%lf, GCrate=%lf GoCrate=%lf \n", trial, MFFRmean, GCFRmean, GoCFRmean);
					   if (t > 1999) {
						   fprintf(f10, "%1f %1f %lf \n", MFFRmean, GCFRmean, GoCFRmean);
					   }
					   // reset variables
					   for (i = 0; i < NMF; i++)    eMF[i].reset();
					   for (i = 0; i < NGC; i++)    GC[i].reset();
					   for (i = 0; i < NUBC; i++)   UBC[i].reset();
					   for (i = 0; i < NGoC; i++)   GoC[i].reset();

				   }// end of one trial 	

			   }

			   fprintf(f11, "%4d %4d %4d %4d \n", -11, -11, -11, -11);
			   fprintf(f1, "%4d %4d %4d  \n", -11, -11, -11);
			   fprintf(f2, "%4d %4d %4d \n", -11, -11, -11);
			   fprintf(f12, "%4d %4d %4d \n", -11, -11, -11);
			   fprintf(f71, "%4d %4d %4d \n", -11, -11, -11);
			   //fprintf(f11, "%4d %4d %4d %4d \n", -1,-1,-1,-1);
		   }
		   //Inh
		   fprintf(f11, "%4d %4d %4d %4d \n", -1, -1, -1, -1);
		   fprintf(f1, "%4d %4d %4d  \n", -1, -1, -1);
		   fprintf(f2, "%4d %4d %4d \n", -1, -1, -1);
		   fprintf(f12, "%4d %4d %4d \n", -1, -1, -1);
		   fprintf(f71, "%4d %4d %4d \n", -1, -1, -1);
	   }
			fprintf(f11, "%4d %4d %4d %4d \n", -2, -2, -2, -2);
			fprintf(f1, "%4d %4d %4d  \n", -2, -2, -2);
			fprintf(f2, "%4d %4d %4d \n", -2, -2, -2);
			fprintf(f12, "%4d %4d %4d \n", -2, -2, -2);
			fprintf(f71, "%4d %4d %4d \n", -2, -2, -2);

  }//--------------------END CALCULATION-------------------------------
      
  //-----------------close ALL files-----------------------------------
  fprintf(f1, " %4d %4d %4d\n",9999, NTRIAL, NSTIM );
  fclose(f1);
  fclose(f2);

  fclose(f3);
  fclose(f4);
  fclose(f5);
  fclose(f6);
  fclose(f7);
  fclose(f8);
  fclose(f9);
  fclose(f10);
  fclose(f11);
  fclose(f12);
  fclose(f71);
  // free memory
  //delete [] ga_IE[0];
  //delete [] ampa_EE[0];
  //delete [] nmda_EE[0]; 
  //delete [] ampa_EI[0];
  //delete [] nmda_EI[0];

  // free memory
  for(i=0; i < NGC; i++) {
	  delete [] syn_iG[i];
	  delete [] syn_eG[i];
	  delete [] syn_GoG[i];
  }

  for(i=0; i < NUBC; i++) {
	  delete [] syn_MU[i];
	  delete [] syn_GoU[i];
	  delete [] syn_UU[i];
  }
  for(i=0; i < NGoC; i++) {
	  delete [] syn_MGo[i];
	  delete [] syn_UGo[i];
	  delete [] syn_GGo[i];
	  delete [] syn_GG[i];
  }


   /*
   delete [] ga_IE;
   delete [] ampa_EE;
   delete [] nmda_EE; 
   delete [] ampa_EI;
   delete [] nmda_EI;
  */

   finish = omp_get_wtime();
   //totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
   totaltime = (double)(finish-start);
   cout<<"\n running time is "<<totaltime<<" sec"<<endl;

   system("PAUSE"); 
   return 0;
}



//----------external functions----------------------------------------------
void fun(double x, double *y_ini, double *f_ini){
	int i, j; 


		//========here the MAIN loop to calculate intrinsic conductances===========
	//--------(f_ini IS changed, y_ini IS NOT changed)-------------------------
#pragma omp parallel private(i,j)
{
#pragma omp for nowait
	for(i=0; i<NGC; ++i)    GC[i].calc( x, y_ini+no_GC[i][0],  f_ini+no_GC[i][0]); 
#pragma omp for nowait
	for(i=0; i<NMF; ++i) 	eMF[i].calc(x, y_ini+no_eMF[i][0], f_ini+no_eMF[i][0]); 
#pragma omp for nowait
	for(i=0; i<NGoC; ++i) 	GoC[i].calc(x, y_ini+no_GoC[i][0], f_ini+no_GoC[i][0]); 
#pragma omp barrier

		//========here the MAIN loop to calculate synaptic conductances=============
		//--------(f_ini IS changed, y_ini IS NOT changed) -------------------------
	    // i-post; j-pre
		// update GC post synapse
#pragma omp for nowait
	for (i = 0; i < NGC; ++i) {

		//-----------AMPA and NMDA from iMF to GC cells--------------------------------------
		/*for(j = 0; j < N_iG; ++j){
			syn_iG[i][j].calc( g_iG[i][j], UBC[ pre_iG[i][j] ].ampaMG, UBC[ pre_iG[i][j] ].nmdaMG, UBC[ pre_iG[i][j] ].ampaMGsl,
				y_ini[no_GC[i][0]], GC[i].postB );
			f_ini[no_GC[i][0]] -= syn_iG[i][j].I / GC[i].Cm ;
		}*/

		//-----------AMPA and NMDA from eMF to GC cells--------------------------------------
		for (j = 0; j < N_eG; ++j) {
			syn_eG[i][j].calc(g_eG[i][j], eMF[pre_eG[i][j]].ampaMG, eMF[pre_eG[i][j]].nmdaMG, eMF[pre_eG[i][j]].ampaMGsl,
				y_ini[no_GC[i][0]], GC[i].postB);
			f_ini[no_GC[i][0]] -= syn_eG[i][j].I / GC[i].Cm;
		}
		//----------- GABA from GoC to GC cells--------------------------------------
			for (j = 0; j < N_GoG; ++j) {
				//cout<<" pre_GOC is "<<pre_GoG[i][j] <<endl;
				syn_GoG[i][j].calc(g_GoG[i][j], GoC[pre_GoG[i][j]].gabaa, GoC[pre_GoG[i][j]].gabab, y_ini[no_GC[i][0]]);
				f_ini[no_GC[i][0]] -= syn_GoG[i][j].I / GC[i].Cm;
			}
	}



#pragma omp for nowait
		// update GoC post synapse
		for(i = 0; i < NGoC; ++i)	{						
			//-----------AMPA and NMDA from MF to GoC cells--------------------------------------
			/*for(j = 0; j < N_MGo; ++j){
				syn_MGo[i][j].calc( g_MGo[i][j], eMF[ pre_MGo[i][j] ].ampaMGo, eMF[ pre_MGo[i][j] ].nmdaMGo, eMF[ pre_MGo[i][j] ].ampaMGsl, 
					y_ini[no_GoC[i][0]], GoC[i].postB );
				f_ini[no_GoC[i][0]] -= syn_MGo[i][j].I / GoC[i].Cm;		
			}*/
			//-----------AMPA and NMDA from UBC to GoC cells--------------------------------------
			/*for(j = 0; j < N_UGo; ++j){
				syn_UGo[i][j].calc( g_UGo[i][j], UBC[ pre_UGo[i][j] ].ampaMGo, UBC[ pre_UGo[i][j] ].nmdaMGo, UBC[ pre_UGo[i][j] ].ampaMGosl,  
					y_ini[no_GoC[i][0]], GoC[i].postB );
				f_ini[no_GoC[i][0]] -= syn_UGo[i][j].I / GoC[i].Cm;		
			}*/
			//-----------AMPA and NMDA from GC to GoC cells--------------------------------------
			for(j = 0; j < N_GGo; ++j){
				syn_GGo[i][j].calc( g_GGo[i][j], GC[ pre_GGo[i][j] ].ampaMG, GC[ pre_GGo[i][j] ].nmdaMG, y_ini[no_GoC[i][0]], GoC[i].postB );
				f_ini[no_GoC[i][0]] -= syn_GGo[i][j].I / GoC[i].Cm;		
			}
			//-----------GAP from GoC to GoC cells--------------------------------------
			/*for(j = 0; j < N_GG; ++j){
				syn_GG[i][j].calc( g_GG[i][j], y_ini[no_GoC[ pre_GG[i][j] ][0]], y_ini[no_GoC[i][0]] );
				f_ini[no_GoC[i][0]] -= syn_GG[i][j].I / GoC[i].Cm;		
			}*/
		}

}
		//=============END of MAIN loop==============================================

}


// 1st order Rounge-Kutta ((forward) Euler method) solver for ODE -------------------------------------------------
//   rk(N_EQ, fun, h, t, y_ini, f_ini, y1, y2);
void euler(int n, void fun(double, double*, double*), 
        double h, double x, double* y, double* f)
{
	int i;

#pragma omp parallel for
	for(i = 0; i < n; ++i) 	{
		y[i] += h * f[i]; 
	}

	//k1
	fun(x, y, f);  

}



// 2nd order Rounge-Kutta solver for ODE -------------------------------------------------
//   rk(N_EQ, fun, h, t, y_ini, f_ini, y1, y2);
void rkSecond(int n, void fun(double, double*, double*), 
        double h, double x, double* y, double* f, double* s, double* yk)
{
	int i;
	double xk;

	//k1
	fun(x, y, f);    

	for(i = 0; i < n; i++)
	{
		yk[i] = y[i] + (h/2.)*f[i]; 
	}
	xk = x + h/2.;    
	//k2
	fun(xk, yk, f);

	for(i = 0; i < n; ++i) y[i] += h * f[i];
}

