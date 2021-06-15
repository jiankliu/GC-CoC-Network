
#include <math.h>

#ifndef PARAM_H
#define PARAM_H

/************** Defines some global paramaters -- *************/
	
//  units:
//  (nS) = 
//  (nA) = (nanoamp)
//  (mV) = (millivolt)
//  (umho) = (micromho)
//  (mM) = (milli/liter)
//  t = msec

#define WINVC 1
#define UNIX  0

//==============================================================================
//     [simulation parameters and settings]
//==============================================================================

//------------------------------------------------------------------------------
// general settings
//------------------------------------------------------------------------------
#define TEST  0
#define POISSON 1
#define BURST 0
#define MFSTIM 0
#define SAVE_VOLTAGE  1
#define SAVE_EPSC  0
#define SAVE_RATE 1
#define CURRENT_CLAMP 0
#define V_CLAMP 0
#define NOUBC  0
#define HOLD   0
#define GABAA  0     

const int NSTIM = 1;
const int NTRIAL =10;
const int ST = 0;         // time without stimuli
const int NT = 1;
const double T = 1000;  //  (ms) period of stimulus 
const double SHORTTIME = T;
const double TMAX =  NTRIAL * SHORTTIME + ST*T;       //simulation time (m sec)

const int SEED = 2947;  // randome seed 
const int RANDIN = 1;

const double OFFUBC = 0.;
const double GK = 2;

//Goc
const double GKA = 0; //1.2;

//UBC
const double GM2 = 3;  
const double GM1 = 1;  
const double GT = 35;  
const double GL = 0;  
const double GH = 0;

const double GKIR = 3;

const double GKSL = 0;  // [2:5]

const double TONICI = 900;  // 
const double TSTIM  = 100;  // 
const double EL = 70;  // 

const int STIMON = 1;  // 1: apply stimuli; 0: no external stim
const double MFRATE = 1000/TSTIM;     // default firing rate
const double A = 10;      // ( o/s) amplitude of stimulus velocity

const double GCRATE  = 5;     // average GC firing rate
const double ALPHA = 0.0;     // time constant of controlling

//------------------------------------------------------------------------------
//  network simulation parameter
//------------------------------------------------------------------------------
const int NMF = 500;
const int NGC = 2000;
const int NUBC = 4; 
const int NGoC = 144;

const double stimp = 10;
const int NINPUT = (int)ceil(NGoC * stimp/100);

//------------------------------------------------------------------------------
//  stimulus parameter
//------------------------------------------------------------------------------
const double THR = 0;   //  threshold to cutoff the amplitude of velocity

// Number of input synapses for post cell 
const int N_iG  = 4; //4;  // iMF -> GC
const int N_eG  = 4; //4;  // eMF -> GC
const int N_MU  = 1;   //  MF -> UBC
const int N_MGo = 10; //10;   //  MF -> GoC
const int N_UGo = 1; //10;   //  UBC-> GoC
const int N_GGo = 50;// (int)(NGC / 16.); // 8=0.6; 16=0.3;   //  GC -> GoC
const int N_GoG = 10;// (int)(NGoC / 4.8); //4.8;   //  GoC -> GC
const int N_GoU = 1; //6;   //  GoC -> UBC
const int N_UU = 1;  //  UBC -> UBC
const int N_GG = NGoC-1;  //10;   //  GoC -> GoC, gap junction
const double DIMGG = 0.4;

const int NMFG  = NGC * N_eG ;
const int NMFU  = NUBC * N_MU;
const int NMGo  = NGoC * N_MGo;

//------------------------------------------------------------------------------
//  synaptic simulation parameter
//------------------------------------------------------------------------------
// synapse weight in units of nS
const double AMPA_MAXiG = 30;     // 1.6  for single MF-GC
const double AMPA_MAXeG = 30;     // 1.6  for single MF-GC
const double AMPA_MAXMU = 70;     // 3.5-IAF; for single MF-UBC
const double AMPA_MAXUU = 70;     // 3.5-IAF; for single MF-UBC
const double AMPA_MAXMGo = 50;     // 2.5 
const double AMPA_MAXUGo = 50;     // 2.5 
const double AMPA_MAXGGo = 50;     // 2.5 
const double AMPA_MAXGoG = 400;    // <20 
const double AMPA_MAXGoU = 600;    // <30 
const double AMPA_MAXGG  = 600;    // <30 
const double weg = 1.0*1.6; //1.6;     // amp = -50 pm 0 pA (Arenz, Science 2008)
const double wig = 0*weg;     // amp = -50 pm 0 pA (Arenz, Science 2008)
const double wmu = 0*2.8;     //
const double wuu = wmu;     //
const double wmgo = 3.0; //20 if it is nicolas's model; 2.;      // amp = -66 pm 26 pA (Kanichay, J. Neurosci. 2006)
const double wugo = 0*wmgo;      // amp = -66 pm 26 pA (Kanichay, J. Neurosci. 2006)
const double wggo =3.0;    //2.0 
const double wgog = 1;// 1.0;// 1.0;     // amp =  54 pm 41 pA (Dugue, J. Neurosci. 2005)
                            //  but see (Crowley, Neuron, 2009)   
const double wgou = 0;     // amp =  53 pm 58 pA (Dugue, J. Neurosci. 2005)

const double wgg = 1.6;// 1.6;//0.4;     // amp =  53 pm 58 pA (Dugue, J. Neurosci. 2005)

const double NAR_MG   = 2.4;        // MF-GC   = NMDA  / AMPA  
const double NAR_MG2  = 2.0;       // MF-GC   = AMPAslow  / AMPA  
const double NAR_MGo   = 0.0;        // MF-GoC  =  NMDA / AMPA  
const double NAR_MGo2  = 2.0;        // MF-GoC  = AMPAslow  / AMPA  

const double NAR_GGo = 0.0;        // GC-GoC  = NMDA  / AMPA  
const double NAR_SL = 0.36;        // 0.36 nS MF-UBC   = AMPASL  / AMPA  
const double NAR_MU = 0.5;         // MF-UBC    NMDA  / AMPA 
const double NAR_M2U = GM2/10.;     // 3nS MGluR2
const double NAR_M1U = GM1/10.;    // 1nS MGluR2
const double NAR_GABA = 0.15;     // GoC-GC(UBC)

// for gaussian
const double var = 0.3;  //this is sacle variance for weights with var = avgW*SDW
const double varUBC = 0.3;  //this is sacle variance for weights with var = avgW*SDW

const double TRECNMDA = 1600;
const double TRECAMPA = 400;

//------------------------------------------------------------------------------
//  intfire cell properties [2]
//------------------------------------------------------------------------------
const double THR_GC = -49;
const double THR_UBC = -50;      //used as average_thr to randomized thr      
const double THR_GoC = -45;
const int SPKTIME = 1000;

const double DT = 0.1;    //IAF
const double TERR = DT; 
const int starttime = 500;

//------------------------------------------------------------------------------

#endif
