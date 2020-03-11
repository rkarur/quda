#pragma once

#include <quda.h>
#include <random_quda.h>
#include <vector>
#include <color_spinor_field.h>

#define gauge_site_size 18 // real numbers per link
#define spinor_site_size 24 // real numbers per wilson spinor
#define stag_spinor_site_size 6 // real numbers per staggered 'spinor'
#define clover_site_size 72 // real numbers per block-diagonal clover matrix
#define mom_site_size    10 // real numbers per momentum
#define hw_site_size    12 // real numbers per half wilson

extern int Z[4];
extern int V;
extern int Vh;
extern int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
extern int faceVolume[4];
extern int E1, E1h, E2, E3, E4; 
extern int E[4];
extern int V_ex, Vh_ex;

extern double kappa5;
extern int Ls;
extern int V5;
extern int V5h;
  
extern int my_spinor_site_size;
extern size_t host_gauge_data_type_size;
extern size_t host_spinor_data_type_size;
extern size_t host_clover_data_type_size;

// QUDA precisions
extern QudaPrecision &cpu_prec;
extern QudaPrecision &cuda_prec;
extern QudaPrecision &cuda_prec_sloppy;
extern QudaPrecision &cuda_prec_precondition;
extern QudaPrecision &cuda_prec_refinement_sloppy;
extern QudaPrecision &cuda_prec_ritz;

void setQudaDefaultPrecs();
void setQudaDefaultMgSolveTypes();
void setQudaDefaultMgTestParams();
void setQudaStaggeredEigTestParams();
void setQudaStaggeredInvTestParams();

void constructQudaGaugeField(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param);
void constructHostGaugeField(void **gauge, QudaGaugeParam &gauge_param, int argc, char **argv);
void constructStaggeredHostGaugeField(void **qdp_inlink, void **qdp_longlink, void **qdp_fatlink, QudaGaugeParam &gauge_param, int argc, char **argv);

void constructFatLongGaugeField(void **fatlink, void** longlink, int type, 
				QudaPrecision precision, QudaGaugeParam*, 
				QudaDslashType dslash_type);



void constructQudaCloverField(void *clover, double norm, double diag, QudaPrecision precision);
void constructHostCloverField(void *clover, void *clover_inv, QudaInvertParam &inv_param);

void constructStaggeredTestSpinorParam(quda::ColorSpinorParam *csParam, const QudaInvertParam *inv_param, const QudaGaugeParam *gauge_param);
void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, const int *const x, quda::RNG &rng);


void computeLongLinkCPU(void** longlink, void **sitelink, QudaPrecision prec, void* act_path_coeff);  

void computeHISQLinksCPU(void** fatlink, void** longlink, void** fatlink_eps, void** longlink_eps,
			 void** sitelink, void* qudaGaugeParamPtr, 
			 double** act_path_coeffs, double eps_naik);

// data reordering routines
void reorderQDPtoMILC(void* milc_out, void** qdp_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision);
void reorderMILCtoQDP(void** qdp_out, void* milc_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision);


void performanceStats(double *time, double *gflops);

void initComms(int argc, char **argv, std::array<int, 4> &commDims);
void initComms(int argc, char **argv, int *const commDims);
void finalizeComms();
void initRand();

void setDims(int *X);
void dw_setDims(int *X, const int L5);
void setSpinorSiteSize(int n);
int dimPartitioned(int dim);

bool last_node_in_t();

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1) ;
  
int neighborIndex(int dim[], int index, int oddBit, int dx[]);
int neighborIndexFullLattice(int dim[], int index, int dx[]);  

int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1);

void printSpinorElement(void *spinor, int X, QudaPrecision precision);
void printGaugeElement(void *gauge, int X, QudaPrecision precision);
  
int fullLatticeIndex(int i, int oddBit);
int fullLatticeIndex(int dim[], int index, int oddBit);
int getOddBit(int X);

void applyGaugeFieldScaling_long(void **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type, QudaPrecision local_prec);



void createSiteLinkCPU(void** link,  QudaPrecision precision, int phase) ;

void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision);
void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision, QudaGaugeParam *param);

void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, QudaPrecision precision);
void strong_check(void *spinor, void *spinorGPU, int len, QudaPrecision precision);
int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision);

void check_gauge(void **, void **, double epsilon, QudaPrecision precision);

int strong_check_link(void ** linkA, const char* msgA,  void **linkB, const char* msgB, int len, QudaPrecision prec);
int strong_check_mom(void * momA, void *momB, int len, QudaPrecision prec);
  
void createMomCPU(void* mom,  QudaPrecision precision);
void createHwCPU(void* hw,  QudaPrecision precision);
  
//used by link fattening code
int x4_from_full_index(int i);

// ---------- gauge_read.cpp ----------

//void readGaugeField(char *filename, float *gauge[], int argc, char *argv[]);

// additions for dw (quickly hacked on)
int fullLatticeIndex_4d(int i, int oddBit);
int fullLatticeIndex_5d(int i, int oddBit);
int fullLatticeIndex_5d_4dpc(int i, int oddBit);
int process_command_line_option(int argc, char** argv, int* idx);
int process_options(int argc, char **argv);

// Use for profiling
void stopwatchStart();
double stopwatchReadSeconds();

#ifdef __cplusplus
extern "C" {
#endif

  // Implemented in face_gauge.cpp
  void exchange_cpu_sitelink(int* X,void** sitelink, void** ghost_sitelink,
			     void** ghost_sitelink_diag,
			     QudaPrecision gPrecision, QudaGaugeParam* param, int optflag);
  void exchange_cpu_sitelink_ex(int* X, int *R, void** sitelink, QudaGaugeFieldOrder cpu_order,
				QudaPrecision gPrecision, int optflag, int geometry);
  void exchange_cpu_staple(int* X, void* staple, void** ghost_staple,
			   QudaPrecision gPrecision);
  void exchange_llfat_init(QudaPrecision prec);
  void exchange_llfat_cleanup(void);

  // Implemented in host_blas.cpp
  double norm_2(void *vector, int len, QudaPrecision precision);
  void mxpy(void *x, void *y, int len, QudaPrecision precision);
  void ax(double a, void *x, int len, QudaPrecision precision);
  void axpy(double a, void *x, void *y, int len, QudaPrecision precision);
  void xpay(void *x, double a, void *y, int len, QudaPrecision precision);
  void cxpay(void *x, double _Complex a, void *y, int len, QudaPrecision precision);
  void cpu_axy(QudaPrecision prec, double a, void* x, void* y, int size);
  void cpu_xpy(QudaPrecision prec, void* x, void* y, int size);
  

#ifdef __cplusplus
}
#endif

inline QudaPrecision getPrecision(int i)
{
  switch (i) {
  case 0: return QUDA_QUARTER_PRECISION;
  case 1: return QUDA_HALF_PRECISION;
  case 2: return QUDA_SINGLE_PRECISION;
  case 3: return QUDA_DOUBLE_PRECISION;
  }
  return QUDA_INVALID_PRECISION;
}

inline int getReconstructNibble(QudaReconstructType recon)
{
  switch (recon) {
  case QUDA_RECONSTRUCT_NO: return 4;
  case QUDA_RECONSTRUCT_13:
  case QUDA_RECONSTRUCT_12: return 2;
  case QUDA_RECONSTRUCT_9:
  case QUDA_RECONSTRUCT_8: return 1;
  default: return 0;
  }
}

// MG param types
void setMultigridParam(QudaMultigridParam &mg_param);
void setStaggeredMultigridParam(QudaMultigridParam &mg_param);

// Eig param types
void setDeflationParam(QudaEigParam &df_param);
void setMultigridEigParam(QudaEigParam &eig_param, int level);
void setEigParam(QudaEigParam &eig_param);

// Invert param types
void setInvertParam(QudaInvertParam &inv_param);
void setContractInvertParam(QudaInvertParam &inv_param);
void setMultigridInvertParam(QudaInvertParam &inv_param);
void setDeflatedInvertParam(QudaInvertParam &inv_param);
void setStaggeredInvertParam(QudaInvertParam &inv_param);

// Gauge param types
void setWilsonGaugeParam(QudaGaugeParam &gauge_param);
void setStaggeredQDPGaugeParam(QudaGaugeParam &gauge_param);
void setStaggeredMILCGaugeParam(QudaGaugeParam &gauge_param);
