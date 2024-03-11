#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <contract_quda.h>

using namespace quda;

// Delclaration of function in interface_quda.cpp
TimeProfile &getProfileSinkProject();

void laphSinkProject(double _Complex *host_sinks, void **host_quark, int n_quark, int tile_quark,
                     void **host_evec, int n_evec, int tile_evec, QudaInvertParam *inv_param, const int X[4])
{
  auto profile = pushProfile(getProfileSinkProject(), inv_param->secs, inv_param->gflops);

  // Parameter object describing the sources and smeared quarks
  lat_dim_t x = {X[0], X[1], X[2], X[3]};
  ColorSpinorParam cpu_quark_param(host_quark, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> quark(n_quark);
  for (auto i = 0; i < n_quark; i++) {
    cpu_quark_param.v = host_quark[i];
    quark[i] = ColorSpinorField(cpu_quark_param);
  }

  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  // Switch to spin 1
  cpu_evec_param.nSpin = 1;
  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> evec(n_evec);
  for (auto i = 0; i < n_evec; i++) {
    cpu_evec_param.v = host_evec[i];
    evec[i] = ColorSpinorField(cpu_evec_param);
  }

  // Create device vectors
  ColorSpinorParam quda_quark_param(cpu_quark_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  quda_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  std::vector<ColorSpinorField> quda_quark(tile_quark, quda_quark_param);

  // Create device vectors for evecs
  ColorSpinorParam quda_evec_param(cpu_evec_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  std::vector<ColorSpinorField> quda_evec(tile_evec, quda_evec_param);

  auto Lt = x[3] * comm_dim(3);
  std::vector<Complex> hostSink(n_quark * n_evec * Lt * 4);

  for (auto i = 0; i < n_quark; i += tile_quark) { // iterate over all quarks
    for (auto tq = 0; tq < tile_quark && i + tq < n_quark; tq++) quda_quark[tq] = quark[i + tq];

    for (auto j = 0; j < n_evec; j += tile_evec) { // iterate over all EV
      for (auto te = 0; te < tile_evec && j + te < n_evec; te++) quda_evec[te] = evec[j + te];

      for (auto tq = 0; tq < tile_quark && i + tq < n_quark; tq++) {
        for (auto te = 0; te < tile_evec && j + te < n_evec; te++) {
          std::vector<Complex> tmp(Lt * 4);

          // We now perform the projection onto the eigenspace. The data
          // is placed in host_sinks in  T, spin order
          evecProjectLaplace3D(tmp, quda_quark[tq], quda_evec[te]);

          for (auto k = 0u; k < tmp.size(); k++)
            hostSink[((i + tq) * n_evec + (j + te)) * 4 * Lt + k] = tmp[k];
        }
      }
    }
  }

  comm_allreduce_sum(hostSink);

  for (auto i = 0; i < n_quark * n_evec * Lt * 4; i++) { // iterate over all quarks
    reinterpret_cast<std::complex<double> *>(host_sinks)[i] = hostSink[i];
  }
}
