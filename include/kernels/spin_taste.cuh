#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaSpinTasteGamma gamma_>
  struct SpinTasteArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaSpinTasteGamma gamma = gamma_;
    using F = typename colorspinor_mapper<Float, 1, nColor, false, false>::type;

    int X[4];

    F out;        /** output vector field */
    const F in;   /** input vector field */

    SpinTasteArg(ColorSpinorField &out_, const ColorSpinorField &in_) :
      kernel_param(dim3(in_.VolumeCB(), in_.SiteSubset(), 1)),
      out(out_),
      in(in_)
    {
      checkOrder(out_, in_);     // check all orders match
      checkPrecision(out_, in_); // check all precisions match
      checkLocation(out_, in_);  // check all locations match
      if (!in_.isNative())
        errorQuda("Unsupported field order colorspinor= %d\n", in_.FieldOrder());
      if (!out_.isNative())
        errorQuda("Unsupported field order colorspinor= %d\n", out_.FieldOrder());
      #pragma unroll
      for (int i=0; i<4; i++) { X[i] = in_.X()[i]; }
    }
  };

  // FIXME only works with even local volumes
  template <typename Arg> constexpr auto getSign(int x, int y, int z, int t, const Arg &arg) {
    typename Arg::Float sign = 1.0;
    if (Arg::gamma == QUDA_SPIN_TASTE_G1) {
      sign = 1.0;
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GX) {
      sign = 1.0 - 2.0 * ((y + z + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GY) {
      sign = 1.0 - 2.0 * ((x + z + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GZ) {
      sign = 1.0 - 2.0 * ((x + y + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GT) {
      sign = 1.0 - 2.0 * ((x + y + z) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_G5) {
      sign = 1.0 - 2.0 * ((x + y + z + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GYGZ) {
      sign = 1.0 - 2.0 * ((y + z) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GZGX) {
      sign = 1.0 - 2.0 * ((z + x) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GXGY) {
      sign = 1.0 - 2.0 * ((x + y) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GXGT) {
      sign = 1.0 - 2.0 * ((x + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GYGT) {
      sign = 1.0 - 2.0 * ((y + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_GZGT) {
      sign = 1.0 - 2.0 * ((z + t) % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_G5GX) {
      sign = 1.0 - 2.0 * (x % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_G5GY) {
      sign = 1.0 - 2.0 * (y % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_G5GZ) {
      sign = 1.0 - 2.0 * (z % 2);
    } else if (Arg::gamma == QUDA_SPIN_TASTE_G5GT) {
      sign = 1.0 - 2.0 * (t % 2);
    }
    return sign;
  }

  template <typename Arg>
  __device__ __host__ void applySign(int idx, int parity, const Arg &arg) {
    using real = typename mapper<typename Arg::Float>::type;
    using Vector = ColorSpinor<real, Arg::nColor, 1>;

    int x[4];

    getCoords(x, idx, arg.X, parity);

    real sign = getSign(x[0], x[1], x[2], x[3], arg);

    Vector out = arg.in(idx, parity);

    arg.out(idx, parity) = sign * out;
  }

  template <typename Arg> struct SpinTastePhase
  {
    const Arg &arg;
    constexpr SpinTastePhase(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      applySign(x_cb, parity, arg);
    }
  };

}
