#pragma once

#include "complex_quda.h"
#include "quda_constants.h"
#include "quda_api.h"
#include <math_helper.cuh>
#include <array.h>

/**
   @file reducer.h

   Implementations of various helper classes used for reductions,
   together with transformers used to aid reduction.  Generally
   speaking the reducers are binary operators, taking two arguments
   and reducing to a single value.  Transformers on the other hand are
   unary operators, though the return type may be different.  The
   transformers are able to work on fixed-point data, as we have
   specializations to handle the rescaling required in these cases.
*/

namespace quda
{

  namespace reducer
  {
    /**
       @return the reduce buffer size allocated
    */
    size_t buffer_size();

    /**
       @return pointer to device reduction buffer
    */
    void *get_device_buffer();

    /**
       @return pointer to device-mapped host reduction buffer
    */
    void *get_mapped_buffer();

    /**
       @return pointer to host reduction buffer
    */
    void *get_host_buffer();

    /**
       @brief get_count returns the pointer to the counter array used
       for tracking the number of completed thread blocks.  We
       template this function, since the return type is target
       dependent.
       @return pointer to the reduction count array.
     */
    template <typename count_t> count_t *get_count();

    /**
       @return reference to the event used for synchronizing
       reductions with the host
     */
    qudaEvent_t &get_event();
  } // namespace reducer

  constexpr int max_n_reduce() { return QUDA_MAX_MULTI_REDUCE; }

  /**
     plus reducer, used for conventional sum reductions
   */
  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    using reducer_t = plus<T>;
    __device__ __host__ inline T operator()(T a, T b) const { return a + b; }
  };

  /**
     maximum reducer, used for max reductions
   */
  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    using reducer_t = maximum<T>;
    __device__ __host__ inline T operator()(T a, T b) const { return quda::max(a,b); }
  };

  /**
     minimum reducer, used for min reductions
   */
  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    using reducer_t = minimum<T>;
    __device__ __host__ inline T operator()(T a, T b) const { return a < b ? a : b; }
  };

  /**
     identity transformer, preserves input
   */
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    __device__ __host__ inline T operator()(T a) const { return a; }
  };

  /**
     square transformer, return the L2 norm squared of the input
   */
  template <typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x) const
    {
      return static_cast<ReduceType>(norm(x));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (int8_t specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (short specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (int specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input
   */
  template <typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const { return abs(x); }
  };

  /**
     abs transformer, return the absolute value of the input (int8_t
     specialization)
   */
  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input (short
     specialization)
   */
  template <typename Float> struct abs_<Float, short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input (int
     specialization)
   */
  template <typename Float> struct abs_<Float, int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components
   */
  template <typename Float, typename storeFloat> struct abs_max_ {
    abs_max_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const
    {
      return maximum<Float>()(abs(x.real()), abs(x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (int8_t specialziation)
   */
  template <typename Float> struct abs_max_<Float, int8_t> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (short specialziation)
   */
  template <typename Float> struct abs_max_<Float, short> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (int specialziation)
   */
  template <typename Float> struct abs_max_<Float, int> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components
   */
  template <typename Float, typename storeFloat> struct abs_min_ {
    abs_min_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const
    {
      return minimum<Float>()(abs(x.real()), abs(x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (int8_t specialziation)
   */
  template <typename Float> struct abs_min_<Float, int8_t> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (short specialziation)
   */
  template <typename Float> struct abs_min_<Float, short> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (int specialziation)
   */
  template <typename Float> struct abs_min_<Float, int> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

} // namespace quda
