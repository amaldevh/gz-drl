// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef SECOND_ORDER_LP_FILTER_HH
#define SECOND_ORDER_LP_FILTER_HH

#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include "print_utils.hh"

namespace sitl
{
    /**
     * @brief Implements a second order low-pass filter
     *
     * @tparam dim dimension of the LP filter
     * @tparam T data type
     */
    template <size_t dim, typename T>
    class SecondOrderLPFilter
    {
        using VecType = Eigen::Matrix<T, dim, 1>;
        using MatType = Eigen::Matrix<T, dim, dim>;

    public:
        /**
         * @brief Constructor
         * @param omega_n Cutoff frequency in rad/s
         * @param eta Damping ratio
         * @param dt Time step in seconds
         */
        SecondOrderLPFilter(T omega_n, T eta, T dt)
            : omega_n_(omega_n), eta_(eta), dt_(dt)
        {
            ComputeCoefficients();
            Reset();
        }
        /**
         * @brief Reset the filter state */
        void Reset()
        {
            x1_.setZero();
            x2_.setZero();
            y1_.setZero();
            y2_.setZero();
        }
        /**
         *  @brief Update the filter with a new input sample
         * @param input New input sample
         * @return Filtered output sample
         */
        const VecType &Update(const VecType &input)
        {

            const VecType y = b0_ * input +
                              b1_ * x1_ +
                              b2_ * x2_ -
                              a1_ * y1_ -
                              a2_ * y2_;
            x2_ = x1_;
            x1_ = input;
            y2_ = y1_;
            y1_ = y;
            return y1_;
        }

    private:
        /**
         * @brief Compute filter coefficients based on cutoff frequency, damping ratio, and time step */
        void ComputeCoefficients()
        {
            const T omega_n2 = omega_n_ * omega_n_;
            const T k = 2.0 / dt_;
            const T k2 = k * k;

            // Denominator coefficients (normalize by a0)
            const T a0 = k2 + 2.0 * eta_ * omega_n_ * k + omega_n2;
            a1_ = (2.0 * omega_n2 - 2.0 * k2) / a0;
            a2_ = (k2 - 2.0 * eta_ * omega_n_ * k + omega_n2) / a0;

            // Numerator coefficients
            b0_ = omega_n2 / a0;
            b1_ = 2.0 * omega_n2 / a0;
            b2_ = omega_n2 / a0;
        }

        T b0_, b1_, b2_, a1_, a2_; ///< Filter coefficients

        T omega_n_, eta_, dt_; ///< Cutoff frequency, damping ratio, and time step
        // State vectors for inputs and outputs
        VecType x1_; ///< Last input
        VecType x2_; ///< Last Last input
        VecType y1_; ///< Last output
        VecType y2_; ///< Last Last output
    };

} // namespace sitl

#endif // SECOND_ORDER_LP_FILTER_HH