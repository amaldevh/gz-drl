// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef ORNSTEINUHLENBECK_HH_
#define ORNSTEINUHLENBECK_HH_
#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

/**
 * @brief A Multivariate Normal distribution in C++
 *
 */
class MVNormal
{
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    /**
     * @brief Construct a new MVNormal object
     *
     * @param mean mean along each dimension
     * @param cov covariance matrix
     * @param jitter  jitter
     */
    MVNormal(const Vec &mean, const Mat &cov, double jitter = 0.0)
        : dim_(mean.size()), mean_(mean)
    {
        assert(cov.rows() == dim_ && cov.cols() == dim_);
        Mat A = cov;
        if (jitter > 0.0)
            A.diagonal().array() += jitter; // for near-PSD covariances
        // LLT is ~2x faster than eigen-decomp when cov is SPD
        Eigen::LLT<Mat> llt(A);
        if (llt.info() != Eigen::Success)
        {
            // fall back to eigen-decomposition (handles semi-definite)
            Eigen::SelfAdjointEigenSolver<Mat> es(A);
            auto evals = es.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
            L_ = es.eigenvectors() * evals; // A ≈ L Lᵀ
        }
        else
        {
            L_ = llt.matrixL(); // A = L Lᵀ
        }
    }

    /**
     * @brief Generates a sample from URBG
     *
     * @tparam URBG Random generator class
     * @param gen generator instance
     * @return Vec generated values
     */
    template <class URBG>
    Vec sample(URBG &gen) const
    {
        thread_local std::normal_distribution<double> N01(0.0, 1.0);
        Vec z(dim_);
        for (int i = 0; i < dim_; ++i)
            z[i] = N01(gen);
        return mean_ + L_ * z;
    }

    /**
     * @brief Multi-sample overload of sample
     *
     * @tparam URBG generator class
     * @param gen generator instance
     * @param n number of samples to generate
     * @return Mat generated samples
     */
    template <class URBG>
    Mat sampleN(URBG &gen, int n) const
    {
        Mat Z(dim_, n);
        thread_local std::normal_distribution<double> N01(0.0, 1.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < dim_; ++i)
                Z(i, j) = N01(gen);
        return mean_.replicate(1, n) + L_ * Z;
    }
    /**
     * @brief Return the lower triangular decomposition of covariance matrix
     *
     * @return const Mat& lower triangle matrix
     */
    const Mat &L() const { return L_; }
    /**
     * @brief Returns the mean
     *
     * @return const Vec& mean
     */
    const Vec &mean() const { return mean_; }

private:
    int dim_;  ///< dimension
    Vec mean_; ///< mean
    Mat L_;    ///< Lower traingular decomposition of covariance
};

/**
 * @brief A struct holding OU params
 *
 * @tparam dims DImension of the state-vector
 */
template <size_t dims>
struct OUParams
{
    typedef Eigen::Matrix<double, dims, 1> VecType;
    typedef Eigen::Matrix<double, dims, dims> MatrixType;
    VecType mu_;          ///< long-term mean (d-dimensional)
    MatrixType theta_;    ///< drift/reversion matrix (d x d)
    MatrixType sigma_;    ///< diffusion matrix (d x d)
    MatrixType lt_sigma_; ///< lower triangular decomposition of cov
    /**
     * @brief Construct a new OUParams object
     *
     * @param mu mean
     * @param theta drift reversion
     * @param sigma  covariance
     */
    OUParams(VecType mu, MatrixType theta, MatrixType sigma) : mu_(mu), theta_(theta), sigma_(sigma)
    {
        lt_sigma_ = Eigen::LLT<MatrixType>(sigma).matrixL();
    };
    /**
     * @brief Construct a new OUParams object
     *
     */
    OUParams() {};
};

/**
 * @brief Multivariate OU process
 *
 * @tparam dims dimension of state
 */
template <size_t dims>
class OrnsteinUhlenbeckMultivar
{
public:
    typedef Eigen::Matrix<double, dims, 1> VecType;
    typedef Eigen::Matrix<double, dims, dims> MatrixType;
    /**
     * @brief Construct a new Ornstein Uhlenbeck Multivar object
     *
     * @param x0 initial state
     * @param params  OUParams config
     * @param dt time step
     */
    OrnsteinUhlenbeckMultivar(
        const VecType &x0, const OUParams<dims> &params, double dt)
        : x_(x0),
          p_(params),
          dt_(dt)
    {
        this->rng_.seed(42);
        this->d_ = x0.size();
        this->dist_ = std::normal_distribution<double>(0.0, 1.0);
        this->eps_.resize(this->d_);
        this->dt_root = std::sqrt(this->dt_);
    }

    /**
     * @brief perform single Euler-Maruyama step
     *
     * @return const VecType& next state
     */
    const VecType &step()
    {

        for (int i = 0; i < this->d_; i++)
        {
            this->eps_(i) = this->dist_(this->rng_);
        }
        // dx = A(mu - x) dt + Sigma sqrt(dt) eps
        this->x_ += this->p_.theta_ * (this->p_.mu_ - this->x_) * this->dt_ + this->p_.lt_sigma_ * (this->dt_root * this->eps_);
        return x_;
    }
    /**
     * @brief Get the current state
     *
     * @return const VecType& current state
     */
    const VecType &state() const { return x_; }
    /**
     * @brief Reset the state to zeros
     *
     */
    void reset() { this->x_ = VecType::Zero(); }

private:
    VecType x_;        ///< state
    OUParams<dims> p_; ///< OUParams config
    double dt_;        ///< time step
    double dt_root;    ///< square root of time step
    int d_;            ///< dimension of state

    std::mt19937 rng_;                      ///< rng generator
    std::normal_distribution<double> dist_; ///< normal distribution
    VecType eps_;                           ///< epsilon
};

#endif