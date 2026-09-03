// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#pragma once
#ifndef ASYNC_RL_SERVER_HH_
#define ASYNC_RL_SERVER_HH_
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "rl_server.hh"

namespace async
{

  /**
   * @brief *A thread-safe blocking queue
   * Used to hold commands for AsyncDRLServerPool workers
   *
   * @tparam T is the type of elements stored in the queue
   */
  template <class T>
  class BlockingQueue
  {
  public:
    /**
     * @brief Pushes an element into the queue
     *
     * @param v Element to push
     *
     * @throws std::runtime_error if the queue is closed
     */
    void push(T v)
    {
      {
        std::lock_guard<std::mutex> lk(m_);
        if (closed_)
          throw std::runtime_error("push on closed queue");
        q_.push(std::move(v));
      }
      cv_.notify_one();
    }
    /**
     * @brief Pops an element from the queue
     * Blocks if the queue is empty until an element is available or the queue is closed
     *
     * @param out Reference to store the popped element
     *
     * @return true if an element was popped, false if the queue is closed and empty
     */
    bool pop(T &out)
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&]
               { return closed_ || !q_.empty(); });
      if (q_.empty())
        return false;
      out = std::move(q_.front());
      q_.pop();
      return true;
    }
    /**
     * @brief Closes the queue
     * After closing, no more elements can be pushed
     * All waiting pop calls will be unblocked
     */
    void close()
    {
      {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
      }
      cv_.notify_all();
    }

  private:
    std::mutex m_;               ///< Mutex for thread safety
    std::condition_variable cv_; ///< Condition variable for blocking pops
    std::queue<T> q_;            ///< Underlying queue
    bool closed_{false};         ///< Closed flag
  };

  /**
   * @brief AsyncDRLServerPool
   *  A pool of DRLServer instances for asynchronous simulation
   *  Allows parallel execution of multiple DRLServer instances in separate threads
   */
  class AsyncDRLServerPool
  {
  public:
    using ServerPtr = std::shared_ptr<DRLServer>;
    using ServerFactory = std::function<ServerPtr(std::size_t)>;

    /**
     * @brief Constructs the pool with a vector of existing servers
     * @param servers Vector of shared pointers to DRLServer instances
     * @throws std::invalid_argument if num_servers is zero
     */
    AsyncDRLServerPool(const std::vector<ServerPtr> &servers) : num_servers_(servers.size()),
                                                                queues_(servers.size()),
                                                                servers_(servers.size())
    {
      const size_t n = servers.size();
      auto factory = [this, n, &servers](std::size_t i)
      {
        return servers[i];
      };
      InitMethod(factory);
    }
    /**
     * @brief Constructs the pool with a number of servers and a factory function
     * @param num_servers Number of DRLServer instances to create
     * @param factory Factory function to create DRLServer instances
     * @throws std::invalid_argument if num_servers is zero
     */
    AsyncDRLServerPool(std::size_t num_servers, ServerFactory factory)
        : num_servers_(num_servers),
          queues_(num_servers),
          servers_(num_servers)
    {
      InitMethod(factory);
    }

    /**
     * @brief Initializes the pool with a factory function
     * @param factory Factory function to create DRLServer instances
     * @throws std::invalid_argument if num_servers is zero
     */
    void InitMethod(ServerFactory factory)
    {

      if (num_servers_ == 0)
        throw std::invalid_argument("num_servers must be > 0");

      for (std::size_t i = 0; i < num_servers_; ++i)
      {
        servers_[i] = factory(i);
        if (!servers_[i])
          throw std::runtime_error("ServerFactory returned null");
      }
      const std::size_t processor_count = std::max<std::size_t>(
          1, std::thread::hardware_concurrency());
      workers_.reserve(num_servers_);
      for (std::size_t i = 0; i < num_servers_; ++i)
      {
        workers_.emplace_back([this, i]
                              { worker_loop(i); });
      }
      // set thread affinity
      for (std::size_t tid = 0; tid < num_servers_; ++tid)
      {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        std::size_t cid = (tid) % processor_count;
        CPU_SET(cid, &cpuset);
        pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t),
                               &cpuset);
      }
    }

    /**
     *  @brief Destructor for AsyncDRLServerPool
     * Joins all worker threads and closes all queues
     */
    ~AsyncDRLServerPool()
    {
      for (auto &q : queues_)
        q.close();
      for (auto &t : workers_)
        if (t.joinable())
          t.join();
    }
    /**
     * @brief Returns the number of servers in the pool
     * @return Number of DRLServer instances
     */
    std::size_t size() const noexcept { return num_servers_; }

    /**
     * @brief Submits a task to a specific server in the pool
     * @param env_id Index of the server to submit the task to
     * @param f Function to execute on the server
     * @return Future representing the result of the task
     * @throws std::out_of_range if env_id is invalid
     */
    template <class F>
    auto submit(std::size_t env_id, F &&f)
        -> std::future<std::invoke_result_t<F, DRLServer &>>
    {
      using R = std::invoke_result_t<F, DRLServer &>;
      check_env(env_id);

      auto task = std::make_shared<std::packaged_task<R(DRLServer &)>>(std::forward<F>(f));
      auto fut = task->get_future();
      queues_[env_id].push(Command{[task = std::move(task)](DRLServer &s) mutable
                                   { (*task)(s); }});
      return fut;
    }

    /**
     * @brief Calls a member function on a specific server in the pool
     * @param env_id Index of the server to call the function on
     * @param mf Member function pointer to call
     * @param args Arguments to pass to the member function
     * @return Future representing the result of the member function call
     * @throws std::out_of_range if env_id is invalid
     */
    template <class MemFn, class... Args>
    auto call(std::size_t env_id, MemFn mf, Args &&...args)
        -> std::future<std::invoke_result_t<MemFn, DRLServer &, Args...>>
    {
      using R = std::invoke_result_t<MemFn, DRLServer &, Args...>;
      using Tup = std::tuple<std::decay_t<Args>...>;
      check_env(env_id);

      // Move/copy args now; we own them. We'll conditionally std::move at invoke time.
      auto caller = [mf, tup = Tup(std::forward<Args>(args)...)](DRLServer &srv) mutable -> R
      {
        return invoke_with_pack<MemFn, R, Tup, Args...>(srv, mf, tup,
                                                        std::make_index_sequence<sizeof...(Args)>{});
      };

      return submit(env_id, std::move(caller));
    }

    /**
     * @brief Closes all queues and joins all worker threads
     * Ensures clean shutdown of the pool
     */
    void close()
    {
      for (auto &q : queues_)
        q.close();
      for (auto &t : workers_)
        if (t.joinable())
          t.join();
      workers_.clear();
      servers_.clear();
      num_servers_ = 0;
    }

  private:
    /**
     * @brief Command structure holding a function to execute on a DRLServer
     *
     */
    struct Command
    {
      std::function<void(DRLServer &)> fn;
    };
    /**
     * @brief Worker loop that processes commands from the queue for a specific server
     *
     * @param env_id environment id to get the DRLServer
     */
    void worker_loop(std::size_t env_id)
    {
      Command c;
      while (queues_[env_id].pop(c))
      {
        try
        {
          c.fn(*servers_[env_id]);
        }
        catch (...)
        { /* packaged_task handles */
        }
      }
    }
    /**
     * @brief Checks if env_id is valid
     *
     * @param env_id Index to check
     *
     * @throws std::out_of_range if env_id is invalid
     */
    void check_env(std::size_t env_id) const
    {
      if (env_id >= num_servers_)
        throw std::out_of_range("env_id out of range");
    }

    /**
     * @brief Helper function to invoke a member function with unpacked tuple arguments
     * @tparam MemFn Member function pointer type
     * @tparam R Return type of the member function
     * @tparam Tup Tuple type holding the arguments
     * @tparam Args Parameter pack of argument types
     * @param srv DRLServer instance to invoke the member function on
     * @param mf Member function pointer to invoke
     * @param tup Tuple holding the arguments
     * @param std::index_sequence for unpacking the tuple
     * @return Result of the member function invocation
     */
    template <class MemFn, class R, class Tup, class... Args, std::size_t... I>
    static R invoke_with_pack(DRLServer &srv, MemFn mf, Tup &tup, std::index_sequence<I...>)
    {
      return (srv.*mf)(forward_arg<I, Args>(std::get<I>(tup))...);
    }
    /**
     * @brief Forwards an argument based on its original value category
     * @tparam I Index of the argument
     * @tparam A Original argument type
     * @tparam T Current type of the argument
     * @param x Argument to forward
     * @return Forwarded argument, moved if originally an rvalue
     */
    template <std::size_t I, class A, class T>
    static decltype(auto) forward_arg(T &x)
    {
      // If A was passed as rvalue at submission, move it now; else keep lvalue.
      if constexpr (std::is_rvalue_reference_v<A &&>)
      {
        return std::move(x); // crucial: enables calling `&&` overloads
      }
      else
      {
        return x;
      }
    }

    std::size_t num_servers_;                    ///< Number of DRLServers managed
    std::vector<BlockingQueue<Command>> queues_; ///< Command queues

    std::vector<std::thread> workers_; ///< Worker threads
  public:
    std::vector<ServerPtr> servers_; ///< DRLServer instances
  };

} // namespace async
#endif // ASYNC_RL_SERVER_HH_
