// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef ONNX_HH
#define ONNX_HH

#include <thread>
#include <Eigen/Core>
#include <onnxruntime_c_api.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cerrno>
#include <codecvt>
#include <locale>
#include <stdexcept>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "cpp11_compat.hh"

class ONNXModel {
public:
    ONNXModel(const std::string& model_path, bool use_gpu = true): model_path_(model_path),
     use_gpu_(use_gpu), api_(OrtGetApiBase()->GetApi(ORT_API_VERSION)), env_(NULL), session_(NULL),
     session_options_(NULL), allocator_(NULL), memory_info_(NULL), input_size_(0), output_size_(0) {
        LoadModel();
    };
    ~ONNXModel() {
        if (memory_info_ != NULL) {
            api_->ReleaseMemoryInfo(memory_info_);
            memory_info_ = NULL;
        }
        if (session_ != NULL) {
            api_->ReleaseSession(session_);
            session_ = NULL;
        }
        if (session_options_ != NULL) {
            api_->ReleaseSessionOptions(session_options_);
            session_options_ = NULL;
        }
        if (env_ != NULL) {
            api_->ReleaseEnv(env_);
            env_ = NULL;
        }
    }
    
    void RunInference(Eigen::VectorXf& observation, Eigen::VectorXf& action){
        if (observation.size() != input_size_){
            throw std::runtime_error("Observation size " + std::to_string(observation.size()) + " does not match model input size " + std::to_string(input_size_) + ".");
        }
        OrtValue* input_tensor = NULL;
        ThrowOnError(api_->CreateTensorWithDataAsOrtValue(
            memory_info_,
            observation.data(),
            static_cast<size_t>(input_size_) * sizeof(float),
            &input_dims_[0],
            input_dims_.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        OrtValue* output_tensor = NULL;
        const OrtValue* input_tensors[] = {input_tensor};
        ThrowOnError(api_->Run(
            session_,
            NULL,
            input_names,
            input_tensors,
            1,
            output_names,
            1,
            &output_tensor));

        float* output_data = NULL;
        ThrowOnError(api_->GetTensorMutableData(output_tensor, reinterpret_cast<void**>(&output_data)));
        
        action.resize(output_size_);
        for (size_t i = 0; i < output_size_; ++i) {
            action[i] = output_data[i];
        }

        api_->ReleaseValue(output_tensor);
        api_->ReleaseValue(input_tensor);
    };
private:
    void ThrowOnError(OrtStatus* status) {
        if (status != NULL) {
            const char* msg = api_->GetErrorMessage(status);
            std::string err = (msg != NULL) ? std::string(msg) : std::string("ONNX runtime error");
            api_->ReleaseStatus(status);
            throw std::runtime_error(err);
        }
    }

    static std::string GetFileName(const std::string& path) {
        const std::string::size_type pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(pos + 1);
    }

    static std::string GetParentPath(const std::string& path) {
        const std::string::size_type pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return ".";
        }
        return path.substr(0, pos);
    }

    static std::string JoinPath(const std::string& left, const std::string& right) {
        if (left.empty()) {
            return right;
        }
        const char last = left[left.size() - 1];
        if (last == '/' || last == '\\') {
            return left + right;
        }
#ifdef _WIN32
        return left + "\\" + right;
#else
        return left + "/" + right;
#endif
    }

    static void CreateDirectoryIfMissing(const std::string& path) {
#ifdef _WIN32
        const int rc = _mkdir(path.c_str());
#else
        const int rc = mkdir(path.c_str(), 0755);
#endif
        if (rc != 0 && errno != EEXIST) {
            std::cout << "[ONNX] Failed to create cache directory: " << path << std::endl;
        }
    }

    void LoadModel(){
        const std::string model_path = model_path_;
        std::string model_name = GetFileName(model_path);
        std::cout << "[ONNX] Loading ONNX model: " << model_name << std::endl;
        
        ThrowOnError(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNX", &env_));
        
        ThrowOnError(api_->CreateSessionOptions(&session_options_));
        ThrowOnError(api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
        ThrowOnError(api_->SetSessionExecutionMode(session_options_, ORT_SEQUENTIAL));
        ThrowOnError(api_->EnableMemPattern(session_options_));
        ThrowOnError(api_->EnableCpuMemArena(session_options_));
        
        if (use_gpu_) {
            std::cout << "[ONNX] C++11 build uses ONNX C API in CPU mode by default" << std::endl;
        }
        
    #ifdef _WIN32
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t> > converter;
        const std::wstring abs_path_str = converter.from_bytes(model_path);
        ThrowOnError(api_->CreateSession(env_, abs_path_str.c_str(), session_options_, &session_));
    #else
        const std::string abs_path_str = model_path;
        ThrowOnError(api_->CreateSession(env_, abs_path_str.c_str(), session_options_, &session_));
    #endif

        ThrowOnError(api_->GetAllocatorWithDefaultOptions(&allocator_));
        ThrowOnError(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));
        
        char* input_name_ptr = NULL;
        ThrowOnError(api_->SessionGetInputName(session_, 0, allocator_, &input_name_ptr));
        input_name_ = input_name_ptr;
        ThrowOnError(api_->AllocatorFree(allocator_, input_name_ptr));

        char* output_name_ptr = NULL;
        ThrowOnError(api_->SessionGetOutputName(session_, 0, allocator_, &output_name_ptr));
        output_name_ = output_name_ptr;
        ThrowOnError(api_->AllocatorFree(allocator_, output_name_ptr));

        OrtTypeInfo* input_type_info = NULL;
        ThrowOnError(api_->SessionGetInputTypeInfo(session_, 0, &input_type_info));
        const OrtTensorTypeAndShapeInfo* input_tensor_info = NULL;
        ThrowOnError(api_->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));
        size_t input_num_dims = 0;
        ThrowOnError(api_->GetDimensionsCount(input_tensor_info, &input_num_dims));
        input_shape_.resize(input_num_dims);
        ThrowOnError(api_->GetDimensions(input_tensor_info, &input_shape_[0], input_num_dims));
        api_->ReleaseTypeInfo(input_type_info);

        OrtTypeInfo* output_type_info = NULL;
        ThrowOnError(api_->SessionGetOutputTypeInfo(session_, 0, &output_type_info));
        const OrtTensorTypeAndShapeInfo* output_tensor_info = NULL;
        ThrowOnError(api_->CastTypeInfoToTensorInfo(output_type_info, &output_tensor_info));
        size_t output_num_dims = 0;
        ThrowOnError(api_->GetDimensionsCount(output_tensor_info, &output_num_dims));
        output_shape_.resize(output_num_dims);
        ThrowOnError(api_->GetDimensions(output_tensor_info, &output_shape_[0], output_num_dims));
        api_->ReleaseTypeInfo(output_type_info);
        
        input_dims_.resize(input_shape_.size());
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            input_dims_[i] = (input_shape_[i] > 0) ? input_shape_[i] : 1;
        }
        input_size_ = 1;
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            input_size_ *= static_cast<size_t>(input_dims_[i]);
        }
        output_size_ = 1;
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            const int64_t dim = (output_shape_[i] > 0) ? output_shape_[i] : 1;
            output_size_ *= static_cast<size_t>(dim);
        }

        

        std::cout<<"[ONNX] Summary for model "<<model_name<<":"<<std::endl;
        std::cout << "[ONNX] Model loaded successfully" << std::endl;
        std::cout << "[ONNX] Input shape: [" << input_shape_[0] << ", " << input_shape_[1] << "]" << std::endl;
        std::cout << "[ONNX] Output shape: [" << output_shape_[0] << ", " << output_shape_[1] << "]" << std::endl;
        
    };
    const OrtApi* api_;
    OrtEnv* env_;
    OrtSession* session_;
    OrtSessionOptions* session_options_;
    OrtAllocator* allocator_;
    OrtMemoryInfo* memory_info_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    // model and inference configuration
    std::string model_path_;
    bool use_gpu_ = true;

    // runtime params
    std::vector<int64_t> input_dims_;
    size_t input_size_;
    size_t output_size_;

};
#endif