//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETNET_H
#define CTDET_TRT_CTDETNET_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "ctdetConfig.h"
#include "utils.h"
#include "logger.h"
#include "common/commons.h"
#include "common/argsParser.h"
//#include "NvOnnxParserRuntime.h"
#define tensorrt8 1

#ifdef tensorrt8

using samplesCommon::SampleUniquePtr;
namespace ctdet
{
    enum class RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };

    class ctdetNet
    {
    public:
        ctdetNet(const std::string& onnxFile,
                 const std::string& calibFile,
                 RUN_MODE mode = RUN_MODE::FLOAT32);

        ctdetNet(const std::string& engineFile);

        // the following config determines the int8 or fp16 model
        bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);
       

        ~ctdetNet(){
            cudaStreamSynchronize(mCudaStream);
            cudaStreamDestroy(mCudaStream);
            for(auto& item : mCudaBuffers)
                cudaFree(item);
            cudaFree(cudaOutputBuffer);
            if(!mRunTime)
                mRunTime->destroy();
            if(!mContext)
                mContext->destroy();
            if(!mEngine)
                mEngine->destroy();

        }

        void saveEngine(const std::string& fileName);

        void doInference(const void* inputData, void* outputData);

        void printTime()
        {
            mProfiler.printTime(runIters) ;
        }

        inline size_t getInputSize() {
            return mBindBufferSizes[0];
        };

        int64_t outputBufferSize;
        // bool forwardFace;
        int forwardFace;
    private:

        void InitEngine();

        nvinfer1::IExecutionContext* mContext;
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
        nvinfer1::IRuntime* mRunTime;

        
        nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
        nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
        int mNumber{0};             //!< The number to classify
        samplesCommon::OnnxSampleParams mParams;

        // subsitute config for the mode
        RUN_MODE runMode;

        std::vector<void*> mCudaBuffers;
        std::vector<int64_t> mBindBufferSizes;
        void * cudaOutputBuffer;

        cudaStream_t mCudaStream;

        int runIters;
        Profiler mProfiler;
    };

}



#else
namespace ctdet
{
    enum class RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };

    class ctdetNet
    {
    public:
        ctdetNet(const std::string& onnxFile,
                 const std::string& calibFile,
                 RUN_MODE mode = RUN_MODE::FLOAT32);

        ctdetNet(const std::string& engineFile);

        ~ctdetNet(){
            cudaStreamSynchronize(mCudaStream);
            cudaStreamDestroy(mCudaStream);
            for(auto& item : mCudaBuffers)
                cudaFree(item);
            cudaFree(cudaOutputBuffer);
            if(!mRunTime)
                mRunTime->destroy();
            if(!mContext)
                mContext->destroy();
            if(!mEngine)
                mEngine->destroy();

        }

        void saveEngine(const std::string& fileName);

        void doInference(const void* inputData, void* outputData);

        void printTime()
        {
            mProfiler.printTime(runIters) ;
        }

        inline size_t getInputSize() {
            return mBindBufferSizes[0];
        };

        int64_t outputBufferSize;
        // bool forwardFace;
        int forwardFace;
    private:

        void InitEngine();

        nvinfer1::IExecutionContext* mContext;
        nvinfer1::ICudaEngine* mEngine;
        nvinfer1::IRuntime* mRunTime;

        RUN_MODE runMode;

        // the following IPluginFactory has been removed from tensorrt8, use XX to subsitude.
        nvonnxparser::IPluginFactory *mPlugins;



        std::vector<void*> mCudaBuffers;
        std::vector<int64_t> mBindBufferSizes;
        void * cudaOutputBuffer;

        cudaStream_t mCudaStream;

        int runIters;
        Profiler mProfiler;
    };

}

#endif




#endif //CTDET_TRT_CTDETNET_H
