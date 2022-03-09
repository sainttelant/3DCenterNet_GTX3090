//
// Created by cao on 19-10-26.
//

#include <assert.h>
#include <fstream>
#include "ctdetNet.h"
#include "ctdetLayer.h"
#include "entroyCalibrator.h"

using namespace std;

#ifdef tensorrt8



namespace ctdet
{
   
    ctdetNet::ctdetNet(const std::string &onnxFile, const std::string &calibFile,
            ctdet::RUN_MODE mode):
            forwardFace(0),
            mContext(nullptr),
            mEngine(nullptr),
            mRunTime(nullptr),
           
            runMode(mode),
            runIters(0)   
    {

        memset(&mParams,0,sizeof(mParams));
         mParams.onnxFileName = onnxFile;
         //mParams.dataDirs = 
         std::cout<<"wilson init ctdet begin<<<<<<<<<<<1 \n"<<std::endl;
        const int maxBatchSize = 1;
        nvinfer1::IHostMemory *modelStream{nullptr};
        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
       std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.1 \n"<<std::endl;     
        //nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger);
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.2 \n"<<std::endl;   
        if (!builder)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
         std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.3 \n"<<std::endl;   
        if (!network)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.4 \n"<<std::endl;  
        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.5 \n"<<std::endl;  
        auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.6 \n"<<std::endl;  
        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.7 \n"<<std::endl;  
        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.8 \n"<<std::endl;  
        config->setProfileStream(*profileStream);

        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            std::cout<<"build failed!"<<std::endl;
        }
std::cout<<"wilson init ctdet begin<<<<<<<<<<<1.9 \n"<<std::endl;  
        SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        if (!runtime)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
        if (!mEngine)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        std::cout<<"wilson init ctdet begin<<<<<<<<<<<2 \n"<<std::endl;

        ASSERT(network->getNbInputs() == 1);
        mInputDims = network->getInput(0)->getDimensions();
        ASSERT(mInputDims.nbDims == 4);

        ASSERT(network->getNbOutputs() == 1);
        mOutputDims = network->getOutput(0)->getDimensions();
        ASSERT(mOutputDims.nbDims == 2);   

        std::cout<<"wilson init ctdet begin<<<<<<<<<<<3 \n"<<std::endl;
        //auto parser = nvonnxparser::createParser(*network, sample::gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), verbosity))
        {
            std::string msg("failed to parse onnx file");
            sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        std::cout<<"wilson init ctdet begin<<<<<<<<<<<4 \n"<<std::endl;

        builder->setMaxBatchSize(maxBatchSize);
       // builder->setMaxWorkspaceSize(1 << 30);// 1G

        nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
        if(calibFile.size()>0) calibrator = new nvinfer1::int8EntroyCalibrator(maxBatchSize,calibFile,"calib.table");


        // use config to determine which mode ultilized
       /*  if (runMode== RUN_MODE::INT8)
        {
            //nvinfer1::IInt8Calibrator* calibrator;
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(calibrator);
        }
        else if (runMode == RUN_MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
            builder->setFp16Mode(true);
        } */
        // config input shape

        std::cout << "Begin building engine..." << std::endl;
        //nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
        if (!engine){
            std::string error_message ="Unable to create engine";
            sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
        }
        std::cout << "End building engine..." << std::endl;

        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        // We don't need the network any more, and we can destroy the parser.


        // Serialize the engine, then close everything down.
        modelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        builder->destroy();
        parser->destroy();
        assert(modelStream != nullptr);
        mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
        assert(mRunTime != nullptr);
        //mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), mPlugins);

        assert(mEngine != nullptr);
        modelStream->destroy();
        InitEngine();
        std::cout<<"wilson init ctdet begin<<<<<<<<<<<5 \n"<<std::endl;

    }
  
   bool ctdetNet::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{

     std::cout<<"wilson  constructNetwork<<<<<<<<<<<1 \n"<<std::endl;
     printf("mParams.onnxFileName:%s \n",mParams.onnxFileName.c_str());
     printf("mParams.dataDirs:%d \n",mParams.dataDirs.size());
     std::string onnxpath = "/home/xuewei/tensort-3D/model/ddd_3dop.onnx";
    //auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        //static_cast<int>(sample::gLogger.getReportableSeverity()));
    auto parsed = parser->parseFromFile(onnxpath.c_str(),static_cast<int>(sample::gLogger.getReportableSeverity()));



        
    printf("parsed = %d \n",parsed);
      std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.1 \n"<<std::endl;
    if (!parsed)
    {
        std::cout<<"parsed is null, return false \n"<<std::endl;
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
     std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.2 \n"<<std::endl;
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }
    std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.3 \n"<<std::endl;
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    std::cout<<"wilson  constructNetwork<<<<<<<<<<<2 \n"<<std::endl;
    return true;
}


    // for runDet
    ctdetNet::ctdetNet(const std::string &engineFile)
            :forwardFace(0),
            mContext(nullptr),
            mEngine(nullptr),
            mRunTime(nullptr),
            runMode(RUN_MODE::FLOAT32),
            runIters(0)
    {
        using namespace std;
        fstream file;

        file.open(engineFile,ios::binary | ios::in);
        if(!file.is_open())
        {
            cout << "read engine file" << engineFile <<" failed" << endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        std::cout << "deserializing" << std::endl;
        mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
        assert(mRunTime != nullptr);
       // mEngine= mRunTime->deserializeCudaEngine(data.get(), length, mPlugins);
        assert(mEngine != nullptr);
        InitEngine();
    }

    void ctdetNet::InitEngine() {
        const int maxBatchSize = 1;
        mContext = mEngine->createExecutionContext();
        assert(mContext != nullptr);
        mContext->setProfiler(&mProfiler);
        int nbBindings = mEngine->getNbBindings();
        // std::cout<<"mEngine->getNbBindings()"<<nbBindings<<std::endl;

        // if (nbBindings > 4) forwardFace= true;
        // face: 5, Helmet: 4, ctnet: 4, ddd: 7
        if (nbBindings == 4) forwardFace = 0;
        else if (nbBindings == 5) forwardFace = 1;
        else forwardFace = 2;
                

        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings);
        int64_t totalSize = 0;
        for (int i = 0; i < nbBindings; ++i)
        {
            nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
            mBindBufferSizes[i] = totalSize;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);
        }
        outputBufferSize = mBindBufferSizes[1] * 6 ;
        cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
    }

    void ctdetNet::doInference(const void *inputData, void *outputData)
    {
        const int batchSize = 1;
        int inputIndex = 0 ;
        CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[inputIndex], inputData, mBindBufferSizes[inputIndex], cudaMemcpyHostToDevice, mCudaStream));
        mContext->execute(batchSize, &mCudaBuffers[inputIndex]);
        CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
        // if (forwardFace){
        //     CTfaceforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
        //                       static_cast<const float *>(mCudaBuffers[3]),static_cast<const float *>(mCudaBuffers[4]),static_cast<float *>(cudaOutputBuffer),
        //                       input_w/4,input_h/4,classNum,kernelSize,visThresh);
        // } else{
        //     CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
        //                  static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
        //                      input_w/4,input_h/4,classNum,kernelSize,visThresh);
        // }

        if (forwardFace==1){
            CTfaceforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                              static_cast<const float *>(mCudaBuffers[3]),static_cast<const float *>(mCudaBuffers[4]),static_cast<float *>(cudaOutputBuffer),
                              input_w/4,input_h/4,classNum,kernelSize,visThresh);
        } 
        else if (forwardFace==0)
        {
            CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
                             input_w/4,input_h/4,classNum,kernelSize,visThresh);
        }
        else if (forwardFace==2)
        {
            CTdddforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]), static_cast<const float *>(mCudaBuffers[4]), static_cast<const float *>(mCudaBuffers[5]), static_cast<const float *>(mCudaBuffers[6]), static_cast<float *>(cudaOutputBuffer),
                             input_w/4,input_h/4,classNum,kernelSize,visThresh);            
        }


        CUDA_CHECK(cudaMemcpyAsync(outputData, cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream));

        runIters++ ;
    }
    
    void ctdetNet::saveEngine(const std::string &fileName)
    {
        cout<<"wilson<<<<<<<<<in saveEngine fuct1 \n"<<endl;
        if(mEngine)
        {
            cout<<"wilson<<<<<<<<<in saveEngine fuct engine in\n"<<endl;
            nvinfer1::IHostMemory* data = mEngine->serialize();
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                return;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
        }
        cout<<"wilson<<<<<<<<<in saveEngine fuctout\n"<<endl;
    }
}






#else
static Logger sample::gLogger;
namespace ctdet
{
   
    ctdetNet::ctdetNet(const std::string &onnxFile, const std::string &calibFile,
            ctdet::RUN_MODE mode):forwardFace(0),mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),
                                  runMode(mode),runIters(0),mPlugins(nullptr)    
    {

        const int maxBatchSize = 1;
        nvinfer1::IHostMemory *modelStream{nullptr};
        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();

        mPlugins = nvonnxparser::createPluginFactory(sample::gLogger);
        auto parser = nvonnxparser::createParser(*network, sample::gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), verbosity))
        {
            std::string msg("failed to parse onnx file");
            sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 30);// 1G

        nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
        if(calibFile.size()>0) calibrator = new nvinfer1::int8EntroyCalibrator(maxBatchSize,calibFile,"calib.table");
        if (runMode== RUN_MODE::INT8)
        {
            //nvinfer1::IInt8Calibrator* calibrator;
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(calibrator);
        }
        else if (runMode == RUN_MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
            builder->setFp16Mode(true);
        }
        // config input shape

        std::cout << "Begin building engine..." << std::endl;
        nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (!engine){
            std::string error_message ="Unable to create engine";
            sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
        }
        std::cout << "End building engine..." << std::endl;

        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        // We don't need the network any more, and we can destroy the parser.


        // Serialize the engine, then close everything down.
        modelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        builder->destroy();
        parser->destroy();
        assert(modelStream != nullptr);
        mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), mPlugins);
        assert(mEngine != nullptr);
        modelStream->destroy();
        InitEngine();

    }
    // for runDet
    // ctdetNet::ctdetNet(const std::string &engineFile)
    //         :forwardFace(false),mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),runMode(RUN_MODE::FLOAT32),runIters(0),
    //         mPlugins(nullptr)
    ctdetNet::ctdetNet(const std::string &engineFile)
            :forwardFace(0),mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),runMode(RUN_MODE::FLOAT32),runIters(0),
            mPlugins(nullptr)
    {
        using namespace std;
        fstream file;

        file.open(engineFile,ios::binary | ios::in);
        if(!file.is_open())
        {
            cout << "read engine file" << engineFile <<" failed" << endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        mPlugins = nvonnxparser::createPluginFactory(sample::gLogger);
        std::cout << "deserializing" << std::endl;
        mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(data.get(), length, mPlugins);
        assert(mEngine != nullptr);
        InitEngine();
    }

    void ctdetNet::InitEngine() {
        const int maxBatchSize = 1;
        mContext = mEngine->createExecutionContext();
        assert(mContext != nullptr);
        mContext->setProfiler(&mProfiler);
        int nbBindings = mEngine->getNbBindings();
        // std::cout<<"mEngine->getNbBindings()"<<nbBindings<<std::endl;

        // if (nbBindings > 4) forwardFace= true;
        // face: 5, Helmet: 4, ctnet: 4, ddd: 7
        if (nbBindings == 4) forwardFace = 0;
        else if (nbBindings == 5) forwardFace = 1;
        else forwardFace = 2;
                

        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings);
        int64_t totalSize = 0;
        for (int i = 0; i < nbBindings; ++i)
        {
            nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
            mBindBufferSizes[i] = totalSize;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);
        }
        outputBufferSize = mBindBufferSizes[1] * 6 ;
        cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
    }

    void ctdetNet::doInference(const void *inputData, void *outputData)
    {
        const int batchSize = 1;
        int inputIndex = 0 ;
        CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[inputIndex], inputData, mBindBufferSizes[inputIndex], cudaMemcpyHostToDevice, mCudaStream));
        mContext->execute(batchSize, &mCudaBuffers[inputIndex]);
        CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
        // if (forwardFace){
        //     CTfaceforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
        //                       static_cast<const float *>(mCudaBuffers[3]),static_cast<const float *>(mCudaBuffers[4]),static_cast<float *>(cudaOutputBuffer),
        //                       input_w/4,input_h/4,classNum,kernelSize,visThresh);
        // } else{
        //     CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
        //                  static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
        //                      input_w/4,input_h/4,classNum,kernelSize,visThresh);
        // }

        if (forwardFace==1){
            CTfaceforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                              static_cast<const float *>(mCudaBuffers[3]),static_cast<const float *>(mCudaBuffers[4]),static_cast<float *>(cudaOutputBuffer),
                              input_w/4,input_h/4,classNum,kernelSize,visThresh);
        } 
        else if (forwardFace==0)
        {
            CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
                             input_w/4,input_h/4,classNum,kernelSize,visThresh);
        }
        else if (forwardFace==2)
        {
            CTdddforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]), static_cast<const float *>(mCudaBuffers[4]), static_cast<const float *>(mCudaBuffers[5]), static_cast<const float *>(mCudaBuffers[6]), static_cast<float *>(cudaOutputBuffer),
                             input_w/4,input_h/4,classNum,kernelSize,visThresh);            
        }


        CUDA_CHECK(cudaMemcpyAsync(outputData, cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream));

        runIters++ ;
    }
    
    void ctdetNet::saveEngine(const std::string &fileName)
    {
        if(mEngine)
        {
            nvinfer1::IHostMemory* data = mEngine->serialize();
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                return;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
        }

    }
}

#endif




