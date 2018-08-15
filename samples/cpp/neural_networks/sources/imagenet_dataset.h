/* file: blob_dataset.h */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    Auxiliary functions used in C++ neural networks samples
!******************************************************************************/

#ifndef _BIN_DATASET_H
#define _BIN_DATASET_H

#include "daal_defines.h"

#include <fstream>
#include <stdint.h>
#include <stdexcept>

class BlobDatasetReader
{
public:
   BlobDatasetReader() { }
   virtual ~BlobDatasetReader() { }

   virtual bool next() = 0;
   virtual void reset() = 0;

   virtual TensorPtr getBatch() = 0;
   virtual TensorPtr getGroundTruthBatch() = 0;

   virtual Collection<size_t> getBatchDimensions() = 0;
   virtual size_t getTotalNumberOfObjects() const = 0;
};

template<typename DataType>
class ImageBlobDatasetReader : public BlobDatasetReader {
private:
   size_t _imagesNumber;
   size_t _imagesInBatch;

   size_t _batchCounter;
   size_t _imageChannels;
   size_t _imageHeight;
   size_t _imageWidth;

   std::fstream _dataFile;
   std::streampos _imagesPosition;
   std::streampos _classesPosition;
   std::streampos _next_classesPosition;

public:
    ImageBlobDatasetReader(const std::string &pathToImages, size_t batchSize=32);
    virtual ~ImageBlobDatasetReader();

    /* Advances the reader to next batch */
    virtual bool next();

    /* Resests reader position to the first batch */
    virtual void reset();

    /* Returns current batch */
    virtual TensorPtr getBatch();

    /* Returns coresponding lables of current batch */
    virtual TensorPtr getGroundTruthBatch();

    /* Returns dimensions of one batch */
    virtual Collection<size_t> getBatchDimensions();

    /* Returns total number of images in the blob */
    virtual size_t getTotalNumberOfObjects() const { return _imagesNumber; }

private:
    /* Opens the file containing dataset */
    void open(const std::string &datasetPath);

    /* Closes the dataset file */
    void close();

    /* Reads batch of images coresponding to the current reader position */
    TensorPtr readBatchFromDataset(std::fstream &file, size_t counter);

    /* Reads batch of labels coresponding to the current reader position */
    TensorPtr readGroundTruthFromDataset(std::fstream &file, size_t counter);

    void checkBeforeReadBatch();
    static uint32_t readDWORD(std::fstream &input);
    static uint16_t readWORD(std::fstream  &input);
    static uint8_t  readBYTE(std::fstream  &input);
};


template<typename DataType>
TensorPtr allocateTensor(size_t d1, size_t d2)
{
    Collection<size_t> dimensionsCollection;
    dimensionsCollection.push_back(d1);
    dimensionsCollection.push_back(d2);

    return TensorPtr(new HomogenTensor<DataType>(dimensionsCollection, Tensor::doAllocate));
}

template<typename DataType>
TensorPtr allocateTensor(size_t d1, size_t d2, size_t d3, size_t d4)
{
    Collection<size_t> dimensionsCollection;
    dimensionsCollection.push_back(d1);
    dimensionsCollection.push_back(d2);
    dimensionsCollection.push_back(d3);
    dimensionsCollection.push_back(d4);

    return TensorPtr(new HomogenTensor<DataType>(dimensionsCollection, Tensor::doAllocate));
}

template<typename DataType>
ImageBlobDatasetReader<DataType>::ImageBlobDatasetReader(const std::string &pathToImages, size_t batchSize) :
    _imagesNumber(0),
    _imagesInBatch(batchSize),
    _batchCounter(0),
    _imageChannels(0),
    _imageHeight(0),
    _imageWidth(0) { open(pathToImages); }

template<typename DataType>
ImageBlobDatasetReader<DataType>::~ImageBlobDatasetReader() { close(); }

template<typename DataType>
bool ImageBlobDatasetReader<DataType>::next()
{
    if (!_dataFile.is_open())
    {
        throw std::runtime_error("Can't open dataset file");
    }

    _batchCounter++;
    return _batchCounter * _imagesInBatch <= _imagesNumber;
}

template<typename DataType>
TensorPtr ImageBlobDatasetReader<DataType>::getBatch()
{
    checkBeforeReadBatch();
    return readBatchFromDataset(_dataFile, _batchCounter - 1);
}

template<typename DataType>
TensorPtr ImageBlobDatasetReader<DataType>::getGroundTruthBatch()
{
    checkBeforeReadBatch();
    return readGroundTruthFromDataset(_dataFile, _batchCounter - 1);
}

template<typename DataType>
Collection<size_t> ImageBlobDatasetReader<DataType>::getBatchDimensions()
{
    Collection<size_t> dims;
    dims.push_back(_imagesInBatch);
    dims.push_back(_imageChannels);
    dims.push_back(_imageHeight);
    dims.push_back(_imageWidth);

#if defined DEBUG
    std::cout << "Batch Dimensions\n";
    for(auto boo = 0; boo != dims.size(); boo++)
    {
        std::cout << "\t" << dims[boo] << "\n";
    }
    std::cout << std::endl;
#endif

    return dims;
}

template<typename DataType>
void ImageBlobDatasetReader<DataType>::reset()
{
    _batchCounter = 0;
}

template<typename DataType>
void ImageBlobDatasetReader<DataType>::open(const std::string &datasetPath)
{
    if (!datasetPath.size())
    {
        throw std::runtime_error("Path to dataset is empty");
    }

    _dataFile.open(datasetPath.c_str(), std::fstream::in | std::fstream::binary);
    if (!_dataFile.is_open())
    {
        throw std::runtime_error("Can't open dataset " + datasetPath);
    }

    std::cout << "Reading config at " << _dataFile.tellg() << "\n";

    //Imagenet 64x64
    _imagesNumber  = 128116;//128116;
    _imageChannels = 3;
    _imageWidth    = 64;
    _imageHeight   = 64;

    size_t imagesDataSize = _imagesNumber * _imageChannels *
                            _imageWidth   * _imageHeight;

    //images are at third byte
    _imagesPosition = 2;

    //lables are first byte
    _classesPosition = 0;

#if defined DEBUG
    std::cout << "_imagesPos " << _imagesPosition << "\n";
    std::cout << "_classesPosition " << _classesPosition << " " << _imagesNumber << " " << _imageChannels << " " << _imageWidth << " " << _imageHeight << "\n";
    std::cout << std::endl;
#endif

}

template<typename DataType>
void ImageBlobDatasetReader<DataType>::close()
{
    if (_dataFile.is_open())
    {
        _dataFile.close();
    }
}

template<typename DataType>
TensorPtr ImageBlobDatasetReader<DataType>::readBatchFromDataset(std::fstream &file, size_t counter)
{
    size_t imagesBatchSize = _imagesInBatch * _imageChannels *
                             _imageWidth    * _imageHeight * sizeof(char);
    size_t batchLabelsSize = _imagesInBatch * sizeof(uint16_t);

    size_t batchPosition = (size_t)_imagesPosition + (batchLabelsSize * counter) + (imagesBatchSize * counter);
    file.seekg(batchPosition);

#if defined DEBUG
    std::cout << "BSpos " << file.tellg() << "\n";
    std::cout << "BPosition " << _imagesPosition << "  Counter " << counter << "\n";
    std::cout << "BBatchSize " << imagesBatchSize << " - " << _imagesInBatch << " " << _imageChannels << " " << _imageWidth << " " << _imageHeight << " " << sizeof(char) << " \n";
    std::cout << "BbatchPosition " << batchPosition << "\n";
#endif

    TensorPtr dataBatch = allocateTensor<DataType>(_imagesInBatch, _imageChannels, _imageHeight, _imageWidth);
    const size_t trainTensorSize = dataBatch->getSize();

#if defined DEBUG
    std::cout << "BtrainTesnorSize " << trainTensorSize << "\n";
#endif

    SubtensorDescriptor<DataType> batchBlock;
    dataBatch->getSubtensor(0, 0, 0, _imagesInBatch, writeOnly, batchBlock);
    DataType *objectsPtr = batchBlock.getPtr();

#if defined DEBUG
    std::cout << "BbatchBlockSize " << batchBlock.getSize() << "\n";
#endif

    unsigned char *objectData = new unsigned char[trainTensorSize];
    file.read((char*)objectData, sizeof(unsigned char) * trainTensorSize);

    for (size_t i = 0; i < trainTensorSize; i++)
    {
        objectsPtr[i] = (DataType)static_cast< int >(objectData[i]);
    }

#if defined DEBUG
    std::cout << std::endl;
    printPredictedClasses(dataBatch, dataBatch, batchBlock.getSize());

    std::cout << "BEpos " << file.tellg() << "\n";
    std::cout << "---------------------------------------------\n";
#endif

    delete[] objectData;
    dataBatch->releaseSubtensor(batchBlock);
    return dataBatch;
}

template<typename DataType>
TensorPtr ImageBlobDatasetReader<DataType>::readGroundTruthFromDataset(std::fstream &file, size_t counter)
{
    size_t imagesBatchSize = _imagesInBatch * _imageChannels *
                            _imageWidth    * _imageHeight * sizeof(char);
    size_t batchLabelsSize = _imagesInBatch * sizeof(uint16_t);

    size_t batchPosition = (size_t)_classesPosition + (imagesBatchSize * counter) + (batchLabelsSize * counter);
    file.seekg(batchPosition);

#if defined DEBUG
    std::cout << "GSpos " << file.tellg() << "\n";
    std::cout << "GbatchLabelsSize " << batchLabelsSize << " - " << _imagesInBatch << " " << sizeof(uint16_t) << "\n";
    std::cout << "GbatchPosition " << batchPosition << " - " << (size_t)_classesPosition << " " << imagesBatchSize << " " << batchLabelsSize << " " << counter << "\n";
#endif

    TensorPtr groundTruthBatch = allocateTensor<int>(_imagesInBatch, 1);

    SubtensorDescriptor<int> groundTruthBlock;
    groundTruthBatch->getSubtensor(0, 0, 0, _imagesInBatch, writeOnly, groundTruthBlock);
    int *groundTruthPtr = groundTruthBlock.getPtr();

    for (size_t i = 0; i < _imagesInBatch; i++)
    {
        groundTruthPtr[i] = (int)readWORD(file);
        #if defined DEBUG
            std::cout << "\t" << groundTruthPtr[i] << "\n";
        #endif
    }

#if defined DEBUG
    std::cout << std::endl;
    printPredictedClasses(groundTruthBatch, groundTruthBatch, groundTruthBlock.getSize());

    std::cout << "GEpos " << file.tellg() << "\n";
    std::cout << "---------------------------------------------\n";
#endif

    groundTruthBatch->releaseSubtensor(groundTruthBlock);
    return groundTruthBatch;
}

template<typename DataType>
void ImageBlobDatasetReader<DataType>::checkBeforeReadBatch()
{
    if (!_dataFile.is_open())
    {
        throw std::runtime_error("Can't open dataset file");
    }
}

template<typename DataType>
uint32_t ImageBlobDatasetReader<DataType>::readDWORD(std::fstream &input)
{
    uint32_t dword;
    input.read((char*)(&dword), sizeof(uint32_t));
    return dword;
}

template<typename DataType>
uint16_t ImageBlobDatasetReader<DataType>::readWORD(std::fstream &input)
{
    uint16_t word;
    input.read((char*)(&word), sizeof(uint16_t));
    return word;
}

template<typename DataType>
uint8_t ImageBlobDatasetReader<DataType>::readBYTE(std::fstream &input)
{
    uint8_t byte;
    input.read((char*)(&byte), sizeof(uint8_t));
    return byte;
}

#endif
