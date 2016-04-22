#include "mnistloader.h"
#include <arpa/inet.h>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

using namespace std;
using namespace boost::iostreams;


//
// Read MNIST digit labels from the given file at path.
//
LABELS
readLabelFile(string const &path)
{
  ifstream file(path, ios::binary);
  filtering_streambuf<input> in;
  in.push(gzip_decompressor());
  in.push(file);
  istream stream(&in);
  stream.exceptions(ios::failbit | ios::badbit);

  uint32_t magicNum;
  stream.read(reinterpret_cast<char*>(&magicNum), sizeof(magicNum));
  magicNum = ntohl(magicNum);
  if (magicNum != 2049) {
    throw runtime_error(path + " has wrong magic number");
  }
  uint32_t nLabels;
  stream.read(reinterpret_cast<char*>(&nLabels), sizeof(nLabels));
  nLabels= ntohl(nLabels);

  LABELS labels;
  uint8_t label;
  for (uint32_t i=0; i < nLabels; i++) {
    stream.read(reinterpret_cast<char*>(&label), sizeof(label));
    labels.emplace_back(label);
  }
  return labels;
}


//
// Read MNIST digit images from the file at path.
//
IMAGES
readImageFile(string const &path)
{
  ifstream file(path, ios::binary);
  filtering_streambuf<input> in;
  in.push(gzip_decompressor());
  in.push(file);
  istream stream(&in);
  stream.exceptions(ios::failbit | ios::badbit);

  uint32_t magicNum;
  stream.read(reinterpret_cast<char*>(&magicNum), sizeof(magicNum));
  magicNum = ntohl(magicNum);
  if (magicNum != 2051) {
    throw runtime_error(path + " has wrong magic number");
  }
  uint32_t nImages;
  stream.read(reinterpret_cast<char*>(&nImages), sizeof(nImages));
  nImages = ntohl(nImages);
  uint32_t nRows;
  stream.read(reinterpret_cast<char*>(&nRows), sizeof(nRows));
  nRows = ntohl(nRows);
  uint32_t nCols;
  stream.read(reinterpret_cast<char*>(&nCols), sizeof(nCols));
  nCols = ntohl(nCols);

  IMAGES images;
  uint8_t pixel;
  for (uint32_t i=0; i < nImages; i++) {
    IMG image;
    for (uint32_t j=0; j < (nRows*nCols); j++) {
        stream.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        image.emplace_back(pixel);
    }
    images.emplace_back(image);
  }
  return images;
}


//
// Read MNIST trainings data from the given directory at path.
//
pair<IMAGES, LABELS>
loadMnistTrainingData(string const &path)
{
  auto img = readImageFile(path + "/train-images-idx3-ubyte.gz");
  auto lbl = readLabelFile(path + "/train-labels-idx1-ubyte.gz");
  return make_pair(img, lbl);
}


//
// Read MNIST validation data from the given directory at path.
//
pair<IMAGES, LABELS>
loadMnistValidationData(string const &path)
{
  auto img = readImageFile(path + "/t10k-images-idx3-ubyte.gz");
  auto lbl = readLabelFile(path + "/t10k-labels-idx1-ubyte.gz");
  return make_pair(img, lbl);
}
