// -*- mode: c++ -*-
#pragma once

#include <stdint.h>
#include <string>
#include <vector>

using IMG    = std::vector<uint8_t>;
using IMAGES = std::vector<IMG>;
using LABELS = std::vector<uint8_t>;

std::pair<IMAGES, LABELS> loadMnistTrainingData(std::string const &path);
std::pair<IMAGES, LABELS> loadMnistValidationData(std::string const &path);
