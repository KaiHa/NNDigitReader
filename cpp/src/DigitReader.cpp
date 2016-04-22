#include "mnistloader.h"
#include <fann.h>
#include <fann_train.h>
#include <iostream>
#include <stdexcept>

using namespace std;

using unit = unsigned int;

static pair<IMAGES, LABELS> raw_training_data;
static pair<IMAGES, LABELS> raw_validation_data;

static void
prepare_data(uint idx, uint num_input, uint num_output,
             float *input, float *output) {
  uint i = 0;
  for (auto x : raw_training_data.first.at(idx)) {
    if (i >= num_input) {
      throw runtime_error("more pixels than expected");
    }
    input[i] = (float) x / 255.0;
    ++i;
  }
  auto label = raw_training_data.second.at(idx);
  for (uint j=0; j < num_output; j++) {
    output[j] = (j == label) ? 1.0 : 0.0;
  }
}


int
main()
{
  raw_training_data = loadMnistTrainingData("../../data/");
  raw_validation_data = loadMnistValidationData("../../data/");
  auto ann = fann_create_standard(3, 784, 30, 10);
  fann_randomize_weights(ann, -0.5, 0.5);
  auto train_data =
    fann_create_train_from_callback(raw_training_data.first.size(),
                                    raw_training_data.first.at(0).size(),
                                    10,
                                    prepare_data);
  fann_train_on_data(ann, train_data, 30, 1, 0.05);

  auto validation_data =
    fann_create_train_from_callback(raw_validation_data.first.size(),
                                    raw_validation_data.first.at(0).size(),
                                    10,
                                    prepare_data);
  fann_test_data(ann, validation_data);
  cout << 100 * fann_get_MSE(ann) << "% error" << endl;
  return 0;
}
