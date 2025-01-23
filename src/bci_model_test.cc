// #include "bci_model.h"
// #include <gtest/gtest.h>
// #include <tvm/runtime/ndarray.h>

// namespace qvac_lib_inference_addon_mlc_bci {
// namespace {

// class BCIModelTest : public ::testing::Test {
//  protected:
//   void SetUp() override {
//     // Setup common test data
//     device_ = "cpu:0";
//     model_lib_path_ = "path/to/test/model.so";
//     config_files_ = {
//       {"mlc-chat-config.json", "{}"},
//       {"ndarray-cache.json", "{\"records\": []}"},
//       {"vocab.json", "{}"}
//     };
//   }

//   std::string device_;
//   std::string model_lib_path_;
//   std::unordered_map<std::string, std::string> config_files_;
// };

// TEST_F(BCIModelTest, ConstructorInitializesCorrectly) {
//   EXPECT_NO_THROW({
//     BCIModel model(device_, model_lib_path_, config_files_);
//   });
// }

// TEST_F(BCIModelTest, ConfigFileOperations) {
//   BCIModel model(device_, model_lib_path_, config_files_);
  
//   // Test getting valid config file
//   EXPECT_NO_THROW({
//     model.get_config_file_contents("mlc-chat-config.json");
//   });

//   // Test setting valid config file
//   EXPECT_TRUE(model.set_config_file_contents("mlc-chat-config.json", "new content"));

//   // Test invalid config file
//   EXPECT_FALSE(model.set_config_file_contents("invalid.json", "content"));
//   EXPECT_THROW(model.get_config_file_contents("invalid.json"), std::runtime_error);
// }

// TEST_F(BCIModelTest, WeightsFileOperations) {
//   BCIModel model(device_, model_lib_path_, config_files_);
  
//   std::vector<uint8_t> test_data = {1, 2, 3, 4};
//   std::span<const uint8_t> data_span(test_data);

//   // Test setting valid weights file
//   EXPECT_TRUE(model.set_weights_for_file("params_shard_0.bin", data_span, true));

//   // Test invalid weights file
//   EXPECT_FALSE(model.set_weights_for_file("invalid.bin", data_span, true));
// }

// TEST_F(BCIModelTest, ProcessInput) {
//   BCIModel model(device_, model_lib_path_, config_files_);
  
//   // Create a test input NDArray
//   std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
//   tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty(
//       {4}, tvm::runtime::DataType::Float(32), tvm::Device{kDLCPU, 0});
//   input_array.CopyFromBytes(input_data.data(), input_data.size() * sizeof(float));

//   // Test processing
//   EXPECT_NO_THROW({
//     tvm::runtime::NDArray output = model.process(input_array);
//     EXPECT_TRUE(output.defined());
//   });
// }

// TEST_F(BCIModelTest, GetAllowedConfigFiles) {
//   auto allowed_files = BCIModel::get_allowed_config_files();
//   EXPECT_FALSE(allowed_files.empty());
//   EXPECT_TRUE(std::find(allowed_files.begin(), allowed_files.end(), "mlc-chat-config.json") 
//               != allowed_files.end());
// }

// } // namespace
// } // namespace qvac_lib_inference_addon_mlc_bci

// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TEST();
// } 