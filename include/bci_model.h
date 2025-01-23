#pragma once

#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <span>
#include <string>
#include <unordered_map>
// #include "qvac-lib-inference-addon-cpp/RuntimeStats.hpp"

namespace qvac_lib_inference_addon_mlc_bci {

namespace mlc_llm {
  class TransformerBCI;
}

std::pair<std::string, int> detect_device(std::string device);

DLDevice get_device(const std::string& device_name, int device_id);

class BCIModel {
public:
  using Input = std::string;
  using InputView = std::string_view;
  using Output = std::string;

  BCIModel() = default;
  BCIModel(const std::string& device_name,
                   const std::string& model_lib_name_with_path,
                   std::unordered_map<std::string, std::string>& config_filemap);

  inline std::string get_device_name() const { return device_name_; }
  inline int get_device_id() const { return device_id_; }
  inline DLDevice get_device() const { return device_; }

  const std::string& get_config_file_contents(const std::string& filename) const;

  const std::string& get_weights_for_file(const std::string& filename) const;
  /* Gradually append to the contents for shard files, and update cache once parameters are there */
  bool set_weights_for_file(const std::string& filename,
                            std::span<const uint8_t> bytes,
                            bool is_finished_current);

  /* Run inference and get  output */
tvm::runtime::NDArray process(tvm::runtime::NDArray& input_bytes);

  void reset();

  virtual ~BCIModel();

public:
  static std::vector<std::string> get_allowed_config_files();
  /* Clear files (mostly shard files) from memory once cache is uploaded */
  bool clear_file_contents(const std::string& filename);

private:
  void load_model(const std::string& model_lib_filename_with_path);
  bool set_config_file_contents(const std::string& filename, const std::string& contents);
  static void clear_global_memory_manager();

private:
  std::string device_name_;
  int device_id_;
  DLDevice device_;

  tvm::runtime::Module model_; // The .so file
  mlc_llm::TransformerBCI* helper_module_ = nullptr;
  size_t params_shard_files_to_load_ = 0;

  std::unordered_map<std::string, std::string> config_filemap_;
  std::unordered_map<std::string, std::string> params_shard_filemap_;
};

} // namespace qvac_lib_inference_addon_mlc_marian_base_iface