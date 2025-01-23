#include "bci_model.h"

#include "cpp/metadata/model.h"
#include "cpp/support/random.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/debug.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility> // for std::pair
#include <vector>

namespace qvac_lib_inference_addon_mlc_bci {

namespace mlc_llm {

using namespace tvm::runtime;
using tvm::Device;

using ::mlc::llm::ModelMetadata;

struct FunctionTable {
  static PackedFunc SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
    return PackedFunc([sess, func = std::move(sess_func), name = std::move(name)]
                      (TVMArgs args, TVMRetValue* rv)
                      -> void {
      std::vector<TVMValue> tvm_values(args.num_args + 3);
      std::vector<int> tvm_type_codes(args.num_args + 3);
      TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
      setter(0, static_cast<int>(DiscoAction::kCallPacked));
      setter(1, 0);
      setter(2, func);

      for (int i = 0; i < args.num_args; ++i) {
        tvm_values[i + 3] = args.values[i];
        tvm_type_codes[i + 3] = args.type_codes[i];
      }

      *rv = sess->CallWithPacked(TVMArgs(tvm_values.data(),
                                         tvm_type_codes.data(),
                                         args.num_args + 3));
    });
  }

  void Init(Module executable, Device device) {
    Device null_device{DLDeviceType(0), 0};

    this->use_disco = false;
    auto fload_exec = executable->GetFunction("vm_load_executable");
    ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    this->local_vm = fload_exec();
    this->local_vm->GetFunction("vm_initialization")(
      static_cast<int>(device.device_type), device.device_id,
      static_cast<int>(memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
      static_cast<int>(memory::AllocatorType::kPooled));
    this->mod_get_func = [this](const std::string& name) -> PackedFunc {
      PackedFunc func = this->local_vm->GetFunction(name, false);
      return func;
    };
    this->get_global_func = [](const std::string& name) -> PackedFunc {
      const auto* f = tvm::runtime::Registry::Get(name);
      CHECK(f != nullptr) << "ValueError: Cannot find function " << name;
      return *f;
    };
    this->model_metadata_ = ModelMetadata::FromModule(this->local_vm, { });

    this->_InitFunctions();
  }

  void UpdateNDArrayCache(const BCIModel& model,
                          const tvm::runtime::relax_vm::NDArrayCacheMetadata& ndarray_cache_metadata,
                          const std::string& filename,
                          bool use_presharded_weights) {
    using tvm::runtime::ShapeTuple;
    using FileRecord = tvm::runtime::relax_vm::NDArrayCacheMetadata::FileRecord;
    using ParamRecord = FileRecord::ParamRecord;

    const size_t filename_chars = std::string("params_shard_").size();
    size_t stop_idx = filename.find_last_of('.');
    std::string remaining = filename.substr(filename_chars, stop_idx - filename_chars);
    size_t file_record_idx = std::stoi(remaining);
    const FileRecord& file_record = ndarray_cache_metadata.records[file_record_idx];
    size_t total_param_records = file_record.records.size();
    Array<NDArray> params;
    const auto& params_shard_file = model.get_weights_for_file(filename);
    Optional<NDArray> staging_buffer;

    CHECK(!use_presharded_weights) << "Use of pre-sharded weights requires more than one GPU";
    std::cerr << filename << " has these many parameter records: " << total_param_records << '\n';

    params.reserve(total_param_records);

    for (size_t i = 0; i < total_param_records; ++i) {
      const ParamRecord& param_record = file_record.records[i];

      params.push_back(param_record.Load(model.get_device(),
                                         &params_shard_file,
                                         &staging_buffer));
    }

    const PackedFunc* fload_cache_update = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.update");
    ICHECK(fload_cache_update) << "TVM runtime cannot find vm.builtin.ndarray_cache.update";

    /* Update the global cache with all the various parameters */
    for (size_t i = 0; i < params.size(); ++i)
      (*fload_cache_update)(file_record.records[i].name, params[i], true);
  }

  ObjectRef LoadParams() {
    Array<NDArray> params;

    if (this->model_metadata_.params.empty()) {
      constexpr const char* name_loader = "vm.builtin.param_array_from_cache";

      const PackedFunc* fload_params = tvm::runtime::Registry::Get(name_loader);
      ICHECK(fload_params) << "Cannot find env function: " << name_loader;

      params = (*fload_params)("param", -1);
    }
    else {
      /* `model_metadata_.params` have been populated with `name` property set at least,
       * so we can just grab their values from the cache using the parameter's names
       */
      constexpr const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
      const PackedFunc* fload_params = tvm::runtime::Registry::Get(name_loader);
      ICHECK(fload_params) << "Cannot find env function: " << name_loader;

      Array<String> param_names;
      param_names.reserve(this->model_metadata_.params.size());

      for (const auto& param : this->model_metadata_.params)
      {
        param_names.push_back(param.name);
        std::cout<<"Adding parameter: "<< param.name<<"\n";
      }

      params = (*fload_params)(param_names);
    }

    /* After we get params, it is safe to simply clear the cached version
     * as these params are referenced by the member `params_`
     */
    const PackedFunc* fclear_ndarray_cache
      = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
    ICHECK(fclear_ndarray_cache) << "Cannot find env function: vm.builtin.ndarray_cache.clear";

    (*fclear_ndarray_cache)();

    return params;
  }

  void _InitFunctions() {
    this->encode_func_ = mod_get_func("encode");
  }

  ObjectRef Empty(ShapeTuple shape, DataType dtype, Device device) const {
    Device null_device { DLDeviceType(0), 0 };

    if (this->use_disco) {
      DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");

      return sess->CallPacked(empty_func, shape, dtype, null_device);
    }

    return NDArray::Empty(shape, dtype, device);
  }

  ObjectRef CopyToWorker0(const NDArray& host_array) {
    Device null_device { DLDeviceType(0), 0 };

    if (this->use_disco) {
      DRef array = Downcast<DRef>(this->Empty(host_array.Shape(),
                                              host_array.DataType(),
                                              null_device));
      sess->CopyToWorker0(host_array, array);

      return array;
    }

    return host_array;
  }

  bool use_disco = false;
  Session sess { nullptr };
  DRef disco_mod { nullptr };
  tvm::runtime::Module local_vm{nullptr};

  TypedPackedFunc<PackedFunc(const std::string&)> mod_get_func;
  TypedPackedFunc<PackedFunc(const std::string&)> get_global_func;

  PackedFunc encode_func_;
 
  ModelMetadata model_metadata_;
};

/*
 * Implements the fcog module wrapper
 */
class TransformerBCI {
public:
  explicit TransformerBCI(DLDevice device) : device_(device) {}

  /* Initialize function table
   */
  void Init(Module model_mod) {
    this->ft_.Init(model_mod, device_);    
  }

  /* Load cached metadata of the parameters
   * and return total file records present
   */
  size_t LoadNDArrayCacheMetadata(const std::string& ndarray_cache_json) {
    ndarray_cache_metadata_ = tvm::runtime::relax_vm::NDArrayCacheMetadata::LoadFromStr(
      ndarray_cache_json,
      ""
    );

    return ndarray_cache_metadata_.records.size();
  }

//   /* Step 5.2 of "reload" instruction. Upload global cache of NDArray. */
  void UpdateNDArrayCacheForFile(const BCIModel& model, const String& filename) {
    use_presharded_weights_ = false;
    ft_.UpdateNDArrayCache(model,
                           ndarray_cache_metadata_,
                           filename,
                           use_presharded_weights_);
  }

  /* Step 5.3 of "reload" instruction. Upload global cache of NDArray. */
  void LoadParamsFromCache() {
    params_ = ft_.LoadParams();
  }

  
  NDArray EncodeStep(NDArray input_data) {
    tvm::runtime::NDArray ndarray;
    CHECK(ft_.encode_func_.defined()) << "Make sure to build the function with the encode";
    ObjectRef encode_out = ft_.encode_func_(ft_.CopyToWorker0(input_data),  params_);
    ndarray = extractNDArrayEncodeStep(encode_out);
    return ndarray;
}

  NDArray extractNDArrayEncodeStep(const tvm::runtime::ObjectRef& obj_ref) {
    tvm::runtime::NDArray ndarray;
    // Try to cast to Optional<ObjectRef>
    auto opt_obj_ref = obj_ref.as<tvm::runtime::ObjectRef>();

    if (opt_obj_ref.defined()) {
      // It's an Optional<ObjectRef>, now check the contained value
      const tvm::runtime::ObjectRef& contained_obj_ref = opt_obj_ref.value();

      // Check if the contained value is an NDArray
      auto opt_ndarray = contained_obj_ref.as<tvm::runtime::NDArray>();

      if (opt_ndarray.defined()) {
        // The contained value is an NDArray
        // std::cout << "Object is an Optional containing an NDArray" << std::endl;
        // get the NDarray
        ndarray = opt_ndarray.value();
      }
    }

    output_ndarray_ = ndarray.CopyTo(DLDevice{ kDLCPU, 0 });
  

    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    output_ndarray_ = ndarray;
    float* output_data = static_cast<float*>(output_ndarray_->data);
    std::cout<<output_data[0]<<"\n";

    return output_ndarray_;
  }

private:
  // Load weights that were saved in sharded form
  bool use_presharded_weights_;
  //----------------------------
  // TVM related states
  //----------------------------
  tvm::runtime::NDArray output_ndarray_;
  // runtime device
  Device device_;
  FunctionTable ft_;
  // local params
  ObjectRef params_;
  // Metadata of the params_shard_x.bin files
  tvm::runtime::relax_vm::NDArrayCacheMetadata ndarray_cache_metadata_;
};

} // namespace mlc_llm

using namespace mlc_llm;

static const std::unordered_set<std::string> ALLOWED_CONFIG_FILES = {
  "mlc-chat-config.json", "ndarray-cache.json", "vocab.json",
  "source.model", "target.model"
};

static const std::unordered_set<std::string> ALLOWED_PARAMS_SHARD_FILES = {
  "params_shard_0.bin", "params_shard_1.bin", "params_shard_2.bin",
  "params_shard_3.bin", "params_shard_4.bin", "params_shard_5.bin",
  "params_shard_6.bin", "params_shard_7.bin", "params_shard_8.bin"  ,
  "params_shard_9.bin", "params_shard_10.bin" 
};

static bool is_valid_config_file(const std::string& filename) {
  return ALLOWED_CONFIG_FILES.find(filename) != ALLOWED_CONFIG_FILES.end();
}

static bool is_valid_params_shard_file(const std::string& filename) {
  return ALLOWED_PARAMS_SHARD_FILES.find(filename) != ALLOWED_PARAMS_SHARD_FILES.end();
}

std::pair<std::string, int> detect_device(std::string device) {
  using tvm::runtime::DeviceAPI;

  std::string device_name;
  int device_id;
  int delimiter_pos = device.find(":");

  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  }
  else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }

  if (device_name == "auto") {
    bool allow_missing = true;

    if (DeviceAPI::Get(DLDevice{ kDLCUDA, 0 }, allow_missing))
      return { "cuda", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLMetal, 0 }, allow_missing))
      return { "metal", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLROCM, 0 }, allow_missing))
      return { "rocm", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLVulkan, 0 }, allow_missing))
      return { "vulkan", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLOpenCL, 0 }, allow_missing))
      return { "opencl", device_id };

    // TODO: Auto-detect devices for mali
    LOG(FATAL) << "Cannot auto detect device-name";
  }

  return { device_name, device_id };
}

DLDevice get_device(const std::string& device_name, int device_id) {
  if (device_name == "cuda")
    return { kDLCUDA, device_id };

  if (device_name == "metal")
    return { kDLMetal, device_id };

  if (device_name == "rocm")
    return { kDLROCM, device_id };

  if (device_name == "vulkan")
    return { kDLVulkan, device_id };

  if (device_name == "opencl" || device_name == "mali")
    return { kDLOpenCL, device_id };

  LOG(FATAL) << "Invalid device name: " << device_name
             << ". Please enter the device in the form 'device_name:device_id'"
                " or 'device_name', where 'device_name' needs to be one of 'cuda', 'metal', "
                "'vulkan', 'rocm', 'opencl', 'auto'.";

  return { kDLCPU, 0 };
}

/*
 * class: TranslationModel impl
 **/
BCIModel::BCIModel(const std::string& device,
                                   const std::string& model_lib_filename_with_path,
                                   std::unordered_map<std::string, std::string>& config_filemap) {
  auto pr = detect_device(device);
  

  device_name_ = pr.first;
  device_id_ = pr.second;
  device_ = ::qvac_lib_inference_addon_mlc_bci::get_device(device_name_, device_id_);
  helper_module_ = new TransformerBCI(device_);
  load_model(model_lib_filename_with_path);
  helper_module_->Init(model_);
  for (auto& [filename, contents] : config_filemap)
    set_config_file_contents(filename, contents);

  clear_global_memory_manager();

  //helper_module_->ReloadConfig(*this, { });

  params_shard_files_to_load_
    = helper_module_->LoadNDArrayCacheMetadata(get_config_file_contents("ndarray-cache.json"));
}

const std::string& BCIModel::get_config_file_contents(const std::string& filename) const {
  if (!is_valid_config_file(filename)) {
    std::string e = "Accessing file: " + filename + ", which is NOT allowed!\n";
    throw std::runtime_error(e.c_str());
  }

  if (config_filemap_.find(filename) == config_filemap_.end()) {
    std::string e = "Accessing file: " + filename + ", before it has been set!\n";
    throw std::runtime_error(e.c_str());
  }

  return config_filemap_.at(filename);
}

bool BCIModel::set_config_file_contents(const std::string& filename,
                                                const std::string& contents) {
  if (!is_valid_config_file(filename)) {
    std::cerr << "Failed to set contents for given file: " << filename
              << "\nThis file is not allowed. Please make sure the file you are sending is correct\n";
    return false;
  }

  config_filemap_[filename] = contents;

  return true;
}

const std::string& BCIModel::get_weights_for_file(const std::string& filename) const {
  if (!is_valid_params_shard_file(filename)) {
    std::string e = "Accessing file: " + filename + ", before it has been set!\n";
    throw std::runtime_error(e.c_str());
  }

  return params_shard_filemap_.at(filename);
  
}


bool BCIModel::set_weights_for_file(const std::string& filename,
                                            std::span<const uint8_t> bytes,
                                            bool is_finished_current) {
  if (!is_valid_params_shard_file(filename)) {
    std::cerr << "Failed to set contents for given file: " << filename
              << "\nThis file is not allowed. Please make sure the file you are sending is correct\n";

    return false;
  }

  auto& params_shard_file = params_shard_filemap_[filename];

  /* Since parameter shard records are quite large upto 80MiB,
   * we may have to send contents for the same file multiple times
   */
  params_shard_file.insert(params_shard_file.end(), bytes.begin(), bytes.end());

  if (is_finished_current) {
    helper_module_->UpdateNDArrayCacheForFile(*this, filename);
    clear_file_contents(filename);

    if (!(--params_shard_files_to_load_)) {
      helper_module_->LoadParamsFromCache();
    }
  }

  return true;
}

bool BCIModel::clear_file_contents(const std::string& filename) {
  std::string extension;


  if (!is_valid_config_file(filename) && !is_valid_params_shard_file(filename)) {
    std::cerr << "Invalid filename: " << filename << ", nothing to clear!\n";

    return false;
  }

  extension = filename.substr(filename.find_last_of('.'));

  if (extension == ".bin")
    params_shard_filemap_.erase(filename);
  else
    config_filemap_.erase(filename);

  return true;
}

std::vector<std::string> BCIModel::get_allowed_config_files() {
  std::vector<std::string> res;

  res.reserve(ALLOWED_CONFIG_FILES.size());

  for (const auto& key : ALLOWED_CONFIG_FILES)
    res.push_back(key);

  return res;
}

NDArray BCIModel::process(tvm::runtime::NDArray& input_bytes) {
    
    NDArray out = helper_module_->EncodeStep(input_bytes);
    return out;
}


void BCIModel::clear_global_memory_manager() {
  std::string func_name = "vm.builtin.memory_manager.clear";
  const PackedFunc* fclear_memory_manager = tvm::runtime::Registry::Get(func_name);

  ICHECK(fclear_memory_manager) << "Cannot find env function: " << func_name;

  (*fclear_memory_manager)();
}

void BCIModel::load_model(const std::string& model_lib_filename_with_path) {
  model_ = tvm::runtime::Module::LoadFromFile(model_lib_filename_with_path, "so");
}

BCIModel::~BCIModel() {
  delete helper_module_;
}

} // namespace qvac_lib_inference_addon_mlc_bci_base_iface