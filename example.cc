#include <fstream>
#include "include/bci_model.h"
#include <filesystem>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>




using namespace qvac_lib_inference_addon_mlc_bci;
using namespace tvm::runtime;
using tvm::Device;

namespace fs = std::filesystem;

std::vector<uint8_t> get_span_from_file(std::string file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    return buffer;  // Return the vector directly
}

int main(int argc, char* argv[]) {
    std::string input_file = argv[1];
    std::string output_dir = argv[2];

    std::cout<< "input_file: "<<input_file<<"\n";
    std::cout<< "output_dir: "<<output_dir<<"\n";

    std::string model_path = "/home/ubuntu/qvac-ext-lib-mlc/ecog_out1";
    std::string model_lib_path;

    //tvm::Device device = tvm::Device{ static_cast<DLDeviceType>(kDLVulkan), 0 };
    int weight_files_num =11;
   

    std::unordered_map<std::string, std::string> config_files;
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".json") {
            std::ifstream file(entry.path());
            std::stringstream buffer;
            buffer << file.rdbuf();
            config_files[entry.path().filename().string()] = buffer.str();
        }
    }
     BCIModel model = BCIModel("vulkan",
                    "/home/ubuntu/qvac-ext-lib-mlc/ecog_out1/model.so",
                   config_files);

    for (int i=0;i<weight_files_num; i++)
    {
        std::string file_name = std::string("params_shard_").append(std::to_string(i)).append(".bin");
        std::string full_path = std::string("").append(model_path).append("/"+file_name);
        std::vector<uint8_t> bytes = get_span_from_file(full_path);
        model.set_weights_for_file(file_name, bytes, true); 
    }

    DLDevice device =  get_device("vulkan",0);
    // Load input data from file
    
    //std::string input_file = "/home/ubuntu/evo-pocs/input_ecog1.bin";
    std::vector<uint8_t> input_bytes = get_span_from_file(input_file);
    
    // Convert bytes to float32 array
    float* float_data = reinterpret_cast<float*>(input_bytes.data());
    size_t num_floats = input_bytes.size() / sizeof(float);
    
    // Create NDArray with target shape
    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({1,1,1600,1,16,16}, DataType::Float(32), device);
    
    // Copy data into NDArray
    input.CopyFromBytes(float_data, input_bytes.size());

    std::cout<<"start processing input\n";
    
    tvm::runtime::NDArray out = model.process(input);

    std::cout<<"done processing input\n";
    // Print output shape
    std::cout << "Output shape: ";
    int size =1;
    for (int i = 0; i < out->ndim; i++) {
        std::cout << out->shape[i];
        if (i < out->ndim - 1) std::cout << "x";
        size *= out->shape[i];
    }
    size*=sizeof(float);
    std::cout << std::endl;

    float* output_data = (float* )malloc(size);
    out.CopyToBytes(output_data, size );	

    // Get dimensions
    int batch = out->shape[0];     // 1
    int seq_len = out->shape[1];   // 4096 llama dimention
    

    //float expected_output[] = { 0.965979, -2.22667, -3.28247, 1.27051, 0.0888224, -0.488834, 0.929905, -0.963828, 0.458228, -0.217912 };
    //jit:     0.9661, -2.2263, -3.2844,  1.2679,  0.0834, -0.4893,  0.9306, -0.9649, 0.4582, -0.2196
    //pytorch: 0.9655, -2.2263, -3.2846,  1.2681,  0.0826, -0.4891,  0.9300, -0.9645, 0.4581, -0.2189
    //cpp:     0.9659, -2.2266, -3.2825,  1.2705,  0.0888, -0.4888,  0.9299, -0.9638, 0.4582, -0.2179
    for (int i = 0; i< 10; i++){
        std::cout << output_data[i];
        //assert(abs(output_data[i] - expected_output[i]) < 0.001);
    }


     std::string output_path = output_dir + std::string("/output.bin");
     std::ofstream outfile(output_path, std::ios::binary | std::ios::app);
     for (int i = 0; i < size/sizeof(float); i++) {
         outfile.write(reinterpret_cast<char*>(&output_data[i]), sizeof(float));
     }
     outfile.close();
    std::cout << "\n";
    free(output_data);

    return 0;
}