#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main()
{
    // --- 1. Setup ONNX Runtime ---
    // Ort::Env is the main environment object
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DEMO");
    // Ort::SessionOptions allows configurting the session
    Ort::SessionOptions session_options;

    // The model to load
    const char *model_path = "simple_model.onnx";

    // --- 2. Create a session and load the model ---
    Ort::Session session(env, model_path, session_options);

    // --- 3. Prepare input tensor ---
    // Define the shape of the input tensor (1 batch, 4 features)
    std::vector<int64_t> input_shape = {1, 4};

    // The input data MUST be the same as in the Python script
    std::vector<float> input_data = {-2.466057, 0.3622862, 0.37654912, -0.1808088};

    // Create a memory info object for CPU
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create the input tensor object from the data and shape
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    // --- 4. Run inference ---
    // Define the names of the input and output nodes. These must match the names provided
    // during the ONNX export in the Python script.
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};

    // Run the session
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,   // input node names
        &input_tensor, // pointer to input tensor
        1,             // number of inputs
        output_names,  // output node names
        1              // number of outputs
    );

    // --- 5. Process output tensor ---
    // Get a pointer to the output tensor's data
    float *output_data = output_tensors[0].GetTensorMutableData<float>();

    // Get the shape of the output tensor
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "C++ ONNX Runtime output: [";
    for (size_t i = 0; i < output_shape[1]; ++i)
    {
        std::cout << output_data[i] << (i == output_shape[1] - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    return 0;
}