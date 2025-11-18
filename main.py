import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort


# define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # a simple linear layer with input of size 4 and output of size 2
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        # flatten the input and pass through the linear layer
        x = x.view(x.size(0), -1)
        return self.linear(x)


def main():
    print("--- 1. Setting up the PyTorch model and getting prediction ---")

    # for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # instantiate the model and set to evaluation mode
    model = SimpleModel()
    model.eval()

    # create a dummy input tensor of batch size 1 and 4 features
    dummy_input = torch.randn(1, 4)
    print("Dummy input:", dummy_input.numpy())

    # get the ground truth prediction from the PyTorch model
    model_output = model(dummy_input)
    print("PyTorch model output:", model_output.detach().numpy())

    print("\n--- 2. Exporting the model to ONNX format ---")

    input_names = ["input"]
    output_names = ["output"]
    onnx_model_path = "simple_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes={"x": {0: "batch_size"}},
    )
    print(f"Model exported to {onnx_model_path}")

    print("\n--- 3. Verifying ONNX model with ONNX Runtime ---")

    # create an ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # prepare the input for ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # run inference
    ort_output = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime output: {ort_output[0]}")

    print("\n--- 4. Comparing PyTorch and ONNX Runtime outputs ---")

    np.testing.assert_allclose(
        model_output.detach().numpy(), ort_output[0], rtol=1e-05, atol=1e-05
    )
    print("Verification successful: The outputs are numerically identical!")


if __name__ == "__main__":
    main()
