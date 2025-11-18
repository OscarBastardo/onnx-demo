# ONNX Demo

This repository demonstrates how to export a simple PyTorch model to ONNX format and run inference using ONNX Runtime in C++.

## Workflow
1. Define simple PyTorch model (Python)
2. Get model predictions
3. Export to `simple_model.onnx`
4. Use Netron to inspect the ONNX model graph
5. Load `simple_model.onnx` in ONNX Runtime (C++)
6. Run inference and compare predictions

## Instructions
1. Initialise and activate a Python virtual environment with `python -m venv venv && source venv/bin/activate`
2. Install Python dependencies with `pip install -r requirements.txt`
3. Ensure onnxruntime is installed in the `onnxruntime` directory
4. Run the Python script to export the model: `python main.py`
5. Use Netron to open and inspect model with `netron simple_model.onnx`
6. Build the C++ inference application with `mkdir build && cd build && cmake .. && make`
7. Run the C++ executable: `cd .. && ./build/onnx_demo`