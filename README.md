# ONNX Demo

This repository demonstrates how to export a simple PyTorch model to ONNX format and run inference using ONNX Runtime in C++.

## Workflow
1. Define simple PyTorch model (Python)
2. Get model predictions
3. Export to `simple_model.onnx`
4. Load `simple_model.onnx` in ONNX Runtime (C++)
5. Run inference and compare predictions

## Instructions
1. Initialise and activate a Python virtual environment with `python -m venv venv && source venv/bin/activate`
2. Install Python dependencies with `pip install -r requirements.txt`
3. Ensure onnxruntime is installed in the `onnxruntime` directory
4. Run the Python script to export the model: `python main.py`
5. Build the C++ inference application with `mkdir build && cd build && cmake .. && make`
6. Run the C++ executable: `cd .. && ./build/onnx_demo`