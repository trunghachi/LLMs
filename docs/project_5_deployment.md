# Project 5: LLM Endpoint with vLLM

This project guides you through deploying a fine-tuned Large Language Model (LLM) as an API endpoint using vLLM for efficient inference and FastAPI for the API layer.

## Objectives
- Deploy an LLM for production use.
- Optimize inference with vLLM.

## Steps
1. **Set Up vLLM**:
   - Load a fine-tuned model (e.g., from Project 4).
   - Configure vLLM for efficient inference.
   - **Code Placeholder**: Implement vLLM setup in `deploy.py`.
2. **Create a FastAPI Endpoint**:
   - Expose the model via a REST API.
   - **Code Placeholder**: Implement FastAPI endpoint in `deploy.py`.
3. **Test the Endpoint**:
   - Send sample queries to the API.
   - Measure latency and throughput.

## Run
```bash
python src/section_5_deployment/vllm_endpoint.py
```

## Expected Output
- A working API endpoint for LLM inference.
- Performance metrics (latency, throughput).

## Next Steps
Proceed to [Section 6: Building the Application Layer](section_6_application.md).
