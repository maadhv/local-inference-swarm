\# Baseline 002 — GTX 1650 Laptop



\## Hardware



\- GPU: NVIDIA GTX 1650 4 GB

\- RAM: 8 GB

\- CPU: \[fill in CPU]

\- OS: Windows



\## Model



\- Model: Qwen3 4B

\- Quantization: Q4\_K\_M

\- Context: 4096



\## Runtime



\- Runtime: llama.cpp

\- Backend: Vulkan

\- GPU device: NVIDIA GTX 1650

\- GPU layers: 99



\## Test Prompt



> Explain distributed systems in simple terms in exactly 100 words.



\## Results



| Metric | Result |

|---|---:|

| Prompt processing | 30.4 tok/s |

| Generation | 35.3 tok/s |



\## Result



Successful local inference using the NVIDIA GTX 1650.



\## Notes



The NVIDIA GPU was explicitly selected through the Vulkan device

configuration. The model was successfully loaded with GPU layer

offloading enabled.

