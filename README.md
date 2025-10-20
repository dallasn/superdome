# Superdome

Superdome: Superposition Density-Optimized Mixture of Experts

A framework for model-to-model knowledge compression

SuperDOME is a large language model architecture that combines orthogonal LoRA superposition with a high-capacity mixture-of-experts framework to bridge the gap between small and large models. By extracting and compressing hundreds of LoRA adapters from teacher-student weight differences into shared supertensors, SuperDOME allows dynamic expert selection at inference while keeping VRAM use low. The approach aims to recover much of a larger 70B model’s capability on a 7B base with minimal VRAM increase.
