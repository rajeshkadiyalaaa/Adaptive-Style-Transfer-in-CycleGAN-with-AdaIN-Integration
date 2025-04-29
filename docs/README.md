# AdaIN-CycleGAN: Adaptive Style Transfer with CycleGAN

This documentation provides a detailed explanation of the AdaIN-CycleGAN project, which combines CycleGAN's unpaired image-to-image translation capabilities with Adaptive Instance Normalization (AdaIN) for arbitrary style transfer.

## Documentation Index

- [Project Overview](overview.md): Introduction to the project and its goals
- [Architecture](architecture.md): Detailed explanation of the system architecture
- [Technical Details](technical_details.md): In-depth exploration of the implementation
- [Usage Guide](usage_guide.md): Instructions for training, testing, and using the model
- [Workflow](workflow.md): Step-by-step explanation of how the system works

## Project Summary

AdaIN-CycleGAN addresses a significant limitation of traditional CycleGAN models: the need to retrain for each new style. By integrating Adaptive Instance Normalization (AdaIN) layers into the CycleGAN architecture, our implementation enables dynamic, arbitrary style transfer while maintaining the content integrity of the original images.

The key innovations include:

1. **Arbitrary Style Transfer**: Apply any style to any content image without retraining
2. **Content Preservation**: Maintains the structural integrity of content while transferring style
3. **User-Defined Stylization**: Enables custom, on-the-fly style definition
4. **Improved Metrics**: Better PSNR, SSIM, and FID values compared to standard CycleGAN

## Quick Start

For those eager to get started, see the [usage guide](usage_guide.md) for installation and basic commands.

## References

1. J. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," in 2017 IEEE International Conference on Computer Vision (ICCV), 2017.
2. X. Huang and S. Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization," in Proceedings of the IEEE International Conference on Computer Vision, 2017. 