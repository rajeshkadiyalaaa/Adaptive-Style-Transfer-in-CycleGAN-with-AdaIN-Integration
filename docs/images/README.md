# Diagrams for AdaIN-CycleGAN

This directory contains diagrams that illustrate the architecture and workflow of the AdaIN-CycleGAN system.

## Diagram Descriptions

### AdaIN-CycleGAN Architecture

The `adain_cyclegan_architecture.png` diagram illustrates the overall architecture of the AdaIN-CycleGAN system, showing the different components and their connections.

### AdaIN Layer

The `adain_layer.png` diagram shows the detailed structure of the Adaptive Instance Normalization (AdaIN) layer, which is a key component of our system.

### Training Workflow

The `training_workflow.png` diagram visualizes the complete training process, including:
- Data loading from multiple domains
- Style feature extraction
- Forward and backward passes
- Loss computation
- Optimization

### Inference Workflow

The `inference_workflow.png` diagram shows the simplified workflow during inference (after training), illustrating how a content image and style image are processed to produce a stylized output.

### CycleGAN vs AdaIN-CycleGAN

The `comparison.png` diagram provides a side-by-side comparison of traditional CycleGAN and our enhanced AdaIN-CycleGAN, highlighting the key differences and improvements.

## Creating Diagrams

These diagrams can be created using tools like:
- Draw.io (diagrams.net)
- Lucidchart
- Microsoft Visio
- Adobe Illustrator

For consistency, it's recommended to use the following color scheme:
- CycleGAN components: Blue
- AdaIN components: Green
- Shared components: Gray
- Data flow: Black arrows
- Loss computation: Red

## Adding New Diagrams

When adding new diagrams:
1. Use a descriptive filename
2. Add an entry to this README
3. Ensure the diagram is in PNG or SVG format
4. Maintain a consistent style with existing diagrams 