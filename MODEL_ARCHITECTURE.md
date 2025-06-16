# VTA-GAN Model Architecture

## Data Flow
```
Input Video → Preprocessing → Generator → Discriminator → Losses
```

## Input Processing
```
Raw Video: UCSD2 frames
├─ Shape: (1, 8, 3, H, W)
├─ Aspect: 360×240 → 96×64 → 64×64
├─ Augmentation: Conservative mode
└─ Decomposition:
   ├─ Laplacian stream: (1, 8, 3, 64, 64)
   └─ Residual stream: (1, 8, 3, 64, 64)
```
## Temporal Processing
```
TemporalFeatureFusion:
├─ Input: combined_streams (1, 8, 3, 64, 64)
├─ Process: Temporal aggregation across 8 frames
├─ Output: temporal_summary (1, 3, 64, 64)
└─ Enhancement: 
   ├─ Broadcast to all frames
   ├─ lap_enhanced = lap + 0.1 * temporal_summary
   └─ res_enhanced = res + 0.1 * temporal_summary
```

## Enhanced Temporal Processing with Optical Flow
```
Input Video Processing:
├─ TemporalFeatureFusion: Temporal context analysis
└─ OpticalFlowFeatureFusion: Motion pattern analysis

OpticalFlowFeatureFusion:
├─ Input: combined_streams (1, 8, 3, 64, 64)
├─ Optical Flow: FlowNetSD between consecutive frames
├─ Motion Analysis: 3D convolution on flow sequence
├─ Motion Features: (1, 3, 64, 64)
├─ Motion Magnitude: Anomaly sensitivity map
└─ Enhancement: 
   ├─ Broadcast to all frames
   ├─ lap_enhanced = lap + temporal_summary + motion_features
   └─ res_enhanced = res + temporal_summary + motion_features
```

## Generator: UnetGenerator_CS
```
Input: 2 streams × (8, 3, 64, 64)

Encoder:
├─ Conv1: 3 → 64   (64×64 → 32×32) + CS_Attention
├─ Conv2: 64 → 128 (32×32 → 16×16) + CS_Attention
├─ Conv3: 128 → 256 (16×16 → 8×8) + CS_Attention
└─ Conv4: 256 → 512 (8×8 → 4×4) + CS_Attention

Decoder:
├─ Deconv4: 512 → 256 (4×4 → 8×8) + CS_Attention + Skip
├─ Deconv3: 256 → 128 (8×8 → 16×16) + CS_Attention + Skip
├─ Deconv2: 128 → 64 (16×16 → 32×32) + CS_Attention + Skip
└─ Deconv1: 64 → 3 (32×32 → 64×64) + CS_Attention

Output: 2 streams × (8, 3, 64, 64)
```

## CS Attention (Applied at each layer)
```
CS Block:
├─ Input: (lap_features, res_features)
├─ Combine: lap + res
├─ GlobalAvgPool: (B, C, H, W) → (B, C)
├─ FC layers: Feature → Attention weights
├─ Softmax: Normalize weights
├─ Apply: 
│  ├─ attended_lap = lap * weight[0]
│  └─ attended_res = res * weight[1]
└─ Output: (attended_lap, attended_res)
```

## Discriminator: BasicDiscriminator
```
Input: Real/Fake streams (8, 3, 64, 64)
├─ Temporal discrimination
├─ Spatial discrimination
└─ Output: Real/Fake classification
```

## Loss Components
```
Total Loss = Adversarial + Reconstruction + Latent + Temporal

Adversarial Loss (w=1.0):
└─ Generator vs Discriminator

Reconstruction Loss (w=50.0):
└─ L1/L2 between input and output

Latent Loss (w=1.0):
└─ Feature space consistency

Temporal Losses:
├─ Consistency (w=0.08): Frame-to-frame smoothness
├─ Motion (w=0.04): Movement pattern consistency
└─ Regularization (w=0.01): Attention stability
```

## Training Configuration
```
Optimizer: Adam (lr=0.0002, β1=0.9)
Scheduler: Step decay every 15 iterations
Gradient Clipping: 1.0
Batch Size: 1
Epochs: 1
Device: CPU/GPU auto-detect
```

## Active Components
```
✅ UnetGenerator_CS with CS attention
✅ TemporalFeatureFusion (input enhancement)
✅ OpticalFlowFeatureFusion (motion-aware enhancement) 
✅ CombinedTemporalLoss (3 components)
✅ Enhanced video augmentation
✅ Dual-stream processing
✅ Aspect ratio preservation

❌ TemporalAttention (generator) - initialized but unused
❌ MultiScaleTemporalAttention
❌ EnhancedTemporalFusion
❌ HierarchicalTemporalAttention
```

## Parameter Count (Estimated)
```
UnetGenerator_CS: ~2-3M parameters
CS Attention blocks: ~50K parameters
TemporalFeatureFusion: ~100K parameters
OpticalFlowFeatureFusion: ~300K parameters
   ├─ FlowNetSD: ~45M parameters (shared/frozen)
   ├─ Motion Encoder (3D CNN): ~200K parameters
   └─ Motion Processing: ~100K parameters
BasicDiscriminator: ~1M parameters
Total: ~3.9M parameters (+ 45M FlowNet if trainable)
```

## Visual Model Graph
python train_video.py --dataset ucsd2 --name ucsd2_enhanced_vta_gan --use_temporal_attention --use_optical_flow --isize 256 --num_frames 16 --batchsize 2 --niter 60 --lr 0.0001 --lr_policy step --lr_decay_iters 20 --w_adv 1.0 --w_con 25.0 --w_lat 0.5 --w_temporal_consistency 0.15 --w_temporal_motion 0.08 --w_temporal_reg 0.02 --optical_flow_weight 0.3 --beta1 0.5 --grad_clip_norm 0.5 --workers 8
```
                            VTA-GAN Architecture
                                    
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT PROCESSING                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    Raw Video: (1, 8, 3, 64, 64)
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                    Laplacian Stream      Residual Stream
                   (1, 8, 3, 64, 64)    (1, 8, 3, 64, 64)
                           │                     │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL FUSION (if --use_temporal_attention)        │
└─────────────────────────────────────────────────────────────────────────────┘
                           │                     │
                           └──────────┬──────────┘
                                      │
                              Combined Input
                           (1, 8, 3, 64, 64)
                                      │
                              TemporalFeatureFusion
                                      │
                             Temporal Summary
                            (1, 3, 64, 64)
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                  Enhanced Lap              Enhanced Res
                (1, 8, 3, 64, 64)        (1, 8, 3, 64, 64)
                          │                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPTICAL FLOW PROCESSING                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          OpticalFlowFeatureFusion
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                Motion Features: (1, 3, 64, 64)    Motion Magnitude
                          │                       │
                          └───────────┬───────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                    Fake Lap Stream         Fake Res Stream
                   (8, 3, 64, 64)         (8, 3, 64, 64)
                          │                       │
                          └───────────┬───────────┘
                                      │
                              Reshape to Video
                           (1, 8, 3, 64, 64)
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DISCRIMINATOR                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                              BasicDiscriminator
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                   Real/Fake Score         Temporal Score
                          │                       │
                          └───────────┬───────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LOSSES                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
        Adversarial Loss         Reconstruction           Temporal Losses
          (w=1.0)                    Loss                 (if enabled)
              │                    (w=50.0)                    │
              │                       │               ┌───────┼───────┐
              │                       │               │       │       │
              │                       │          Consistency Motion Reg
              │                       │           (w=0.08) (w=0.04) (w=0.01)
              │                       │               │       │       │
              └───────────────────────┼───────────────┴───────┴───────┘
                                      │
                                 Total Loss
                                      │
                              Backpropagation
```

## CS Attention Detail
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CS Attention Block                                │
└─────────────────────────────────────────────────────────────────────────────┘

Input: (Laplacian Features, Residual Features)
         │                      │
         └──────────┬───────────┘
                    │
              Combine (Add)
                    │
            Global Average Pool
                    │
              FC Layer 1
                    │
              FC Layer 2
                    │
               Softmax
                    │
           Attention Weights
        ┌──────────┴──────────┐
        │                     │
    Weight[0]             Weight[1]
        │                     │
        ▼                     ▼
   Attended Lap         Attended Res
```

## TEMPORAL FUSION Detailed Explanation

### What TemporalFeatureFusion Does:

```
Input: Combined streams (1, 8, 3, 64, 64)
       ↓
TemporalFeatureFusion Module:
├─ Analyzes motion patterns across 8 frames
├─ Extracts temporal relationships between frames
├─ Identifies consistent vs. anomalous temporal patterns
├─ Compresses temporal information into single frame
       ↓
Output: Temporal summary (1, 3, 64, 64)
```

### Step-by-Step Process:

1. **Input Combination**: `combined = input_lap + input_res`
   - Merges Laplacian (details) and Residual (structure) streams
   - Creates unified representation of scene content

2. **Temporal Analysis**: `temporal_fused = self.temporal_fusion(combined)`
   - Processes all 8 frames simultaneously
   - Learns temporal dependencies and motion patterns
   - Identifies frame-to-frame correlations

3. **Feature Compression**: 8 frames → 1 frame
   - Distills temporal information into single representative frame
   - Preserves important motion cues and temporal context

4. **Enhancement Broadcasting**: 
   ```python
   temporal_fused = temporal_fused.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
   input_lap_enhanced = self.input_lap + 0.1 * temporal_fused
   input_res_enhanced = self.input_res + 0.1 * temporal_fused
   ```
   - Broadcasts temporal summary back to all 8 frames
   - Adds 10% temporal context to each original frame
   - Creates "temporally-aware" input features

### Purpose for Anomaly Detection:

- **Normal patterns**: Consistent motion, predictable frame transitions
- **Anomalous patterns**: Sudden changes, irregular motion, unusual events
- **Context enhancement**: Each frame gets global temporal understanding
- **Improved discrimination**: Generator produces more temporally consistent outputs

### Effect on Model:

**Without Temporal Fusion**: Each frame processed independently
**With Temporal Fusion**: Each frame enhanced with 8-frame temporal context

This makes the model more sensitive to temporal anomalies while maintaining spatial detail.

## Optical Flow Integration Details
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPTICAL FLOW PROCESSING                              │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Combined streams (1, 8, 3, 64, 64)
       ↓
OpticalFlowFeatureFusion Module:
├─ FlowNetSD: Computes flow between consecutive frames (7 flow maps)
├─ Motion Encoder: 3D CNN analyzes temporal flow patterns
├─ Motion Compression: Reduces to input feature dimensions  
├─ Motion Attention: Focuses on important motion regions
├─ Magnitude Analysis: Detects motion intensity for anomalies
       ↓
Output: Motion-enhanced features (1, 3, 64, 64)

Integration with TemporalFeatureFusion:
├─ Temporal features: Context across frames
├─ Motion features: Movement patterns and anomalies
└─ Combined enhancement: lap/res + temporal + motion
```

## 🚀 OPTICAL FLOW ENHANCEMENT SUMMARY

### What We Added:

1. **OpticalFlowFeatureFusion Module** (`lib/models/attention_modules.py`)
   - **FlowNetSD Integration**: Uses existing FlowNetSD for optical flow computation
   - **Motion Pattern Analysis**: 3D CNN analyzes temporal flow patterns  
   - **Motion Attention**: Focuses on important motion regions
   - **Motion Magnitude**: Detects motion intensity for anomaly sensitivity
   - **Multi-modal Fusion**: Combines with TemporalFeatureFusion

2. **GAN Model Integration** (`lib/models/gan_model.py`)
   - **Import**: Added OpticalFlowFeatureFusion import
   - **Initialization**: Module initialized alongside TemporalFeatureFusion
   - **Device Placement**: Proper GPU/CPU handling
   - **Training Mode**: Included in training setup
   - **Optimizer**: Parameters included in optimizer
   - **Forward Pass**: Integrated in forward_g() method

3. **Configuration Options** (`options.py`)
   - `--use_optical_flow`: Enable/disable optical flow (default: True)
   - `--optical_flow_weight`: Control optical flow influence (default: 0.2)
   - `--motion_magnitude_weight`: Control motion magnitude sensitivity (default: 0.1)

4. **Enhanced Architecture**:
   ```
   Original: Raw Input → TemporalFusion → Enhanced Input → Generator
   Enhanced: Raw Input → TemporalFusion + OpticalFlowFusion → Multi-Enhanced Input → Generator
   ```

### Benefits:

✅ **Motion-Aware Anomaly Detection**: Captures unusual movement patterns
✅ **Complementary Features**: Temporal context + Motion dynamics  
✅ **Configurable**: Can be enabled/disabled and tuned
✅ **Backward Compatible**: Works with existing TemporalFeatureFusion
✅ **Scalable**: Adapts to different video resolutions and frame counts

### How It Works:

1. **Optical Flow Computation**: FlowNetSD computes flow between consecutive frames
2. **Motion Analysis**: 3D CNN processes flow sequence to extract motion patterns
3. **Feature Enhancement**: Motion features enhance both Laplacian and Residual streams
4. **Anomaly Sensitivity**: Motion magnitude helps identify sudden/unusual movements
5. **Multi-Modal Fusion**: Combines temporal context with motion dynamics

### Expected Improvements:

🎯 **Better Anomaly Detection**: Motion patterns help identify unusual events
🎯 **Temporal Consistency**: Motion-aware features improve frame coherence
🎯 **Reduced False Positives**: Better understanding of normal vs abnormal motion
🎯 **Enhanced Generalization**: Motion features transfer across different scenarios
