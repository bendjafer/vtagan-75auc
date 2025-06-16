# VTA-GAN Architecture: Complete Model with Temporal Attention and Optical Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                        VTA-GAN COMPLETE ARCHITECTURE                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                              INPUT STAGE                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                           Video Snippet Input (batch, 8, 3, 64, 64)
                                                                    │
                                                                    │
                                           ┌────────────────────────▼─────────────────────────┐
                                           │         Frequency Decomposition (FD)              │
                                           │     - Gaussian Pyramid Decomposition              │
                                           │     - Dynamic Sizing (size=opt.isize)             │
                                           │     - pyrDown() → pyrUp() → Subtract              │
                                           └────────────────────┬─────────────────────────────┘
                                                                │
                                     ┌──────────────────────────┴──────────────────────────┐
                                     │                                                     │
                           ┌─────────▼─────────┐                                ┌─────────▼─────────┐
                           │   Laplacian (lap) │                                │  Residual (res)   │
                           │ (batch,8,3,64,64) │                                │ (batch,8,3,64,64) │
                           └─────────┬─────────┘                                └─────────┬─────────┘
                                     │                                                     │
                                     └──────────────────────────┬──────────────────────────┘
                                                                │
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                         ENHANCEMENT MODULES                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                                │
                                              ┌─────────────────▼─────────────────┐
                                              │      OPTICAL FLOW FUSION         │
                                              │   OpticalFlowFeatureFusion        │
                                              │                                   │
                                              │  ┌─────────────────────────────┐  │
                                              │  │        FlowNetSD            │  │
                                              │  │   - Encoder-Decoder         │  │
                                              │  │   - Correlation Layers      │  │
                                              │  │   - Flow Estimation         │  │
                                              │  │   Output: (B*T-1, 2, H, W)  │  │
                                              │  └─────────────────────────────┘  │
                                              │                │                  │
                                              │  ┌─────────────▼─────────────┐    │
                                              │  │    Flow Processing        │    │
                                              │  │ - Upsample to input size  │    │
                                              │  │ - Magnitude calculation   │    │
                                              │  │ - Direction extraction    │    │
                                              │  │ - Temporal padding        │    │
                                              │  └─────────────┬─────────────┘    │
                                              │                │                  │
                                              │  ┌─────────────▼─────────────┐    │
                                              │  │   Motion Convolutions     │    │
                                              │  │ - 3x3 Conv + GroupNorm    │    │
                                              │  │ - Motion features extract │    │
                                              │  │ - Adaptive scaling        │    │
                                              │  └─────────────┬─────────────┘    │
                                              │                │                  │
                                              │  ┌─────────────▼─────────────┐    │
                                              │  │   Feature Enhancement     │    │
                                              │  │ - Weighted combination    │    │
                                              │  │ - Residual connections    │    │
                                              │  │ - Dynamic resize to input │    │
                                              │  └─────────────────────────────┘  │
                                              └─────────────────┬─────────────────┘
                                                                │
                                                   ┌────────────▼────────────┐
                                                   │  Enhanced LAP & RES     │
                                                   │ (batch,8,3,64,64) each  │
                                                   └────────────┬────────────┘
                                                                │
                                              ┌─────────────────▼─────────────────┐
                                              │     TEMPORAL ATTENTION FUSION    │
                                              │   TemporalFeatureFusion           │
                                              │                                   │
                                              │  ┌─────────────────────────────┐  │
                                              │  │   Combined Input            │  │
                                              │  │   lap + res                 │  │
                                              │  │ (batch, 8, 3, 64, 64)       │  │
                                              │  └─────────────┬───────────────┘  │
                                              │                │                  │
                                              │  ┌─────────────▼───────────────┐  │
                                              │  │    Spatial Encoder          │  │
                                              │  │ - Conv2d(3→32) + ReLU       │  │
                                              │  │ - Conv2d(32→64) + ReLU      │  │ 
                                              │  │ - Conv2d(64→128) + ReLU     │  │
                                              │  │ - AdaptiveAvgPool2d(4,4)    │  │
                                              │  │ Output: (B, 8, 128, 4, 4)   │  │
                                              │  └─────────────┬───────────────┘  │
                                              │                │                  │
                                              │  ┌─────────────▼───────────────┐  │
                                              │  │   Temporal Attention        │  │
                                              │  │ - Reshape to (B, 8, 2048)   │  │
                                              │  │ - Multi-Head Self-Attention │  │
                                              │  │ - 4 heads, head_dim=512     │  │
                                              │  │ - Positional Encoding       │  │
                                              │  │ - LayerNorm + Dropout       │  │
                                              │  └─────────────┬───────────────┘  │
                                              │                │                  │
                                              │  ┌─────────────▼───────────────┐  │
                                              │  │   Temporal Aggregation      │  │
                                              │  │ - Global avg over time      │  │
                                              │  │ - Linear(2048→3*64*64)      │  │
                                              │  │ - Reshape to (B,3,64,64)    │  │
                                              │  └─────────────────────────────┘  │
                                              └─────────────────┬─────────────────┘
                                                                │
                                                   ┌────────────▼────────────┐
                                                   │   Temporal Features     │
                                                   │  (batch, 3, 64, 64)     │
                                                   └────────────┬────────────┘
                                                                │
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                         GENERATOR STAGE                                                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                                │
                                                   ┌────────────▼────────────┐
                                                   │    Feature Combination  │
                                                   │ - Optical Flow Enhanced │
                                                   │ - Temporal Attended     │
                                                   │ - Residual Connections  │
                                                   └────────────┬────────────┘
                                                                │
                                             ┌──────────────────┴───────────────────┐
                                             │                                      │
                                   ┌─────────▼─────────┐                  ┌─────────▼─────────┐
                                   │  Enhanced LAP     │                  │  Enhanced RES     │
                                   │ + Temporal * 0.1  │                  │ + Temporal * 0.1  │
                                   │ (batch,8,3,64,64) │                  │ (batch,8,3,64,64) │
                                   └─────────┬─────────┘                  └─────────┬─────────┘
                                             │                                      │
                                             │          ┌─────────────┐             │
                                             │          │    Noise    │             │
                                             │          │(batch,8,3,  │             │
                                             │          │  64,64)     │             │
                                             │          └──────┬──────┘             │
                                             │                 │                    │
                                   ┌─────────▼─────────┐       │       ┌─────────▼─────────┐
                                   │   LAP + Noise     │       │       │   RES + Noise     │
                                   │(batch*8,3,64,64)  │       │       │(batch*8,3,64,64)  │
                                   └─────────┬─────────┘       │       └─────────┬─────────┘
                                             │                 │                 │
                                             └─────────────────┼─────────────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │     GENERATOR       │
                                                    │      (NetG)         │
                                                    │                     │
                                                    │ ┌─────────────────┐ │
                                                    │ │   Encoder       │ │
                                                    │ │ - Conv Layers   │ │
                                                    │ │ - Downsampling  │ │
                                                    │ │ - Feature Ext   │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │   Latent Space  │ │
                                                    │ │ - Bottleneck    │ │
                                                    │ │ - Compressed    │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │    Decoder      │ │
                                                    │ │ - Deconv Layers │ │
                                                    │ │ - Upsampling    │ │
                                                    │ │ - Reconstruction│ │
                                                    │ └─────────────────┘ │
                                                    └──────────┬──────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │   Generated Output  │
                                                    │  Fake LAP + Fake RES│
                                                    │ (batch*8,3,64,64)    │
                                                    └──────────┬──────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │    Reshape Back     │
                                                    │ (batch,8,3,64,64)   │
                                                    └──────────┬──────────┘
                                                               │
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                       DISCRIMINATOR STAGE                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                               │
                                      ┌────────────────────────┼─────────────────────────┐
                                      │                        │                         │
                            ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
                            │    Real Data      │    │   Fake Data       │    │  Fake Aug Data    │
                            │  LAP + RES        │    │  Generated        │    │   (Augmented)     │
                            │(batch*8,3,64,64)  │    │(batch*8,3,64,64)  │    │(batch*8,3,64,64)  │
                            └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                                      │                        │                         │
                                      └────────────────────────┼─────────────────────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │    DISCRIMINATOR    │
                                                    │       (NetD)        │
                                                    │                     │
                                                    │ ┌─────────────────┐ │
                                                    │ │  Conv Layers    │ │
                                                    │ │ - Feature Ext   │ │
                                                    │ │ - Downsampling  │ │
                                                    │ │ - LeakyReLU     │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │ Feature Maps    │ │
                                                    │ │ - Intermediate  │ │
                                                    │ │ - Representations│ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │  Classifier     │ │
                                                    │ │ - Real/Fake     │ │
                                                    │ │ - Probability   │ │
                                                    │ └─────────────────┘ │
                                                    └──────────┬──────────┘
                                                               │
                                             ┌─────────────────┼─────────────────┐
                                             │                 │                 │
                                   ┌─────────▼─────────┐       │       ┌─────────▼─────────┐
                                   │   Predictions     │       │       │   Feature Maps    │
                                   │ - pred_real       │       │       │ - feat_real       │
                                   │ - pred_fake       │       │       │ - feat_fake       │
                                   │ - pred_fake_aug   │       │       │ - feat_fake_aug   │
                                   └───────────────────┘       │       └───────────────────┘
                                                               │
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           LOSS COMPUTATION                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                               │
                                                    ┌──────────▼──────────┐
                                                    │    LOSS FUNCTIONS   │
                                                    │                     │
                                                    │ ┌─────────────────┐ │
                                                    │ │ Adversarial     │ │
                                                    │ │ - Generator     │ │
                                                    │ │ - Discriminator │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │ Reconstruction  │ │
                                                    │ │ - L2 Loss       │ │
                                                    │ │ - Content Loss  │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │ Feature Match   │ │
                                                    │ │ - Latent Loss   │ │
                                                    │ │ - Feature Align │ │
                                                    │ └─────────────────┘ │
                                                    │         │           │
                                                    │ ┌───────▼─────────┐ │
                                                    │ │ Temporal Loss   │ │
                                                    │ │ - Consistency   │ │
                                                    │ │ - Smoothness    │ │
                                                    │ │ - Flow-based    │ │
                                                    │ └─────────────────┘ │
                                                    └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                        TRAINING FLOW                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Input Video → FD → Optical Flow → Temporal Attention → Generator → Discriminator → Loss → Backprop

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                      TESTING/INFERENCE                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Input Video → Enhanced Processing → Reconstruction → Error Calculation → Anomaly Score → AUC = 1.0000

```

## Key Architecture Components:

### 1. **Input Processing**
- Frequency Decomposition (FD) with dynamic sizing
- Splits into Laplacian and Residual components

### 2. **Enhancement Modules**
- **OpticalFlowFeatureFusion**: Motion-aware feature enhancement
- **TemporalFeatureFusion**: Temporal attention across frames

### 3. **Generator Network**
- Encoder-decoder architecture
- Processes enhanced features with noise injection

### 4. **Discriminator Network**
- Adversarial training for realistic reconstruction
- Feature extraction for latent loss

### 5. **Loss Functions**
- Adversarial, Reconstruction, Feature Matching, Temporal losses
- Combined training for robust anomaly detection

### 6. **Dynamic Sizing**
- Supports any input size (64x64, 128x128, 256x256, etc.)
- Automatic tensor shape matching throughout pipeline
