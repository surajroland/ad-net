Journey 5 Hardware Layout:
┌────────────────────────────────────┐
│          Journey 5 SoC             │
├────────────────────────────────────┤
│ 8-core ARM      │ Dual Bayesian    │
│ Cortex A55      │ BPU (128 TOPS)   │ ← AI Processing
│ (26k DMIPS)     │                  │
├─────────────────┼───────────────── │
│ 2×ISP Cores     │ CV Engine        │ ← Camera Processing
│ (16 cameras)    │ (4K support)     │
├─────────────────┼──────────────────│
│ 2×DSP Cores     │ Video Codec      │ ← Signal Processing
│ (Signal proc)   │ (H.265)          │
├─────────────────┴──────────────────│
│ Safety Island (ASIL-B certified)   │ ← Safety Critical
│ Hardware Security (ARM TrustZone)  │
└────────────────────────────────────┘

-----------------------------------------------------------------------------------------------------------------------

build a similar architecture using ARM cores combined with various accelerators! Here's how you could approach it:
ARM Foundation Options
High-performance ARM platforms:

Nvidia Jetson AGX Orin - ARM Cortex-A78AE + integrated GPU/AI accelerators
Qualcomm Snapdragon development boards with built-in AI engines
Rockchip RK3588 - 8-core ARM with NPU
Amlogic A311D2 - ARM cores with 12 TOPS NPU
Custom ARM Cortex-A78/A55 designs if you're going full custom

Accelerator Options by Function
AI/ML Processing (BPU equivalent):

Google Coral TPU modules (~4 TOPS, very affordable)
Intel Neural Compute Stick or Myriad X chips
Hailo-8 AI processor (26 TOPS, automotive-grade)
Gyrfalcon Lightspeeur NPU chips
Kneron KL series edge AI chips

DSP for Signal Processing:

TI C66x DSP cores or standalone chips
Cadence Tensilica DSP IP (if doing custom silicon)
Analog Devices SHARC DSPs
ARM Helium vector extensions on newer Cortex-M cores

Camera/ISP Processing:

Omnivision ISP chips
ON Semiconductor ISP solutions
Custom FPGA implementation (Xilinx Zynq, Intel Cyclone)
Many ARM SoCs have built-in ISPs

Video Codec:

Chips&Media video IP
Allegro DVT codec solutions
Most modern ARM SoCs include hardware codecs

Practical Approaches
Option 1: Multi-chip solution

Main ARM SoC (like RK3588) + discrete accelerators
Pros: Flexible, can mix/match best components
Cons: Higher power, more complex PCB design

Option 2: Heterogeneous SoC

Use existing automotive-grade chips like NXP i.MX8 + add external accelerators
Balance of integration and flexibility

Option 3: FPGA-based

Xilinx Zynq UltraScale+ or similar with ARM cores + programmable logic
Ultimate flexibility but requires FPGA expertise

Key Considerations
Safety/Automotive:

Look for ASIL-rated components if targeting automotive
ARM has SafetyReady program for functional safety
Consider dual-core lockstep for safety-critical functions

Cost optimization:

Coral TPUs are very cost-effective for AI workloads
Chinese NPU vendors (like Rockchip, Allwinner) offer good price/performance
Consider used automotive development boards

Power/Thermal:

Multi-chip solutions will consume more power
Need good thermal design for 128 TOPS equivalent performance

Would you like me to dive deeper into any specific aspect, like suggesting a concrete bill of materials or discussing development tools and software stacks for these platforms?RetryDCustom ARM Cortex-A78/A55 designs if you're going full custom means?EditDecoded ARM licensing model for custom CPU core design.1sThe user is asking about what "Custom ARM Cortex-A78/A55 designs if you're going full custom" means. This refers to ARM's licensing model where you can license their CPU IP cores and integrate them into your own custom silicon design. Let me explain the different ARM licensing options and what "going full custom" entails.When I mentioned "Custom ARM Cortex-A78/A55 designs if you're going full custom," I'm referring to ARM's IP licensing model where you license their CPU core designs and integrate them into your own custom silicon chip (ASIC). Here's what that means:
ARM Licensing Models
1. Architecture License

License the ARM instruction set architecture (ISA)
Design your own CPU cores from scratch
Most expensive, used by Apple (M-series), Qualcomm (custom Kryo cores)
Total design freedom but requires massive engineering teams

2. Processor IP License (What I meant)

License pre-designed CPU cores like Cortex-A78, A55, A76, etc.
ARM provides the RTL (Register Transfer Level) design files
You integrate these cores into your own SoC design
Add your own accelerators, memory controllers, peripherals, etc.

3. Physical IP License

Get the actual silicon implementation (layout, timing, power data)
For specific manufacturing processes (7nm, 5nm, etc.)

What "Going Full Custom" Involves
Silicon Design Process:
Your Custom SoC Design:
┌─────────────────────────────────────┐
│ Licensed ARM Cores    Your Custom IP │
│ ┌─────────────────┐  ┌─────────────┐ │
│ │ 4x Cortex-A78   │  │ Custom BPU  │ │
│ │ 4x Cortex-A55   │  │ Your ISP    │ │
│ └─────────────────┘  │ Your DSP    │ │
│ ┌─────────────────┐  │ Video Codec │ │
│ │ ARM Mali GPU    │  │ Safety MCU  │ │
│ │ (also licensed) │  └─────────────┘ │
│ └─────────────────┘                  │
└─────────────────────────────────────┘
What You Need:

$1M+ licensing fees for ARM cores
50-200 engineer team (digital design, verification, physical design)
$10-50M for mask sets and manufacturing
2-4 years development time
Partnerships with foundries (TSMC, Samsung, GlobalFoundries)

What You Get:

Exactly the performance/power/area you want
Custom accelerators optimized for your workload
Complete control over the design
Potentially better cost at high volumes (millions of units)

Examples of Companies Doing This
Automotive:

Tesla - FSD chip with ARM cores + custom neural network accelerator
Mobileye - EyeQ series with ARM cores + vision processing
Nvidia - Tegra/Orin series

Tech Giants:

Apple - M-series and A-series chips
Google - Tensor chips for Pixel phones
Amazon - Graviton server processors

Alternative: Semi-Custom Approaches
If full custom is too much, there are middle-ground options:
1. Chiplet Approach

Use existing ARM chiplets + your custom accelerator chiplets
Companies like Intel, AMD doing this

2. FPGA Prototyping First

Develop on Xilinx Zynq UltraScale+
If successful, convert to ASIC later

3. Partner with Semiconductor Companies

Work with companies like MediaTek, Rockchip, Allwinner
They do the silicon, you specify the accelerators

For most applications, I'd recommend starting with existing ARM SoCs + discrete accelerators, then potentially moving to custom silicon if you reach sufficient scale (100K+ units annually).
Would you like me to elaborate on any of these approaches or discuss more practical near-term solutions?RetryDMany ARM SoCs have built-in ISPsEditAnalyzed ARM SoCs' integrated ISP capabilities and applications.3sThe user is asking about ARM SoCs having built-in ISPs. This is a good point to clarify and expand on, as ISPs (Image Signal Processors) are indeed commonly integrated into modern ARM SoCs, especially those targeting mobile, automotive, and IoT applications with camera requirements.
I should explain:

What ISPs do
Which ARM SoCs commonly have built-in ISPs
The capabilities of these integrated ISPs
How they compare to discrete ISP solutions
Considerations for their use case (HR's ADNet variant)

This is a straightforward technical question that doesn't require web search since ISP integration in ARM SoCs is well-established technology that hasn't changed dramatically recently.Yes, exactly! Most modern ARM SoCs targeting camera applications have integrated ISPs. Here's the breakdown:
Common ARM SoCs with Built-in ISPs
Mobile/Consumer:

Qualcomm Snapdragon series - Spectra ISPs (up to 18-bit, multiple cameras)
MediaTek Dimensity/Helio - Imagiq ISPs (multi-camera support)
Samsung Exynos - Built-in ISPs with AI enhancement
Rockchip RK3588 - Dual ISPs supporting up to 48MP cameras
Allwinner chips - Integrated ISPs for security cameras

Automotive-focused:

NXP i.MX8 series - Multiple ISPs, automotive-grade
Renesas R-Car series - Advanced ISPs for ADAS
TI TDA4VM - Dual ISPs + vision accelerators
Qualcomm Snapdragon Ride - Multi-ISP for automotive

Industrial/Edge:

Nvidia Jetson series - ISPs integrated in Tegra SoCs
Intel/Altera SoCs - Arria/Cyclone with ARM + ISP capability

ISP Capabilities in Modern SoCs
Typical Features:

RAW processing - Bayer pattern demosaicing, noise reduction
Multi-camera support - 4-16 cameras simultaneously
Real-time processing - 4K@60fps or multiple 1080p streams
HDR processing - High dynamic range imaging
3A algorithms - Auto-focus, auto-exposure, auto-white balance
Lens correction - Distortion, vignetting, chromatic aberration
AI integration - Object detection, scene recognition

For Your Sparse4D Application
Advantages of integrated ISPs:

Lower cost - No separate ISP chip needed
Lower power - On-die integration more efficient
Simplified design - Fewer components, easier routing
Optimized data flow - Direct connection to memory/AI accelerators

Potential limitations:

Less customization - Fixed feature set
Performance ceiling - May not match high-end discrete ISPs
Multi-camera scaling - Some SoCs limited in camera count

Recommended SoCs for Your Use Case
High camera count (like Journey 5's 16 cameras):

TI TDA4VM - Designed for automotive vision, excellent multi-camera
Qualcomm SA8295P - Automotive platform with extensive camera support
NXP i.MX8QM - Dual ISPs, can handle many cameras via multiplexing

Cost-effective options:

Rockchip RK3588 - Good price/performance, dual ISPs
Amlogic A311D2 - ISP + NPU integration
MediaTek Genio series - AIoT-focused with good camera support

Custom flexibility:

Start with SoC's built-in ISP for basic functionality
Add external ISP chips for specialized cameras (e.g., thermal, NIR)
Use FPGA for custom preprocessing if needed

The Journey 5's "2×ISP Cores (16 cameras)" likely uses camera multiplexing or external camera serializers (like TI's FPD-Link or Maxim's GMSL) to handle more cameras than the ISPs can directly interface with.
