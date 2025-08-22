## **Tier 1: Standard Multi-Camera Temporal Datasets**

### **nuScenes** (2019) - Boston/Singapore
- **Cameras**: 6 surround-view (360°)
- **Temporal**: 20Hz, 20s sequences
- **Annotations**: 3D boxes, tracking IDs, attributes
- **Access**: Free registration

### **Waymo Open Dataset** (2019) - Multiple US cities
- **Cameras**: 5 cameras (front + 4 side)
- **Temporal**: 10Hz, ~20s sequences
- **Annotations**: 3D boxes, tracking, motion labels
- **Access**: Free with terms acceptance

### **Argoverse 1.0** (2019) - Pittsburgh/Miami
- **Cameras**: 7 cameras + 2 stereo
- **Temporal**: 10Hz sequences
- **Annotations**: 3D tracking, motion forecasting
- **Access**: Free download

### **Argoverse 2.0** (2021) - 6 cities worldwide
- **Cameras**: 7 surround cameras
- **Temporal**: 10Hz, longer sequences
- **Annotations**: Improved 3D boxes, sensor suite
- **Access**: Free registration

## **Tier 2: Large-Scale Public Datasets**

### **ONCE** (2021) - China
- **Cameras**: 7 cameras
- **Temporal**: 10Hz sequences
- **Annotations**: 3D detection + tracking
- **Access**: Free registration

### **Zenseact Open Dataset (ZOD)** (2022) - Europe
- **Cameras**: 8 cameras
- **Temporal**: Variable Hz, long sequences
- **Annotations**: Dense 3D annotations
- **Access**: Free registration

### **KITTI-360** (2021) - Germany
- **Cameras**: 2 stereo pairs
- **Temporal**: Continuous sequences
- **Annotations**: 3D semantic segmentation
- **Access**: Free download

### **A2D2** (2020) - Audi/Germany
- **Cameras**: 6 cameras
- **Temporal**: 10Hz sequences
- **Annotations**: 3D boxes, semantic labels
- **Access**: Free registration

## **Tier 3: Specialized/Regional Datasets**

### **PandaSet** (2021) - San Francisco
- **Cameras**: 6 cameras
- **Temporal**: 10Hz sequences
- **Annotations**: 3D boxes + semantic segmentation
- **Access**: Free registration

### **Lyft Level 5** (2020) - Palo Alto
- **Cameras**: 7 cameras
- **Temporal**: 10Hz sequences
- **Annotations**: Similar to nuScenes format
- **Access**: Free (archived but available)

### **CADC** (2020) - Canada/Winter
- **Cameras**: 8 cameras
- **Temporal**: Adverse weather sequences
- **Annotations**: 3D detection in snow/rain
- **Access**: Free registration

### **ApolloScape** (2018) - China/Baidu
- **Cameras**: Multiple camera views
- **Temporal**: Various sequence lengths
- **Annotations**: 3D detection, tracking, trajectory
- **Access**: Free registration

## **Tier 4: Academic/Research Datasets**

### **H3D** (2019) - Honda Research
- **Cameras**: 3 front cameras
- **Temporal**: Highway driving sequences
- **Annotations**: 3D boxes, occlusion labels
- **Access**: Research license

### **DAIR-V2X** (2022) - V2X Cooperative
- **Cameras**: Vehicle + infrastructure cameras
- **Temporal**: Cooperative perception sequences
- **Annotations**: 3D detection from multiple viewpoints
- **Access**: Free registration

### **Rope3D** (2022) - Roadside Perception
- **Cameras**: Roadside fixed cameras
- **Temporal**: Traffic monitoring sequences
- **Annotations**: 3D detection from roadside view
- **Access**: Free registration

### **DDAD** (2020) - Toyota Research
- **Cameras**: 6 cameras
- **Temporal**: Dense depth + detection
- **Annotations**: 3D boxes + dense depth
- **Access**: Research agreement

## **Tier 5: Synthetic/Simulation Datasets**

### **CARLA** (2017-ongoing) - Open Source Simulator
- **Cameras**: Configurable multi-camera setup
- **Temporal**: Any temporal resolution
- **Annotations**: Perfect ground truth
- **Access**: Open source

### **AirSim** (2017) - Microsoft Simulator
- **Cameras**: Configurable cameras
- **Temporal**: Configurable sequences
- **Annotations**: Synthetic ground truth
- **Access**: Open source

### **V2X-Sim** (2022) - V2X Simulation
- **Cameras**: Multi-vehicle/infrastructure cameras
- **Temporal**: Cooperative scenarios
- **Annotations**: Multi-agent annotations
- **Access**: Research use

## **Tier 6: Emerging/New Datasets**

### **Shifts** (2022) - Distribution Shift Evaluation
- **Cameras**: Various camera configurations
- **Temporal**: Cross-domain sequences
- **Annotations**: 3D detection across domains
- **Access**: Competition/research

### **nuPlan** (2021) - Planning Dataset
- **Cameras**: 6 cameras (nuScenes extension)
- **Temporal**: Planning-oriented sequences
- **Annotations**: Planning labels + 3D detection
- **Access**: Free registration

## **Implementation Priority Recommendation:**

### **Phase 1** (Core benchmarks):
- nuScenes, Waymo, Argoverse 2.0, KITTI-360

### **Phase 2** (Scale + diversity):
- ONCE, ZOD, A2D2, PandaSet

### **Phase 3** (Specialized):
- CADC, DAIR-V2X, H3D, Lyft Level 5

### **Phase 4** (Synthetic):
- CARLA, AirSim

**Total: ~20 public datasets for comprehensive 4D detection coverage**

Would you like me to add all of these to your framework configuration?
