# ADNet Implementation Roadmap

## Current Status (Completed)
- ✅ Comprehensive type checking and best practices
- ✅ Unified dataset interfaces and abstract classes
- ✅ Multi-dataset loader with coordinate/class harmonization
- ✅ Cross-dataset validation framework
- ✅ Pre-commit hooks and development environment

## Tomorrow's Implementation Plan

### Phase 1: Dataset Integration
1. **Research dataset file formats and download requirements**
2. **Implement nuScenes dataset loader with real data integration**
3. **Add Waymo dataset support to multi-dataset framework**
4. **Implement KITTI dataset loader for comparison**
5. **Test unified taxonomy and coordinate harmonization**
6. **Create dataset validation and statistics pipeline**

### Phase 2: Visualization System
7. **Implement 3D point cloud visualization with Open3D**
8. **Create multi-view camera visualization dashboard**
9. **Build interactive dataset exploration interface**
10. **Add temporal sequence visualization for 4D data**

## Architecture Decisions

### Dataset Unification
- **No external tools needed** - built-in harmonization
- **UnifiedTaxonomy** for class mapping
- **CoordinateHarmonizer** for spatial alignment
- **MultiDatasetLoader** for combined training

### Visualization Approach
- **Primary**: Pipeline + Config pattern (matches ADNet architecture)
- **Optional**: MCP agent integration for natural language exploration
- **Backends**: Open3D (3D), Plotly (interactive), Streamlit (web)
- **No additional containers** - runs in existing dev environment

### File Structure
```
src/adnet/visualization/
├── pipeline.py          # Main VisualizationPipeline
├── backends/
│   ├── open3d_backend.py
│   ├── plotly_backend.py
│   └── streamlit_backend.py
├── exporters/
└── configs/
```

## Key Implementation Notes

### Dataset Requirements
- **nuScenes**: JSON + binary files from official website
- **Waymo**: TensorFlow records, convert to standard format
- **KITTI**: Text annotations + images
- **Argoverse**: Parquet/JSON format

### Naming Convention
- Repository: `ad-net` (with hyphen)
- Python package: `adnet` (no hyphen) - correct practice

### Visualization Capabilities
- 3D scene visualization (point clouds + bounding boxes)
- Multi-view camera displays
- Temporal sequence tracking
- Cross-dataset comparison
- Interactive exploration
- Export to various formats

## Next Session Commands
```bash
# Resume this exact session
claude-code --resume

# Or start fresh in project directory
cd /path/to/ad-net
claude-code
```

## Git State
- Current branch: master
- Latest commit: Cleanup and type checking implementation
- All framework code committed and pushed
- Ready for dataset implementation phase
