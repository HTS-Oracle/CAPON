# 🧬 AtlasNMR

A comprehensive Streamlit-based application for advanced protein structure conformational analysis using multiple PDB models.

## 📋 Features

- **RMSD Analysis**: Calculate and visualize Root Mean Square Deviation between conformers
- **TM-Score Calculation**: Structure similarity assessment with scale-independent metrics
- **Secondary Structure Analysis**: DSSP-based secondary structure assignment and visualization
- **Binding Pocket Detection**: Identify and analyze potential drug binding sites
- **Clustering Analysis**: Multiple methods (K-means, Hierarchical, DBSCAN) with validation
- **PCA Analysis**: Principal Component Analysis for conformational space exploration
- **Hydrogen Bond Analysis**: Detect and quantify hydrogen bonds across conformations
- **Contact Analysis**: Monitor residue-residue contacts and their changes
- **Ramachandran Analysis**: Backbone dihedral angle validation
- **Statistical Analysis**: Bootstrap, cross-validation, and correlation studies
- **Interactive Visualizations**: 3D molecular viewers and comprehensive plots
- **Export Options**: Download results in multiple formats (PDB, CSV, JSON, PyMOL scripts)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HTS-Oracle/CAPON.git
cd CAPON
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

#### DSSP (Required for Secondary Structure Analysis)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install dssp
```

**macOS:**
```bash
brew install dssp
```

**Alternative - Install mkdssp:**
Download from: https://github.com/PDB-REDO/dssp

#### Optional: AutoDock Vina (for Docking)
```bash
pip install vina
```

## 💻 Usage

### Running the Application

```bash
streamlit run enhanced_pdb_analyzer.py
```

The application will open in your default web browser at `http://localhost:8501`

### Quick Start Guide

1. **Upload PDB File**: Upload a PDB file containing multiple models/conformations
2. **Configure Analysis**: Select atom selection (default: protein and name CA)
3. **Choose Modules**: Enable desired analysis modules from the sidebar
4. **Run Analysis**: Click "🚀 Run Complete Analysis"
5. **Explore Results**: Navigate through different analysis sections
6. **Export Data**: Download results in various formats

## 📊 Analysis Modules

### Core Analysis
- **RMSD Matrix**: Pairwise RMSD calculations
- **TM-Score**: Structure similarity scores
- **Clustering**: Conformational clustering with validation

### Structural Analysis
- **Secondary Structure**: Helix, sheet, loop assignments
- **Contact Analysis**: Residue-residue interactions
- **Hydrogen Bonds**: H-bond network analysis

### Advanced Features
- **PCA**: Dimensionality reduction and visualization
- **Binding Pockets**: Druggable pocket detection
- **Statistical Tests**: Bootstrap and cross-validation
- **Ramachandran**: Backbone geometry validation

## 📁 Input Format

The application accepts PDB files with multiple MODEL entries:

```pdb
MODEL        1
ATOM      1  N   MET A   1      ...
ATOM      2  CA  MET A   1      ...
...
ENDMDL
MODEL        2
ATOM      1  N   MET A   1      ...
...
ENDMDL
```

## 📤 Export Options

The application provides various export formats:

- **PDB Files**: Representative structures for each cluster
- **CSV Files**: RMSD matrices, TM-scores, and statistical data
- **JSON**: Complete analysis summary
- **PyMOL Scripts**: Visualization scripts for PyMOL
- **Plots**: High-resolution PNG/SVG images

## 🔧 Configuration

### Atom Selection

The app uses MDAnalysis selection syntax. Examples:

- `protein and name CA` - Alpha carbons only (default)
- `protein` - All protein atoms
- `backbone` - Backbone atoms (N, CA, C, O)
- `resid 1:100` - Specific residue range
- `protein and not type H` - Non-hydrogen protein atoms

### Clustering Parameters

- **Number of Clusters (k)**: 2-10 (default: auto-detect)
- **Method**: Hierarchical Ward, K-means, DBSCAN
- **Distance Cutoff**: For DBSCAN clustering

## 📚 Dependencies

### Required
- streamlit
- MDAnalysis
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- scipy
- scikit-learn

### Optional
- biopython (for secondary structure)
- py3Dmol (for 3D visualization)
- vina (for docking simulations)

## 🐛 Troubleshooting

### Common Issues

**1. DSSP Not Found**
```
Solution: Install DSSP system package or disable secondary structure analysis
```

**2. Memory Error with Large Files**
```
Solution: Use smaller selections (e.g., CA atoms only) or reduce number of models
```

**3. BioPython Import Error**
```bash
pip install biopython
```

**4. Streamlit Not Opening**
```bash
# Check if port 8501 is available
streamlit run enhanced_pdb_analyzer.py --server.port 8502
```

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{pdb_analyzer_pro,
  title = {PDB Conformational Analyzer Pro+},
  author = {HTS-Oracle},
  year = {2026},
  url = {https://github.com/HTS-Oracle/CAPON}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Hossam Nada**

## 🙏 Acknowledgments

- MDAnalysis Development Team
- BioPython Contributors
- Streamlit Team
- Scientific Python Community

## 📧 Contact

For questions and support, please open an issue on GitHub.

## 🔗 Resources

- [MDAnalysis Documentation](https://www.mdanalysis.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [BioPython Tutorial](https://biopython.org/wiki/Documentation)
- [DSSP](https://github.com/PDB-REDO/dssp)

---

**Note**: This application requires computational resources. For very large systems (>10,000 atoms) or many models (>100), consider using a high-performance computing environment.
