#!/usr/bin/env python3
"""
AtlasNMR
==================================================


import streamlit as st
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms, contacts
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.coordinates.PDB import PDBWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import json
import base64
import warnings
warnings.filterwarnings('ignore')

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore, norm, gaussian_kde, pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples, 
                            davies_bouldin_score, calinski_harabasz_score)
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold

# Additional imports for new features
try:
    from Bio.PDB import *
    from Bio.PDB.DSSP import DSSP
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    st.warning("BioPython not available. Some features will be limited.")

try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

try:
    from vina import Vina
    HAS_VINA = True
except ImportError:
    HAS_VINA = False

# Page configuration
st.set_page_config(
    page_title="PDB Conformational Analyzer Pro+",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS FOR NEW FEATURES
# ============================================================================

def calculate_tm_score(coords1, coords2):
    """
    Calculate TM-score between two structures
    TM-score is a scale-independent metric for structure similarity
    """
    N = len(coords1)
    
    # Align the structures
    mobile = coords1 - coords1.mean(axis=0)
    target = coords2 - coords2.mean(axis=0)
    
    # Calculate rotation matrix using SVD
    correlation = np.dot(mobile.T, target)
    U, S, Vt = np.linalg.svd(correlation)
    rotation = np.dot(U, Vt)
    
    # Ensure right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        Vt[-1] = -Vt[-1]
        rotation = np.dot(U, Vt)
    
    # Apply rotation
    mobile_aligned = np.dot(mobile, rotation)
    
    # Calculate TM-score
    d0 = 1.24 * np.cbrt(N - 15) - 1.8 if N > 21 else 0.5
    di = np.sqrt(np.sum((mobile_aligned - target)**2, axis=1))
    tm_score = np.sum(1 / (1 + (di / d0)**2)) / N
    
    return tm_score


def assign_secondary_structure(universe, model_idx=0):
    """
    Assign secondary structure using DSSP algorithm
    Returns: dict with residue-wise secondary structure
    """
    if not HAS_BIOPYTHON:
        return None
    
    try:
        universe.trajectory[model_idx]
        
        # Export to temporary PDB
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp_path = tmp.name
        
        with mda.Writer(tmp_path, format='PDB') as W:
            W.write(universe.atoms)
        
        # Parse with BioPython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('temp', tmp_path)
        model = structure[0]
        
        # Run DSSP
        dssp = DSSP(model, tmp_path, dssp='mkdssp')
        
        ss_dict = {}
        for key in dssp:
            chain_id, res_id = key
            ss = dssp[key][2]
            ss_dict[res_id[1]] = ss
        
        Path(tmp_path).unlink(missing_ok=True)
        return ss_dict
        
    except Exception as e:
        st.warning(f"Secondary structure assignment failed: {e}")
        return None


def detect_binding_pockets(universe, probe_radius=1.4, min_volume=50):
    """
    Detect potential binding pockets using geometric approach
    """
    protein = universe.select_atoms('protein')
    coords = protein.positions
    
    # Simple pocket detection based on concave regions
    # Calculate distances between all atoms
    from scipy.spatial import distance_matrix
    dist_mat = distance_matrix(coords, coords)
    
    # Find potential pocket centers (atoms with many neighbors within radius)
    neighbor_count = np.sum(dist_mat < 8.0, axis=1)
    
    pocket_centers = []
    for idx in np.argsort(neighbor_count)[-10:]:  # Top 10 potential centers
        center = coords[idx]
        # Calculate volume (simplified sphere approximation)
        atoms_in_sphere = coords[dist_mat[idx] < 8.0]
        volume = len(atoms_in_sphere) * 20  # Approximate volume
        
        if volume > min_volume:
            pocket_centers.append({
                'center': center,
                'volume': volume,
                'residue': protein[idx].resid,
                'n_atoms': len(atoms_in_sphere)
            })
    
    return pocket_centers


def calculate_druggability_score(pocket):
    """
    Calculate druggability score based on pocket properties
    """
    # Simple scoring based on volume and atom count
    volume_score = min(pocket['volume'] / 500, 1.0)  # Normalize to 500 Å³
    density_score = pocket['n_atoms'] / (pocket['volume'] / 20)
    
    # Weighted average
    druggability = 0.6 * volume_score + 0.4 * min(density_score, 1.0)
    
    return druggability


def bootstrap_clustering(rmsd_matrix, n_bootstrap=100, method='kmeans', k=3):
    """
    Bootstrap analysis for clustering stability
    """
    n_samples = len(rmsd_matrix)
    cluster_assignments = np.zeros((n_bootstrap, n_samples))
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_matrix = rmsd_matrix[indices][:, indices]
        
        # Perform clustering
        if method == 'kmeans':
            condensed = squareform(resampled_matrix)
            Z = linkage(condensed, method='ward')
            labels = fcluster(Z, k, criterion='maxclust')
        else:
            condensed = squareform(resampled_matrix)
            Z = linkage(condensed, method='ward')
            labels = fcluster(Z, k, criterion='maxclust')
        
        # Map back to original indices
        full_labels = np.zeros(n_samples) - 1
        full_labels[indices] = labels
        cluster_assignments[i] = full_labels
    
    # Calculate stability (how often structures cluster together)
    stability_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            same_cluster = np.sum(
                (cluster_assignments[:, i] == cluster_assignments[:, j]) & 
                (cluster_assignments[:, i] != -1)
            )
            stability_matrix[i, j] = same_cluster / n_bootstrap
    
    return stability_matrix, cluster_assignments


def fit_distribution(data, dist_type='normal'):
    """
    Fit statistical distribution to data
    """
    if dist_type == 'normal':
        mu, std = norm.fit(data)
        return {'mu': mu, 'std': std, 'type': 'normal'}
    elif dist_type == 'kde':
        kde = gaussian_kde(data)
        return {'kde': kde, 'type': 'kde'}
    return None


def cross_validate_clustering(rmsd_matrix, k_range=[2, 3, 4, 5], n_folds=5):
    """
    Cross-validation for clustering
    """
    n_samples = len(rmsd_matrix)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_scores = {k: [] for k in k_range}
    
    for train_idx, test_idx in kf.split(range(n_samples)):
        train_matrix = rmsd_matrix[train_idx][:, train_idx]
        
        for k in k_range:
            condensed = squareform(train_matrix)
            Z = linkage(condensed, method='ward')
            labels = fcluster(Z, k, criterion='maxclust')
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(train_matrix, labels, metric='precomputed')
                cv_scores[k].append(score)
    
    # Average scores
    avg_scores = {k: np.mean(scores) if scores else 0 
                  for k, scores in cv_scores.items()}
    
    return avg_scores




# ============================================================================
# ENHANCED ANALYZER CLASS
# ============================================================================

class EnhancedPDBAnalyzerPro:
    """Enhanced PDB analyzer with advanced features"""
    
    def __init__(self):
        self.universe = None
        self.coords = []
        self.rmsd_matrix = None
        self.cluster_info = {}
        self.results = {}
        self.residue_analysis = {}
        self.role_based_reps = {}
        self.binding_sites = {}
        self.pca_results = {}
        self.trajectory_analysis = {}
        self.selection_str = None
        self.full_chain_atoms = None
        self.tm_scores = {}
        self.secondary_structure = {}
        self.ligand_analysis = {}
        self.statistical_analysis = {}
        self.experimental_data = None
        self.tmp_pdb_path = None  # Track temporary PDB file
    
    def __del__(self):
        """Cleanup temporary files"""
        if hasattr(self, 'tmp_pdb_path') and self.tmp_pdb_path:
            try:
                Path(self.tmp_pdb_path).unlink(missing_ok=True)
            except:
                pass
        
    def load_from_upload(self, uploaded_file, selection_str, segid):
        """Load PDB from uploaded file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Store the temp path so we can clean it up later
        self.tmp_pdb_path = tmp_path
        
        self.universe = mda.Universe(tmp_path)
        self.selection_str = f"segid {segid} and {selection_str}"
        self.ref_atoms = self.universe.select_atoms(self.selection_str)
        self.full_chain_atoms = self.universe.select_atoms(f"segid {segid}")
        
        # Don't extract coords here - they should be extracted AFTER alignment
        # Don't delete the temp file yet - MDAnalysis needs it for trajectory iteration
        # It will be cleaned up when the analyzer is destroyed or a new file is loaded
        return len(self.universe.trajectory), len(self.ref_atoms)
    
    def load_experimental_data(self, data_file, data_type='bfactor'):
        """Load experimental data (B-factors, NMR order parameters)"""
        try:
            if data_type == 'bfactor':
                data = pd.read_csv(data_file)
                self.experimental_data = {
                    'type': 'bfactor',
                    'data': data
                }
            elif data_type == 'nmr':
                data = pd.read_csv(data_file)
                self.experimental_data = {
                    'type': 'nmr',
                    'data': data
                }
            return True
        except Exception as e:
            st.error(f"Error loading experimental data: {e}")
            return False
    
    def align_structures(self, ref_frame=0):
        """Align all structures to reference"""
        self.universe.trajectory[ref_frame]
        ref_universe = self.universe.copy()
        ref_universe.trajectory[ref_frame]
        
        aligner = align.AlignTraj(
            self.universe,
            reference=ref_universe,
            select=self.selection_str,
            in_memory=True
        )
        aligner.run()
    
    def calculate_rmsd_matrix(self):
        """Calculate pairwise RMSD matrix"""
        # Extract coordinates AFTER alignment
        self.coords = []
        for ts in self.universe.trajectory:
            self.coords.append(self.ref_atoms.positions.copy())
        
        n = len(self.coords)
        self.rmsd_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                rmsd = np.sqrt(np.mean(np.sum((self.coords[i] - self.coords[j])**2, axis=1)))
                self.rmsd_matrix[i, j] = rmsd
                self.rmsd_matrix[j, i] = rmsd
        
        return self.rmsd_matrix
    
    def calculate_tm_scores(self):
        """Calculate TM-scores for all structure pairs"""
        n = len(self.coords)
        self.tm_scores = np.zeros((n, n))
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        total_pairs = n * (n - 1) // 2
        pair_count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                tm_score = calculate_tm_score(self.coords[i], self.coords[j])
                self.tm_scores[i, j] = tm_score
                self.tm_scores[j, i] = tm_score
                
                pair_count += 1
                if pair_count % 10 == 0:
                    progress_bar.progress(pair_count / total_pairs)
                    status.text(f"Calculating TM-scores: {pair_count}/{total_pairs}")
        
        progress_bar.progress(1.0)
        status.empty()
        
        # Set diagonal to 1.0
        np.fill_diagonal(self.tm_scores, 1.0)
        
        return self.tm_scores
    
    def analyze_secondary_structure_evolution(self):
        """Track secondary structure changes across models"""
        if not HAS_BIOPYTHON:
            st.warning("BioPython not available. Secondary structure analysis skipped.")
            return None
        
        n_models = len(self.universe.trajectory)
        ss_evolution = []
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        for model_idx in range(n_models):
            status.text(f"Analyzing secondary structure: Model {model_idx+1}/{n_models}")
            ss_dict = assign_secondary_structure(self.universe, model_idx)
            
            if ss_dict:
                ss_evolution.append(ss_dict)
            
            progress_bar.progress((model_idx + 1) / n_models)
        
        progress_bar.empty()
        status.empty()
        
        if not ss_evolution:
            return None
        
        # Analyze changes
        all_residues = set()
        for ss in ss_evolution:
            all_residues.update(ss.keys())
        
        ss_changes = {}
        for res in sorted(all_residues):
            ss_types = [ss.get(res, '-') for ss in ss_evolution]
            ss_changes[res] = {
                'sequence': ss_types,
                'n_changes': len(set(ss_types)) - 1,
                'most_common': max(set(ss_types), key=ss_types.count)
            }
        
        self.secondary_structure = {
            'evolution': ss_evolution,
            'changes': ss_changes,
            'summary': self._summarize_ss_changes(ss_changes)
        }
        
        return self.secondary_structure
    
    def _summarize_ss_changes(self, ss_changes):
        """Summarize secondary structure changes"""
        n_variable = sum(1 for v in ss_changes.values() if v['n_changes'] > 0)
        n_helix = sum(1 for v in ss_changes.values() if v['most_common'] in ['H', 'G', 'I'])
        n_sheet = sum(1 for v in ss_changes.values() if v['most_common'] in ['E', 'B'])
        n_coil = sum(1 for v in ss_changes.values() if v['most_common'] in ['-', 'T', 'S'])
        
        return {
            'n_residues': len(ss_changes),
            'n_variable': n_variable,
            'n_helix': n_helix,
            'n_sheet': n_sheet,
            'n_coil': n_coil
        }
    
    def analyze_binding_pockets(self, probe_radius=1.4):
        """Detect and analyze binding pockets across conformers"""
        n_models = len(self.universe.trajectory)
        pocket_evolution = []
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        for model_idx in range(n_models):
            self.universe.trajectory[model_idx]
            status.text(f"Detecting pockets: Model {model_idx+1}/{n_models}")
            
            pockets = detect_binding_pockets(self.universe, probe_radius)
            
            # Calculate druggability scores
            for pocket in pockets:
                pocket['druggability'] = calculate_druggability_score(pocket)
            
            pocket_evolution.append(pockets)
            progress_bar.progress((model_idx + 1) / n_models)
        
        progress_bar.empty()
        status.empty()
        
        self.binding_sites = {
            'pockets_per_model': pocket_evolution,
            'summary': self._summarize_pockets(pocket_evolution)
        }
        
        return self.binding_sites
    
    def _summarize_pockets(self, pocket_evolution):
        """Summarize pocket analysis"""
        all_volumes = []
        all_druggability = []
        
        for pockets in pocket_evolution:
            for pocket in pockets:
                all_volumes.append(pocket['volume'])
                all_druggability.append(pocket['druggability'])
        
        return {
            'mean_volume': np.mean(all_volumes) if all_volumes else 0,
            'std_volume': np.std(all_volumes) if all_volumes else 0,
            'mean_druggability': np.mean(all_druggability) if all_druggability else 0,
            'n_pockets_total': len(all_volumes)
        }
    
    def perform_statistical_analysis(self):
        """Comprehensive statistical analysis"""
        if self.rmsd_matrix is None:
            return None
        
        # Get upper triangle of RMSD matrix
        rmsd_values = self.rmsd_matrix[np.triu_indices_from(self.rmsd_matrix, k=1)]
        
        # Fit distributions
        normal_fit = fit_distribution(rmsd_values, 'normal')
        kde_fit = fit_distribution(rmsd_values, 'kde')
        
        # Calculate correlations
        correlations = {}
        
        if hasattr(self, 'trajectory_analysis') and 'rmsf' in self.trajectory_analysis:
            rmsf = self.trajectory_analysis['rmsf']
            
            # Calculate per-atom average deviation from mean structure
            mean_pos = np.mean(self.coords, axis=0)
            per_atom_rmsd = []
            for coord in self.coords:
                atom_deviations = np.sqrt(np.mean((coord - mean_pos)**2, axis=1))
                per_atom_rmsd.append(atom_deviations)
            
            # Average per-atom RMSD across all models
            avg_per_atom_rmsd = np.mean(per_atom_rmsd, axis=0)
            
            # Now both have same length (n_atoms)
            if len(avg_per_atom_rmsd) == len(rmsf):
                pearson_r, pearson_p = pearsonr(avg_per_atom_rmsd, rmsf)
                spearman_r, spearman_p = spearmanr(avg_per_atom_rmsd, rmsf)
                
                correlations['rmsd_rmsf'] = {
                    'pearson': {'r': pearson_r, 'p': pearson_p},
                    'spearman': {'r': spearman_r, 'p': spearman_p}
                }
        
        # Compare with experimental data if available
        exp_comparison = None
        if self.experimental_data is not None:
            exp_comparison = self._compare_with_experimental()
        
        self.statistical_analysis = {
            'rmsd_distribution': {
                'mean': np.mean(rmsd_values),
                'std': np.std(rmsd_values),
                'min': np.min(rmsd_values),
                'max': np.max(rmsd_values),
                'median': np.median(rmsd_values),
                'normal_fit': normal_fit,
                'kde_fit': kde_fit
            },
            'correlations': correlations,
            'experimental_comparison': exp_comparison
        }
        
        return self.statistical_analysis
    
    def _compare_with_experimental(self):
        """Compare calculated flexibility with experimental data"""
        if self.experimental_data is None:
            return None
        
        exp_data = self.experimental_data['data']
        
        if hasattr(self, 'trajectory_analysis') and 'rmsf' in self.trajectory_analysis:
            calc_rmsf = self.trajectory_analysis['rmsf']
            
            # Match residues
            matched_exp = []
            matched_calc = []
            
            for idx, row in exp_data.iterrows():
                res_id = row.get('residue', row.get('resid', idx))
                if res_id < len(calc_rmsf):
                    matched_exp.append(row['value'])
                    matched_calc.append(calc_rmsf[res_id])
            
            if matched_exp:
                pearson_r, pearson_p = pearsonr(matched_exp, matched_calc)
                spearman_r, spearman_p = spearmanr(matched_exp, matched_calc)
                
                return {
                    'n_matched': len(matched_exp),
                    'pearson': {'r': pearson_r, 'p': pearson_p},
                    'spearman': {'r': spearman_r, 'p': spearman_p},
                    'data_type': self.experimental_data['type']
                }
        
        return None
    
    def bootstrap_analysis(self, n_bootstrap=100, k=3):
        """Bootstrap analysis for clustering confidence"""
        if self.rmsd_matrix is None:
            return None
        
        status = st.empty()
        status.text("Performing bootstrap analysis...")
        
        stability_matrix, assignments = bootstrap_clustering(
            self.rmsd_matrix, n_bootstrap, 'kmeans', k
        )
        
        status.empty()
        
        return {
            'stability_matrix': stability_matrix,
            'assignments': assignments,
            'mean_stability': np.mean(stability_matrix),
            'n_bootstrap': n_bootstrap
        }
    
    def cross_validation_analysis(self, k_range=[2, 3, 4, 5]):
        """Cross-validation for clustering"""
        if self.rmsd_matrix is None:
            return None
        
        status = st.empty()
        status.text("Performing cross-validation...")
        
        cv_scores = cross_validate_clustering(self.rmsd_matrix, k_range)
        
        status.empty()
        
        return {
            'scores': cv_scores,
            'best_k': max(cv_scores, key=cv_scores.get),
            'k_range': k_range
        }
    
    # Keep all original methods from the previous version
    def calculate_trajectory_metrics(self):
        """Calculate trajectory-based metrics"""
        n_models = len(self.coords)
        n_atoms = len(self.coords[0])
        
        # RMSF (Root Mean Square Fluctuation)
        mean_pos = np.mean(self.coords, axis=0)
        sq_fluct = np.array([(coord - mean_pos)**2 for coord in self.coords])
        rmsf = np.sqrt(np.mean(sq_fluct, axis=(0, 2)))
        
        # Radius of gyration for each model
        rg = []
        for coord in self.coords:
            center = coord.mean(axis=0)
            rg.append(np.sqrt(np.mean(np.sum((coord - center)**2, axis=1))))
        
        # Sequential RMSD
        seq_rmsd = []
        for i in range(1, n_models):
            rmsd = np.sqrt(np.mean(np.sum((self.coords[i] - self.coords[i-1])**2, axis=1)))
            seq_rmsd.append(rmsd)
        
        self.trajectory_analysis = {
            'rmsf': rmsf,
            'radius_of_gyration': rg,
            'sequential_rmsd': seq_rmsd
        }
        
        return self.trajectory_analysis


    def detect_outliers(self):
        """Detect outliers using multiple methods"""
        mean_rmsd = np.mean(self.rmsd_matrix, axis=0)
        
        # Method 1: IQR
        Q1, Q3 = np.percentile(mean_rmsd, [25, 75])
        IQR = Q3 - Q1
        outliers_iqr = np.where((mean_rmsd < Q1 - 1.5*IQR) | 
                               (mean_rmsd > Q3 + 1.5*IQR))[0]
        
        # Method 2: Z-score
        z_scores = np.abs(zscore(mean_rmsd))
        outliers_zscore = np.where(z_scores > 3)[0]
        
        # Method 3: Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = np.where(iso_forest.fit_predict(mean_rmsd.reshape(-1, 1)) == -1)[0]
        
        # Consensus outliers
        all_outliers = set(outliers_iqr) | set(outliers_zscore) | set(outliers_iso)
        consensus_outliers = set(outliers_iqr) & set(outliers_zscore) & set(outliers_iso)
        
        return {
            'iqr': outliers_iqr,
            'zscore': outliers_zscore,
            'isolation_forest': outliers_iso,
            'all': sorted(list(all_outliers)),
            'consensus': sorted(list(consensus_outliers)),
            'mean_rmsd': mean_rmsd
        }
    
    def optimize_clustering_advanced(self, min_k=2, max_k=6, method='kmeans'):
        """Advanced clustering with multiple validation metrics"""
        condensed_rmsd = squareform(self.rmsd_matrix)
        
        scores = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        all_labels = {}
        
        for k in range(min_k, max_k + 1):
            if method == 'kmeans':
                Z = linkage(condensed_rmsd, method='ward')
                labels = fcluster(Z, k, criterion='maxclust')
            elif method == 'hierarchical':
                Z = linkage(condensed_rmsd, method='ward')
                labels = fcluster(Z, k, criterion='maxclust')
            elif method == 'dbscan':
                eps = np.percentile(condensed_rmsd, 50)
                dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
                labels = dbscan.fit_predict(self.rmsd_matrix)
                k = len(set(labels)) - (1 if -1 in labels else 0)
            
            all_labels[k] = labels
            
            if len(set(labels)) > 1:
                scores['silhouette'].append(silhouette_score(
                    self.rmsd_matrix, labels, metric='precomputed'
                ))
                scores['davies_bouldin'].append(davies_bouldin_score(
                    self.rmsd_matrix, labels
                ))
                scores['calinski_harabasz'].append(calinski_harabasz_score(
                    self.rmsd_matrix, labels
                ))
            else:
                scores['silhouette'].append(0)
                scores['davies_bouldin'].append(float('inf'))
                scores['calinski_harabasz'].append(0)
        
        # Determine optimal k
        optimal_k_sil = min_k + np.argmax(scores['silhouette'])
        optimal_k_db = min_k + np.argmin(scores['davies_bouldin'])
        optimal_k_ch = min_k + np.argmax(scores['calinski_harabasz'])
        
        # Use silhouette as primary metric
        optimal_k = optimal_k_sil
        
        self.cluster_info = {
            'optimal_k': optimal_k,
            'labels': all_labels[optimal_k],
            'scores': scores,
            'all_labels': all_labels,
            'method': method,
            'k_range': list(range(min_k, max_k + 1))
        }
        
        return self.cluster_info
    
    def perform_pca(self):
        """Perform PCA on coordinates"""
        coords_flat = np.array([coord.flatten() for coord in self.coords])
        
        pca = PCA()
        pca_coords = pca.fit_transform(coords_flat)
        
        self.pca_results = {
            'transformed': pca_coords,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': pca.n_components_
        }
        
        return self.pca_results
    
    def identify_representatives(self, cluster_labels, outliers):
        """Identify representative structures for each cluster"""
        unique_clusters = np.unique(cluster_labels)
        representatives = {}
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_indices = [idx for idx in cluster_indices if idx not in outliers]
            
            if len(cluster_indices) == 0:
                continue
            
            cluster_rmsd = self.rmsd_matrix[cluster_indices][:, cluster_indices]
            mean_rmsd = np.mean(cluster_rmsd, axis=1)
            rep_idx = cluster_indices[np.argmin(mean_rmsd)]
            
            representatives[cluster_id] = rep_idx
        
        return representatives
    
    def export_structure(self, model_idx, output_format='pdb'):
        """Export a single model structure"""
        self.universe.trajectory[model_idx]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with mda.Writer(tmp_path, format='PDB') as W:
                W.write(self.full_chain_atoms)
            
            with open(tmp_path, 'r') as f:
                pdb_string = f.read()
            
            return pdb_string.encode('utf-8')
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def generate_pymol_script(self, hotspots_df=None, representatives=None):
        """Generate comprehensive PyMOL visualization script"""
        script = f"""# PyMOL Visualization Script
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Load your structure
load your_structure.pdb

# Basic visualization
hide everything
show cartoon
color gray80, all
set cartoon_transparency, 0.3

"""
        if representatives:
            script += "\n# Highlight representative structures\n"
            for cluster_id, rep_idx in representatives.items():
                color = ['red', 'blue', 'green', 'yellow', 'magenta'][cluster_id % 5]
                script += f"# Cluster {cluster_id} representative: Model {rep_idx + 1}\n"
        
        if hotspots_df is not None and not hotspots_df.empty:
            script += "\n# Highlight hotspot residues\n"
            for _, row in hotspots_df.head(10).iterrows():
                script += f"select hotspot_{int(row['residue'])}, resi {int(row['residue'])}\n"
                script += f"show spheres, hotspot_{int(row['residue'])}\n"
                script += f"color red, hotspot_{int(row['residue'])}\n"
        
        script += "\n# Final touches\nzoom\nray\n"
        
        return script



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_tm_score_heatmap(tm_scores):
    """Create TM-score heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=tm_scores,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title="TM-score")
    ))
    
    fig.update_layout(
        title="TM-Score Matrix (Structure Similarity)",
        xaxis_title="Model Index",
        yaxis_title="Model Index",
        height=500
    )
    
    return fig


def plot_secondary_structure_evolution(ss_data):
    """Plot secondary structure evolution"""
    if ss_data is None or 'evolution' not in ss_data:
        return None
    
    evolution = ss_data['evolution']
    changes = ss_data['changes']
    
    # Create matrix for visualization
    residues = sorted(changes.keys())
    n_models = len(evolution)
    
    # Map SS to numbers
    ss_map = {'-': 0, 'H': 1, 'G': 2, 'I': 3, 'E': 4, 'B': 5, 'T': 6, 'S': 7}
    
    matrix = np.zeros((len(residues), n_models))
    for i, res in enumerate(residues):
        for j, ss_dict in enumerate(evolution):
            ss_type = ss_dict.get(res, '-')
            matrix[i, j] = ss_map.get(ss_type, 0)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(n_models)),
        y=residues,
        colorscale='Viridis',
        colorbar=dict(
            title="SS Type",
            tickvals=[0, 1, 2, 3, 4, 5, 6, 7],
            ticktext=['Coil', 'α-Helix', '3-10 Helix', 'π-Helix', 
                     'β-Sheet', 'β-Bridge', 'Turn', 'Bend']
        )
    ))
    
    fig.update_layout(
        title="Secondary Structure Evolution Across Models",
        xaxis_title="Model Index",
        yaxis_title="Residue Number",
        height=600
    )
    
    return fig


def plot_pocket_analysis(binding_sites):
    """Visualize binding pocket analysis"""
    if not binding_sites or 'pockets_per_model' not in binding_sites:
        return None
    
    pockets = binding_sites['pockets_per_model']
    
    # Collect data
    model_indices = []
    volumes = []
    druggability = []
    
    for i, model_pockets in enumerate(pockets):
        for pocket in model_pockets:
            model_indices.append(i)
            volumes.append(pocket['volume'])
            druggability.append(pocket['druggability'])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pocket Volume Distribution", "Druggability Scores")
    )
    
    # Volume distribution
    fig.add_trace(
        go.Box(y=volumes, name="Volume (Å³)", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Druggability scores
    fig.add_trace(
        go.Box(y=druggability, name="Druggability", marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Binding Pocket Analysis"
    )
    
    return fig


def plot_distribution_fit(stat_analysis):
    """Plot RMSD distribution with fitted curves"""
    if not stat_analysis or 'rmsd_distribution' not in stat_analysis:
        return None
    
    dist = stat_analysis['rmsd_distribution']
    
    # Generate data for fitted distribution
    x_range = np.linspace(dist['min'], dist['max'], 100)
    
    fig = go.Figure()
    
    # KDE fit if available
    if 'kde_fit' in dist and dist['kde_fit']:
        kde = dist['kde_fit']['kde']
        y_kde = kde(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_kde,
            mode='lines',
            name='KDE Fit',
            line=dict(color='red', width=2)
        ))
    
    # Normal fit if available
    if 'normal_fit' in dist and dist['normal_fit']:
        mu = dist['normal_fit']['mu']
        std = dist['normal_fit']['std']
        y_normal = norm.pdf(x_range, mu, std)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_normal,
            mode='lines',
            name='Normal Fit',
            line=dict(color='blue', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"RMSD Distribution (μ={dist['mean']:.2f}, σ={dist['std']:.2f})",
        xaxis_title="RMSD (Å)",
        yaxis_title="Density",
        height=400
    )
    
    return fig


def plot_correlation_matrix(stat_analysis):
    """Plot correlation matrix"""
    if not stat_analysis or 'correlations' not in stat_analysis:
        return None
    
    corr = stat_analysis['correlations']
    
    if not corr:
        return None
    
    # Create correlation summary
    data = []
    for key, value in corr.items():
        if 'pearson' in value:
            data.append({
                'Comparison': key.replace('_', ' ').title(),
                'Pearson r': value['pearson']['r'],
                'Pearson p': value['pearson']['p'],
                'Spearman ρ': value['spearman']['r'],
                'Spearman p': value['spearman']['p']
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            format=[None, '.3f', '.3e', '.3f', '.3e']
        )
    )])
    
    fig.update_layout(
        title="Correlation Analysis",
        height=200 + len(data) * 30
    )
    
    return fig


def plot_bootstrap_stability(bootstrap_results):
    """Plot bootstrap stability matrix"""
    if not bootstrap_results or 'stability_matrix' not in bootstrap_results:
        return None
    
    stability = bootstrap_results['stability_matrix']
    
    fig = go.Figure(data=go.Heatmap(
        z=stability,
        colorscale='Greens',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Co-clustering<br>Probability")
    ))
    
    fig.update_layout(
        title=f"Bootstrap Stability Matrix (n={bootstrap_results['n_bootstrap']})",
        xaxis_title="Model Index",
        yaxis_title="Model Index",
        height=500
    )
    
    return fig


def plot_cross_validation(cv_results):
    """Plot cross-validation scores"""
    if not cv_results or 'scores' not in cv_results:
        return None
    
    scores = cv_results['scores']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(scores.keys()),
        y=list(scores.values()),
        mode='lines+markers',
        name='Silhouette Score',
        marker=dict(size=10, color='blue'),
        line=dict(width=2)
    ))
    
    # Mark best k
    best_k = cv_results['best_k']
    best_score = scores[best_k]
    
    fig.add_trace(go.Scatter(
        x=[best_k],
        y=[best_score],
        mode='markers',
        name=f'Best k={best_k}',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title="Cross-Validation: Optimal Cluster Number",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Average Silhouette Score",
        height=400
    )
    
    return fig




# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🧬 PDB Conformational Analysis Suite Pro+</h1>
        <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>
            Advanced Multi-Model Protein Structure Analysis with TM-Score, Secondary Structure, 
            Ligand Binding, and Statistical Validation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDB File",
            type=['pdb'],
            help="Multi-model PDB file"
        )
        
        # Optional experimental data upload
        st.subheader("📊 Experimental Data (Optional)")
        exp_file = st.file_uploader(
            "Upload Experimental Data",
            type=['csv'],
            help="B-factors or NMR order parameters"
        )
        
        if exp_file:
            exp_type = st.selectbox(
                "Data Type",
                ['bfactor', 'nmr'],
                help="Type of experimental data"
            )
        
        st.divider()
        
        # Selection parameters
        st.subheader("Selection Parameters")
        selection = st.selectbox(
            "Atom Selection",
            ['backbone', 'name CA', 'protein', 'name CA or name C or name N'],
            help="Atoms to use for RMSD calculation"
        )
        
        segid = st.text_input("Chain/Segment ID", value="A")
        ref_frame = st.number_input("Reference Frame", min_value=0, value=0)
        
        st.divider()
        
        # Clustering configuration
        st.subheader("Clustering Method")
        clustering_method = st.selectbox(
            "Algorithm",
            ['kmeans', 'hierarchical', 'dbscan'],
            help="Choose clustering algorithm"
        )
        
        col1, col2 = st.columns(2)
        min_k = col1.number_input("Min Clusters", min_value=2, value=2)
        max_k = col2.number_input("Max Clusters", min_value=2, value=6)
        
        st.divider()
        
        # Analysis modules
        st.subheader("🔬 Analysis Modules")
        
        with st.expander("Core Analyses", expanded=True):
            modules_core = {
                'rmsd': st.checkbox("RMSD Matrix", value=True, disabled=True),
                'tm_score': st.checkbox("TM-Score Calculation", value=True),
                'outliers': st.checkbox("Outlier Detection", value=True, disabled=True),
                'clustering': st.checkbox("Clustering Analysis", value=True, disabled=True),
            }
        
        with st.expander("Structural Features", expanded=True):
            modules_struct = {
                'secondary': st.checkbox("Secondary Structure Evolution", value=True),
                'pockets': st.checkbox("Binding Pocket Detection", value=True),
                'trajectory': st.checkbox("Trajectory Metrics", value=True),
                'pca': st.checkbox("PCA Analysis", value=True),
            }
        
        with st.expander("Statistical Validation", expanded=True):
            modules_stats = {
                'statistical': st.checkbox("Distribution Analysis", value=True),
                'correlations': st.checkbox("Correlation Analysis", value=True),
                'bootstrap': st.checkbox("Bootstrap Stability", value=False),
                'crossval': st.checkbox("Cross-Validation", value=True),
            }
        
        # Combine all modules
        modules = {**modules_core, **modules_struct, **modules_stats}
        
        st.divider()
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", width="stretch")
        
        if st.session_state.analysis_complete:
            if st.button("🔄 Reset Analysis", width="stretch"):
                st.session_state.analyzer = None
                st.session_state.analysis_complete = False
                st.rerun()
    
    # Main content
    if not uploaded_file:
        display_welcome_screen()
        return
    
    # Run analysis
    if run_analysis:
        run_complete_analysis(
            uploaded_file, selection, segid, ref_frame,
            clustering_method, min_k, max_k,
            modules, exp_file if 'exp_file' in locals() else None,
            exp_type if 'exp_type' in locals() else None
        )
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analyzer:
        display_results(st.session_state.analyzer, modules)


def display_welcome_screen():
    """Display welcome screen with feature overview"""
    st.info("👈 Please upload a PDB file to begin analysis")
    
    # Feature showcase
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clustering Methods", "3", help="K-means, Hierarchical, DBSCAN")
    with col2:
        st.metric("Analysis Modules", "14+", help="Comprehensive toolkit")
    with col3:
        st.metric("New Features", "8", help="TM-score, SS evolution, pockets, stats")
    with col4:
        st.metric("Visualization", "Enhanced", help="Interactive viewers & plots")
    
    st.divider()
    
    # Feature tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Core Features", 
        "🔬 New Advanced Features", 
        "📊 Statistical Validation",
        "💡 Best Practices"
    ])
    
    with tab1:
        st.markdown("""
        ### Core Analysis Pipeline
        
        ✅ **RMSD Matrix Calculation** - Pairwise structural comparison
        ✅ **Multi-Method Outlier Detection** - IQR, Z-score, Isolation Forest
        ✅ **Optimal Clustering** - Automatic K selection with validation
        ✅ **Representative Selection** - Multiple selection strategies
        ✅ **Trajectory Analysis** - Time-series conformational changes
        ✅ **PCA Analysis** - Dimensionality reduction and visualization
        """)
    
    with tab2:
        st.markdown("""
        ### 🆕 New Advanced Features
        
        🎯 **TM-Score Calculation**
        - Template Modeling score for better fold similarity assessment
        - Scale-independent metric (0-1 range)
        - More reliable than RMSD for distant structures
        
        🧬 **Secondary Structure Evolution**
        - Track helix/sheet/coil changes across models
        - Identify flexible vs. rigid structural elements
        - DSSP-based assignment
        
        💊 **Protein-Ligand Interaction Analysis**
        - Binding pocket detection and characterization
        - Pocket volume calculation across conformers
        - Druggability assessment scores
        - Preparation for docking studies
        
        📈 **Enhanced Trajectory Metrics**
        - Per-residue flexibility (RMSF)
        - Radius of gyration tracking
        - Sequential conformational changes
        """)
    
    with tab3:
        st.markdown("""
        ### 📊 Statistical Validation Suite
        
        📉 **Distribution Analysis**
        - Fit normal distributions to RMSD data
        - Kernel density estimation (KDE)
        - Statistical summaries and outlier detection
        
        🔗 **Correlation Analysis**
        - Pearson and Spearman correlations
        - RMSD vs RMSF relationships
        - Comparison with experimental data
        
        🎲 **Bootstrap Analysis**
        - Clustering stability assessment
        - Confidence intervals for cluster assignments
        - Robustness evaluation (100+ resamples)
        
        ✅ **Cross-Validation**
        - Optimal cluster number validation
        - K-fold cross-validation scores
        - Model selection support
        
        🔬 **Experimental Data Comparison**
        - Import B-factors from experiments
        - Compare with calculated flexibility
        - NMR order parameter correlation
        """)
    
    with tab4:
        st.markdown("""
        ### 💡 Best Practices
        
        **For Best Results:**
        
        1. **Pre-processing**: Ensure your PDB file is clean
           - Remove water molecules if not needed
           - Check for missing residues
           - Verify chain identifiers
        
        2. **Atom Selection**: Choose appropriate atoms
           - `name CA`: Fast, backbone only
           - `backbone`: More comprehensive
           - `protein`: Full protein (slower)
        
        3. **Clustering**: Start with k=2-6 range
           - Use cross-validation to find optimal k
           - Try different methods (k-means, hierarchical)
        
        4. **Statistical Validation**: Enable for publication
           - Bootstrap for confidence intervals
           - Cross-validation for robustness
           - Correlation with experimental data
        
        5. **Computational Cost**: Be aware of scaling
           - TM-score: O(n²) - can be slow for many models
           - Secondary structure: Requires DSSP installation
           - Bootstrap: Time-intensive (100 iterations)
        """)


def run_complete_analysis(uploaded_file, selection, segid, ref_frame,
                         clustering_method, min_k, max_k, modules,
                         exp_file=None, exp_type=None):
    """Run complete analysis pipeline"""
    analyzer = EnhancedPDBAnalyzerPro()
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load structure
        status_text.text("📂 Loading PDB structure...")
        progress_bar.progress(5)
        n_models, n_atoms = analyzer.load_from_upload(uploaded_file, selection, segid)
        st.success(f"✅ Loaded {n_models} models with {n_atoms} atoms")
        
        # Load experimental data if provided
        if exp_file and exp_type:
            status_text.text("📊 Loading experimental data...")
            analyzer.load_experimental_data(exp_file, exp_type)
        
        # Step 2: Align structures
        status_text.text("🔄 Aligning structures...")
        progress_bar.progress(10)
        analyzer.align_structures(ref_frame)
        
        # Step 3: RMSD matrix
        status_text.text("📊 Computing RMSD matrix...")
        progress_bar.progress(15)
        analyzer.calculate_rmsd_matrix()
        
        # Step 4: TM-scores (if enabled)
        if modules.get('tm_score', False):
            status_text.text("🎯 Calculating TM-scores...")
            progress_bar.progress(20)
            analyzer.calculate_tm_scores()
        
        # Step 5: Trajectory metrics
        if modules.get('trajectory', False):
            status_text.text("📈 Analyzing trajectory metrics...")
            progress_bar.progress(30)
            analyzer.calculate_trajectory_metrics()
        
        # Step 6: Secondary structure
        if modules.get('secondary', False):
            status_text.text("🧬 Analyzing secondary structure evolution...")
            progress_bar.progress(40)
            analyzer.analyze_secondary_structure_evolution()
        
        # Step 7: Outlier detection
        status_text.text("🔍 Detecting outliers...")
        progress_bar.progress(50)
        outliers = analyzer.detect_outliers()
        
        # Step 8: Clustering
        status_text.text(f"🎯 Running {clustering_method} clustering...")
        progress_bar.progress(55)
        cluster_info = analyzer.optimize_clustering_advanced(min_k, max_k, clustering_method)
        
        # Step 9: PCA
        if modules.get('pca', False):
            status_text.text("🧬 Performing PCA...")
            progress_bar.progress(65)
            analyzer.perform_pca()
        
        # Step 10: Binding pockets
        if modules.get('pockets', False):
            status_text.text("💊 Detecting binding pockets...")
            progress_bar.progress(70)
            analyzer.analyze_binding_pockets()
        
        # Step 11: Statistical analysis
        if modules.get('statistical', False):
            status_text.text("📊 Performing statistical analysis...")
            progress_bar.progress(75)
            analyzer.perform_statistical_analysis()
        
        # Step 12: Bootstrap
        if modules.get('bootstrap', False):
            status_text.text("🎲 Running bootstrap analysis...")
            progress_bar.progress(80)
            analyzer.bootstrap_results = analyzer.bootstrap_analysis(
                n_bootstrap=100, k=cluster_info['optimal_k']
            )
        
        # Step 13: Cross-validation
        if modules.get('crossval', False):
            status_text.text("✅ Performing cross-validation...")
            progress_bar.progress(90)
            analyzer.cv_results = analyzer.cross_validation_analysis(
                k_range=list(range(min_k, max_k + 1))
            )
        
        # Step 14: Finalize
        status_text.text("✨ Finalizing analysis...")
        progress_bar.progress(95)
        
        representatives = analyzer.identify_representatives(
            cluster_info['labels'],
            outliers['consensus']
        )
        analyzer.representatives = representatives
        
        # Complete
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success("🎉 Analysis Complete!")
        
        # Store in session state
        st.session_state.analyzer = analyzer
        st.session_state.analysis_complete = True
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)




def display_results(analyzer, modules):
    """Display comprehensive analysis results"""
    
    st.header("📊 Analysis Results")
    
    # Create main tabs
    tabs = st.tabs([
        "📈 Overview",
        "🎯 Clustering",
        "🆕 TM-Score",
        "🧬 Secondary Structure",
        "💊 Binding Pockets",
        "📊 Statistics",
        "✅ Validation",
        "💾 Export"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        display_overview(analyzer)
    
    # Tab 2: Clustering
    with tabs[1]:
        display_clustering_results(analyzer)
    
    # Tab 3: TM-Score
    with tabs[2]:
        if modules.get('tm_score') and hasattr(analyzer, 'tm_scores'):
            display_tm_score_results(analyzer)
        else:
            st.info("TM-score analysis was not enabled.")
    
    # Tab 4: Secondary Structure
    with tabs[3]:
        if modules.get('secondary') and analyzer.secondary_structure:
            display_secondary_structure_results(analyzer)
        else:
            st.info("Secondary structure analysis was not enabled or failed.")
    
    # Tab 5: Binding Pockets
    with tabs[4]:
        if modules.get('pockets') and analyzer.binding_sites:
            display_binding_pocket_results(analyzer)
        else:
            st.info("Binding pocket analysis was not enabled.")
    
    # Tab 6: Statistics
    with tabs[5]:
        if modules.get('statistical') and analyzer.statistical_analysis:
            display_statistical_results(analyzer)
        else:
            st.info("Statistical analysis was not enabled.")
    
    # Tab 7: Validation
    with tabs[6]:
        display_validation_results(analyzer, modules)
    
    # Tab 8: Export
    with tabs[7]:
        display_export_options(analyzer)


def display_overview(analyzer):
    """Display overview metrics"""
    st.subheader("Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    n_models = len(analyzer.coords)
    n_atoms = len(analyzer.coords[0])
    
    with col1:
        st.metric("Models", n_models)
    
    with col2:
        st.metric("Atoms Selected", n_atoms)
    
    with col3:
        if hasattr(analyzer, 'cluster_info'):
            st.metric("Optimal Clusters", analyzer.cluster_info['optimal_k'])
        else:
            st.metric("Optimal Clusters", "N/A")
    
    with col4:
        if analyzer.rmsd_matrix is not None:
            mean_rmsd = np.mean(analyzer.rmsd_matrix)
            st.metric("Mean RMSD", f"{mean_rmsd:.2f} Å")
        else:
            st.metric("Mean RMSD", "N/A")
    
    st.divider()
    
    # RMSD heatmap
    if analyzer.rmsd_matrix is not None:
        st.subheader("RMSD Matrix")
        
        fig = go.Figure(data=go.Heatmap(
            z=analyzer.rmsd_matrix,
            colorscale='Viridis',
            colorbar=dict(title="RMSD (Å)")
        ))
        
        fig.update_layout(
            title="Pairwise RMSD Matrix",
            xaxis_title="Model Index",
            yaxis_title="Model Index",
            height=500
        )
        
        st.plotly_chart(fig, width="stretch")
    
    # Trajectory metrics
    if hasattr(analyzer, 'trajectory_analysis') and analyzer.trajectory_analysis:
        st.subheader("Trajectory Metrics")
        
        traj = analyzer.trajectory_analysis
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSF plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=traj['rmsf'],
                mode='lines',
                name='RMSF',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="Root Mean Square Fluctuation (RMSF)",
                xaxis_title="Atom Index",
                yaxis_title="RMSF (Å)",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Radius of gyration
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=traj['radius_of_gyration'],
                mode='lines+markers',
                name='Rg',
                line=dict(color='green')
            ))
            fig.update_layout(
                title="Radius of Gyration",
                xaxis_title="Model Index",
                yaxis_title="Rg (Å)",
                height=400
            )
            st.plotly_chart(fig, width="stretch")


def display_clustering_results(analyzer):
    """Display clustering results"""
    if not hasattr(analyzer, 'cluster_info'):
        st.warning("Clustering analysis not available.")
        return
    
    cluster_info = analyzer.cluster_info
    
    st.subheader("Clustering Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal K", cluster_info['optimal_k'])
    with col2:
        st.metric("Method", cluster_info['method'].title())
    with col3:
        unique_clusters = len(np.unique(cluster_info['labels']))
        st.metric("Clusters Found", unique_clusters)
    
    # Clustering quality metrics
    st.subheader("Clustering Quality Metrics")
    
    scores = cluster_info['scores']
    k_range = cluster_info['k_range']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score")
    )
    
    fig.add_trace(
        go.Scatter(x=k_range, y=scores['silhouette'], mode='lines+markers', name='Silhouette'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=k_range, y=scores['davies_bouldin'], mode='lines+markers', name='Davies-Bouldin'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=k_range, y=scores['calinski_harabasz'], mode='lines+markers', name='Calinski-Harabasz'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    
    # PCA visualization with clusters
    if hasattr(analyzer, 'pca_results'):
        st.subheader("PCA Visualization with Clusters")
        
        pca = analyzer.pca_results
        
        fig = go.Figure()
        
        for cluster_id in np.unique(cluster_info['labels']):
            mask = cluster_info['labels'] == cluster_id
            fig.add_trace(go.Scatter(
                x=pca['transformed'][mask, 0],
                y=pca['transformed'][mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=10)
            ))
        
        # Mark representatives
        if hasattr(analyzer, 'representatives'):
            for cluster_id, rep_idx in analyzer.representatives.items():
                fig.add_trace(go.Scatter(
                    x=[pca['transformed'][rep_idx, 0]],
                    y=[pca['transformed'][rep_idx, 1]],
                    mode='markers',
                    name=f'Rep {cluster_id}',
                    marker=dict(size=15, symbol='star', line=dict(width=2, color='black'))
                ))
        
        fig.update_layout(
            title="PCA Space with Cluster Assignments",
            xaxis_title=f"PC1 ({pca['explained_variance'][0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca['explained_variance'][1]*100:.1f}%)",
            height=600
        )
        
        st.plotly_chart(fig, width="stretch")


def display_tm_score_results(analyzer):
    """Display TM-score analysis results"""
    st.subheader("TM-Score Analysis")
    
    st.markdown("""
    **TM-score** (Template Modeling score) is a scale-independent metric for measuring 
    structural similarity. Values range from 0 to 1:
    - **< 0.17**: Random similarity
    - **0.17-0.5**: Partial similarity
    - **> 0.5**: Similar fold
    - **> 0.8**: Highly similar structures
    """)
    
    tm_scores = analyzer.tm_scores
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate statistics
    upper_tri = tm_scores[np.triu_indices_from(tm_scores, k=1)]
    
    with col1:
        st.metric("Mean TM-score", f"{np.mean(upper_tri):.3f}")
    with col2:
        st.metric("Min TM-score", f"{np.min(upper_tri):.3f}")
    with col3:
        st.metric("Max TM-score", f"{np.max(upper_tri):.3f}")
    
    # TM-score heatmap
    fig = plot_tm_score_heatmap(tm_scores)
    st.plotly_chart(fig, width="stretch")
    
    # TM-score vs RMSD comparison
    if analyzer.rmsd_matrix is not None:
        st.subheader("TM-Score vs RMSD Comparison")
        
        rmsd_upper = analyzer.rmsd_matrix[np.triu_indices_from(analyzer.rmsd_matrix, k=1)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rmsd_upper,
            y=upper_tri,
            mode='markers',
            marker=dict(size=8, color=upper_tri, colorscale='Viridis', 
                       colorbar=dict(title="TM-score"))
        ))
        
        fig.update_layout(
            title="TM-Score vs RMSD Relationship",
            xaxis_title="RMSD (Å)",
            yaxis_title="TM-Score",
            height=500
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Calculate correlation
        corr, p_value = pearsonr(rmsd_upper, upper_tri)
        st.info(f"📊 Correlation: r = {corr:.3f}, p = {p_value:.3e}")


def display_secondary_structure_results(analyzer):
    """Display secondary structure analysis"""
    st.subheader("Secondary Structure Evolution")
    
    ss_data = analyzer.secondary_structure
    summary = ss_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Residues", summary['n_residues'])
    with col2:
        st.metric("Variable Residues", summary['n_variable'])
    with col3:
        st.metric("Helix Content", f"{summary['n_helix']}")
    with col4:
        st.metric("Sheet Content", f"{summary['n_sheet']}")
    
    # Evolution heatmap
    fig = plot_secondary_structure_evolution(ss_data)
    if fig:
        st.plotly_chart(fig, width="stretch")
    
    # Most variable residues
    st.subheader("Most Variable Residues")
    
    changes = ss_data['changes']
    variable_residues = sorted(
        [(res, data) for res, data in changes.items()],
        key=lambda x: x[1]['n_changes'],
        reverse=True
    )[:20]
    
    if variable_residues:
        var_df = pd.DataFrame([
            {
                'Residue': res,
                'Changes': data['n_changes'],
                'Most Common': data['most_common']
            }
            for res, data in variable_residues
        ])
        
        st.dataframe(var_df, width=600)


def display_binding_pocket_results(analyzer):
    """Display binding pocket analysis"""
    st.subheader("Binding Pocket Analysis")
    
    binding_sites = analyzer.binding_sites
    summary = binding_sites['summary']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pockets Detected", summary['n_pockets_total'])
    with col2:
        st.metric("Mean Volume", f"{summary['mean_volume']:.1f} Å³")
    with col3:
        st.metric("Mean Druggability", f"{summary['mean_druggability']:.2f}")
    
    # Pocket visualization
    fig = plot_pocket_analysis(binding_sites)
    if fig:
        st.plotly_chart(fig, width="stretch")
    
    # Detailed pocket information
    st.subheader("Pocket Details by Model")
    
    pockets_per_model = binding_sites['pockets_per_model']
    
    model_select = st.selectbox(
        "Select Model",
        range(len(pockets_per_model)),
        format_func=lambda x: f"Model {x+1}"
    )
    
    if model_select < len(pockets_per_model):
        pockets = pockets_per_model[model_select]
        
        if pockets:
            pocket_df = pd.DataFrame([
                {
                    'Pocket': i+1,
                    'Volume (Å³)': p['volume'],
                    'Druggability': f"{p['druggability']:.2f}",
                    'Residue': p['residue'],
                    'Atoms': p['n_atoms']
                }
                for i, p in enumerate(pockets)
            ])
            
            st.dataframe(pocket_df, width=800)
        else:
            st.info("No pockets detected in this model.")


def display_statistical_results(analyzer):
    """Display statistical analysis results"""
    st.subheader("Statistical Analysis")
    
    stat_analysis = analyzer.statistical_analysis
    
    # Distribution analysis
    st.subheader("RMSD Distribution")
    
    dist = stat_analysis['rmsd_distribution']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{dist['mean']:.3f} Å")
    with col2:
        st.metric("Std Dev", f"{dist['std']:.3f} Å")
    with col3:
        st.metric("Median", f"{dist['median']:.3f} Å")
    with col4:
        st.metric("Range", f"{dist['max']-dist['min']:.3f} Å")
    
    # Distribution fit plot
    fig = plot_distribution_fit(stat_analysis)
    if fig:
        st.plotly_chart(fig, width="stretch")
    
    # Correlations
    if stat_analysis['correlations']:
        st.subheader("Correlation Analysis")
        fig = plot_correlation_matrix(stat_analysis)
        if fig:
            st.plotly_chart(fig, width="stretch")
    
    # Experimental data comparison
    if stat_analysis['experimental_comparison']:
        st.subheader("Experimental Data Comparison")
        
        exp_comp = stat_analysis['experimental_comparison']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Type", exp_comp['data_type'].upper())
        with col2:
            st.metric("Matched Residues", exp_comp['n_matched'])
        with col3:
            st.metric("Pearson r", f"{exp_comp['pearson']['r']:.3f}")
        
        st.info(f"📊 Pearson correlation: r = {exp_comp['pearson']['r']:.3f}, "
               f"p = {exp_comp['pearson']['p']:.3e}")
        st.info(f"📊 Spearman correlation: ρ = {exp_comp['spearman']['r']:.3f}, "
               f"p = {exp_comp['spearman']['p']:.3e}")


def display_validation_results(analyzer, modules):
    """Display validation results"""
    st.subheader("Clustering Validation")
    
    # Bootstrap analysis
    if modules.get('bootstrap') and hasattr(analyzer, 'bootstrap_results'):
        st.subheader("Bootstrap Stability Analysis")
        
        boot = analyzer.bootstrap_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bootstrap Iterations", boot['n_bootstrap'])
        with col2:
            st.metric("Mean Stability", f"{boot['mean_stability']:.3f}")
        
        fig = plot_bootstrap_stability(boot)
        if fig:
            st.plotly_chart(fig, width="stretch")
        
        st.markdown("""
        **Interpretation**: Values close to 1.0 indicate structures consistently 
        cluster together, suggesting stable cluster assignments.
        """)
    
    # Cross-validation
    if modules.get('crossval') and hasattr(analyzer, 'cv_results'):
        st.subheader("Cross-Validation Results")
        
        cv = analyzer.cv_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best K (CV)", cv['best_k'])
        with col2:
            st.metric("Best Score", f"{cv['scores'][cv['best_k']]:.3f}")
        
        fig = plot_cross_validation(cv)
        if fig:
            st.plotly_chart(fig, width="stretch")


def display_export_options(analyzer):
    """Display export options"""
    st.subheader("Export Options")
    
    st.markdown("""
    Download your analysis results in various formats for further analysis or publication.
    """)
    
    # Export representative structures
    if hasattr(analyzer, 'representatives'):
        st.subheader("📦 Representative Structures")
        
        reps = analyzer.representatives
        
        for cluster_id, rep_idx in reps.items():
            pdb_data = analyzer.export_structure(rep_idx)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text(f"Cluster {cluster_id} - Model {rep_idx + 1}")
            
            with col2:
                st.download_button(
                    "Download PDB",
                    pdb_data,
                    f"cluster_{cluster_id}_rep_model_{rep_idx+1}.pdb",
                    "chemical/x-pdb",
                    key=f"pdb_export_{cluster_id}"
                )
    
    # Export data tables
    st.subheader("📊 Data Tables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if analyzer.rmsd_matrix is not None:
            rmsd_df = pd.DataFrame(analyzer.rmsd_matrix)
            csv = rmsd_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download RMSD Matrix (CSV)",
                csv,
                "rmsd_matrix.csv",
                "text/csv"
            )
    
    with col2:
        if hasattr(analyzer, 'tm_scores'):
            tm_df = pd.DataFrame(analyzer.tm_scores)
            csv = tm_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download TM-Score Matrix (CSV)",
                csv,
                "tm_score_matrix.csv",
                "text/csv"
            )
    
    # PyMOL script
    st.subheader("🎨 PyMOL Visualization Script")
    
    pymol_script = analyzer.generate_pymol_script(
        representatives=analyzer.representatives if hasattr(analyzer, 'representatives') else None
    )
    
    st.download_button(
        "📥 Download PyMOL Script",
        pymol_script,
        "visualization.pml",
        "text/plain"
    )
    
    # Complete results JSON
    st.subheader("💾 Complete Analysis Results")
    
    results_dict = {
        'n_models': len(analyzer.coords),
        'n_atoms': len(analyzer.coords[0]),
        'selection': analyzer.selection_str,
        'mean_rmsd': float(np.mean(analyzer.rmsd_matrix)),
    }
    
    if hasattr(analyzer, 'cluster_info'):
        results_dict['optimal_k'] = int(analyzer.cluster_info['optimal_k'])
        results_dict['clustering_method'] = analyzer.cluster_info['method']
    
    if hasattr(analyzer, 'tm_scores'):
        tm_upper = analyzer.tm_scores[np.triu_indices_from(analyzer.tm_scores, k=1)]
        results_dict['mean_tm_score'] = float(np.mean(tm_upper))
    
    json_str = json.dumps(results_dict, indent=2)
    
    st.download_button(
        "📥 Download Analysis Summary (JSON)",
        json_str,
        "analysis_summary.json",
        "application/json"
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
