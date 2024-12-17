# To run this code please ensure all the files(both the data and the code) are in the same directory.
# Ensure to run this in the terminal to install all the packages required: 
# pip install numpy pandas biopython scipy scikit-learn matplotlib typing-extensions
# It takes 5-8 minutes to produce the results in the output directory called results. 
# Import required libraries for scientific computing, data analysis, and visualization
import os                                # Operating system operations (file/directory management)
import logging                           # Logging events and errors during analysis
import numpy as np                       # Numerical computing and array operations
import pandas as pd                      # Data manipulation and CSV export
from Bio import SeqIO                    # Biological sequence file parsing
from scipy.fft import fft                # Fast Fourier Transform for spectral analysis
from scipy.spatial.distance import pdist, squareform  # Distance matrix computations
from scipy.stats import pearsonr         # Statistical correlation calculations
from sklearn.manifold import MDS         # Multidimensional scaling for visualization
import matplotlib.pyplot as plt          # Plotting and visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting capabilities
from typing import List, Dict, Tuple, Union  # Type hinting for code clarity
import scipy.cluster.hierarchy as sch    # Hierarchical clustering
from sklearn.decomposition import PCA    # Principal Component Analysis
from matplotlib.colors import LinearSegmentedColormap  # Custom color mapping

class ViralSequenceAnalyzer:
    """
    Advanced class for analyzing viral sequences using signal processing and machine learning techniques.
    
    This class provides methods to:
    1. Convert DNA sequences to numerical representations
    2. Apply Fourier Transform for spectral analysis
    3. Calculate genetic distances between sequences
    4. Visualize sequence relationships through various dimensionality reduction techniques
    """
    
    def __init__(self, numeric_mapping: str = 'electron_ion'):
        """
        Initialize the ViralSequenceAnalyzer with a specific numeric mapping strategy.
        
        Different mapping strategies allow for various interpretations of DNA base numerical representation:
        - 'electron_ion': Maps bases to charge-like values
        - 'atomic': Maps bases to atomic values
        - 'molecular_mass': Maps bases to molecular mass values
        
        Args:
            numeric_mapping (str): Type of numerical mapping to use for DNA sequences
        
        Raises:
            ValueError: If an invalid numeric mapping is provided
        """
        # Define multiple numerical mapping strategies for DNA bases
        self.numeric_mappings = {
            # Charge-like representation (positive/negative/neutral)
            'electron_ion': {'A': 1.0, 'T': -1.0, 'C': 0.0, 'G': 0.0},
            
            # Atomic value representation
            'atomic': {'A': 70.0, 'T': 66.0, 'C': 58.0, 'G': 78.0},
            
            # Molecular mass representation
            'molecular_mass': {'A': 135.128, 'T': 126.115, 'C': 111.103, 'G': 151.128}
        }
        
        # Validate the chosen numeric mapping
        if numeric_mapping not in self.numeric_mappings:
            raise ValueError(f"Invalid numeric mapping. Choose from {list(self.numeric_mappings.keys())}")
        
        # Set the selected mapping
        self.mapping = self.numeric_mappings[numeric_mapping]
    
    def sequence_to_numerical(self, sequence: str) -> np.ndarray:
        """
        Convert a DNA sequence to a numerical representation based on the predefined mapping.
        
        Transforms each DNA base to its corresponding numerical value:
        - Uses the predefined mapping strategy
        - Converts sequence to uppercase
        - Handles unknown bases by assigning 0.0
        
        Args:
            sequence (str): Input DNA sequence to convert
        
        Returns:
            np.ndarray: Numerical representation of the DNA sequence
        
        Handles potential conversion errors by logging and returning an empty array
        """
        try:
            # Convert each base to its numerical value, defaulting to 0.0 for unknown bases
            return np.array([self.mapping.get(base, 0.0) for base in sequence.upper()])
        except Exception as e:
            # Log any errors during conversion
            logging.error(f"Error converting sequence to numerical: {e}")
            return np.array([])
    
    def pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate a numerical sequence to a specified target length.
        
        Ensures consistent sequence length for further analysis:
        - If sequence is longer than target, truncate
        - If sequence is shorter than target, pad with zeros
        
        Args:
            sequence (np.ndarray): Numerical sequence to adjust
            target_length (int): Desired length of the sequence
        
        Returns:
            np.ndarray: Padded or truncated sequence
        """
        current_length = len(sequence)
        if current_length >= target_length:
            # Truncate sequence if longer than target length
            return sequence[:target_length]
        
        # Pad sequence with zeros to reach target length
        return np.pad(sequence, (0, target_length - current_length), mode='constant')
    
    def apply_dft(self, numerical_sequence: np.ndarray, target_length: int = None) -> np.ndarray:
        """
        Apply Discrete Fourier Transform (DFT) to a numerical sequence.
        
        Converts the sequence to its frequency domain representation:
        - Pads sequence to target length if specified
        - Computes magnitude spectrum
        - Normalizes the spectrum
        
        Args:
            numerical_sequence (np.ndarray): Numerical representation of sequence
            target_length (int, optional): Length to pad sequence before DFT
        
        Returns:
            np.ndarray: Normalized magnitude spectrum of the sequence
        """
        # Check for empty sequence
        if numerical_sequence is None or len(numerical_sequence) == 0:
            logging.warning("Empty sequence provided to apply_dft")
            return np.array([])
        
        # Pad sequence to target length if specified
        if target_length is not None:
            numerical_sequence = self.pad_sequence(numerical_sequence, target_length)
        
        # Compute Fourier Transform
        dft_result = fft(numerical_sequence)
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(dft_result)
        n = len(magnitude_spectrum)
        
        # Return normalized first half of magnitude spectrum
        return magnitude_spectrum[:n//2 + 1] / n
    
    def calculate_pairwise_distances(self, spectra: List[np.ndarray], metric: str = 'pearson') -> np.ndarray:
        """
        Calculate pairwise distances between magnitude spectra of sequences.
        
        Computes a symmetric distance matrix using various distance metrics:
        - Pearson correlation-based distance
        - Euclidean distance
        - Manhattan distance
        
        Args:
            spectra (List[np.ndarray]): List of magnitude spectra
            metric (str, optional): Distance calculation method
        
        Returns:
            np.ndarray: Symmetric distance matrix
        """
        n_sequences = len(spectra)
        distances = np.zeros((n_sequences, n_sequences))
        
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                try:
                    # Ensure consistent length by taking minimum length
                    min_length = min(len(spectra[i]), len(spectra[j]))
                    spec1 = spectra[i][:min_length]
                    spec2 = spectra[j][:min_length]
                    
                    # Calculate distance based on chosen metric
                    if metric == 'pearson':
                        # 1 - absolute correlation provides a distance measure
                        corr, _ = pearsonr(spec1, spec2)
                        distance = 1 - abs(corr)
                    elif metric == 'euclidean':
                        # Euclidean distance between spectral vectors
                        distance = np.linalg.norm(spec1 - spec2)
                    elif metric == 'manhattan':
                        # Manhattan (city block) distance
                        distance = np.sum(np.abs(spec1 - spec2))
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                    
                    # Populate symmetric distance matrix
                    distances[i, j] = distance
                    distances[j, i] = distance
                except Exception as e:
                    logging.error(f"Error calculating distance between spectra {i} and {j}: {e}")
        
        return distances
    
    def create_momap3d(self, distance_matrix: np.ndarray, labels: List[str]) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a 3D Molecular Distance Map using Multidimensional Scaling (MDS).
        
        Visualizes sequence relationships in 3D space:
        - Reduces distance matrix to 3D coordinates
        - Plots sequences with color-coded labels
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            labels (List[str]): Labels for each sequence
        
        Returns:
            Tuple containing:
            - Matplotlib Figure
            - MDS coordinates
            - Unique labels
            - Color palette
        """
        # Validate input distance matrix
        if distance_matrix is None or len(distance_matrix) == 0:
            raise ValueError("Invalid distance matrix")
        
        # Apply Multidimensional Scaling to reduce dimensionality to 3D
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate color palette for unique labels
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each group with a different color
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                      c=[color], label=label, alpha=0.7, s=50)
        
        # Set axis labels and title
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        ax.set_zlabel('MDS Component 3')
        ax.legend(bbox_to_anchor=(1.15, 1))
        plt.title('3D Molecular Distance Map of Viral Sequences')
        
        return fig, coords, unique_labels, colors

def analyze_viral_sequences(fasta_files: List[str], output_dir: str = 'results') -> None:
    """
    Comprehensive analysis of viral sequences from multiple FASTA files.
    
    Main workflow:
    1. Process FASTA files
    2. Convert sequences to numerical representations
    3. Compute Discrete Fourier Transform
    4. Calculate genetic distances
    5. Perform multiple visualization and analysis techniques:
       - 3D Molecular Distance Map
       - Distance Matrix Heatmap
       - Hierarchical Clustering Dendrogram
       - Principal Component Analysis
    
    Args:
        fasta_files (List[str]): Paths to FASTA files containing viral sequences
        output_dir (str, optional): Directory to save analysis results
    """
    # Create output directory for storing results
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to track analysis progress and potential errors
    log_file = os.path.join(output_dir, 'analysis_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )
    
    logging.info("Starting viral sequence analysis")
    
    # Initialize sequence analyzer
    analyzer = ViralSequenceAnalyzer()
    
    # Initialize lists to store sequence data
    all_sequences = []    # Raw sequences
    labels = []           # Sequence labels (virus types)
    numerical_seqs = []   # Numerical representations
    max_length = 0        # Maximum sequence length
    
    # Process each FASTA file
    logging.info("Reading and processing FASTA files...")
    for file in fasta_files:
        try:
            sequence_count = 0
            for record in SeqIO.parse(file, "fasta"):
                sequence = str(record.seq)
                max_length = max(max_length, len(sequence))
                all_sequences.append(sequence)
                
                # Extract virus type from filename
                virus_type = os.path.splitext(os.path.basename(file))[0].replace("_", " ")
                
                labels.append(virus_type)
                numerical_seqs.append(analyzer.sequence_to_numerical(sequence))
                sequence_count += 1
            
            logging.info(f"Processed {sequence_count} sequences from {file}")
        
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            continue
    
    # Validate sequence processing
    if not all_sequences:
        logging.error("No sequences were successfully processed. Exiting.")
        return
    
    # Compute Discrete Fourier Transform for all sequences
    logging.info("Computing DFT for all sequences...")
    spectra = [analyzer.apply_dft(seq, max_length) for seq in numerical_seqs]
    
    # Calculate pairwise distance matrix
    logging.info("Calculating pairwise distances...")
    distance_matrix = analyzer.calculate_pairwise_distances(spectra)
    
    # Save distance matrix for further analysis
    np.save(os.path.join(output_dir, 'distance_matrix.npy'), distance_matrix)
    logging.info(f"Saved distance matrix to {output_dir}/distance_matrix.npy")
    
    # Create and save 3D Molecular Distance Map
    logging.info("Generating 3D Molecular Distance Map...")
    fig, coords, unique_labels, colors = analyzer.create_momap3d(distance_matrix, labels)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'modmap3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved 3D visualization to {output_dir}/modmap3d.png")
    
    # Save MDS coordinates
    np.save(os.path.join(output_dir, 'mds_coordinates.npy'), coords)
    
    # Additional distance matrix analysis
    logging.info("Performing additional analysis on distance matrix...")
    
    # Save distance matrix as CSV for external analysis
    df = pd.DataFrame(distance_matrix)
    df.to_csv(os.path.join(output_dir, 'distance_matrix.csv'), index=False, header=False)
    
    # Generate heatmap visualization of distance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis', interpolation='none')
    
    # Add colorbar with descriptive label
    cbar = plt.colorbar()
    cbar.set_label('Genetic Distance', rotation=270, labelpad=15)
    
    # Add title and axis labels
    plt.title('Distance Matrix Heatmap')
    plt.xlabel('Viral Sequences')
    plt.ylabel('Viral Sequences')
    
    # Add text explanation of color scale
    plt.figtext(1.02, 0.97, 'Color Scale:', fontweight='bold')
    plt.figtext(1.02, 0.93, 'Dark purple: Lower genetic distance\n(more similar sequences)')
    plt.figtext(1.02, 0.87, 'Yellow: Higher genetic distance\n(more divergent sequences)')
    
    plt.tight_layout()

    # Save the distance matrix heatmap figure
    plt.savefig(os.path.join(output_dir, 'distance_matrix_heatmap.png'),
            bbox_inches='tight',
            dpi=300)
    plt.close()

    # Perform hierarchical clustering on the distance matrix
    # Convert square distance matrix to condensed form required by scipy
    condensed_distance_matrix = squareform(distance_matrix)
    # Perform hierarchical clustering using Ward's method
    linked = sch.linkage(condensed_distance_matrix, method='ward')

    # Create dendrogram to visualize hierarchical clustering
    plt.figure(figsize=(12, 8))
    sch.dendrogram(
        linked,
        no_labels=True,  # Hide individual sequence labels
        color_threshold=0.7 * max(linked[:, 2]),  # Set color threshold for clustering
        above_threshold_color='grey'  # Color for clusters above threshold
    )

   # Add title and axis labels to dendrogram
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('Distance')
    plt.xlabel('Viral Sequences')
    plt.tight_layout()
    # Save dendrogram figure
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
    plt.close()

    # Perform Multidimensional Scaling (MDS) for dimensionality reduction
    # Use precomputed distance matrix to preserve pairwise distances
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10)
    mds_result = mds.fit_transform(distance_matrix)

    # Apply Principal Component Analysis (PCA) to MDS results
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(mds_result)

    # Calculate variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_ * 100

    # Create PCA plot with color-coded labels
    plt.figure(figsize=(10, 8))

    # Plot points for each unique label with consistent coloring
    for label, color in zip(unique_labels, colors):
    # Find indices for points with this specific label
       label_indices = [i for i, l in enumerate(labels) if l == label]
    
       # Scatter plot of PCA results, colored by label
       plt.scatter(
            pca_result[label_indices, 0],  # X-coordinates (PC1)
            pca_result[label_indices, 1],  # Y-coordinates (PC2)
            c=[color],  # Color for this label
            label=label,  # Label for legend
            alpha=0.7,  # Slight transparency
            s=50  # Marker size
        )

    # Add title and labels to PCA plot
    plt.title("PCA of Distance Matrix (after MDS)")
    plt.xlabel(f"PC1 ({explained_variance[0]:.2f}% variance explained)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2f}% variance explained)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save PCA plot
    plt.savefig(os.path.join(output_dir, 'pca_of_distance_matrix.png'), bbox_inches='tight')
    plt.close()

    # Log completion of analysis
    logging.info(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    # List of FASTA files to be analyzed
    fasta_files = [
        'DENGUE.txt',
        'YELLOW FEVER.txt',
        'TICK BORNE ENCEPHILITIS.txt',
        'WEST NILE.txt',
        'ZIKA.txt'
    ]
    
    # Run the viral sequence analysis
    analyze_viral_sequences(fasta_files)