"""
MyoAnalyzer: Muscle Composition Quantifier
Clean Streamlit Interface - Researcher Focused
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import tempfile
from pathlib import Path
import sys
from datetime import datetime
import time

# Add muscle_pipeline to path
sys.path.append(str(Path(__file__).parent / 'muscle_pipeline' / 'src'))

# Import analyzer classes
try:
    from muscle_analyzer import MuscleAnalyzer
    ANALYZERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import muscle analyzer: {e}")
    ANALYZERS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="MyoAnalyzer",
    page_icon="muscle",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .error-section {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<p class="main-header">MyoAnalyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated Dixon Fat-Water MRI Analysis for Neuromuscular Research</p>', unsafe_allow_html=True)
    
    if not ANALYZERS_AVAILABLE:
        st.markdown('<div class="error-section">Analysis modules not found. Please check your muscle_pipeline/src/ directory.</div>', unsafe_allow_html=True)
        return
    
    # Initialize session state for results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Main workflow
    upload_and_analyze_section()
    
    # Show results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_current_results()

def upload_and_analyze_section():
    """Clean upload and analysis section"""
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.header("Upload Dixon MRI Images")
    
    # File upload in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fat Image")
        fat_file = st.file_uploader(
            "Select Fat Image",
            type=['nii', 'nii.gz'],
            help="T1-weighted fat-only Dixon MRI (.nii or .nii.gz)",
            key="fat_upload"
        )
        
    with col2:
        st.subheader("Water Image")
        water_file = st.file_uploader(
            "Select Water Image", 
            type=['nii', 'nii.gz'],
            help="T1-weighted water-only Dixon MRI (.nii or .nii.gz)",
            key="water_upload"
        )
    
    # Analysis configuration
    with st.expander("Analysis Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Segmentation Confidence",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence for muscle segmentation"
            )
            
        with col2:
            use_gpu = st.checkbox("Use GPU Acceleration", value=True)
    
    # Analysis button and status
    if fat_file and water_file:
        # Show file info
        st.subheader("File Information")
        file_df = pd.DataFrame({
            'Image Type': ['Fat', 'Water'],
            'Filename': [fat_file.name, water_file.name],
            'Size (MB)': [f"{fat_file.size/1024/1024:.1f}", f"{water_file.size/1024/1024:.1f}"]
        })
        st.table(file_df)
        
        # Analysis button
        if st.button("Start Muscle Analysis", type="primary", use_container_width=True):
            run_analysis(fat_file, water_file, confidence_threshold, use_gpu)
    
    elif fat_file or water_file:
        st.info("Please upload both fat and water images to proceed with analysis.")
    
    else:
        st.info("Upload your Dixon MRI fat and water images to begin analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def run_analysis(fat_file, water_file, confidence_threshold, use_gpu):
    """Run the actual muscle analysis"""
    
    # Clear previous results
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = None
    
    # Progress tracking
    progress_container = st.container()
    
    with progress_container:
        st.subheader("Analysis in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize
            status_text.text("Initializing analyzer...")
            progress_bar.progress(10)
            analyzer = MuscleAnalyzer()
            time.sleep(0.5)
            
            # Step 2: Prepare files with proper directory structure
            status_text.text("Preparing image files...")
            progress_bar.progress(20)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create proper subject directory structure
                subject_name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                subject_dir = temp_path / subject_name
                image_data_dir = subject_dir / "ImageData"
                
                # Create fat and water directories with proper names
                fat_dir = image_data_dir / f"{subject_name}_FAT"
                water_dir = image_data_dir / f"{subject_name}_WATER"
                
                fat_dir.mkdir(parents=True)
                water_dir.mkdir(parents=True)
                
                # Save fat image with proper naming convention
                fat_path = fat_dir / f"{subject_name}_FAT_stack1.nii"
                fat_file.seek(0)
                with open(fat_path, 'wb') as f:
                    f.write(fat_file.read())
                    f.flush()
                
                progress_bar.progress(30)
                
                # Save water image with proper naming convention
                water_path = water_dir / f"{subject_name}_WATER_stack1.nii"
                water_file.seek(0)
                with open(water_path, 'wb') as f:
                    f.write(water_file.read())
                    f.flush()
                
                progress_bar.progress(40)
                
                # Verify files were saved correctly
                status_text.text("Verifying uploaded files...")
                progress_bar.progress(45)
                
                if not fat_path.exists() or not water_path.exists():
                    raise Exception("Failed to save uploaded files")
                
                # Check file sizes
                fat_size = fat_path.stat().st_size / (1024 * 1024)  # MB
                water_size = water_path.stat().st_size / (1024 * 1024)  # MB
                
                if fat_size < 0.1 or water_size < 0.1:
                    raise Exception(f"Files too small: Fat={fat_size:.1f}MB, Water={water_size:.1f}MB")
                
                # Step 3: Run segmentation
                status_text.text("Running muscle segmentation...")
                progress_bar.progress(50)
                time.sleep(1)  # Simulate processing time
                
                # Step 4: Calculate composition
                status_text.text("Calculating muscle composition...")
                progress_bar.progress(70)
                
                # Create output directory
                output_dir = temp_path / "output"
                output_dir.mkdir()
                
                # Run actual analysis
                results = analyzer.analyze_subject(subject_dir, "stack1", output_dir)
                
                progress_bar.progress(90)
                
                if results:
                    # Step 5: Format results
                    status_text.text("Analysis complete!")
                    progress_bar.progress(100)
                    
                    # Store results in session state
                    formatted_results = format_results_for_display(results, fat_file.name, water_file.name)
                    st.session_state.analysis_results = formatted_results
                    st.session_state.analysis_complete = True
                    
                    # Clear progress and show success
                    progress_container.empty()
                    st.success("Analysis completed successfully! Results are displayed below.")
                    
                    # Auto-scroll to results
                    st.rerun()
                    
                else:
                    st.error("Analysis failed - no results returned from muscle analyzer")
                    
        except Exception as e:
            progress_container.empty()
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)

def format_results_for_display(results, fat_filename, water_filename):
    """Format analyzer results for clean display"""
    
    muscle_data = []
    total_volume = 0
    fat_fractions = []
    
    for muscle_name, metrics in results.items():
        muscle_data.append({
            'Muscle Group': muscle_name,
            'Total Volume (cm³)': round(metrics['total_volume_cm3'], 1),
            'Lean Volume (cm³)': round(metrics['lean_volume_cm3'], 1),
            'Fat Volume (cm³)': round(metrics['fat_volume_cm3'], 1),
            'Fat Fraction (%)': round(metrics['mean_fat_fraction_percent'], 1),
            'Fat Fraction SD (%)': round(metrics['std_fat_fraction_percent'], 1),
            'Median Fat Fraction (%)': round(metrics['median_fat_fraction_percent'], 1),
            'Voxel Count': metrics['n_voxels']
        })
        
        total_volume += metrics['total_volume_cm3']
        fat_fractions.append(metrics['mean_fat_fraction_percent'])
    
    # Calculate summary statistics
    summary = {
        'total_muscle_volume': round(total_volume, 1),
        'mean_fat_fraction': round(np.mean(fat_fractions), 1),
        'min_fat_fraction': round(min(fat_fractions), 1),
        'max_fat_fraction': round(max(fat_fractions), 1),
        'std_fat_fraction': round(np.std(fat_fractions), 1),
        'muscles_analyzed': len(results),
        'fat_filename': fat_filename,
        'water_filename': water_filename,
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return {
        'muscle_data': muscle_data,
        'summary': summary,
        'raw_results': results  # Keep original format for downloads
    }

def display_current_results():
    """Display results from current analysis only"""
    
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.header("Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Summary metrics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Muscle Volume", 
            f"{results['summary']['total_muscle_volume']} cm³"
        )
    
    with col2:
        st.metric(
            "Mean Fat Fraction", 
            f"{results['summary']['mean_fat_fraction']}%"
        )
    
    with col3:
        st.metric(
            "Fat Fraction Range", 
            f"{results['summary']['min_fat_fraction']}% - {results['summary']['max_fat_fraction']}%"
        )
    
    with col4:
        st.metric(
            "Muscles Analyzed", 
            results['summary']['muscles_analyzed']
        )
    
    # Input file information
    st.subheader("Input Files")
    input_df = pd.DataFrame({
        'Input Type': ['Fat Image', 'Water Image'],
        'Filename': [results['summary']['fat_filename'], results['summary']['water_filename']],
        'Analysis Time': [results['summary']['analysis_timestamp']] * 2
    })
    st.table(input_df)
    
    # Detailed muscle results
    st.subheader("Muscle-by-Muscle Results")
    muscle_df = pd.DataFrame(results['muscle_data'])
    st.dataframe(muscle_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    create_clean_visualizations(muscle_df)
    
    # Download section
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV download
        csv_data = muscle_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            f"muscle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON download (raw results)
        json_data = json.dumps(results['raw_results'], indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            f"muscle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        # Summary report
        summary_data = pd.DataFrame([results['summary']]).to_csv(index=False)
        st.download_button(
            "Download Summary",
            summary_data,
            f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Option to analyze another subject
    st.subheader("Analyze Another Subject")
    if st.button("Start New Analysis", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = None
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_clean_visualizations(muscle_df):
    """Create focused visualizations for current results"""
    
    st.subheader("Visualizations")
    
    # Two-column layout for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fat fraction by muscle
        fig1 = px.bar(
            muscle_df, 
            x='Muscle Group', 
            y='Fat Fraction (%)',
            title="Fat Fraction by Muscle Group",
            color='Fat Fraction (%)',
            color_continuous_scale='RdYlBu_r'
        )
        fig1.update_layout(
            showlegend=False, 
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Volume distribution
        fig2 = px.pie(
            muscle_df, 
            values='Total Volume (cm³)', 
            names='Muscle Group',
            title="Muscle Volume Distribution"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Fat vs lean volume comparison
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        name='Lean Volume',
        x=muscle_df['Muscle Group'],
        y=muscle_df['Lean Volume (cm³)'],
        marker_color='lightblue'
    ))
    
    fig3.add_trace(go.Bar(
        name='Fat Volume',
        x=muscle_df['Muscle Group'],
        y=muscle_df['Fat Volume (cm³)'],
        marker_color='coral'
    ))
    
    fig3.update_layout(
        title='Lean vs Fat Volume by Muscle',
        barmode='stack',
        xaxis_tickangle=-45,
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Summary statistics table
    st.subheader("Statistical Summary")
    stats_df = muscle_df[['Fat Fraction (%)', 'Total Volume (cm³)', 'Fat Volume (cm³)']].describe().round(1)
    st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    main()
