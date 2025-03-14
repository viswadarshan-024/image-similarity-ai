import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine

# Set page configuration
st.set_page_config(
    page_title="Image Similarity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .uploadedImages {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem; 
        font-weight: bold;
        color: #333333;
    }
    .header-text {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def resize_image(img, max_size=224):
    """Resize image while maintaining aspect ratio"""
    ratio = max_size / max(img.shape[0], img.shape[1])
    return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for comparison"""
    img = cv2.resize(img, target_size)
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Handle RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def compute_histogram_similarity(img1, img2):
    """Compute histogram-based similarity between images"""
    # Convert images to HSV color space
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    else:
        img1_hsv = img1
        img2_hsv = img2
    
    # Calculate histograms
    hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Calculate histogram similarity
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity * 100  # Scale to percentage

def compute_ssim_similarity(img1, img2):
    """Compute structural similarity index between images"""
    # Resize images to same dimensions
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Compute SSIM
    (score, _) = ssim(img1_gray, img2_gray, full=True)
    return score * 100  # Scale to percentage

def compute_orb_feature_similarity(img1, img2):
    """Compute ORB feature similarity between images"""
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Convert to grayscale for feature detection
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    # Check if descriptors were found
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    
    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate match percentage
    total_keypoints = min(len(kp1), len(kp2))
    if total_keypoints == 0:
        return 0.0
        
    # Limit to max 100 matches
    good_matches = matches[:min(100, len(matches))]
    
    # Calculate match quality (lower distance is better)
    average_distance = sum(match.distance for match in good_matches) / len(good_matches) if good_matches else 0
    # Convert distance to similarity score
    similarity = max(0, 100 - (average_distance / 2))
    
    return similarity

def compute_overall_similarity(scores):
    """Compute weighted average of similarity scores"""
    weights = {
        "Histogram Similarity": 0.3,
        "Structural Similarity": 0.4,
        "Feature Similarity": 0.3
    }
    
    weighted_sum = sum(scores[metric] * weights[metric] for metric in weights)
    return weighted_sum

def perform_image_comparison(img1, img2):
    """Perform comprehensive image comparison"""
    # Process images
    img1_rgb = preprocess_image(img1)
    img2_rgb = preprocess_image(img2)
    
    # Compute histogram similarity
    hist_similarity = compute_histogram_similarity(img1_rgb, img2_rgb)
    
    # Compute structural similarity
    ssim_similarity = compute_ssim_similarity(img1_rgb, img2_rgb)
    
    # Compute ORB feature similarity
    orb_similarity = compute_orb_feature_similarity(img1_rgb, img2_rgb)
    
    # Collect all similarity scores
    similarity_scores = {
        "Histogram Similarity": hist_similarity,
        "Structural Similarity": ssim_similarity,
        "Feature Similarity": orb_similarity
    }
    
    # Compute overall similarity
    overall_similarity = compute_overall_similarity(similarity_scores)
    
    return similarity_scores, overall_similarity

def main():
    st.title("Image Similarity Analyzer")
    
    st.markdown("""
    ## 📊 Analyze Similarity Between Two Images
    Upload two images to analyze their similarity using multiple comparison methods.
    The system will provide detailed metrics and an overall similarity score.
    """)
    
    # Create columns for file uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Image")
        uploaded_file1 = st.file_uploader("Choose an image...", key="file1", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    with col2:
        st.subheader("Second Image")
        uploaded_file2 = st.file_uploader("Choose an image...", key="file2", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    # Display images and perform comparison when both are uploaded
    if uploaded_file1 and uploaded_file2:
        # Read images
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)
        
        # Convert to array
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="Image 1", use_container_width=True)
            st.text(f"Dimensions: {img1_array.shape[1]} x {img1_array.shape[0]}")
        
        with col2:
            st.image(image2, caption="Image 2", use_container_width=True)
            st.text(f"Dimensions: {img2_array.shape[1]} x {img2_array.shape[0]}")
        
        # Add a compare button
        if st.button("Compare Images", key="compare_btn"):
            with st.spinner("Computing similarity scores..."):
                # Perform image comparison
                similarity_scores, overall_similarity = perform_image_comparison(img1_array, img2_array)
                
                # Display overall similarity score with gauge
                st.markdown("### 📝 Overall Similarity Score")
                
                # Create a progress bar for overall similarity
                progress_color = "green" if overall_similarity > 75 else "orange" if overall_similarity > 50 else "red"
                st.markdown(f'<div style="text-align: center; font-size: 3rem; font-weight: bold; color: {progress_color};">{overall_similarity:.1f}%</div>', unsafe_allow_html=True)
                st.progress(overall_similarity/100)
                
                # Interpretation text
                if overall_similarity > 90:
                    st.success("The images are nearly identical!")
                elif overall_similarity > 75:
                    st.success("The images are very similar.")
                elif overall_similarity > 50:
                    st.warning("The images have moderate similarity.")
                else:
                    st.error("The images are quite different.")
                
                # Display detailed metrics
                st.markdown("### 📊 Detailed Similarity Metrics")
                
                # Create columns for the metrics
                metric_cols = st.columns(3)
                
                # Define colors for each metric
                colors = [
                    "#FF9500",  # Histogram
                    "#00BA7C",  # Structural
                    "#0080FF"   # Feature
                ]
                
                # Display each metric in its card
                for i, (metric, score) in enumerate(similarity_scores.items()):
                    with metric_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="header-text" style="color: {colors[i]};">{metric}</div>
                            <div class="metric-value">{score:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Option to view match visualization
                if st.checkbox("Show Feature Matches"):
                    # Calculate feature matches for visualization
                    img1_rgb = preprocess_image(img1_array)
                    img2_rgb = preprocess_image(img2_array)
                    
                    # Convert to grayscale
                    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
                    
                    # Find ORB keypoints
                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(img1_gray, None)
                    kp2, des2 = orb.detectAndCompute(img2_gray, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                        # Match features
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        # Draw matches
                        good_matches = matches[:min(30, len(matches))]
                        img_matches = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        
                        # Display match visualization
                        st.image(img_matches, caption="Feature Matches Between Images", use_column_width=True)
                    else:
                        st.warning("Not enough features detected for visualization")
                
                # Add explanations
                st.markdown("### 🔍 What These Metrics Mean")
                st.markdown("""
                - **Histogram Similarity**: Compares color distributions between images
                - **Structural Similarity**: Analyzes patterns and structural information
                - **Feature Similarity**: Uses ORB features to compare key points between images
                """)
    
    # Add information about the app
    st.sidebar.title("About")
    st.sidebar.info("""
    ## Image Similarity Analyzer
    
    This application measures the similarity between two images using multiple computer vision techniques:
    
    1. **Histogram Comparison**: Analyzes color distribution
    2. **Structural Similarity Index (SSIM)**: Measures structural patterns
    3. **ORB Feature Matching**: Identifies and compares key points in both images
    
    The final score is a weighted average of these metrics.
    """)
    
    # Add application settings
    st.sidebar.title("Settings")
    
    # Add download sample images option
    st.sidebar.markdown("### Try with sample images")
    if st.sidebar.button("Generate Sample Images"):
        # Create sample images
        sample1 = np.ones((300, 300, 3), dtype=np.uint8) * 255
        sample2 = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # Draw different shapes on them
        cv2.circle(sample1, (150, 150), 100, (255, 0, 0), -1)
        cv2.rectangle(sample2, (50, 50), (250, 250), (0, 0, 255), -1)
        
        # Convert to PIL Images
        pil_sample1 = Image.fromarray(sample1)
        pil_sample2 = Image.fromarray(sample2)
        
        # Save to BytesIO
        buf1 = io.BytesIO()
        buf2 = io.BytesIO()
        pil_sample1.save(buf1, format='PNG')
        pil_sample2.save(buf2, format='PNG')
        
        # Provide download links
        st.sidebar.download_button(
            label="Download Sample Image 1",
            data=buf1.getvalue(),
            file_name="sample1.png",
            mime="image/png"
        )
        
        st.sidebar.download_button(
            label="Download Sample Image 2",
            data=buf2.getvalue(),
            file_name="sample2.png",
            mime="image/png"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
