# Image Similarity Analyzer

This application is a Streamlit web app that measures the similarity between two images using various computer vision techniques. It provides detailed metrics and an overall similarity score.

[Image Similarity Analyzer App](https://simageai.streamlit.app/)

## Features

-   **Multiple Comparison Methods:** Utilizes histogram comparison, structural similarity index (SSIM), and ORB feature matching.
-   **Detailed Metrics:** Displays individual similarity scores for each method.
-   **Overall Similarity Score:** Provides a weighted average of the metrics.
-   **Feature Match Visualization:** Option to visualize feature matches between images.
-   **User-Friendly Interface:** Built with Streamlit for easy interaction.
-   **Sample Image Generation:** Generates sample images for testing.

## Technologies Used

-   Streamlit
-   OpenCV (opencv-python)
-   NumPy
-   Pillow (PIL)
-   scikit-image
-   SciPy

## Getting Started

### Prerequisites

-   Python 3.7+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/viswadarshan-024/image-similarity-ai.git
    cd image-similarity-analyzer
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit app:**
    ```bash
    streamlit run imageapp.py
    ```
2.  Open your web browser and navigate to the local URL displayed in the terminal.

## Usage

1.  Upload two images using the file uploaders.
2.  Click the "Compare Images" button.
3.  View the overall similarity score and detailed metrics.
4.  Optionally, check the "Show Feature Matches" checkbox to visualize feature matches.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is not licensed.
