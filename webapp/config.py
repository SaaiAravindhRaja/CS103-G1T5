"""
Configuration settings for the web application.
"""

from pathlib import Path

# Application settings
APP_TITLE = "SVD Image Compression"
APP_ICON = "🖼️"
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ["png", "jpg", "jpeg"]

# Default compression settings
DEFAULT_K_VALUE = 20
MIN_K_VALUE = 1
MAX_K_VALUE = 100
K_STEP = 1

# Image processing settings
DEFAULT_IMAGE_SIZE = (256, 256)
MAX_IMAGE_DIMENSION = 1024

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)

# UI settings
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 800

# Tailwind CSS Color Palette
COLORS = {
    "primary": {
        50: "#f8fafc",
        500: "#64748b", 
        700: "#334155",
        900: "#0f172a"
    },
    "secondary": {
        50: "#eff6ff",
        500: "#3b82f6",
        600: "#2563eb",
        700: "#1d4ed8"
    },
    "accent": {
        50: "#ecfdf5",
        500: "#10b981",
        600: "#059669"
    },
    "surface": "#ffffff",
    "background": "#f8fafc",
    "text_primary": "#1f2937",
    "text_secondary": "#6b7280",
    "border": "#e5e7eb"
}

# Tailwind CSS Classes for Components
TAILWIND_CLASSES = {
    "button_primary": "bg-gradient-to-r from-blue-500 to-blue-600 text-white border-0 rounded-lg px-6 py-3 font-medium text-base transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5",
    "button_secondary": "bg-white text-blue-600 border border-blue-300 rounded-lg px-6 py-3 font-medium text-base transition-all duration-200 hover:bg-blue-50",
    "card": "bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-all duration-200",
    "upload_zone": "border-2 border-dashed border-gray-300 rounded-xl p-8 text-center bg-slate-50 transition-all duration-200 hover:border-blue-500 hover:bg-blue-50",
    "metric_card": "bg-white border border-gray-200 rounded-xl p-6 text-center shadow-sm hover:shadow-md transition-all duration-200",
    "loading_spinner": "border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin"
}

# Quality thresholds for visual indicators
QUALITY_THRESHOLDS = {
    "psnr": {
        "excellent": 40,
        "good": 30,
        "fair": 20,
        "poor": 0
    },
    "ssim": {
        "excellent": 0.9,
        "good": 0.7,
        "fair": 0.5,
        "poor": 0
    }
}