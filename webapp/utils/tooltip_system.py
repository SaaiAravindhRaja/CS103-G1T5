"""
Interactive tooltip and help system for the SVD Image Compression webapp.
Provides contextual help, tooltips, and keyboard shortcuts for enhanced user experience.
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class TooltipSystem:
    """Comprehensive tooltip and help system for the webapp."""
    
    def __init__(self):
        """Initialize the tooltip system with predefined help content."""
        self.help_content = self._load_help_content()
        self.keyboard_shortcuts = self._load_keyboard_shortcuts()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for tooltip system."""
        if 'tooltip_system' not in st.session_state:
            st.session_state.tooltip_system = {
                'help_mode': False,
                'show_tooltips': True,
                'tooltip_delay': 500,
                'help_panel_open': False,
                'current_help_topic': None,
                'keyboard_shortcuts_enabled': True
            }
    
    def _load_help_content(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive help content for all components."""
        return {
            'compression_controls': {
                'k_value': {
                    'title': 'K-Value (Compression Level)',
                    'description': 'Controls how many singular values to keep from the SVD decomposition',
                    'details': [
                        'Lower k = Higher compression, smaller file size, lower quality',
                        'Higher k = Lower compression, larger file size, higher quality',
                        'Recommended ranges:',
                        '  • 2-10: High compression (thumbnails, previews)',
                        '  • 20-50: Balanced (web images, documents)',
                        '  • 100+: High quality (archival, detailed images)'
                    ],
                    'tips': [
                        'Use energy-based selection for optimal results',
                        'Start with presets and fine-tune as needed',
                        'Consider image content when choosing k'
                    ]
                },
                'processing_mode': {
                    'title': 'Processing Mode',
                    'description': 'Determines how color channels are processed during compression',
                    'details': [
                        'RGB (Color): Processes red, green, and blue channels separately',
                        '  • Best for: Color photos, artwork, detailed images',
                        '  • Higher quality but larger compressed size',
                        'Grayscale: Converts image to grayscale first, then compresses',
                        '  • Best for: Text documents, diagrams, black & white photos',
                        '  • Higher compression ratios, smaller file sizes'
                    ],
                    'tips': [
                        'Use RGB for color-critical applications',
                        'Use Grayscale for documents and diagrams',
                        'Grayscale mode is faster and uses less memory'
                    ]
                },
                'real_time_preview': {
                    'title': 'Real-time Preview',
                    'description': 'Automatically compresses image as you adjust parameters',
                    'details': [
                        'Enabled: Immediate feedback as you change settings',
                        'Disabled: Manual compression with "Compress Image" button',
                        'Uses smaller image size for faster processing',
                        'Automatically disabled for very large images'
                    ],
                    'tips': [
                        'Disable for large images to improve performance',
                        'Enable for quick experimentation with settings',
                        'Preview uses downsampled image for speed'
                    ]
                },
                'quality_presets': {
                    'title': 'Quality Presets',
                    'description': 'Pre-configured compression settings for common use cases',
                    'details': [
                        'Ultra Low (k=2): Maximum compression, lowest quality',
                        'Low (k=5): High compression, basic quality',
                        'Medium (k=20): Balanced compression and quality',
                        'High (k=50): Low compression, high quality',
                        'Ultra High (k=100+): Minimal compression, maximum quality'
                    ],
                    'tips': [
                        'Start with Medium preset for most images',
                        'Use Low for thumbnails and previews',
                        'Use High for professional applications'
                    ]
                }
            },
            'quality_metrics': {
                'psnr': {
                    'title': 'PSNR (Peak Signal-to-Noise Ratio)',
                    'description': 'Measures the quality of compressed image compared to original',
                    'details': [
                        'Measured in decibels (dB)',
                        'Higher values indicate better quality',
                        'Quality ranges:',
                        '  • > 35 dB: Excellent quality',
                        '  • 25-35 dB: Good quality',
                        '  • 20-25 dB: Fair quality',
                        '  • < 20 dB: Poor quality'
                    ],
                    'formula': 'PSNR = 20 × log₁₀(MAX / √MSE)',
                    'tips': [
                        'PSNR > 30 dB is generally acceptable',
                        'Higher PSNR doesn\'t always mean better visual quality',
                        'Use in combination with SSIM for complete assessment'
                    ]
                },
                'ssim': {
                    'title': 'SSIM (Structural Similarity Index)',
                    'description': 'Measures structural similarity between original and compressed images',
                    'details': [
                        'Range: 0 to 1 (1 = identical images)',
                        'Based on luminance, contrast, and structure',
                        'Quality ranges:',
                        '  • > 0.9: Excellent similarity',
                        '  • 0.7-0.9: Good similarity',
                        '  • 0.5-0.7: Fair similarity',
                        '  • < 0.5: Poor similarity'
                    ],
                    'formula': 'SSIM considers luminance, contrast, and structure',
                    'tips': [
                        'SSIM > 0.8 indicates high perceptual quality',
                        'More aligned with human visual perception than PSNR',
                        'Better metric for assessing visual quality'
                    ]
                },
                'compression_ratio': {
                    'title': 'Compression Ratio',
                    'description': 'Ratio of original size to compressed size',
                    'details': [
                        'Higher values mean more space saved',
                        'Calculated as: Original Size / Compressed Size',
                        'Typical ranges:',
                        '  • 2:1 to 5:1: Moderate compression',
                        '  • 5:1 to 10:1: High compression',
                        '  • 10:1+: Very high compression'
                    ],
                    'tips': [
                        'Balance compression ratio with quality metrics',
                        'Higher ratios may sacrifice visual quality',
                        'Consider your specific use case requirements'
                    ]
                },
                'mse': {
                    'title': 'MSE (Mean Squared Error)',
                    'description': 'Average squared difference between original and compressed pixels',
                    'details': [
                        'Lower values indicate better quality',
                        'Range: 0 to maximum pixel value squared',
                        'Used in PSNR calculation',
                        'Sensitive to outlier pixel differences'
                    ],
                    'formula': 'MSE = (1/N) × Σ(original - compressed)²',
                    'tips': [
                        'Lower MSE generally means better quality',
                        'Use alongside PSNR and SSIM for complete picture',
                        'May not reflect perceptual quality differences'
                    ]
                }
            },
            'svd_concepts': {
                'singular_values': {
                    'title': 'Singular Values',
                    'description': 'Core components of the SVD decomposition that capture image information',
                    'details': [
                        'Ordered from largest to smallest',
                        'Larger values contain more image information',
                        'Energy is proportional to the square of singular values',
                        'Keeping top k values approximates the original image'
                    ],
                    'tips': [
                        'First few values capture most image information',
                        'Plot shows energy distribution across values',
                        'Elbow in the plot suggests optimal k value'
                    ]
                },
                'energy_retention': {
                    'title': 'Energy Retention',
                    'description': 'Percentage of original image energy preserved after compression',
                    'details': [
                        'Based on sum of squared singular values',
                        'Higher retention = better quality approximation',
                        'Common thresholds:',
                        '  • 90%: Good visual quality',
                        '  • 95%: Excellent quality',
                        '  • 99%: Near-lossless quality'
                    ],
                    'formula': 'Energy = Σ(singular_values²) / Total_Energy',
                    'tips': [
                        '90% energy retention is often optimal',
                        'Use energy-based k selection for consistency',
                        'Higher retention requires more singular values'
                    ]
                },
                'svd_decomposition': {
                    'title': 'SVD Decomposition',
                    'description': 'Mathematical foundation of the compression algorithm',
                    'details': [
                        'Factorizes image matrix: A = U × Σ × V^T',
                        'U: Left singular vectors (spatial patterns)',
                        'Σ: Singular values (importance weights)',
                        'V^T: Right singular vectors (spatial patterns)',
                        'Approximation: A_k = U_k × Σ_k × V_k^T'
                    ],
                    'tips': [
                        'SVD finds optimal low-rank approximation',
                        'Compression quality depends on singular value distribution',
                        'Works best for images with smooth regions'
                    ]
                }
            },
            'interface_elements': {
                'upload_zone': {
                    'title': 'Image Upload',
                    'description': 'Drag and drop or click to upload images for compression',
                    'details': [
                        'Supported formats: PNG, JPEG, JPG, BMP, TIFF',
                        'Maximum file size: 10 MB',
                        'Recommended size: Under 2048x2048 pixels',
                        'Multiple files supported in batch mode'
                    ],
                    'tips': [
                        'Drag and drop for quick upload',
                        'Larger images take longer to process',
                        'Use appropriate format for your image type'
                    ]
                },
                'results_display': {
                    'title': 'Results Display',
                    'description': 'Shows original vs compressed images with quality metrics',
                    'details': [
                        'Side-by-side comparison view',
                        'Zoom and pan functionality',
                        'Difference visualization option',
                        'Download compressed image'
                    ],
                    'tips': [
                        'Use zoom to inspect fine details',
                        'Toggle difference view to see compression artifacts',
                        'Download preserves original format when possible'
                    ]
                }
            }
        }
    
    def _load_keyboard_shortcuts(self) -> Dict[str, Dict[str, str]]:
        """Load keyboard shortcuts configuration."""
        return {
            'global': {
                'h': 'Toggle help mode',
                '?': 'Show keyboard shortcuts',
                'Escape': 'Close help panels',
                'Tab': 'Navigate between elements'
            },
            'compression_controls': {
                'ArrowUp': 'Increase k-value by 1',
                'ArrowDown': 'Decrease k-value by 1',
                'Shift+ArrowUp': 'Increase k-value by 10',
                'Shift+ArrowDown': 'Decrease k-value by 10',
                'r': 'Toggle real-time preview',
                'g': 'Switch to grayscale mode',
                'c': 'Switch to color mode',
                '1': 'Apply Ultra Low preset',
                '2': 'Apply Low preset',
                '3': 'Apply Medium preset',
                '4': 'Apply High preset',
                '5': 'Apply Ultra High preset'
            },
            'results_view': {
                '+': 'Zoom in',
                '-': 'Zoom out',
                '0': 'Reset zoom',
                'd': 'Toggle difference view',
                's': 'Download compressed image',
                'f': 'Toggle fullscreen view'
            }
        }
    
    def create_tooltip(self, content: str, tooltip_text: str, position: str = "top", 
                      delay: int = 500, max_width: str = "300px") -> str:
        """Create a tooltip for any content."""
        tooltip_id = f"tooltip_{hash(content + tooltip_text)}"
        
        return f"""
        <div class="tooltip-container" style="position: relative; display: inline-block;">
            {content}
            <div class="tooltip-text" id="{tooltip_id}" style="
                visibility: hidden;
                opacity: 0;
                position: absolute;
                z-index: 1000;
                background-color: #1f2937;
                color: white;
                text-align: center;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 0.875rem;
                line-height: 1.4;
                max-width: {max_width};
                word-wrap: break-word;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: opacity 0.3s, visibility 0.3s;
                {self._get_tooltip_position_styles(position)}
            ">
                {tooltip_text}
                <div class="tooltip-arrow" style="
                    position: absolute;
                    {self._get_arrow_position_styles(position)}
                    border: 5px solid transparent;
                    {self._get_arrow_border_styles(position)}
                "></div>
            </div>
        </div>
        
        <script>
        (function() {{
            const container = document.querySelector('.tooltip-container:last-of-type');
            const tooltip = document.getElementById('{tooltip_id}');
            let timeoutId;
            
            if (container && tooltip) {{
                container.addEventListener('mouseenter', function() {{
                    timeoutId = setTimeout(function() {{
                        tooltip.style.visibility = 'visible';
                        tooltip.style.opacity = '1';
                    }}, {delay});
                }});
                
                container.addEventListener('mouseleave', function() {{
                    clearTimeout(timeoutId);
                    tooltip.style.visibility = 'hidden';
                    tooltip.style.opacity = '0';
                }});
            }}
        }})();
        </script>
        """
    
    def _get_tooltip_position_styles(self, position: str) -> str:
        """Get CSS styles for tooltip positioning."""
        styles = {
            'top': 'bottom: 100%; left: 50%; transform: translateX(-50%); margin-bottom: 8px;',
            'bottom': 'top: 100%; left: 50%; transform: translateX(-50%); margin-top: 8px;',
            'left': 'right: 100%; top: 50%; transform: translateY(-50%); margin-right: 8px;',
            'right': 'left: 100%; top: 50%; transform: translateY(-50%); margin-left: 8px;'
        }
        return styles.get(position, styles['top'])
    
    def _get_arrow_position_styles(self, position: str) -> str:
        """Get CSS styles for tooltip arrow positioning."""
        styles = {
            'top': 'top: 100%; left: 50%; transform: translateX(-50%);',
            'bottom': 'bottom: 100%; left: 50%; transform: translateX(-50%);',
            'left': 'left: 100%; top: 50%; transform: translateY(-50%);',
            'right': 'right: 100%; top: 50%; transform: translateY(-50%);'
        }
        return styles.get(position, styles['top'])
    
    def _get_arrow_border_styles(self, position: str) -> str:
        """Get CSS styles for tooltip arrow border."""
        styles = {
            'top': 'border-top-color: #1f2937;',
            'bottom': 'border-bottom-color: #1f2937;',
            'left': 'border-left-color: #1f2937;',
            'right': 'border-right-color: #1f2937;'
        }
        return styles.get(position, styles['top'])
    
    def create_help_button(self, topic: str, subtopic: str = None, 
                          button_text: str = "?", button_style: str = "icon") -> None:
        """Create a help button that opens detailed help for a specific topic."""
        help_key = f"help_{topic}_{subtopic}" if subtopic else f"help_{topic}"
        
        if button_style == "icon":
            if st.button(
                button_text, 
                key=help_key,
                help=f"Get help about {topic.replace('_', ' ').title()}",
                type="secondary"
            ):
                self._show_help_modal(topic, subtopic)
        else:
            if st.button(
                f"📖 {button_text}",
                key=help_key,
                help=f"Get detailed help about {topic.replace('_', ' ').title()}"
            ):
                self._show_help_modal(topic, subtopic)
    
    def _show_help_modal(self, topic: str, subtopic: str = None):
        """Show detailed help modal for a specific topic."""
        help_data = self.help_content.get(topic, {})
        if subtopic:
            help_data = help_data.get(subtopic, {})
        
        if not help_data:
            st.error(f"Help content not found for {topic}/{subtopic}")
            return
        
        # Create modal-like display using expander
        with st.expander(f"📖 Help: {help_data.get('title', topic.title())}", expanded=True):
            st.markdown(f"**{help_data.get('description', '')}**")
            
            if 'details' in help_data:
                st.markdown("### Details")
                for detail in help_data['details']:
                    st.markdown(f"• {detail}")
            
            if 'formula' in help_data:
                st.markdown("### Formula")
                st.code(help_data['formula'])
            
            if 'tips' in help_data:
                st.markdown("### Tips")
                for tip in help_data['tips']:
                    st.markdown(f"💡 {tip}")
    
    def create_contextual_help_panel(self, context: str = "general") -> None:
        """Create a contextual help panel based on current page/context."""
        if not st.session_state.tooltip_system.get('help_panel_open', False):
            return
        
        st.sidebar.markdown("# 📖 Help & Tips")
        
        # Context-specific help
        if context == "compression_controls":
            self._show_compression_help()
        elif context == "quality_metrics":
            self._show_metrics_help()
        elif context == "svd_concepts":
            self._show_svd_help()
        else:
            self._show_general_help()
        
        # Keyboard shortcuts
        with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
            self._show_keyboard_shortcuts(context)
    
    def _show_compression_help(self):
        """Show compression-specific help in sidebar."""
        st.sidebar.markdown("## Compression Controls")
        
        topics = ['k_value', 'processing_mode', 'real_time_preview', 'quality_presets']
        for topic in topics:
            help_data = self.help_content['compression_controls'][topic]
            with st.sidebar.expander(help_data['title']):
                st.markdown(help_data['description'])
                if 'tips' in help_data:
                    for tip in help_data['tips'][:2]:  # Show first 2 tips
                        st.markdown(f"💡 {tip}")
    
    def _show_metrics_help(self):
        """Show metrics-specific help in sidebar."""
        st.sidebar.markdown("## Quality Metrics")
        
        topics = ['psnr', 'ssim', 'compression_ratio', 'mse']
        for topic in topics:
            help_data = self.help_content['quality_metrics'][topic]
            with st.sidebar.expander(help_data['title']):
                st.markdown(help_data['description'])
                if 'tips' in help_data:
                    for tip in help_data['tips'][:2]:
                        st.markdown(f"💡 {tip}")
    
    def _show_svd_help(self):
        """Show SVD concepts help in sidebar."""
        st.sidebar.markdown("## SVD Concepts")
        
        topics = ['singular_values', 'energy_retention', 'svd_decomposition']
        for topic in topics:
            help_data = self.help_content['svd_concepts'][topic]
            with st.sidebar.expander(help_data['title']):
                st.markdown(help_data['description'])
    
    def _show_general_help(self):
        """Show general help in sidebar."""
        st.sidebar.markdown("## General Help")
        
        st.sidebar.markdown("""
        ### Getting Started
        1. Upload an image using the upload zone
        2. Adjust compression settings
        3. View results and quality metrics
        4. Download compressed image
        
        ### Tips
        💡 Start with Medium quality preset
        💡 Use real-time preview for quick adjustments
        💡 Check energy retention for optimal k-value
        💡 Compare PSNR and SSIM for quality assessment
        """)
    
    def _show_keyboard_shortcuts(self, context: str):
        """Show keyboard shortcuts for current context."""
        shortcuts = self.keyboard_shortcuts.get('global', {})
        context_shortcuts = self.keyboard_shortcuts.get(context, {})
        
        # Global shortcuts
        st.markdown("**Global:**")
        for key, description in shortcuts.items():
            st.markdown(f"`{key}` - {description}")
        
        # Context-specific shortcuts
        if context_shortcuts:
            st.markdown(f"**{context.replace('_', ' ').title()}:**")
            for key, description in context_shortcuts.items():
                st.markdown(f"`{key}` - {description}")
    
    def create_interactive_tutorial(self, tutorial_type: str = "basic") -> None:
        """Create an interactive tutorial overlay."""
        if tutorial_type == "basic":
            self._create_basic_tutorial()
        elif tutorial_type == "advanced":
            self._create_advanced_tutorial()
    
    def _create_basic_tutorial(self):
        """Create basic tutorial for new users."""
        tutorial_steps = [
            {
                'title': 'Welcome to SVD Image Compression',
                'content': 'This tutorial will guide you through the basic features.',
                'target': None
            },
            {
                'title': 'Upload Your Image',
                'content': 'Start by uploading an image using the drag-and-drop zone.',
                'target': 'upload_zone'
            },
            {
                'title': 'Adjust Compression Level',
                'content': 'Use the k-value slider to control compression level.',
                'target': 'k_slider'
            },
            {
                'title': 'View Results',
                'content': 'Compare original and compressed images side by side.',
                'target': 'results_display'
            },
            {
                'title': 'Check Quality Metrics',
                'content': 'Review PSNR, SSIM, and compression ratio.',
                'target': 'metrics_display'
            }
        ]
        
        # Store tutorial state
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        
        current_step = tutorial_steps[st.session_state.tutorial_step]
        
        # Tutorial overlay
        st.markdown(
            f"""
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border: 2px solid #3b82f6;
                border-radius: 12px;
                padding: 20px;
                max-width: 300px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                z-index: 1000;
            ">
                <h4 style="margin: 0 0 10px 0; color: #3b82f6;">
                    Step {st.session_state.tutorial_step + 1} of {len(tutorial_steps)}
                </h4>
                <h5 style="margin: 0 0 8px 0;">{current_step['title']}</h5>
                <p style="margin: 0 0 15px 0; font-size: 0.9rem;">
                    {current_step['content']}
                </p>
                <div style="display: flex; justify-content: space-between;">
                    <button onclick="previousTutorialStep()" 
                            style="background: #6b7280; color: white; border: none; 
                                   padding: 8px 16px; border-radius: 6px; cursor: pointer;"
                            {'disabled' if st.session_state.tutorial_step == 0 else ''}>
                        Previous
                    </button>
                    <button onclick="nextTutorialStep()" 
                            style="background: #3b82f6; color: white; border: none; 
                                   padding: 8px 16px; border-radius: 6px; cursor: pointer;">
                        {'Finish' if st.session_state.tutorial_step == len(tutorial_steps) - 1 else 'Next'}
                    </button>
                </div>
            </div>
            
            <script>
            function nextTutorialStep() {{
                // This would need to be implemented with Streamlit's session state
                console.log('Next tutorial step');
            }}
            
            function previousTutorialStep() {{
                // This would need to be implemented with Streamlit's session state
                console.log('Previous tutorial step');
            }}
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def enable_keyboard_shortcuts(self) -> None:
        """Enable keyboard shortcuts for the application."""
        if not st.session_state.tooltip_system.get('keyboard_shortcuts_enabled', True):
            return
        
        # JavaScript for keyboard shortcuts
        st.markdown(
            """
            <script>
            document.addEventListener('keydown', function(event) {
                // Prevent shortcuts when typing in input fields
                if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                    return;
                }
                
                switch(event.key) {
                    case 'h':
                        // Toggle help mode
                        console.log('Toggle help mode');
                        break;
                    case '?':
                        // Show keyboard shortcuts
                        console.log('Show keyboard shortcuts');
                        break;
                    case 'Escape':
                        // Close help panels
                        console.log('Close help panels');
                        break;
                    case 'ArrowUp':
                        if (event.shiftKey) {
                            console.log('Increase k-value by 10');
                        } else {
                            console.log('Increase k-value by 1');
                        }
                        event.preventDefault();
                        break;
                    case 'ArrowDown':
                        if (event.shiftKey) {
                            console.log('Decrease k-value by 10');
                        } else {
                            console.log('Decrease k-value by 1');
                        }
                        event.preventDefault();
                        break;
                    case 'r':
                        console.log('Toggle real-time preview');
                        break;
                    case 'g':
                        console.log('Switch to grayscale mode');
                        break;
                    case 'c':
                        console.log('Switch to color mode');
                        break;
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                        console.log('Apply preset ' + event.key);
                        break;
                }
            });
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def create_help_overlay(self) -> None:
        """Create a help overlay that can be toggled on/off."""
        if st.session_state.tooltip_system.get('help_mode', False):
            st.markdown(
                """
                <div style="
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    z-index: 999;
                    pointer-events: none;
                "></div>
                
                <div style="
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    max-width: 500px;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                    z-index: 1000;
                ">
                    <h2 style="margin: 0 0 20px 0; color: #3b82f6;">Help Mode Active</h2>
                    <p>Click on any element to get contextual help and tips.</p>
                    <p>Press <kbd>Escape</kbd> or click outside to exit help mode.</p>
                    <button onclick="exitHelpMode()" style="
                        background: #3b82f6;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 6px;
                        cursor: pointer;
                        margin-top: 15px;
                    ">Exit Help Mode</button>
                </div>
                
                <script>
                function exitHelpMode() {
                    // This would need to be implemented with Streamlit's session state
                    console.log('Exit help mode');
                }
                </script>
                """,
                unsafe_allow_html=True
            )
    
    def create_smart_tooltip(self, element_id: str, content: str, 
                           auto_show: bool = False, context_aware: bool = True) -> str:
        """Create a smart tooltip that adapts based on user behavior and context."""
        tooltip_data = {
            'id': element_id,
            'content': content,
            'auto_show': auto_show,
            'context_aware': context_aware,
            'show_count': 0,
            'last_shown': None
        }
        
        # Store tooltip data in session state
        if 'smart_tooltips' not in st.session_state:
            st.session_state.smart_tooltips = {}
        
        st.session_state.smart_tooltips[element_id] = tooltip_data
        
        # Generate smart tooltip HTML
        return f"""
        <div class="smart-tooltip" data-tooltip-id="{element_id}">
            {content}
        </div>
        
        <script>
        (function() {{
            const element = document.querySelector('[data-tooltip-id="{element_id}"]');
            if (element) {{
                // Smart tooltip behavior
                let showTimeout;
                let hideTimeout;
                
                element.addEventListener('mouseenter', function() {{
                    clearTimeout(hideTimeout);
                    showTimeout = setTimeout(function() {{
                        showSmartTooltip('{element_id}');
                    }}, {st.session_state.tooltip_system.get('tooltip_delay', 500)});
                }});
                
                element.addEventListener('mouseleave', function() {{
                    clearTimeout(showTimeout);
                    hideTimeout = setTimeout(function() {{
                        hideSmartTooltip('{element_id}');
                    }}, 100);
                }});
            }}
        }})();
        
        function showSmartTooltip(elementId) {{
            // Implementation would depend on Streamlit integration
            console.log('Show smart tooltip for:', elementId);
        }}
        
        function hideSmartTooltip(elementId) {{
            console.log('Hide smart tooltip for:', elementId);
        }}
        </script>
        """


# Global tooltip system instance
tooltip_system = TooltipSystem()


def create_tooltip(content: str, tooltip_text: str, position: str = "top") -> str:
    """Convenience function to create a tooltip."""
    return tooltip_system.create_tooltip(content, tooltip_text, position)


def create_help_button(topic: str, subtopic: str = None, button_text: str = "?") -> None:
    """Convenience function to create a help button."""
    tooltip_system.create_help_button(topic, subtopic, button_text)


def enable_tooltips_and_help() -> None:
    """Enable the complete tooltip and help system."""
    tooltip_system.enable_keyboard_shortcuts()
    tooltip_system.create_help_overlay()
    tooltip_system.create_contextual_help_panel()