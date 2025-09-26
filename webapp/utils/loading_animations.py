"""
Enhanced loading animations and progress feedback components for the SVD Image Compression webapp.
"""

import streamlit as st
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager


class LoadingAnimations:
    """Enhanced loading animations with smooth transitions and progress feedback."""
    
    def __init__(self):
        self.animation_styles = {
            'spinner': self._create_spinner_animation,
            'pulse': self._create_pulse_animation,
            'wave': self._create_wave_animation,
            'dots': self._create_dots_animation,
            'progress_ring': self._create_progress_ring_animation,
            'skeleton': self._create_skeleton_animation
        }
    
    def show_loading_animation(self, 
                             animation_type: str = 'spinner',
                             text: str = "Processing...",
                             progress: Optional[float] = None,
                             show_percentage: bool = True,
                             color_scheme: str = 'blue') -> Dict[str, Any]:
        """
        Display a loading animation with optional progress.
        
        Args:
            animation_type: Type of animation ('spinner', 'pulse', 'wave', 'dots', 'progress_ring', 'skeleton')
            text: Loading text to display
            progress: Progress value (0.0 to 1.0) for progress-based animations
            show_percentage: Whether to show percentage for progress animations
            color_scheme: Color scheme ('blue', 'green', 'purple', 'orange')
            
        Returns:
            Dictionary with animation controls
        """
        
        # Color schemes
        colors = {
            'blue': {'primary': '#3b82f6', 'secondary': '#93c5fd', 'background': '#eff6ff'},
            'green': {'primary': '#10b981', 'secondary': '#6ee7b7', 'background': '#ecfdf5'},
            'purple': {'primary': '#8b5cf6', 'secondary': '#c4b5fd', 'background': '#f5f3ff'},
            'orange': {'primary': '#f59e0b', 'secondary': '#fcd34d', 'background': '#fffbeb'}
        }
        
        color_config = colors.get(color_scheme, colors['blue'])
        
        # Create animation container
        animation_container = st.empty()
        
        # Generate animation HTML
        if animation_type in self.animation_styles:
            animation_html = self.animation_styles[animation_type](
                text, progress, show_percentage, color_config
            )
        else:
            animation_html = self._create_spinner_animation(text, progress, show_percentage, color_config)
        
        # Display animation
        animation_container.markdown(animation_html, unsafe_allow_html=True)
        
        return {
            'container': animation_container,
            'update_progress': lambda p: self._update_progress(animation_container, animation_type, text, p, show_percentage, color_config),
            'update_text': lambda t: self._update_text(animation_container, animation_type, t, progress, show_percentage, color_config),
            'complete': lambda: self._show_completion(animation_container, color_config),
            'hide': lambda: animation_container.empty()
        }
    
    def _create_spinner_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a spinner loading animation."""
        
        progress_html = ""
        if progress is not None:
            percentage = int(progress * 100)
            progress_html = f"""
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {percentage}%; background: linear-gradient(90deg, {colors['primary']} 0%, {colors['secondary']} 100%);"></div>
            </div>
            {f'<div class="progress-percentage">{percentage}%</div>' if show_percentage else ''}
            """
        
        return f"""
        <div class="loading-animation-container" style="background: {colors['background']};">
            <div class="spinner-container">
                <div class="spinner" style="border-top-color: {colors['primary']};"></div>
            </div>
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
            {progress_html}
        </div>
        
        <style>
        .loading-animation-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .spinner-container {{
            margin-bottom: 1.5rem;
        }}
        
        .spinner {{
            border: 4px solid #e5e7eb;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .loading-text {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        .progress-bar-container {{
            width: 300px;
            height: 8px;
            background-color: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0 0.5rem 0;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .progress-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}
        
        .progress-percentage {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #6b7280;
            font-family: monospace;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """
    
    def _create_pulse_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a pulsing loading animation."""
        
        progress_html = ""
        if progress is not None:
            percentage = int(progress * 100)
            progress_html = f"""
            <div class="pulse-progress-ring">
                <svg class="progress-ring" width="80" height="80">
                    <circle class="progress-ring-circle" 
                            cx="40" cy="40" r="35"
                            style="stroke: {colors['primary']}; stroke-dasharray: {2 * 3.14159 * 35}; stroke-dashoffset: {2 * 3.14159 * 35 * (1 - progress)};">
                    </circle>
                </svg>
                {f'<div class="ring-percentage" style="color: {colors["primary"]};">{percentage}%</div>' if show_percentage else ''}
            </div>
            """
        else:
            progress_html = f"""
            <div class="pulse-circle" style="background: {colors['primary']};"></div>
            """
        
        return f"""
        <div class="loading-animation-container pulse-container" style="background: {colors['background']};">
            {progress_html}
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
        </div>
        
        <style>
        .pulse-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .pulse-circle {{
            width: 60px;
            height: 60px;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
            margin-bottom: 1.5rem;
        }}
        
        .pulse-progress-ring {{
            position: relative;
            margin-bottom: 1.5rem;
        }}
        
        .progress-ring {{
            transform: rotate(-90deg);
        }}
        
        .progress-ring-circle {{
            fill: none;
            stroke-width: 4;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.5s ease;
        }}
        
        .ring-percentage {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1rem;
            font-weight: 700;
            font-family: monospace;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.2); opacity: 0.7; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        </style>
        """
    
    def _create_wave_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a wave loading animation."""
        
        progress_html = ""
        if progress is not None:
            percentage = int(progress * 100)
            progress_html = f"""
            <div class="wave-progress">
                <div class="wave-fill" style="height: {percentage}%; background: linear-gradient(180deg, {colors['secondary']} 0%, {colors['primary']} 100%);"></div>
                {f'<div class="wave-percentage" style="color: {colors["primary"]};">{percentage}%</div>' if show_percentage else ''}
            </div>
            """
        else:
            progress_html = """
            <div class="wave-container">
                <div class="wave" style="background: linear-gradient(90deg, transparent, {}, transparent);"></div>
                <div class="wave" style="background: linear-gradient(90deg, transparent, {}, transparent); animation-delay: -0.5s;"></div>
                <div class="wave" style="background: linear-gradient(90deg, transparent, {}, transparent); animation-delay: -1s;"></div>
            </div>
            """.format(colors['primary'], colors['secondary'], colors['primary'])
        
        return f"""
        <div class="loading-animation-container wave-loading" style="background: {colors['background']};">
            {progress_html}
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
        </div>
        
        <style>
        .wave-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .wave-container {{
            width: 80px;
            height: 60px;
            position: relative;
            margin-bottom: 1.5rem;
            overflow: hidden;
            border-radius: 10px;
            background: #f3f4f6;
        }}
        
        .wave {{
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            animation: wave 2s linear infinite;
            opacity: 0.7;
        }}
        
        .wave-progress {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 4px solid #e5e7eb;
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }}
        
        .wave-fill {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            transition: height 0.5s ease;
            border-radius: 0 0 50px 50px;
        }}
        
        .wave-percentage {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1rem;
            font-weight: 700;
            font-family: monospace;
            z-index: 2;
        }}
        
        @keyframes wave {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
        </style>
        """
    
    def _create_dots_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a dots loading animation."""
        
        progress_html = ""
        if progress is not None:
            percentage = int(progress * 100)
            num_dots = 8
            active_dots = int((progress * num_dots))
            
            dots_html = ""
            for i in range(num_dots):
                dot_color = colors['primary'] if i < active_dots else '#e5e7eb'
                dots_html += f'<div class="progress-dot" style="background-color: {dot_color};"></div>'
            
            progress_html = f"""
            <div class="dots-progress-container">
                {dots_html}
            </div>
            {f'<div class="dots-percentage" style="color: {colors["primary"]};">{percentage}%</div>' if show_percentage else ''}
            """
        else:
            progress_html = f"""
            <div class="dots-container">
                <div class="dot" style="background-color: {colors['primary']};"></div>
                <div class="dot" style="background-color: {colors['primary']};"></div>
                <div class="dot" style="background-color: {colors['primary']};"></div>
            </div>
            """
        
        return f"""
        <div class="loading-animation-container dots-loading" style="background: {colors['background']};">
            {progress_html}
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
        </div>
        
        <style>
        .dots-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .dots-container {{
            display: flex;
            gap: 8px;
            margin-bottom: 1.5rem;
        }}
        
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: bounce 1.4s ease-in-out infinite both;
        }}
        
        .dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .dot:nth-child(2) {{ animation-delay: -0.16s; }}
        .dot:nth-child(3) {{ animation-delay: 0s; }}
        
        .dots-progress-container {{
            display: flex;
            gap: 6px;
            margin-bottom: 1.5rem;
        }}
        
        .progress-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            transition: background-color 0.3s ease;
        }}
        
        .dots-percentage {{
            font-size: 0.9rem;
            font-weight: 600;
            font-family: monospace;
        }}
        
        @keyframes bounce {{
            0%, 80%, 100% {{ 
                transform: scale(0);
                opacity: 0.5;
            }}
            40% {{ 
                transform: scale(1);
                opacity: 1;
            }}
        }}
        </style>
        """
    
    def _create_progress_ring_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a progress ring animation."""
        
        progress_value = progress if progress is not None else 0.0
        percentage = int(progress_value * 100)
        circumference = 2 * 3.14159 * 35
        stroke_dashoffset = circumference * (1 - progress_value)
        
        return f"""
        <div class="loading-animation-container ring-loading" style="background: {colors['background']};">
            <div class="progress-ring-container">
                <svg class="progress-ring" width="100" height="100">
                    <circle class="progress-ring-bg" 
                            cx="50" cy="50" r="35"
                            style="stroke: #e5e7eb;">
                    </circle>
                    <circle class="progress-ring-fill" 
                            cx="50" cy="50" r="35"
                            style="stroke: {colors['primary']}; stroke-dasharray: {circumference}; stroke-dashoffset: {stroke_dashoffset};">
                    </circle>
                </svg>
                {f'<div class="ring-text" style="color: {colors["primary"]};">{percentage}%</div>' if show_percentage else ''}
            </div>
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
        </div>
        
        <style>
        .ring-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .progress-ring-container {{
            position: relative;
            margin-bottom: 1.5rem;
        }}
        
        .progress-ring {{
            transform: rotate(-90deg);
        }}
        
        .progress-ring-bg,
        .progress-ring-fill {{
            fill: none;
            stroke-width: 6;
            stroke-linecap: round;
        }}
        
        .progress-ring-fill {{
            transition: stroke-dashoffset 0.5s ease;
        }}
        
        .ring-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.2rem;
            font-weight: 700;
            font-family: monospace;
        }}
        </style>
        """
    
    def _create_skeleton_animation(self, text: str, progress: Optional[float], show_percentage: bool, colors: Dict) -> str:
        """Create a skeleton loading animation."""
        
        progress_html = ""
        if progress is not None:
            percentage = int(progress * 100)
            progress_html = f"""
            <div class="skeleton-progress">
                <div class="skeleton-bar" style="width: {percentage}%; background: {colors['primary']};"></div>
            </div>
            {f'<div class="skeleton-percentage" style="color: {colors["primary"]};">{percentage}%</div>' if show_percentage else ''}
            """
        
        return f"""
        <div class="loading-animation-container skeleton-loading" style="background: {colors['background']};">
            <div class="skeleton-container">
                <div class="skeleton-line skeleton-line-1"></div>
                <div class="skeleton-line skeleton-line-2"></div>
                <div class="skeleton-line skeleton-line-3"></div>
            </div>
            {progress_html}
            <div class="loading-text" style="color: {colors['primary']};">{text}</div>
        </div>
        
        <style>
        .skeleton-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .skeleton-container {{
            margin-bottom: 1.5rem;
        }}
        
        .skeleton-line {{
            height: 12px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: skeleton-shimmer 2s infinite;
            border-radius: 6px;
            margin: 8px 0;
        }}
        
        .skeleton-line-1 {{ width: 200px; }}
        .skeleton-line-2 {{ width: 150px; }}
        .skeleton-line-3 {{ width: 180px; }}
        
        .skeleton-progress {{
            width: 250px;
            height: 6px;
            background-color: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0 0.5rem 0;
        }}
        
        .skeleton-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        
        .skeleton-percentage {{
            font-size: 0.9rem;
            font-weight: 600;
            font-family: monospace;
        }}
        
        @keyframes skeleton-shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        </style>
        """
    
    def _update_progress(self, container, animation_type: str, text: str, progress: float, show_percentage: bool, colors: Dict):
        """Update progress for an existing animation."""
        if animation_type in self.animation_styles:
            updated_html = self.animation_styles[animation_type](text, progress, show_percentage, colors)
            container.markdown(updated_html, unsafe_allow_html=True)
    
    def _update_text(self, container, animation_type: str, text: str, progress: Optional[float], show_percentage: bool, colors: Dict):
        """Update text for an existing animation."""
        if animation_type in self.animation_styles:
            updated_html = self.animation_styles[animation_type](text, progress, show_percentage, colors)
            container.markdown(updated_html, unsafe_allow_html=True)
    
    def _show_completion(self, container, colors: Dict):
        """Show completion animation."""
        completion_html = f"""
        <div class="completion-animation" style="background: {colors['background']};">
            <div class="completion-icon" style="color: {colors['primary']};">✅</div>
            <div class="completion-text" style="color: {colors['primary']};">Complete!</div>
        </div>
        
        <style>
        .completion-animation {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            animation: completionBounce 0.6s ease-out;
        }}
        
        .completion-icon {{
            font-size: 3rem;
            margin-bottom: 0.5rem;
            animation: checkmark 0.6s ease-out;
        }}
        
        .completion-text {{
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        @keyframes completionBounce {{
            0% {{ transform: scale(0.3); opacity: 0; }}
            50% {{ transform: scale(1.05); }}
            70% {{ transform: scale(0.9); }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        
        @keyframes checkmark {{
            0% {{ transform: scale(0); }}
            50% {{ transform: scale(1.2); }}
            100% {{ transform: scale(1); }}
        }}
        </style>
        """
        container.markdown(completion_html, unsafe_allow_html=True)


class ProgressManager:
    """Manages multiple progress indicators and loading states."""
    
    def __init__(self):
        self.active_operations = {}
        self.animations = LoadingAnimations()
    
    def start_operation(self, 
                       operation_id: str,
                       title: str,
                       steps: List[str],
                       animation_type: str = 'spinner',
                       color_scheme: str = 'blue') -> Dict[str, Any]:
        """
        Start a multi-step operation with progress tracking.
        
        Args:
            operation_id: Unique identifier for the operation
            title: Title for the operation
            steps: List of step descriptions
            animation_type: Type of loading animation
            color_scheme: Color scheme for the animation
            
        Returns:
            Dictionary with operation controls
        """
        
        # Create operation state
        operation_state = {
            'title': title,
            'steps': steps,
            'current_step': 0,
            'progress': 0.0,
            'status': 'running',
            'animation_type': animation_type,
            'color_scheme': color_scheme
        }
        
        # Start animation
        animation_controls = self.animations.show_loading_animation(
            animation_type=animation_type,
            text=f"{title}: {steps[0] if steps else 'Starting...'}",
            progress=0.0,
            color_scheme=color_scheme
        )
        
        operation_state['animation_controls'] = animation_controls
        self.active_operations[operation_id] = operation_state
        
        return {
            'next_step': lambda: self._next_step(operation_id),
            'update_progress': lambda p: self._update_operation_progress(operation_id, p),
            'complete': lambda: self._complete_operation(operation_id),
            'error': lambda msg: self._error_operation(operation_id, msg),
            'get_progress': lambda: self._get_operation_progress(operation_id)
        }
    
    def _next_step(self, operation_id: str):
        """Move to the next step in the operation."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        
        if operation['current_step'] < len(operation['steps']) - 1:
            operation['current_step'] += 1
            operation['progress'] = operation['current_step'] / len(operation['steps'])
            
            # Update animation
            current_step_text = operation['steps'][operation['current_step']]
            operation['animation_controls']['update_text'](
                f"{operation['title']}: {current_step_text}"
            )
            operation['animation_controls']['update_progress'](operation['progress'])
    
    def _update_operation_progress(self, operation_id: str, progress: float):
        """Update progress within the current step."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        
        # Calculate overall progress
        step_progress = operation['current_step'] / len(operation['steps'])
        step_increment = 1.0 / len(operation['steps'])
        overall_progress = step_progress + (progress * step_increment)
        
        operation['progress'] = min(overall_progress, 1.0)
        operation['animation_controls']['update_progress'](operation['progress'])
    
    def _complete_operation(self, operation_id: str):
        """Complete the operation."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        operation['status'] = 'completed'
        operation['progress'] = 1.0
        
        # Show completion animation
        operation['animation_controls']['complete']()
        
        # Clean up after delay
        def cleanup():
            time.sleep(2)
            if operation_id in self.active_operations:
                self.active_operations[operation_id]['animation_controls']['hide']()
                del self.active_operations[operation_id]
        
        threading.Thread(target=cleanup, daemon=True).start()
    
    def _error_operation(self, operation_id: str, error_message: str):
        """Handle operation error."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        operation['status'] = 'error'
        
        # Show error state
        error_html = f"""
        <div class="error-animation">
            <div class="error-icon">❌</div>
            <div class="error-text">Error: {error_message}</div>
        </div>
        
        <style>
        .error-animation {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            background: #fee2e2;
            border: 2px solid #ef4444;
            animation: errorShake 0.6s ease-out;
        }}
        
        .error-icon {{
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }}
        
        .error-text {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #991b1b;
            text-align: center;
        }}
        
        @keyframes errorShake {{
            0%, 100% {{ transform: translateX(0); }}
            10%, 30%, 50%, 70%, 90% {{ transform: translateX(-5px); }}
            20%, 40%, 60%, 80% {{ transform: translateX(5px); }}
        }}
        </style>
        """
        
        operation['animation_controls']['container'].markdown(error_html, unsafe_allow_html=True)
    
    def _get_operation_progress(self, operation_id: str) -> Dict[str, Any]:
        """Get current operation progress."""
        if operation_id not in self.active_operations:
            return {'status': 'not_found'}
        
        operation = self.active_operations[operation_id]
        return {
            'status': operation['status'],
            'progress': operation['progress'],
            'current_step': operation['current_step'],
            'total_steps': len(operation['steps']),
            'current_step_name': operation['steps'][operation['current_step']] if operation['current_step'] < len(operation['steps']) else 'Complete'
        }


@contextmanager
def loading_context(text: str = "Processing...", 
                   animation_type: str = 'spinner',
                   color_scheme: str = 'blue'):
    """
    Context manager for showing loading animation during code execution.
    
    Usage:
        with loading_context("Processing image..."):
            # Your processing code here
            time.sleep(2)
    """
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type=animation_type,
        text=text,
        color_scheme=color_scheme
    )
    
    try:
        yield animation_controls
    finally:
        animation_controls['complete']()
        time.sleep(1)
        animation_controls['hide']()


def create_smooth_transition(from_content: str, to_content: str, duration: float = 0.5) -> str:
    """
    Create a smooth transition between two content states.
    
    Args:
        from_content: Initial content HTML
        to_content: Target content HTML
        duration: Transition duration in seconds
        
    Returns:
        HTML with transition animation
    """
    
    transition_id = f"transition_{int(time.time() * 1000)}"
    
    return f"""
    <div id="{transition_id}" class="smooth-transition-container">
        <div class="transition-content transition-from">{from_content}</div>
        <div class="transition-content transition-to">{to_content}</div>
    </div>
    
    <style>
    .smooth-transition-container {{
        position: relative;
        overflow: hidden;
    }}
    
    .transition-content {{
        transition: all {duration}s ease-in-out;
    }}
    
    .transition-from {{
        opacity: 1;
        transform: translateX(0);
    }}
    
    .transition-to {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        opacity: 0;
        transform: translateX(100%);
    }}
    
    .transition-from.fade-out {{
        opacity: 0;
        transform: translateX(-100%);
    }}
    
    .transition-to.fade-in {{
        opacity: 1;
        transform: translateX(0);
    }}
    </style>
    
    <script>
    setTimeout(function() {{
        const container = document.getElementById('{transition_id}');
        if (container) {{
            const fromElement = container.querySelector('.transition-from');
            const toElement = container.querySelector('.transition-to');
            
            if (fromElement && toElement) {{
                fromElement.classList.add('fade-out');
                toElement.classList.add('fade-in');
                
                setTimeout(function() {{
                    fromElement.style.display = 'none';
                    toElement.style.position = 'relative';
                }}, {duration * 1000});
            }}
        }}
    }}, 100);
    </script>
    """


# Convenience functions for easy use
def show_spinner(text: str = "Processing...", progress: Optional[float] = None, color: str = 'blue'):
    """Show a spinner loading animation."""
    animations = LoadingAnimations()
    return animations.show_loading_animation('spinner', text, progress, color_scheme=color)


def show_progress_ring(text: str = "Processing...", progress: float = 0.0, color: str = 'blue'):
    """Show a progress ring animation."""
    animations = LoadingAnimations()
    return animations.show_loading_animation('progress_ring', text, progress, color_scheme=color)


def show_pulse(text: str = "Processing...", color: str = 'blue'):
    """Show a pulse loading animation."""
    animations = LoadingAnimations()
    return animations.show_loading_animation('pulse', text, color_scheme=color)


def create_multi_step_progress(operation_id: str, title: str, steps: List[str], animation_type: str = 'spinner'):
    """Create a multi-step progress manager."""
    manager = ProgressManager()
    return manager.start_operation(operation_id, title, steps, animation_type)