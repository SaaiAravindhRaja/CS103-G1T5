"""
Enhanced animations and micro-interactions for the SVD Image Compression webapp.
"""

import streamlit as st
import time
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager


class AnimationManager:
    """Manages animations and micro-interactions throughout the application."""
    
    def __init__(self):
        self.animation_queue = []
        self.active_animations = {}
    
    def add_entrance_animation(self, element_key: str, animation_type: str = "fade-in", delay: float = 0):
        """Add entrance animation to an element."""
        animation_class = f"animate-{animation_type}"
        if delay > 0:
            animation_class += f" animation-delay-{int(delay * 1000)}"
        
        return f'<div class="{animation_class}" key="{element_key}">'
    
    def add_hover_effect(self, element_key: str, effect_type: str = "lift"):
        """Add hover effect to an element."""
        effect_class = f"hover-{effect_type} transition-smooth"
        return f'<div class="{effect_class}" key="{element_key}">'
    
    def create_staggered_animation(self, items: List[str], animation_type: str = "fade-in", stagger_delay: float = 0.1):
        """Create staggered animations for a list of items."""
        animated_items = []
        for i, item in enumerate(items):
            delay = i * stagger_delay
            animation_class = f"animate-{animation_type}"
            style = f"animation-delay: {delay}s;"
            animated_items.append(f'<div class="{animation_class}" style="{style}">{item}</div>')
        return animated_items
    
    def show_loading_state(self, container, message: str = "Loading...", animation_type: str = "spinner"):
        """Show animated loading state."""
        loading_html = f"""
        <div class="loading-state animate-fade-in">
            <div class="loading-{animation_type}"></div>
            <p class="loading-message">{message}</p>
        </div>
        
        <style>
        .loading-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
        }}
        
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #e5e7eb;
            border-top: 4px solid var(--secondary-500);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }}
        
        .loading-pulse {{
            width: 40px;
            height: 40px;
            background: var(--secondary-500);
            border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
            margin-bottom: 1rem;
        }}
        
        .loading-dots {{
            display: flex;
            gap: 4px;
            margin-bottom: 1rem;
        }}
        
        .loading-dots::before,
        .loading-dots::after,
        .loading-dots {{
            content: '';
            width: 8px;
            height: 8px;
            background: var(--secondary-500);
            border-radius: 50%;
            animation: bounce 1.4s ease-in-out infinite both;
        }}
        
        .loading-dots::before {{ animation-delay: -0.32s; }}
        .loading-dots::after {{ animation-delay: -0.16s; }}
        
        .loading-message {{
            color: var(--text-secondary);
            font-weight: 500;
            margin: 0;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
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
        container.markdown(loading_html, unsafe_allow_html=True)
    
    def show_success_animation(self, container, message: str = "Success!", duration: float = 3.0):
        """Show success animation with auto-hide."""
        success_html = f"""
        <div class="success-animation animate-bounce-in">
            <div class="success-icon">✅</div>
            <p class="success-message">{message}</p>
        </div>
        
        <style>
        .success-animation {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
            background: var(--accent-50);
            border: 2px solid var(--accent-500);
            border-radius: var(--radius-xl);
            margin: 1rem 0;
        }}
        
        .success-icon {{
            font-size: 3rem;
            margin-bottom: 0.5rem;
            animation: checkmark 0.6s ease-out;
        }}
        
        .success-message {{
            color: var(--accent-600);
            font-weight: 600;
            margin: 0;
        }}
        
        @keyframes checkmark {{
            0% {{ transform: scale(0); }}
            50% {{ transform: scale(1.2); }}
            100% {{ transform: scale(1); }}
        }}
        </style>
        """
        container.markdown(success_html, unsafe_allow_html=True)
        
        # Auto-hide after duration
        if duration > 0:
            time.sleep(duration)
            container.empty()
    
    def show_error_animation(self, container, message: str = "Error occurred", duration: float = 5.0):
        """Show error animation with auto-hide."""
        error_html = f"""
        <div class="error-animation animate-wiggle">
            <div class="error-icon">❌</div>
            <p class="error-message">{message}</p>
        </div>
        
        <style>
        .error-animation {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
            background: #fee2e2;
            border: 2px solid #ef4444;
            border-radius: var(--radius-xl);
            margin: 1rem 0;
        }}
        
        .error-icon {{
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }}
        
        .error-message {{
            color: #991b1b;
            font-weight: 600;
            margin: 0;
        }}
        </style>
        """
        container.markdown(error_html, unsafe_allow_html=True)
        
        # Auto-hide after duration
        if duration > 0:
            time.sleep(duration)
            container.empty()
    
    def create_progress_animation(self, container, progress: float, message: str = "Processing..."):
        """Create animated progress indicator."""
        progress_percent = int(progress * 100)
        
        progress_html = f"""
        <div class="progress-animation animate-fade-in">
            <div class="progress-label">{message}</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {progress_percent}%"></div>
            </div>
            <div class="progress-percentage">{progress_percent}%</div>
        </div>
        
        <style>
        .progress-animation {{
            padding: 1.5rem;
            text-align: center;
        }}
        
        .progress-label {{
            color: var(--text-secondary);
            font-weight: 500;
            margin-bottom: 1rem;
        }}
        
        .progress-bar-container {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: var(--radius-full);
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        
        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--secondary-500), var(--accent-500));
            border-radius: var(--radius-full);
            transition: width 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        .progress-bar-fill::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }}
        
        .progress-percentage {{
            color: var(--text-primary);
            font-weight: 600;
            font-family: monospace;
        }}
        </style>
        """
        container.markdown(progress_html, unsafe_allow_html=True)
    
    def create_notification(self, message: str, notification_type: str = "info", duration: float = 4.0):
        """Create animated notification toast."""
        notification_id = f"notification_{int(time.time() * 1000)}"
        
        type_styles = {
            "success": {"bg": "#ecfdf5", "border": "#10b981", "text": "#065f46", "icon": "✅"},
            "error": {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b", "icon": "❌"},
            "warning": {"bg": "#fffbeb", "border": "#f59e0b", "text": "#92400e", "icon": "⚠️"},
            "info": {"bg": "#eff6ff", "border": "#3b82f6", "text": "#1e40af", "icon": "ℹ️"}
        }
        
        style = type_styles.get(notification_type, type_styles["info"])
        
        notification_html = f"""
        <div id="{notification_id}" class="notification-toast animate-slide-in">
            <div class="notification-content">
                <span class="notification-icon">{style['icon']}</span>
                <span class="notification-message">{message}</span>
            </div>
            <button class="notification-close" onclick="closeNotification('{notification_id}')">&times;</button>
        </div>
        
        <style>
        .notification-toast {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            max-width: 400px;
            background: {style['bg']};
            border: 2px solid {style['border']};
            color: {style['text']};
            border-radius: var(--radius-lg);
            padding: 1rem;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }}
        
        .notification-content {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .notification-icon {{
            font-size: 1.25rem;
        }}
        
        .notification-message {{
            font-weight: 500;
        }}
        
        .notification-close {{
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: {style['text']};
            opacity: 0.7;
            transition: opacity 0.2s;
        }}
        
        .notification-close:hover {{
            opacity: 1;
        }}
        </style>
        
        <script>
        function closeNotification(id) {{
            const notification = document.getElementById(id);
            if (notification) {{
                notification.style.animation = 'slideOut 0.3s ease-in forwards';
                setTimeout(() => notification.remove(), 300);
            }}
        }}
        
        // Auto-hide after duration
        setTimeout(() => closeNotification('{notification_id}'), {duration * 1000});
        
        @keyframes slideOut {{
            to {{
                transform: translateX(100%);
                opacity: 0;
            }}
        }}
        </script>
        """
        
        st.markdown(notification_html, unsafe_allow_html=True)
    
    def create_skeleton_loader(self, lines: int = 3, width_pattern: List[str] = None):
        """Create skeleton loading animation."""
        if width_pattern is None:
            width_pattern = ["100%", "75%", "90%"]
        
        skeleton_lines = []
        for i in range(lines):
            width = width_pattern[i % len(width_pattern)]
            skeleton_lines.append(f'<div class="skeleton-line" style="width: {width};"></div>')
        
        skeleton_html = f"""
        <div class="skeleton-loader animate-fade-in">
            {''.join(skeleton_lines)}
        </div>
        
        <style>
        .skeleton-loader {{
            padding: 1rem;
        }}
        
        .skeleton-line {{
            height: 1rem;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: var(--radius-base);
            margin-bottom: 0.75rem;
        }}
        
        .skeleton-line:last-child {{
            margin-bottom: 0;
        }}
        </style>
        """
        
        return skeleton_html
    
    @contextmanager
    def loading_context(self, container, message: str = "Loading...", animation_type: str = "spinner"):
        """Context manager for loading states."""
        self.show_loading_state(container, message, animation_type)
        try:
            yield
        finally:
            container.empty()
    
    def add_micro_interaction(self, element_html: str, interaction_type: str = "hover-lift"):
        """Add micro-interaction to any HTML element."""
        interaction_class = f"micro-interaction {interaction_type}"
        
        # Wrap the element with interaction class
        if element_html.startswith('<div'):
            # Add class to existing div
            element_html = element_html.replace('<div', f'<div class="{interaction_class}"', 1)
        else:
            # Wrap in new div
            element_html = f'<div class="{interaction_class}">{element_html}</div>'
        
        return element_html
    
    def create_animated_counter(self, container, target_value: float, duration: float = 2.0, 
                              prefix: str = "", suffix: str = "", decimal_places: int = 0):
        """Create animated counter that counts up to target value."""
        counter_id = f"counter_{int(time.time() * 1000)}"
        
        counter_html = f"""
        <div id="{counter_id}" class="animated-counter">0{suffix}</div>
        
        <style>
        .animated-counter {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--secondary-500);
            text-align: center;
            font-family: monospace;
        }}
        </style>
        
        <script>
        function animateCounter(id, target, duration, prefix, suffix, decimals) {{
            const element = document.getElementById(id);
            const start = 0;
            const increment = target / (duration * 60); // 60 FPS
            let current = start;
            
            const timer = setInterval(() => {{
                current += increment;
                if (current >= target) {{
                    current = target;
                    clearInterval(timer);
                }}
                
                const displayValue = decimals > 0 ? current.toFixed(decimals) : Math.floor(current);
                element.textContent = prefix + displayValue + suffix;
            }}, 1000 / 60);
        }}
        
        animateCounter('{counter_id}', {target_value}, {duration}, '{prefix}', '{suffix}', {decimal_places});
        </script>
        """
        
        container.markdown(counter_html, unsafe_allow_html=True)


# Global animation manager instance
animation_manager = AnimationManager()


def add_entrance_animation(element_key: str, animation_type: str = "fade-in", delay: float = 0):
    """Add entrance animation to an element."""
    return animation_manager.add_entrance_animation(element_key, animation_type, delay)


def add_hover_effect(element_key: str, effect_type: str = "lift"):
    """Add hover effect to an element."""
    return animation_manager.add_hover_effect(element_key, effect_type)


def show_loading_state(container, message: str = "Loading...", animation_type: str = "spinner"):
    """Show animated loading state."""
    return animation_manager.show_loading_state(container, message, animation_type)


def show_success_animation(container, message: str = "Success!", duration: float = 3.0):
    """Show success animation with auto-hide."""
    return animation_manager.show_success_animation(container, message, duration)


def show_error_animation(container, message: str = "Error occurred", duration: float = 5.0):
    """Show error animation with auto-hide."""
    return animation_manager.show_error_animation(container, message, duration)


def create_notification(message: str, notification_type: str = "info", duration: float = 4.0):
    """Create animated notification toast."""
    return animation_manager.create_notification(message, notification_type, duration)


def create_skeleton_loader(lines: int = 3, width_pattern: List[str] = None):
    """Create skeleton loading animation."""
    return animation_manager.create_skeleton_loader(lines, width_pattern)


def create_animated_counter(container, target_value: float, duration: float = 2.0, 
                          prefix: str = "", suffix: str = "", decimal_places: int = 0):
    """Create animated counter that counts up to target value."""
    return animation_manager.create_animated_counter(
        container, target_value, duration, prefix, suffix, decimal_places
    )