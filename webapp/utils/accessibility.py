"""
Accessibility utilities for the SVD Image Compression webapp.
Ensures ARIA labels, keyboard navigation, and screen reader compatibility.
"""

import streamlit as st
from typing import Dict, List, Optional, Any


class AccessibilityManager:
    """Manages accessibility features throughout the application."""
    
    def __init__(self):
        self.aria_labels = {}
        self.focus_management = {}
        self.screen_reader_announcements = []
    
    def add_aria_label(self, element_id: str, label: str, description: str = None):
        """Add ARIA label to an element."""
        aria_attrs = f'aria-label="{label}"'
        if description:
            aria_attrs += f' aria-describedby="{element_id}-desc"'
        
        self.aria_labels[element_id] = {
            'label': label,
            'description': description,
            'attrs': aria_attrs
        }
        
        return aria_attrs
    
    def create_accessible_button(self, text: str, onclick_action: str = None, 
                                button_type: str = "button", disabled: bool = False,
                                aria_label: str = None, aria_describedby: str = None):
        """Create an accessible button with proper ARIA attributes."""
        
        aria_label_attr = f'aria-label="{aria_label}"' if aria_label else f'aria-label="{text}"'
        aria_describedby_attr = f'aria-describedby="{aria_describedby}"' if aria_describedby else ""
        disabled_attr = "disabled" if disabled else ""
        onclick_attr = f'onclick="{onclick_action}"' if onclick_action else ""
        
        button_html = f"""
        <button 
            type="{button_type}"
            class="accessible-button btn-primary focus-ring"
            {aria_label_attr}
            {aria_describedby_attr}
            {disabled_attr}
            {onclick_attr}
        >
            {text}
        </button>
        """
        
        return button_html
    
    def create_accessible_form_field(self, field_type: str, name: str, label: str,
                                   required: bool = False, placeholder: str = None,
                                   help_text: str = None, error_message: str = None):
        """Create an accessible form field with proper labels and descriptions."""
        
        field_id = f"field-{name}"
        help_id = f"{field_id}-help" if help_text else None
        error_id = f"{field_id}-error" if error_message else None
        
        # Build aria-describedby
        describedby_parts = []
        if help_id:
            describedby_parts.append(help_id)
        if error_id:
            describedby_parts.append(error_id)
        
        aria_describedby = f'aria-describedby="{" ".join(describedby_parts)}"' if describedby_parts else ""
        aria_required = 'aria-required="true"' if required else ""
        aria_invalid = 'aria-invalid="true"' if error_message else ""
        
        placeholder_attr = f'placeholder="{placeholder}"' if placeholder else ""
        
        form_field_html = f"""
        <div class="form-field-container">
            <label for="{field_id}" class="form-label">
                {label}
                {' <span class="required-indicator" aria-label="required">*</span>' if required else ''}
            </label>
            
            <input 
                type="{field_type}"
                id="{field_id}"
                name="{name}"
                class="form-input focus-ring"
                {aria_describedby}
                {aria_required}
                {aria_invalid}
                {placeholder_attr}
            />
            
            {f'<div id="{help_id}" class="form-help-text">{help_text}</div>' if help_text else ''}
            {f'<div id="{error_id}" class="form-error-text" role="alert">{error_message}</div>' if error_message else ''}
        </div>
        
        <style>
        .form-field-container {{
            margin-bottom: 1.5rem;
        }}
        
        .form-label {{
            display: block;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}
        
        .required-indicator {{
            color: #ef4444;
            margin-left: 0.25rem;
        }}
        
        .form-help-text {{
            font-size: var(--font-size-sm);
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}
        
        .form-error-text {{
            font-size: var(--font-size-sm);
            color: #ef4444;
            margin-top: 0.25rem;
            font-weight: 500;
        }}
        </style>
        """
        
        return form_field_html
    
    def create_accessible_image(self, src: str, alt_text: str, caption: str = None,
                              long_description: str = None, decorative: bool = False):
        """Create an accessible image with proper alt text and descriptions."""
        
        if decorative:
            alt_attr = 'alt="" role="presentation"'
        else:
            alt_attr = f'alt="{alt_text}"'
        
        img_id = f"img-{hash(src) % 10000}"
        longdesc_id = f"{img_id}-longdesc" if long_description else None
        
        aria_describedby = f'aria-describedby="{longdesc_id}"' if longdesc_id else ""
        
        image_html = f"""
        <figure class="accessible-image-container">
            <img 
                id="{img_id}"
                src="{src}"
                {alt_attr}
                {aria_describedby}
                class="accessible-image"
            />
            {f'<figcaption class="image-caption">{caption}</figcaption>' if caption else ''}
            {f'<div id="{longdesc_id}" class="image-long-description sr-only">{long_description}</div>' if long_description else ''}
        </figure>
        
        <style>
        .accessible-image-container {{
            margin: 1rem 0;
        }}
        
        .accessible-image {{
            max-width: 100%;
            height: auto;
            border-radius: var(--radius-lg);
        }}
        
        .image-caption {{
            font-size: var(--font-size-sm);
            color: var(--text-secondary);
            text-align: center;
            margin-top: 0.5rem;
            font-style: italic;
        }}
        
        .image-long-description {{
            /* Hidden but available to screen readers */
        }}
        
        .sr-only {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }}
        </style>
        """
        
        return image_html
    
    def create_accessible_navigation(self, nav_items: List[Dict[str, str]], current_page: str = None):
        """Create accessible navigation with proper ARIA attributes."""
        
        nav_html = """
        <nav role="navigation" aria-label="Main navigation" class="accessible-nav">
            <ul class="nav-list" role="menubar">
        """
        
        for item in nav_items:
            name = item.get('name', '')
            url = item.get('url', '#')
            icon = item.get('icon', '')
            description = item.get('description', '')
            
            is_current = name == current_page
            aria_current = 'aria-current="page"' if is_current else ''
            
            nav_html += f"""
                <li class="nav-item" role="none">
                    <a 
                        href="{url}"
                        class="nav-link {'nav-link-current' if is_current else ''}"
                        role="menuitem"
                        {aria_current}
                        aria-label="{name}: {description}"
                    >
                        {f'<span class="nav-icon" aria-hidden="true">{icon}</span>' if icon else ''}
                        <span class="nav-text">{name}</span>
                    </a>
                </li>
            """
        
        nav_html += """
            </ul>
        </nav>
        
        <style>
        .accessible-nav {{
            margin-bottom: 2rem;
        }}
        
        .nav-list {{
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
            gap: 0.5rem;
            background: #f1f5f9;
            border-radius: var(--radius-lg);
            padding: 0.5rem;
        }}
        
        .nav-item {{
            flex: 1;
        }}
        
        .nav-link {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            text-decoration: none;
            color: var(--text-secondary);
            border-radius: var(--radius-md);
            transition: all var(--transition-base);
            font-weight: 500;
        }}
        
        .nav-link:hover,
        .nav-link:focus {{
            background: white;
            color: var(--text-primary);
            box-shadow: var(--shadow-sm);
        }}
        
        .nav-link-current {{
            background: var(--secondary-500);
            color: white;
            box-shadow: var(--shadow-md);
        }}
        
        .nav-icon {{
            font-size: 1.125rem;
        }}
        
        .nav-text {{
            font-size: var(--font-size-sm);
        }}
        
        @media (max-width: 768px) {{
            .nav-list {{
                flex-direction: column;
            }}
            
            .nav-link {{
                justify-content: flex-start;
            }}
        }}
        </style>
        """
        
        return nav_html
    
    def create_accessible_modal(self, modal_id: str, title: str, content: str,
                              close_button_text: str = "Close"):
        """Create an accessible modal dialog."""
        
        modal_html = f"""
        <div 
            id="{modal_id}"
            class="modal-overlay"
            role="dialog"
            aria-modal="true"
            aria-labelledby="{modal_id}-title"
            aria-hidden="true"
        >
            <div class="modal-container">
                <div class="modal-header">
                    <h2 id="{modal_id}-title" class="modal-title">{title}</h2>
                    <button 
                        class="modal-close"
                        aria-label="{close_button_text}"
                        onclick="closeModal('{modal_id}')"
                    >
                        &times;
                    </button>
                </div>
                <div class="modal-content">
                    {content}
                </div>
                <div class="modal-footer">
                    <button 
                        class="btn-secondary"
                        onclick="closeModal('{modal_id}')"
                    >
                        {close_button_text}
                    </button>
                </div>
            </div>
        </div>
        
        <style>
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all var(--transition-base);
        }}
        
        .modal-overlay[aria-hidden="false"] {{
            opacity: 1;
            visibility: visible;
        }}
        
        .modal-container {{
            background: white;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-2xl);
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            transform: scale(0.9);
            transition: transform var(--transition-base);
        }}
        
        .modal-overlay[aria-hidden="false"] .modal-container {{
            transform: scale(1);
        }}
        
        .modal-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .modal-title {{
            margin: 0;
            font-size: var(--font-size-xl);
            font-weight: 600;
        }}
        
        .modal-close {{
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-secondary);
            padding: 0.25rem;
            border-radius: var(--radius-base);
            transition: all var(--transition-fast);
        }}
        
        .modal-close:hover,
        .modal-close:focus {{
            background: var(--primary-50);
            color: var(--text-primary);
        }}
        
        .modal-content {{
            padding: 1.5rem;
        }}
        
        .modal-footer {{
            padding: 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            gap: 1rem;
        }}
        </style>
        
        <script>
        function openModal(modalId) {{
            const modal = document.getElementById(modalId);
            if (modal) {{
                modal.setAttribute('aria-hidden', 'false');
                // Focus management
                const firstFocusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
                if (firstFocusable) {{
                    firstFocusable.focus();
                }}
                // Trap focus within modal
                trapFocus(modal);
            }}
        }}
        
        function closeModal(modalId) {{
            const modal = document.getElementById(modalId);
            if (modal) {{
                modal.setAttribute('aria-hidden', 'true');
                // Return focus to trigger element
                const trigger = document.querySelector(`[data-modal-trigger="${modalId}"]`);
                if (trigger) {{
                    trigger.focus();
                }}
            }}
        }}
        
        function trapFocus(element) {{
            const focusableElements = element.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const firstFocusable = focusableElements[0];
            const lastFocusable = focusableElements[focusableElements.length - 1];
            
            element.addEventListener('keydown', function(e) {{
                if (e.key === 'Tab') {{
                    if (e.shiftKey) {{
                        if (document.activeElement === firstFocusable) {{
                            lastFocusable.focus();
                            e.preventDefault();
                        }}
                    }} else {{
                        if (document.activeElement === lastFocusable) {{
                            firstFocusable.focus();
                            e.preventDefault();
                        }}
                    }}
                }}
                if (e.key === 'Escape') {{
                    closeModal(element.id);
                }}
            }});
        }}
        </script>
        """
        
        return modal_html
    
    def announce_to_screen_reader(self, message: str, priority: str = "polite"):
        """Announce message to screen readers."""
        announcement_id = f"announcement-{len(self.screen_reader_announcements)}"
        
        announcement_html = f"""
        <div 
            id="{announcement_id}"
            aria-live="{priority}"
            aria-atomic="true"
            class="sr-only"
        >
            {message}
        </div>
        """
        
        self.screen_reader_announcements.append(announcement_id)
        return announcement_html
    
    def create_skip_links(self, links: List[Dict[str, str]]):
        """Create skip navigation links for keyboard users."""
        
        skip_links_html = """
        <div class="skip-links">
        """
        
        for link in links:
            target = link.get('target', '')
            text = link.get('text', '')
            
            skip_links_html += f"""
                <a href="#{target}" class="skip-link">{text}</a>
            """
        
        skip_links_html += """
        </div>
        
        <style>
        .skip-links {{
            position: absolute;
            top: -40px;
            left: 6px;
            z-index: 1000;
        }}
        
        .skip-link {{
            position: absolute;
            top: -40px;
            left: 6px;
            background: var(--secondary-500);
            color: white;
            padding: 8px;
            text-decoration: none;
            border-radius: var(--radius-base);
            font-weight: 600;
            transition: top var(--transition-fast);
        }}
        
        .skip-link:focus {{
            top: 6px;
        }}
        </style>
        """
        
        return skip_links_html
    
    def add_keyboard_navigation(self):
        """Add keyboard navigation support."""
        
        keyboard_nav_script = """
        <script>
        // Enhanced keyboard navigation
        document.addEventListener('keydown', function(e) {
            // Tab navigation enhancement
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
            
            // Arrow key navigation for custom components
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                const focusedElement = document.activeElement;
                const navContainer = focusedElement.closest('[role="menubar"], [role="tablist"]');
                
                if (navContainer) {
                    e.preventDefault();
                    const items = navContainer.querySelectorAll('[role="menuitem"], [role="tab"]');
                    const currentIndex = Array.from(items).indexOf(focusedElement);
                    
                    let nextIndex;
                    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                        nextIndex = (currentIndex + 1) % items.length;
                    } else {
                        nextIndex = (currentIndex - 1 + items.length) % items.length;
                    }
                    
                    items[nextIndex].focus();
                }
            }
            
            // Escape key handling
            if (e.key === 'Escape') {
                const modal = document.querySelector('.modal-overlay[aria-hidden="false"]');
                if (modal) {
                    closeModal(modal.id);
                }
            }
        });
        
        // Remove keyboard navigation class on mouse use
        document.addEventListener('mousedown', function() {
            document.body.classList.remove('keyboard-navigation');
        });
        </script>
        
        <style>
        /* Enhanced focus styles for keyboard navigation */
        body.keyboard-navigation *:focus {
            outline: 2px solid var(--secondary-500);
            outline-offset: 2px;
        }
        
        body.keyboard-navigation button:focus,
        body.keyboard-navigation a:focus,
        body.keyboard-navigation input:focus,
        body.keyboard-navigation select:focus,
        body.keyboard-navigation textarea:focus {
            box-shadow: 0 0 0 2px var(--secondary-500);
        }
        </style>
        """
        
        return keyboard_nav_script


# Global accessibility manager instance
accessibility_manager = AccessibilityManager()


def add_aria_label(element_id: str, label: str, description: str = None):
    """Add ARIA label to an element."""
    return accessibility_manager.add_aria_label(element_id, label, description)


def create_accessible_button(text: str, onclick_action: str = None, 
                           button_type: str = "button", disabled: bool = False,
                           aria_label: str = None, aria_describedby: str = None):
    """Create an accessible button with proper ARIA attributes."""
    return accessibility_manager.create_accessible_button(
        text, onclick_action, button_type, disabled, aria_label, aria_describedby
    )


def create_accessible_image(src: str, alt_text: str, caption: str = None,
                          long_description: str = None, decorative: bool = False):
    """Create an accessible image with proper alt text and descriptions."""
    return accessibility_manager.create_accessible_image(
        src, alt_text, caption, long_description, decorative
    )


def announce_to_screen_reader(message: str, priority: str = "polite"):
    """Announce message to screen readers."""
    return accessibility_manager.announce_to_screen_reader(message, priority)


def add_keyboard_navigation():
    """Add keyboard navigation support."""
    return accessibility_manager.add_keyboard_navigation()