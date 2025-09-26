"""
Navigation utilities for single-page Streamlit application with tab-based navigation.
"""

import streamlit as st


def create_main_layout():
    """Create the main single-page layout with header navigation and return current tab."""
    
    # Create header with logo and navigation
    create_header()
    
    # Create tab navigation
    current_tab = create_tab_navigation()
    
    return current_tab


def create_header():
    """Create the main header with logo and branding."""
    
    st.markdown(
        """
        <div class="main-header">
            <div class="header-content">
                <div class="logo-section">
                    <h1 class="app-title">🖼️ SVD Image Compression</h1>
                    <p class="app-subtitle">Interactive Academic Tool for Image Analysis</p>
                </div>
                <div class="header-actions">
                    <div class="status-badge">
                        <span class="status-dot"></span>
                        <span>Ready</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_tab_navigation():
    """Create responsive tab-style navigation and return the selected tab."""
    
    # Define tabs with icons and descriptions
    tabs = {
        "Overview": {
            "icon": "🏠",
            "description": "Project overview and getting started",
            "short": "Home"
        },
        "Single Image": {
            "icon": "📷", 
            "description": "Compress and analyze individual images",
            "short": "Single"
        },
        "Batch Processing": {
            "icon": "📊",
            "description": "Process multiple images simultaneously",
            "short": "Batch"
        },
        "Comparison": {
            "icon": "⚖️",
            "description": "Compare different compression levels",
            "short": "Compare"
        },
        "Tutorial": {
            "icon": "📚",
            "description": "Interactive tutorials and help",
            "short": "Help"
        },
        "Loading Demo": {
            "icon": "⏳",
            "description": "Demo of loading animations and progress feedback",
            "short": "Demo"
        }
    }
    
    # Get the active tab from session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    # Create responsive navigation HTML
    tab_html = f"""
    <div class="responsive-tab-navigation">
        <div class="tab-container-responsive">
    """
    
    for tab_name, tab_info in tabs.items():
        active_class = "active" if st.session_state.active_tab == tab_name else ""
        tab_html += f"""
            <div class="tab-item-responsive {active_class}" data-tab="{tab_name}">
                <span class="tab-icon-responsive">{tab_info['icon']}</span>
                <span class="tab-label-full">{tab_name}</span>
                <span class="tab-label-short">{tab_info['short']}</span>
                <div class="tab-tooltip-responsive">{tab_info['description']}</div>
            </div>
        """
    
    tab_html += """
        </div>
    </div>
    
    <style>
    .responsive-tab-navigation {
        margin-bottom: 2rem;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    .tab-container-responsive {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        background: #f1f5f9;
        padding: 0.5rem;
        border-radius: 0.75rem;
        max-width: 100%;
        min-width: max-content;
    }
    
    .tab-item-responsive {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        white-space: nowrap;
        position: relative;
        min-height: 44px;
        min-width: 44px;
        justify-content: center;
    }
    
    .tab-item-responsive:hover {
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .tab-item-responsive.active {
        background: #3b82f6;
        color: white;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .tab-icon-responsive {
        font-size: 1.125rem;
        flex-shrink: 0;
    }
    
    .tab-label-full {
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .tab-label-short {
        display: none;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .tab-tooltip-responsive {
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-top: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        font-size: 0.75rem;
        border-radius: 0.375rem;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s;
        z-index: 10;
        white-space: nowrap;
    }
    
    .tab-item-responsive:hover .tab-tooltip-responsive {
        opacity: 1;
    }
    
    /* Tablet Styles */
    @media (max-width: 1024px) {
        .tab-container-responsive {
            gap: 0.25rem;
            padding: 0.375rem;
        }
        
        .tab-item-responsive {
            padding: 0.5rem 0.75rem;
        }
        
        .tab-label-full {
            font-size: 0.8125rem;
        }
    }
    
    /* Mobile Styles */
    @media (max-width: 768px) {
        .tab-container-responsive {
            justify-content: flex-start;
            padding: 0.25rem;
            gap: 0.125rem;
        }
        
        .tab-item-responsive {
            padding: 0.5rem;
            min-width: 44px;
        }
        
        .tab-label-full {
            display: none;
        }
        
        .tab-label-short {
            display: inline;
            font-size: 0.75rem;
        }
        
        .tab-icon-responsive {
            font-size: 1rem;
        }
    }
    
    /* Small Mobile Styles */
    @media (max-width: 480px) {
        .tab-container-responsive {
            flex-direction: column;
            align-items: stretch;
        }
        
        .tab-item-responsive {
            justify-content: flex-start;
            padding: 0.75rem 1rem;
        }
        
        .tab-label-full {
            display: inline;
        }
        
        .tab-label-short {
            display: none;
        }
        
        .tab-icon-responsive {
            font-size: 1.125rem;
        }
    }
    
    /* Touch device optimizations */
    @media (hover: none) and (pointer: coarse) {
        .tab-item-responsive:active {
            transform: scale(0.95);
        }
        
        .tab-tooltip-responsive {
            display: none;
        }
    }
    </style>
    """
    
    st.markdown(tab_html, unsafe_allow_html=True)
    
    # Create responsive button layout for functionality
    # Use different layouts based on screen size
    
    # Desktop/Tablet layout (6 columns)
    desktop_cols = st.columns(6)
    mobile_cols = st.columns(2)  # Mobile layout (2 columns, 3 rows)
    
    # Create buttons with responsive text
    tab_buttons = []
    for i, (tab_name, tab_info) in enumerate(tabs.items()):
        # Desktop button
        with desktop_cols[i]:
            desktop_text = f"{tab_info['icon']} {tab_name}"
            if st.button(
                desktop_text, 
                key=f"desktop_tab_{i}",
                use_container_width=True,
                type="primary" if st.session_state.active_tab == tab_name else "secondary"
            ):
                st.session_state.active_tab = tab_name
                st.rerun()
        
        # Mobile buttons (2 per row)
        mobile_col_index = i % 2
        if i < 6:  # Only show first 6 tabs in mobile layout
            with mobile_cols[mobile_col_index]:
                mobile_text = f"{tab_info['icon']} {tab_info['short']}"
                if st.button(
                    mobile_text,
                    key=f"mobile_tab_{i}",
                    use_container_width=True,
                    type="primary" if st.session_state.active_tab == tab_name else "secondary"
                ):
                    st.session_state.active_tab = tab_name
                    st.rerun()
    
    # Add CSS to hide appropriate button sets based on screen size
    st.markdown("""
    <style>
    /* Hide mobile buttons on desktop */
    @media (min-width: 769px) {
        div[data-testid="column"]:has(button[key*="mobile_tab"]) {
            display: none !important;
        }
    }
    
    /* Hide desktop buttons on mobile */
    @media (max-width: 768px) {
        div[data-testid="column"]:has(button[key*="desktop_tab"]) {
            display: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add separator
    st.markdown('<div class="content-separator"></div>', unsafe_allow_html=True)
    
    return st.session_state.active_tab


def create_responsive_grid():
    """Create a responsive grid system for main content areas."""
    
    # Create main content grid
    st.markdown(
        """
        <div class="main-content-grid">
            <div class="content-area">
                <!-- Main content will be inserted here -->
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_progress_indicator(current_step, total_steps, step_name=""):
    """Show a progress indicator for multi-step processes."""
    progress = current_step / total_steps
    
    st.markdown("### 🔄 Progress")
    st.progress(progress)
    
    if step_name:
        st.markdown(f"**Current step:** {step_name}")
    
    st.markdown(f"Step {current_step} of {total_steps}")


def add_content_section(title, content, section_type="default"):
    """Add a content section with consistent styling."""
    
    section_class = f"{section_type}-section" if section_type != "default" else "content-section"
    
    st.markdown(
        f"""
        <div class="{section_class}">
            <h3>{title}</h3>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


# Backward compatibility functions
def setup_navigation():
    """Backward compatibility wrapper for create_main_layout."""
    return create_main_layout()


def add_sidebar_info(title, content, info_type="info"):
    """Add informational content (now displayed in main content area)."""
    
    card_class = f"{info_type}-card"
    
    st.markdown(
        f"""
        <div class="{card_class}">
            <h4>{title}</h4>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_progress_indicator(current_step, total_steps, step_name=""):
    """Show a progress indicator for multi-step processes."""
    progress = current_step / total_steps
    
    st.sidebar.markdown("### 🔄 Progress")
    st.sidebar.progress(progress)
    
    if step_name:
        st.sidebar.markdown(f"**Current step:** {step_name}")
    
    st.sidebar.markdown(f"Step {current_step} of {total_steps}")


def add_sidebar_info(title, content, info_type="info"):
    """Add informational content to sidebar."""
    
    card_class = f"{info_type}-card"
    
    st.sidebar.markdown(
        f"""
        <div class="{card_class}">
            <h4>{title}</h4>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )