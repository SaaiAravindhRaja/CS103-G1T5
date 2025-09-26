"""
Keyboard shortcuts system for the SVD Image Compression webapp.
Provides comprehensive keyboard navigation and quick actions.
"""

import streamlit as st
from typing import Dict, List, Callable, Optional
import json


class KeyboardShortcutsManager:
    """Manages keyboard shortcuts for the entire application."""
    
    def __init__(self):
        """Initialize the keyboard shortcuts manager."""
        self.shortcuts = self._load_shortcuts_config()
        self.enabled = True
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for keyboard shortcuts."""
        if 'keyboard_shortcuts' not in st.session_state:
            st.session_state.keyboard_shortcuts = {
                'enabled': True,
                'help_visible': False,
                'current_context': 'global',
                'last_action': None,
                'action_history': []
            }
    
    def _load_shortcuts_config(self) -> Dict[str, Dict[str, Dict]]:
        """Load keyboard shortcuts configuration."""
        return {
            'global': {
                'h': {
                    'description': 'Toggle help mode',
                    'action': 'toggle_help_mode',
                    'category': 'Navigation'
                },
                '?': {
                    'description': 'Show keyboard shortcuts',
                    'action': 'show_shortcuts_help',
                    'category': 'Help'
                },
                'Escape': {
                    'description': 'Close help panels/modals',
                    'action': 'close_help_panels',
                    'category': 'Navigation'
                },
                'Tab': {
                    'description': 'Navigate between elements',
                    'action': 'navigate_elements',
                    'category': 'Navigation'
                },
                'f': {
                    'description': 'Toggle fullscreen mode',
                    'action': 'toggle_fullscreen',
                    'category': 'View'
                }
            },
            'compression_controls': {
                'ArrowUp': {
                    'description': 'Increase k-value by 1',
                    'action': 'increase_k_value',
                    'category': 'Controls',
                    'modifier': None
                },
                'ArrowDown': {
                    'description': 'Decrease k-value by 1',
                    'action': 'decrease_k_value',
                    'category': 'Controls',
                    'modifier': None
                },
                'Shift+ArrowUp': {
                    'description': 'Increase k-value by 10',
                    'action': 'increase_k_value_large',
                    'category': 'Controls',
                    'modifier': 'shift'
                },
                'Shift+ArrowDown': {
                    'description': 'Decrease k-value by 10',
                    'action': 'decrease_k_value_large',
                    'category': 'Controls',
                    'modifier': 'shift'
                },
                'r': {
                    'description': 'Toggle real-time preview',
                    'action': 'toggle_realtime_preview',
                    'category': 'Controls'
                },
                'g': {
                    'description': 'Switch to grayscale mode',
                    'action': 'set_grayscale_mode',
                    'category': 'Controls'
                },
                'c': {
                    'description': 'Switch to color mode',
                    'action': 'set_color_mode',
                    'category': 'Controls'
                },
                '1': {
                    'description': 'Apply Ultra Low preset (k=2)',
                    'action': 'apply_preset_1',
                    'category': 'Presets'
                },
                '2': {
                    'description': 'Apply Low preset (k=5)',
                    'action': 'apply_preset_2',
                    'category': 'Presets'
                },
                '3': {
                    'description': 'Apply Medium preset (k=20)',
                    'action': 'apply_preset_3',
                    'category': 'Presets'
                },
                '4': {
                    'description': 'Apply High preset (k=50)',
                    'action': 'apply_preset_4',
                    'category': 'Presets'
                },
                '5': {
                    'description': 'Apply Ultra High preset (k=100+)',
                    'action': 'apply_preset_5',
                    'category': 'Presets'
                },
                'a': {
                    'description': 'Auto-optimize k-value',
                    'action': 'auto_optimize',
                    'category': 'Smart Actions'
                },
                'e': {
                    'description': 'Apply 90% energy retention',
                    'action': 'apply_90_energy',
                    'category': 'Smart Actions'
                },
                'E': {
                    'description': 'Apply 95% energy retention',
                    'action': 'apply_95_energy',
                    'category': 'Smart Actions',
                    'modifier': 'shift'
                }
            },
            'results_view': {
                '+': {
                    'description': 'Zoom in on images',
                    'action': 'zoom_in',
                    'category': 'View'
                },
                '-': {
                    'description': 'Zoom out on images',
                    'action': 'zoom_out',
                    'category': 'View'
                },
                '0': {
                    'description': 'Reset zoom to fit',
                    'action': 'reset_zoom',
                    'category': 'View'
                },
                'd': {
                    'description': 'Toggle difference view',
                    'action': 'toggle_difference_view',
                    'category': 'View'
                },
                's': {
                    'description': 'Download compressed image',
                    'action': 'download_image',
                    'category': 'Actions'
                },
                'S': {
                    'description': 'Download with options',
                    'action': 'download_with_options',
                    'category': 'Actions',
                    'modifier': 'shift'
                },
                'p': {
                    'description': 'Print/export report',
                    'action': 'export_report',
                    'category': 'Actions'
                },
                'm': {
                    'description': 'Toggle metrics visibility',
                    'action': 'toggle_metrics',
                    'category': 'View'
                }
            },
            'upload_zone': {
                'u': {
                    'description': 'Focus upload zone',
                    'action': 'focus_upload',
                    'category': 'Navigation'
                },
                'Ctrl+v': {
                    'description': 'Paste image from clipboard',
                    'action': 'paste_image',
                    'category': 'Upload',
                    'modifier': 'ctrl'
                }
            },
            'navigation': {
                '1': {
                    'description': 'Go to Overview tab',
                    'action': 'goto_overview',
                    'category': 'Navigation',
                    'modifier': 'alt'
                },
                '2': {
                    'description': 'Go to Single Image tab',
                    'action': 'goto_single',
                    'category': 'Navigation',
                    'modifier': 'alt'
                },
                '3': {
                    'description': 'Go to Batch Processing tab',
                    'action': 'goto_batch',
                    'category': 'Navigation',
                    'modifier': 'alt'
                },
                '4': {
                    'description': 'Go to Comparison tab',
                    'action': 'goto_comparison',
                    'category': 'Navigation',
                    'modifier': 'alt'
                },
                '5': {
                    'description': 'Go to Tutorial tab',
                    'action': 'goto_tutorial',
                    'category': 'Navigation',
                    'modifier': 'alt'
                }
            }
        }
    
    def enable_keyboard_shortcuts(self, context: str = 'global') -> None:
        """Enable keyboard shortcuts for the specified context."""
        if not st.session_state.keyboard_shortcuts.get('enabled', True):
            return
        
        # Update current context
        st.session_state.keyboard_shortcuts['current_context'] = context
        
        # Generate JavaScript for keyboard handling
        shortcuts_js = self._generate_shortcuts_javascript(context)
        
        st.markdown(
            f"""
            <script>
            {shortcuts_js}
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def _generate_shortcuts_javascript(self, context: str) -> str:
        """Generate JavaScript code for handling keyboard shortcuts."""
        
        # Get shortcuts for current context and global shortcuts
        context_shortcuts = self.shortcuts.get(context, {})
        global_shortcuts = self.shortcuts.get('global', {})
        all_shortcuts = {**global_shortcuts, **context_shortcuts}
        
        # Generate JavaScript event handlers
        js_code = """
        // Remove existing keyboard event listeners
        if (window.svdKeyboardHandler) {
            document.removeEventListener('keydown', window.svdKeyboardHandler);
        }
        
        // Keyboard shortcut handler
        window.svdKeyboardHandler = function(event) {
            // Skip if user is typing in input fields
            if (event.target.tagName === 'INPUT' || 
                event.target.tagName === 'TEXTAREA' || 
                event.target.contentEditable === 'true') {
                return;
            }
            
            // Build key combination string
            let keyCombo = '';
            if (event.ctrlKey) keyCombo += 'Ctrl+';
            if (event.altKey) keyCombo += 'Alt+';
            if (event.shiftKey) keyCombo += 'Shift+';
            keyCombo += event.key;
            
            // Handle shortcuts
            switch(keyCombo) {
        """
        
        # Add case statements for each shortcut
        for key_combo, shortcut_info in all_shortcuts.items():
            action = shortcut_info['action']
            js_code += f"""
                case '{key_combo}':
                    event.preventDefault();
                    handleShortcutAction('{action}', '{key_combo}');
                    break;
            """
        
        js_code += """
            }
        };
        
        // Add event listener
        document.addEventListener('keydown', window.svdKeyboardHandler);
        
        // Shortcut action handler
        function handleShortcutAction(action, keyCombo) {
            console.log('Keyboard shortcut:', keyCombo, '->', action);
            
            // Store action in session state (would need Streamlit integration)
            if (window.streamlitController) {
                window.streamlitController.setComponentValue('keyboard_action', {
                    action: action,
                    keyCombo: keyCombo,
                    timestamp: Date.now()
                });
            }
            
            // Handle specific actions that can be done client-side
            switch(action) {
                case 'show_shortcuts_help':
                    showKeyboardShortcutsHelp();
                    break;
                case 'toggle_fullscreen':
                    toggleFullscreen();
                    break;
                case 'close_help_panels':
                    closeHelpPanels();
                    break;
                default:
                    // Most actions need server-side handling
                    console.log('Action requires server-side handling:', action);
            }
        }
        
        // Client-side action implementations
        function showKeyboardShortcutsHelp() {
            // Create and show shortcuts help modal
            const modal = createShortcutsModal();
            document.body.appendChild(modal);
        }
        
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }
        
        function closeHelpPanels() {
            // Close any open help panels or modals
            const helpPanels = document.querySelectorAll('.help-panel, .help-modal, .tooltip-container');
            helpPanels.forEach(panel => {
                if (panel.style.display !== 'none') {
                    panel.style.display = 'none';
                }
            });
        }
        
        function createShortcutsModal() {
            const modal = document.createElement('div');
            modal.className = 'shortcuts-modal';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10000;
            `;
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: white;
                border-radius: 12px;
                padding: 30px;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            `;
            
            content.innerHTML = `
                <h2 style="margin: 0 0 20px 0; color: #3b82f6;">⌨️ Keyboard Shortcuts</h2>
                <div id="shortcuts-content"></div>
                <button onclick="this.closest('.shortcuts-modal').remove()" style="
                    background: #3b82f6;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    cursor: pointer;
                    margin-top: 20px;
                ">Close</button>
            `;
            
            modal.appendChild(content);
            
            // Add shortcuts content
            const shortcutsContent = content.querySelector('#shortcuts-content');
            shortcutsContent.innerHTML = generateShortcutsHTML();
            
            // Close on background click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                }
            });
            
            return modal;
        }
        
        function generateShortcutsHTML() {
            return `
        """ + self._generate_shortcuts_html() + """
            `;
        }
        """
        
        return js_code
    
    def _generate_shortcuts_html(self) -> str:
        """Generate HTML content for shortcuts help modal."""
        
        # Group shortcuts by category
        categories = {}
        for context, shortcuts in self.shortcuts.items():
            for key, info in shortcuts.items():
                category = info.get('category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    'key': key,
                    'description': info['description'],
                    'context': context
                })
        
        html = ""
        for category, shortcuts in categories.items():
            html += f"""
            <div style="margin-bottom: 20px;">
                <h4 style="color: #374151; margin-bottom: 10px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px;">
                    {category}
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 8px; font-size: 0.9rem;">
            """
            
            for shortcut in shortcuts:
                key_display = shortcut['key'].replace('Arrow', '').replace('Shift+', '⇧').replace('Ctrl+', '⌘').replace('Alt+', '⌥')
                html += f"""
                <div style="display: flex; align-items: center;">
                    <kbd style="
                        background: #f3f4f6;
                        border: 1px solid #d1d5db;
                        border-radius: 4px;
                        padding: 2px 6px;
                        font-family: monospace;
                        font-size: 0.8rem;
                    ">{key_display}</kbd>
                </div>
                <div style="color: #6b7280;">
                    {shortcut['description']}
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        return html
    
    def create_shortcuts_help_panel(self, context: str = None) -> None:
        """Create a help panel showing available keyboard shortcuts."""
        
        if context is None:
            context = st.session_state.keyboard_shortcuts.get('current_context', 'global')
        
        with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
            
            # Context selector
            available_contexts = list(self.shortcuts.keys())
            selected_context = st.selectbox(
                "Show shortcuts for:",
                available_contexts,
                index=available_contexts.index(context) if context in available_contexts else 0,
                key="shortcuts_context_selector"
            )
            
            # Display shortcuts for selected context
            shortcuts = self.shortcuts.get(selected_context, {})
            
            if shortcuts:
                # Group by category
                categories = {}
                for key, info in shortcuts.items():
                    category = info.get('category', 'Other')
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((key, info['description']))
                
                # Display each category
                for category, shortcut_list in categories.items():
                    st.markdown(f"**{category}:**")
                    for key, description in shortcut_list:
                        # Format key display
                        key_display = key.replace('Arrow', '').replace('Shift+', '⇧ ').replace('Ctrl+', '⌘ ').replace('Alt+', '⌥ ')
                        st.markdown(f"• `{key_display}` - {description}")
                    st.markdown("")
            else:
                st.info(f"No shortcuts available for {selected_context}")
    
    def create_shortcuts_overlay(self) -> None:
        """Create a floating shortcuts overlay."""
        
        if st.session_state.keyboard_shortcuts.get('help_visible', False):
            
            # Get current context shortcuts
            context = st.session_state.keyboard_shortcuts.get('current_context', 'global')
            shortcuts = self.shortcuts.get(context, {})
            
            # Create floating overlay
            overlay_html = """
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border: 2px solid #3b82f6;
                border-radius: 12px;
                padding: 20px;
                max-width: 350px;
                max-height: 400px;
                overflow-y: auto;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                z-index: 1000;
                font-size: 0.9rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: #3b82f6;">⌨️ Shortcuts</h4>
                    <button onclick="this.closest('div').style.display='none'" style="
                        background: none;
                        border: none;
                        font-size: 1.2rem;
                        cursor: pointer;
                        color: #6b7280;
                    ">×</button>
                </div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 10px;">
                    Context: {context.replace('_', ' ').title()}
                </div>
            """
            
            # Add shortcuts
            for key, info in list(shortcuts.items())[:8]:  # Show first 8 shortcuts
                key_display = key.replace('Arrow', '').replace('Shift+', '⇧ ')
                overlay_html += f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <kbd style="
                        background: #f3f4f6;
                        border: 1px solid #d1d5db;
                        border-radius: 3px;
                        padding: 2px 5px;
                        font-family: monospace;
                        font-size: 0.75rem;
                    ">{key_display}</kbd>
                    <span style="color: #6b7280; font-size: 0.8rem; margin-left: 10px;">
                        {info['description'][:30]}{'...' if len(info['description']) > 30 else ''}
                    </span>
                </div>
                """
            
            if len(shortcuts) > 8:
                overlay_html += f"""
                <div style="text-align: center; margin-top: 10px; color: #6b7280; font-size: 0.8rem;">
                    +{len(shortcuts) - 8} more shortcuts
                </div>
                """
            
            overlay_html += """
                <div style="text-align: center; margin-top: 15px;">
                    <small style="color: #9ca3af;">Press ? for full list</small>
                </div>
            </div>
            """
            
            st.markdown(overlay_html, unsafe_allow_html=True)
    
    def handle_keyboard_action(self, action: str, context: str = None) -> bool:
        """Handle a keyboard shortcut action."""
        
        if context is None:
            context = st.session_state.keyboard_shortcuts.get('current_context', 'global')
        
        # Log the action
        st.session_state.keyboard_shortcuts['last_action'] = action
        if 'action_history' not in st.session_state.keyboard_shortcuts:
            st.session_state.keyboard_shortcuts['action_history'] = []
        st.session_state.keyboard_shortcuts['action_history'].append({
            'action': action,
            'context': context,
            'timestamp': time.time()
        })
        
        # Keep only last 10 actions
        if len(st.session_state.keyboard_shortcuts['action_history']) > 10:
            st.session_state.keyboard_shortcuts['action_history'] = \
                st.session_state.keyboard_shortcuts['action_history'][-10:]
        
        # Handle the action
        return self._execute_action(action, context)
    
    def _execute_action(self, action: str, context: str) -> bool:
        """Execute a specific keyboard action."""
        
        try:
            # Global actions
            if action == 'toggle_help_mode':
                st.session_state.keyboard_shortcuts['help_visible'] = \
                    not st.session_state.keyboard_shortcuts.get('help_visible', False)
                return True
            
            elif action == 'show_shortcuts_help':
                st.session_state.keyboard_shortcuts['help_visible'] = True
                return True
            
            elif action == 'close_help_panels':
                st.session_state.keyboard_shortcuts['help_visible'] = False
                return True
            
            # Compression control actions
            elif action in ['increase_k_value', 'decrease_k_value', 'increase_k_value_large', 'decrease_k_value_large']:
                if 'compression_params' in st.session_state:
                    current_k = st.session_state.compression_params.get('k_value', 20)
                    if action == 'increase_k_value':
                        st.session_state.compression_params['k_value'] = min(current_k + 1, 256)
                    elif action == 'decrease_k_value':
                        st.session_state.compression_params['k_value'] = max(current_k - 1, 1)
                    elif action == 'increase_k_value_large':
                        st.session_state.compression_params['k_value'] = min(current_k + 10, 256)
                    elif action == 'decrease_k_value_large':
                        st.session_state.compression_params['k_value'] = max(current_k - 10, 1)
                    return True
            
            elif action == 'toggle_realtime_preview':
                if 'compression_params' in st.session_state:
                    st.session_state.compression_params['real_time_enabled'] = \
                        not st.session_state.compression_params.get('real_time_enabled', True)
                    return True
            
            elif action in ['set_grayscale_mode', 'set_color_mode']:
                if 'compression_params' in st.session_state:
                    mode = 'Grayscale' if action == 'set_grayscale_mode' else 'RGB (Color)'
                    st.session_state.compression_params['mode'] = mode
                    return True
            
            elif action.startswith('apply_preset_'):
                preset_values = {
                    'apply_preset_1': 2,
                    'apply_preset_2': 5,
                    'apply_preset_3': 20,
                    'apply_preset_4': 50,
                    'apply_preset_5': 100
                }
                if 'compression_params' in st.session_state and action in preset_values:
                    st.session_state.compression_params['k_value'] = preset_values[action]
                    return True
            
            # Add more action handlers as needed
            
            return False
            
        except Exception as e:
            st.error(f"Error executing keyboard action {action}: {str(e)}")
            return False
    
    def get_action_history(self) -> List[Dict]:
        """Get the history of keyboard actions."""
        return st.session_state.keyboard_shortcuts.get('action_history', [])
    
    def clear_action_history(self) -> None:
        """Clear the keyboard action history."""
        st.session_state.keyboard_shortcuts['action_history'] = []


# Global keyboard shortcuts manager instance
keyboard_manager = KeyboardShortcutsManager()


def enable_keyboard_shortcuts(context: str = 'global') -> None:
    """Convenience function to enable keyboard shortcuts."""
    keyboard_manager.enable_keyboard_shortcuts(context)


def create_shortcuts_help() -> None:
    """Convenience function to create shortcuts help panel."""
    keyboard_manager.create_shortcuts_help_panel()


def handle_keyboard_action(action: str, context: str = None) -> bool:
    """Convenience function to handle keyboard actions."""
    return keyboard_manager.handle_keyboard_action(action, context)