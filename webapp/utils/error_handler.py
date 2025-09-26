"""
Comprehensive error handling system for the SVD image compression webapp.
Provides user-friendly error messages, fallback states, and recovery mechanisms.
"""

import streamlit as st
import traceback
import logging
import sys
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from pathlib import Path
import time
from functools import wraps
import numpy as np
from PIL import Image


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling."""
    UPLOAD = "upload"
    PROCESSING = "processing"
    MEMORY = "memory"
    VALIDATION = "validation"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {}
        self.user_messages = {}
        self._setup_logging()
        self._register_default_strategies()
        self._register_user_messages()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Ensure log directory exists first
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'error.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file), mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _register_default_strategies(self):
        """Register default recovery strategies for different error types."""
        
        self.recovery_strategies = {
            ErrorCategory.UPLOAD: {
                'file_too_large': self._handle_large_file,
                'invalid_format': self._handle_invalid_format,
                'corrupted_file': self._handle_corrupted_file,
                'upload_timeout': self._handle_upload_timeout
            },
            ErrorCategory.PROCESSING: {
                'svd_failure': self._handle_svd_failure,
                'dimension_mismatch': self._handle_dimension_mismatch,
                'invalid_k_value': self._handle_invalid_k_value,
                'processing_timeout': self._handle_processing_timeout
            },
            ErrorCategory.MEMORY: {
                'out_of_memory': self._handle_out_of_memory,
                'memory_threshold': self._handle_memory_threshold,
                'allocation_failure': self._handle_allocation_failure
            },
            ErrorCategory.VALIDATION: {
                'invalid_input': self._handle_invalid_input,
                'parameter_out_of_range': self._handle_parameter_range,
                'missing_required_field': self._handle_missing_field
            },
            ErrorCategory.SYSTEM: {
                'disk_space': self._handle_disk_space,
                'permission_denied': self._handle_permission_denied,
                'system_overload': self._handle_system_overload
            }
        }
    
    def _register_user_messages(self):
        """Register user-friendly error messages."""
        
        self.user_messages = {
            # Upload errors
            'file_too_large': {
                'title': '📁 File Too Large',
                'message': 'The uploaded file exceeds the maximum size limit.',
                'suggestions': [
                    'Reduce image resolution or quality',
                    'Use image compression tools before upload',
                    'Try uploading a smaller portion of the image'
                ],
                'severity': ErrorSeverity.WARNING
            },
            'invalid_format': {
                'title': '🖼️ Invalid File Format',
                'message': 'The uploaded file format is not supported.',
                'suggestions': [
                    'Convert to PNG, JPG, or JPEG format',
                    'Ensure the file is a valid image',
                    'Check if the file extension matches the content'
                ],
                'severity': ErrorSeverity.ERROR
            },
            'corrupted_file': {
                'title': '⚠️ Corrupted File',
                'message': 'The uploaded file appears to be corrupted or incomplete.',
                'suggestions': [
                    'Try uploading the file again',
                    'Check the original file for corruption',
                    'Use a different image file'
                ],
                'severity': ErrorSeverity.ERROR
            },
            
            # Processing errors
            'svd_failure': {
                'title': '🔧 Processing Failed',
                'message': 'SVD compression could not be completed.',
                'suggestions': [
                    'Try a different k-value',
                    'Switch to grayscale mode',
                    'Reduce image size',
                    'Check if the image has valid content'
                ],
                'severity': ErrorSeverity.ERROR
            },
            'out_of_memory': {
                'title': '💾 Memory Limit Exceeded',
                'message': 'The image is too large to process with available memory.',
                'suggestions': [
                    'Reduce the image size',
                    'Use a smaller k-value',
                    'Switch to grayscale mode',
                    'Close other applications to free memory'
                ],
                'severity': ErrorSeverity.WARNING
            },
            'invalid_k_value': {
                'title': '🎛️ Invalid Compression Level',
                'message': 'The selected k-value is not valid for this image.',
                'suggestions': [
                    'Choose a k-value between 1 and the image\'s smaller dimension',
                    'Use the auto-optimize feature',
                    'Try one of the quality presets'
                ],
                'severity': ErrorSeverity.WARNING
            },
            
            # System errors
            'disk_space': {
                'title': '💽 Insufficient Disk Space',
                'message': 'Not enough disk space to complete the operation.',
                'suggestions': [
                    'Free up disk space',
                    'Clear temporary files',
                    'Use a smaller image or lower k-value'
                ],
                'severity': ErrorSeverity.ERROR
            },
            'system_overload': {
                'title': '⚡ System Overloaded',
                'message': 'The system is currently overloaded. Please try again.',
                'suggestions': [
                    'Wait a moment and try again',
                    'Reduce the complexity of your request',
                    'Try during off-peak hours'
                ],
                'severity': ErrorSeverity.WARNING
            }
        }
    
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory, 
                    context: Dict[str, Any] = None,
                    show_user_message: bool = True) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            category: Error category for appropriate handling
            context: Additional context information
            show_user_message: Whether to show user-friendly message
            
        Returns:
            Dictionary containing error handling results
        """
        
        # Log the error
        error_info = {
            'error': str(error),
            'category': category.value,
            'context': context or {},
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        self.logger.error(f"Error in {category.value}: {str(error)}", exc_info=True)
        
        # Determine error type and recovery strategy
        error_type = self._classify_error(error, category)
        recovery_result = self._attempt_recovery(error, category, error_type, context)
        
        # Show user-friendly message if requested
        if show_user_message:
            self._show_user_message(error_type, recovery_result)
        
        return {
            'error_type': error_type,
            'recovery_attempted': recovery_result['attempted'],
            'recovery_successful': recovery_result['successful'],
            'fallback_data': recovery_result.get('fallback_data'),
            'user_message_shown': show_user_message,
            'suggestions': recovery_result.get('suggestions', [])
        }
    
    def _classify_error(self, error: Exception, category: ErrorCategory) -> str:
        """Classify the error type for appropriate handling."""
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Memory-related errors
        if 'memory' in error_str or isinstance(error, MemoryError):
            return 'out_of_memory'
        
        # File-related errors
        if category == ErrorCategory.UPLOAD:
            if 'size' in error_str or 'large' in error_str:
                return 'file_too_large'
            elif 'format' in error_str or 'invalid' in error_str:
                return 'invalid_format'
            elif 'corrupt' in error_str or 'damaged' in error_str:
                return 'corrupted_file'
            elif 'timeout' in error_str:
                return 'upload_timeout'
        
        # Processing errors
        if category == ErrorCategory.PROCESSING:
            if 'svd' in error_str or 'decomposition' in error_str:
                return 'svd_failure'
            elif 'dimension' in error_str or 'shape' in error_str:
                return 'dimension_mismatch'
            elif 'k' in error_str or 'value' in error_str:
                return 'invalid_k_value'
            elif 'timeout' in error_str:
                return 'processing_timeout'
        
        # System errors
        if 'disk' in error_str or 'space' in error_str:
            return 'disk_space'
        elif 'permission' in error_str:
            return 'permission_denied'
        elif 'overload' in error_str or 'busy' in error_str:
            return 'system_overload'
        
        # Default classification
        return 'unknown_error'
    
    def _attempt_recovery(self, 
                         error: Exception, 
                         category: ErrorCategory, 
                         error_type: str, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from the error using registered strategies."""
        
        recovery_result = {
            'attempted': False,
            'successful': False,
            'fallback_data': None,
            'suggestions': []
        }
        
        # Get recovery strategy
        if category in self.recovery_strategies and error_type in self.recovery_strategies[category]:
            strategy = self.recovery_strategies[category][error_type]
            recovery_result['attempted'] = True
            
            try:
                fallback_result = strategy(error, context)
                recovery_result['successful'] = fallback_result['success']
                recovery_result['fallback_data'] = fallback_result.get('data')
                recovery_result['suggestions'] = fallback_result.get('suggestions', [])
                
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                recovery_result['successful'] = False
        
        return recovery_result
    
    def _show_user_message(self, error_type: str, recovery_result: Dict[str, Any]):
        """Show user-friendly error message."""
        
        if error_type in self.user_messages:
            message_info = self.user_messages[error_type]
            
            # Choose appropriate Streamlit function based on severity
            if message_info['severity'] == ErrorSeverity.INFO:
                st.info(f"ℹ️ {message_info['title']}")
            elif message_info['severity'] == ErrorSeverity.WARNING:
                st.warning(f"⚠️ {message_info['title']}")
            elif message_info['severity'] == ErrorSeverity.ERROR:
                st.error(f"❌ {message_info['title']}")
            elif message_info['severity'] == ErrorSeverity.CRITICAL:
                st.error(f"🚨 {message_info['title']}")
            
            # Show message and suggestions
            st.markdown(f"**{message_info['message']}**")
            
            if message_info['suggestions'] or recovery_result.get('suggestions'):
                st.markdown("**💡 Suggestions:**")
                all_suggestions = message_info['suggestions'] + recovery_result.get('suggestions', [])
                for suggestion in all_suggestions:
                    st.markdown(f"• {suggestion}")
            
            # Show recovery status if attempted
            if recovery_result['attempted']:
                if recovery_result['successful']:
                    st.success("✅ Automatic recovery was successful!")
                else:
                    st.info("🔄 Automatic recovery was attempted but may not have fully resolved the issue.")
        
        else:
            # Generic error message
            st.error("❌ An unexpected error occurred")
            st.markdown("Please try again or contact support if the problem persists.")
    
    # Recovery strategy implementations
    def _handle_large_file(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle large file errors by suggesting compression."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Try resizing the image to a smaller resolution',
                'Use image compression before upload',
                'Consider processing the image in smaller sections'
            ]
        }
    
    def _handle_invalid_format(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invalid format errors."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Convert the image to PNG, JPG, or JPEG format',
                'Ensure the file is actually an image',
                'Check that the file is not corrupted'
            ]
        }
    
    def _handle_corrupted_file(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle corrupted file errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Try uploading the file again',
                'Use a different image file',
                'Check the original file for corruption'
            ]
        }
    
    def _handle_upload_timeout(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle upload timeout errors."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Check your internet connection',
                'Try uploading a smaller file',
                'Retry the upload after a moment'
            ]
        }
    
    def _handle_svd_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SVD processing failures with fallback strategies."""
        
        fallback_data = None
        success = False
        
        if context and 'image' in context and 'k_value' in context:
            try:
                # Try with reduced k value
                image = context['image']
                original_k = context['k_value']
                fallback_k = max(1, original_k // 2)
                
                # Attempt processing with fallback k
                from .performance_optimizer import get_image_processor
                processor = get_image_processor()
                result = processor.process_with_fallback(image, fallback_k, context.get('mode', 'RGB'))
                
                if result['success']:
                    fallback_data = result
                    success = True
                
            except Exception:
                pass
        
        return {
            'success': success,
            'data': fallback_data,
            'suggestions': [
                'Try a smaller k-value',
                'Switch to grayscale mode for better stability',
                'Ensure the image has valid numerical data'
            ]
        }
    
    def _handle_out_of_memory(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle out of memory errors with size reduction."""
        
        fallback_data = None
        success = False
        
        if context and 'image' in context:
            try:
                # Try with smaller image
                image = context['image']
                
                # Reduce image size by 50%
                if image.ndim == 3:
                    h, w, c = image.shape
                    new_h, new_w = h // 2, w // 2
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                    resized_pil = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    reduced_image = np.array(resized_pil) / 255.0
                else:
                    h, w = image.shape
                    new_h, new_w = h // 2, w // 2
                    pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
                    resized_pil = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    reduced_image = np.array(resized_pil) / 255.0
                
                # Try processing with reduced image
                from .performance_optimizer import get_image_processor
                processor = get_image_processor()
                result = processor.process_with_fallback(
                    reduced_image, 
                    context.get('k_value', 20), 
                    context.get('mode', 'RGB')
                )
                
                if result['success']:
                    fallback_data = result
                    success = True
                
            except Exception:
                pass
        
        return {
            'success': success,
            'data': fallback_data,
            'suggestions': [
                'Reduce the image size before processing',
                'Use a smaller k-value',
                'Switch to grayscale mode',
                'Close other applications to free memory'
            ]
        }
    
    def _handle_invalid_k_value(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invalid k-value errors."""
        
        fallback_data = None
        success = False
        
        if context and 'image' in context:
            try:
                image = context['image']
                # Use a safe k value (10% of smaller dimension)
                safe_k = max(1, min(image.shape[:2]) // 10)
                
                from .performance_optimizer import get_image_processor
                processor = get_image_processor()
                result = processor.process_with_fallback(
                    image, 
                    safe_k, 
                    context.get('mode', 'RGB')
                )
                
                if result['success']:
                    fallback_data = result
                    success = True
                
            except Exception:
                pass
        
        return {
            'success': success,
            'data': fallback_data,
            'suggestions': [
                'Use the auto-optimize feature to find the best k-value',
                'Try one of the quality presets',
                'Ensure k is between 1 and the smaller image dimension'
            ]
        }
    
    def _handle_dimension_mismatch(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dimension mismatch errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Check that the image has valid dimensions',
                'Ensure the image is properly loaded',
                'Try reloading the image file'
            ]
        }
    
    def _handle_processing_timeout(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle processing timeout errors."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Try with a smaller image or k-value',
                'Use grayscale mode for faster processing',
                'Wait a moment and try again'
            ]
        }
    
    def _handle_memory_threshold(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory threshold warnings."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Consider reducing the image size',
                'Use a smaller k-value',
                'Monitor system memory usage'
            ]
        }
    
    def _handle_allocation_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory allocation failures."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Restart the application',
                'Free up system memory',
                'Use smaller images or parameters'
            ]
        }
    
    def _handle_invalid_input(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invalid input errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Check that all required fields are filled',
                'Ensure input values are within valid ranges',
                'Verify the input format is correct'
            ]
        }
    
    def _handle_parameter_range(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parameter out of range errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Adjust parameters to be within valid ranges',
                'Use the suggested default values',
                'Check the parameter documentation'
            ]
        }
    
    def _handle_missing_field(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing required field errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Fill in all required fields',
                'Check for any missing inputs',
                'Ensure all necessary data is provided'
            ]
        }
    
    def _handle_disk_space(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle disk space errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Free up disk space',
                'Clear temporary files',
                'Use smaller images or lower quality settings'
            ]
        }
    
    def _handle_permission_denied(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle permission denied errors."""
        
        return {
            'success': False,
            'data': None,
            'suggestions': [
                'Check file and directory permissions',
                'Run with appropriate privileges',
                'Contact system administrator'
            ]
        }
    
    def _handle_system_overload(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system overload errors."""
        
        return {
            'success': True,
            'data': None,
            'suggestions': [
                'Wait a moment and try again',
                'Reduce the complexity of your request',
                'Try during off-peak hours'
            ]
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        
        if not self.error_log:
            return {'total_errors': 0}
        
        # Count errors by category
        category_counts = {}
        for error_info in self.error_log:
            category = error_info['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Recent errors (last hour)
        current_time = time.time()
        recent_errors = [
            e for e in self.error_log 
            if current_time - e['timestamp'] < 3600
        ]
        
        return {
            'total_errors': len(self.error_log),
            'recent_errors': len(recent_errors),
            'category_breakdown': category_counts,
            'most_common_category': max(category_counts, key=category_counts.get) if category_counts else None
        }
    
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log.clear()


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(error: Exception, 
                category: ErrorCategory, 
                context: Dict[str, Any] = None,
                show_user_message: bool = True) -> Dict[str, Any]:
    """
    Convenience function to handle errors.
    
    Args:
        error: The exception that occurred
        category: Error category
        context: Additional context
        show_user_message: Whether to show user message
        
    Returns:
        Error handling result
    """
    handler = get_error_handler()
    return handler.handle_error(error, category, context, show_user_message)


def error_boundary(category: ErrorCategory, context: Dict[str, Any] = None):
    """
    Decorator to create an error boundary around functions.
    
    Args:
        category: Error category for this boundary
        context: Additional context information
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle_error(e, category, context, show_user_message=True)
        return wrapper
    return decorator