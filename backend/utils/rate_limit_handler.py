"""
Rate Limit Handler for OpenAI API
Handles rate limits with exponential backoff and detailed error logging
"""

import time
import asyncio
from typing import Callable, Any, Optional
from openai import RateLimitError, APIError
from core.logging import logger


def handle_rate_limit_with_backoff(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Execute a function with exponential backoff on rate limit errors
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
    
    Returns:
        Result of the function call
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Extract detailed rate limit info from error
                error_msg = str(e)
                error_code = getattr(e, 'code', None)
                error_type = getattr(e, 'type', None)
                error_param = getattr(e, 'param', None)
                
                logger.warning(
                    f"⚠️ Rate Limit Error (attempt {attempt + 1}/{max_retries}):\n"
                    f"   Type: {error_type}\n"
                    f"   Code: {error_code}\n"
                    f"   Param: {error_param}\n"
                    f"   Message: {error_msg}"
                )
                
                # Try to extract retry-after from headers if available
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after_header = e.response.headers.get('retry-after')
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                            logger.info(f"⏳ Server suggests waiting {retry_after} seconds")
                        except ValueError:
                            pass
                    
                    # Log response headers for debugging
                    logger.debug(f"Response headers: {dict(e.response.headers)}")
                
                # Use retry-after if available, otherwise use exponential backoff
                wait_time = retry_after if retry_after else delay
                wait_time = min(wait_time, max_delay)  # Cap at max_delay
                
                logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
                
                # Increase delay for next attempt (exponential backoff)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise
        except APIError as e:
            # For other API errors, log and re-raise
            logger.error(f"OpenAI API error: {e}")
            raise
    
    raise Exception(f"Failed after {max_retries} attempts")


async def handle_rate_limit_with_backoff_async(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Execute an async function with exponential backoff on rate limit errors
    
    Args:
        func: Async function to execute
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
    
    Returns:
        Result of the function call
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Extract detailed rate limit info from error
                error_msg = str(e)
                error_code = getattr(e, 'code', None)
                error_type = getattr(e, 'type', None)
                error_param = getattr(e, 'param', None)
                
                logger.warning(
                    f"⚠️ Rate Limit Error (attempt {attempt + 1}/{max_retries}):\n"
                    f"   Type: {error_type}\n"
                    f"   Code: {error_code}\n"
                    f"   Param: {error_param}\n"
                    f"   Message: {error_msg}"
                )
                
                # Try to extract retry-after from headers if available
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after_header = e.response.headers.get('retry-after')
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                            logger.info(f"⏳ Server suggests waiting {retry_after} seconds")
                        except ValueError:
                            pass
                    
                    # Log response headers for debugging
                    logger.debug(f"Response headers: {dict(e.response.headers)}")
                
                # Use retry-after if available, otherwise use exponential backoff
                wait_time = retry_after if retry_after else delay
                wait_time = min(wait_time, max_delay)  # Cap at max_delay
                
                logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                await asyncio.sleep(wait_time)
                
                # Increase delay for next attempt (exponential backoff)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise
        except APIError as e:
            # For other API errors, log and re-raise
            error_code = getattr(e, 'code', None)
            error_type = getattr(e, 'type', None)
            logger.error(f"OpenAI API error: Type={error_type}, Code={error_code}, Message={str(e)}")
            raise
    
    raise Exception(f"Failed after {max_retries} attempts")

