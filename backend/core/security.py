"""
🔒 Security & Rate Limiting
Simple in-memory rate limiter
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import HTTPException, Request
from core.config import settings


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        # Structure: {client_ip: [(timestamp, count)]}
        self.clients: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client is within rate limits
        
        Returns:
            (is_allowed, remaining_requests)
        """
        current_time = time.time()
        window_start = current_time - settings.RATE_LIMIT_WINDOW
        
        # Remove old entries
        self.clients[client_ip] = [
            (ts, count) for ts, count in self.clients[client_ip]
            if ts > window_start
        ]
        
        # Count requests in current window
        request_count = sum(count for _, count in self.clients[client_ip])
        
        if request_count >= settings.RATE_LIMIT_REQUESTS:
            return False, 0
        
        # Add current request
        self.clients[client_ip].append((current_time, 1))
        
        remaining = settings.RATE_LIMIT_REQUESTS - request_count - 1
        return True, remaining
    
    def cleanup(self):
        """Cleanup old entries (call periodically)"""
        current_time = time.time()
        window_start = current_time - settings.RATE_LIMIT_WINDOW
        
        for client_ip in list(self.clients.keys()):
            self.clients[client_ip] = [
                (ts, count) for ts, count in self.clients[client_ip]
                if ts > window_start
            ]
            
            # Remove empty entries
            if not self.clients[client_ip]:
                del self.clients[client_ip]


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """Dependency for rate limiting"""
    client_ip = request.client.host
    
    is_allowed, remaining = rate_limiter.is_allowed(client_ip)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {settings.RATE_LIMIT_WINDOW} seconds."
        )
    
    # Add rate limit headers to request state
    request.state.rate_limit_remaining = remaining
    
    return True
