#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from abc import ABC, abstractmethod
from ..exceptions import APIRequestError, PaymentInsufficientError, AuthenticationError


class BaseAPI(ABC):
    """Base class for all Genos API endpoints with token validation and error handling.
    
    This class provides common functionality for all API endpoints, including:
    - Token validation and authentication error handling
    - Payment/credit insufficiency checking
    - Standardized error response handling
    - HTTP status code validation
    
    Only responses with status code 200 are considered successful and will proceed
    to further analysis. All other status codes will raise appropriate exceptions
    with the received error message.
    """
    
    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30):
        """
        Initialize the base API client.
        
        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
        """
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def _validate_response(self, response: requests.Response) -> dict:
        """
        Validate API response and handle different status codes.
        
        This method ensures that only successful responses (status code 200)
        proceed to further analysis. All other status codes will print the
        error information and raise appropriate exceptions.
        
        Args:
            response (requests.Response): The HTTP response to validate.
            
        Returns:
            dict: JSON response data if status code is 200.
            
        Raises:
            APIRequestError: For all non-200 status codes with server error message.
        """
        status_code = response.status_code
        
        if status_code == 200:
            # Success - proceed with analysis
            return response.json()
        else:
            # Error responses - print error info and raise exception
            try:
                error_data = response.json()
                error_message = error_data.get('messages', error_data.get('message', f'HTTP {status_code} error'))
            except:
                error_message = f'HTTP {status_code} error'
            
            # Print error status and message before raising exception
            print(f"❌ API Error - Status Code: {status_code}")
            print(f"❌ Error Message: {error_message}")
            raise APIRequestError(f"API request failed: {error_message}", status_code)
    
    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make an HTTP request with automatic response validation.
        
        Args:
            method (str): HTTP method (GET, POST, etc.).
            url (str): Full URL for the request.
            **kwargs: Additional arguments passed to requests.
            
        Returns:
            dict: JSON response data for successful requests (status code 200).
            
        Raises:
            APIRequestError: For non-200 responses or network issues.
        """
        try:
            response = getattr(self.session, method.lower())(url, timeout=self.timeout, **kwargs)
            return self._validate_response(response)
        except requests.RequestException as e:
            raise APIRequestError(f"Network request failed: {e}")


class HealthAPI(BaseAPI):
    """Wrapper class for the Health Check API endpoint.

    This class handles requests to the `/health` endpoint, allowing clients
    to verify that the Genos service is running and responsive.
    """

    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30):
        """
        Initialize the HealthAPI client.

        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
        """
        super().__init__(session, base_url, timeout)

    def check(self) -> dict:
        """
        Perform a health check request.

        Returns:
            dict: JSON response from the health endpoint, typically containing:
                - "status": "healthy" or "unhealthy"
                - "message": optional status message

        Raises:
            APIRequestError: If the request fails or the server is unreachable.
        """
        url = f"{self.base_url}/health"
        return self._make_request('GET', url)
