#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class GenosError(Exception):
    """Base exception class for all Genos-related errors.

    All custom exceptions in the Genos SDK should inherit from this class.
    It serves as the root of the exception hierarchy, allowing users to
    catch all Genos-specific errors with a single `except GenosError` block.
    """
    pass


class APIRequestError(GenosError):
    """Raised when an API request or response fails.

    This exception is typically thrown when the HTTP response status code
    indicates a failure (e.g., 4xx or 5xx), or when a network error occurs.

    Attributes:
        message (str): Description of the error.
        status_code (int, optional): HTTP status code returned by the API.
    """

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(GenosError):
    """Raised when authentication with the Genos API fails.

    For example, this may occur if the provided API token is missing,
    invalid, or expired.
    """
    pass


class ValidationError(GenosError):
    """Raised when input parameters fail validation.

    This exception indicates that the client provided invalid data to the API,
    such as a malformed variant format or missing required fields.
    """
    pass


class PaymentInsufficientError(GenosError):
    """Raised when the user's account has insufficient payment/credits for the API request.

    This exception is raised when the API returns a non-200 status code
    indicating payment-related issues, such as insufficient credits or
    expired subscription.
    """
    pass
