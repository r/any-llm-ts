/**
 * Tests for error handling and error classes.
 */

import { describe, it, expect } from 'vitest';
import {
  MissingApiKeyError,
  UnsupportedProviderError,
  ProviderRequestError,
  RateLimitError,
  ProviderUnavailableError,
  TimeoutError,
  isAnyLLMError,
  isRateLimitError,
  wrapError,
  isOpenAIError,
  isAnthropicError,
  isConnectionError,
} from '../errors.js';
import { AnyLLMError, AnyLLMErrorCode } from '../types.js';

describe('Error Classes', () => {
  describe('MissingApiKeyError', () => {
    it('should create error with correct message', () => {
      const error = new MissingApiKeyError('openai', 'OPENAI_API_KEY');
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('MissingApiKeyError');
      expect(error.code).toBe(AnyLLMErrorCode.MissingApiKey);
      expect(error.provider).toBe('openai');
      expect(error.message).toContain('openai');
      expect(error.message).toContain('OPENAI_API_KEY');
    });
  });
  
  describe('UnsupportedProviderError', () => {
    it('should create error with supported providers list', () => {
      const error = new UnsupportedProviderError('invalid', ['openai', 'anthropic']);
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('UnsupportedProviderError');
      expect(error.code).toBe(AnyLLMErrorCode.UnsupportedProvider);
      expect(error.message).toContain('invalid');
      expect(error.message).toContain('openai');
      expect(error.message).toContain('anthropic');
    });
  });
  
  describe('ProviderRequestError', () => {
    it('should create error with status code', () => {
      const error = new ProviderRequestError('openai', 'Bad request', 400);
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('ProviderRequestError');
      expect(error.code).toBe(AnyLLMErrorCode.RequestFailed);
      expect(error.provider).toBe('openai');
      expect(error.statusCode).toBe(400);
    });
    
    it('should create error with cause', () => {
      const cause = new Error('Original error');
      const error = new ProviderRequestError('anthropic', 'Request failed', 500, cause);
      
      expect(error.cause).toBe(cause);
    });
  });
  
  describe('RateLimitError', () => {
    it('should create error without retry after', () => {
      const error = new RateLimitError('openai');
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('RateLimitError');
      expect(error.code).toBe(AnyLLMErrorCode.RateLimited);
      expect(error.statusCode).toBe(429);
    });
    
    it('should create error with retry after', () => {
      const error = new RateLimitError('anthropic', 60);
      
      expect(error.message).toContain('60 seconds');
    });
  });
  
  describe('ProviderUnavailableError', () => {
    it('should create error with reason', () => {
      const error = new ProviderUnavailableError('ollama', 'Not running');
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('ProviderUnavailableError');
      expect(error.code).toBe(AnyLLMErrorCode.ProviderUnavailable);
      expect(error.message).toContain('Not running');
    });
    
    it('should create error without reason', () => {
      const error = new ProviderUnavailableError('llamafile');
      
      expect(error.message).toContain('llamafile');
      expect(error.message).toContain('unavailable');
    });
  });
  
  describe('TimeoutError', () => {
    it('should create error with timeout duration', () => {
      const error = new TimeoutError('openai', 30000);
      
      expect(error).toBeInstanceOf(AnyLLMError);
      expect(error.name).toBe('TimeoutError');
      expect(error.code).toBe(AnyLLMErrorCode.Timeout);
      expect(error.message).toContain('30000ms');
    });
  });
});

describe('Error Type Guards', () => {
  describe('isAnyLLMError', () => {
    it('should return true for AnyLLMError instances', () => {
      expect(isAnyLLMError(new MissingApiKeyError('test', 'TEST_KEY'))).toBe(true);
      expect(isAnyLLMError(new RateLimitError('test'))).toBe(true);
      expect(isAnyLLMError(new TimeoutError('test', 1000))).toBe(true);
    });
    
    it('should return false for regular errors', () => {
      expect(isAnyLLMError(new Error('test'))).toBe(false);
      expect(isAnyLLMError(null)).toBe(false);
      expect(isAnyLLMError(undefined)).toBe(false);
      expect(isAnyLLMError('error string')).toBe(false);
    });
  });
  
  describe('isRateLimitError', () => {
    it('should return true for RateLimitError', () => {
      expect(isRateLimitError(new RateLimitError('test'))).toBe(true);
    });
    
    it('should return false for other errors', () => {
      expect(isRateLimitError(new TimeoutError('test', 1000))).toBe(false);
      expect(isRateLimitError(new Error('rate limit'))).toBe(false);
    });
  });
  
  describe('isOpenAIError', () => {
    it('should detect OpenAI SDK errors', () => {
      const sdkError = { status: 429, message: 'Rate limit exceeded' };
      expect(isOpenAIError(sdkError)).toBe(true);
    });
    
    it('should return false for regular errors', () => {
      expect(isOpenAIError(new Error('test'))).toBe(false);
      expect(isOpenAIError(null)).toBe(false);
    });
  });
  
  describe('isAnthropicError', () => {
    it('should detect Anthropic SDK errors', () => {
      const sdkError = { status: 401, message: 'Invalid API key', error: { type: 'authentication_error' } };
      expect(isAnthropicError(sdkError)).toBe(true);
    });
    
    it('should return false for regular errors', () => {
      expect(isAnthropicError(new Error('test'))).toBe(false);
    });
  });
  
  describe('isConnectionError', () => {
    it('should detect connection refused errors', () => {
      expect(isConnectionError(new Error('ECONNREFUSED'))).toBe(true);
      expect(isConnectionError(new Error('fetch failed'))).toBe(true);
      expect(isConnectionError(new Error('network error'))).toBe(true);
    });
    
    it('should return false for other errors', () => {
      expect(isConnectionError(new Error('Invalid JSON'))).toBe(false);
      expect(isConnectionError(null)).toBe(false);
    });
  });
});

describe('wrapError', () => {
  it('should return AnyLLMError unchanged', () => {
    const original = new RateLimitError('test');
    const wrapped = wrapError(original, 'test');
    
    expect(wrapped).toBe(original);
  });
  
  it('should convert SDK errors with 429 to RateLimitError', () => {
    const sdkError = { status: 429, message: 'Rate limit' };
    const wrapped = wrapError(sdkError, 'openai');
    
    expect(wrapped).toBeInstanceOf(RateLimitError);
  });
  
  it('should convert SDK errors with 401 to MissingApiKeyError', () => {
    const sdkError = { status: 401, message: 'Invalid key' };
    const wrapped = wrapError(sdkError, 'openai');
    
    expect(wrapped).toBeInstanceOf(MissingApiKeyError);
  });
  
  it('should convert SDK errors with 403 to MissingApiKeyError', () => {
    const sdkError = { status: 403, message: 'Forbidden' };
    const wrapped = wrapError(sdkError, 'anthropic');
    
    expect(wrapped).toBeInstanceOf(MissingApiKeyError);
  });
  
  it('should convert connection errors to ProviderUnavailableError', () => {
    const connectionError = new Error('ECONNREFUSED');
    const wrapped = wrapError(connectionError, 'ollama');
    
    expect(wrapped).toBeInstanceOf(ProviderUnavailableError);
  });
  
  it('should detect rate limit from error message', () => {
    const error = new Error('Rate limit exceeded, please try again later');
    const wrapped = wrapError(error, 'test');
    
    expect(wrapped).toBeInstanceOf(RateLimitError);
  });
  
  it('should detect auth errors from error message', () => {
    const error = new Error('Unauthorized access');
    const wrapped = wrapError(error, 'test');
    
    expect(wrapped).toBeInstanceOf(MissingApiKeyError);
  });
  
  it('should detect timeout from AbortError', () => {
    const error = new Error('Request aborted');
    error.name = 'AbortError';
    const wrapped = wrapError(error, 'test');
    
    expect(wrapped).toBeInstanceOf(TimeoutError);
  });
  
  it('should wrap unknown errors as ProviderRequestError', () => {
    const error = new Error('Something went wrong');
    const wrapped = wrapError(error, 'test');
    
    expect(wrapped).toBeInstanceOf(ProviderRequestError);
    expect(wrapped.message).toContain('Something went wrong');
  });
  
  it('should handle string errors', () => {
    const wrapped = wrapError('string error', 'test');
    
    expect(wrapped).toBeInstanceOf(ProviderRequestError);
    expect(wrapped.message).toContain('string error');
  });
  
  it('should handle null/undefined', () => {
    const wrapped = wrapError(null, 'test');
    
    expect(wrapped).toBeInstanceOf(ProviderRequestError);
  });
});

describe('Error Inheritance', () => {
  it('all errors should be instances of Error', () => {
    expect(new MissingApiKeyError('test', 'KEY')).toBeInstanceOf(Error);
    expect(new UnsupportedProviderError('test', [])).toBeInstanceOf(Error);
    expect(new ProviderRequestError('test', 'msg')).toBeInstanceOf(Error);
    expect(new RateLimitError('test')).toBeInstanceOf(Error);
    expect(new ProviderUnavailableError('test')).toBeInstanceOf(Error);
    expect(new TimeoutError('test', 1000)).toBeInstanceOf(Error);
  });
  
  it('all errors should be instances of AnyLLMError', () => {
    expect(new MissingApiKeyError('test', 'KEY')).toBeInstanceOf(AnyLLMError);
    expect(new UnsupportedProviderError('test', [])).toBeInstanceOf(AnyLLMError);
    expect(new ProviderRequestError('test', 'msg')).toBeInstanceOf(AnyLLMError);
    expect(new RateLimitError('test')).toBeInstanceOf(AnyLLMError);
    expect(new ProviderUnavailableError('test')).toBeInstanceOf(AnyLLMError);
    expect(new TimeoutError('test', 1000)).toBeInstanceOf(AnyLLMError);
  });
  
  it('errors should have correct error codes', () => {
    expect(new MissingApiKeyError('test', 'KEY').code).toBe(AnyLLMErrorCode.MissingApiKey);
    expect(new UnsupportedProviderError('test', []).code).toBe(AnyLLMErrorCode.UnsupportedProvider);
    expect(new ProviderRequestError('test', 'msg').code).toBe(AnyLLMErrorCode.RequestFailed);
    expect(new RateLimitError('test').code).toBe(AnyLLMErrorCode.RateLimited);
    expect(new ProviderUnavailableError('test').code).toBe(AnyLLMErrorCode.ProviderUnavailable);
    expect(new TimeoutError('test', 1000).code).toBe(AnyLLMErrorCode.Timeout);
  });
});

