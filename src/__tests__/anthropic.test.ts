/**
 * Tests for Anthropic provider.
 * 
 * These tests focus on the provider configuration and metadata
 * since mocking the SDK is complex in ES modules.
 */

import { describe, it, expect } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic.js';

describe('AnthropicProvider', () => {
  describe('constructor and metadata', () => {
    it('should create provider with correct metadata', () => {
      const provider = new AnthropicProvider({ apiKey: 'sk-ant-test-key' });
      
      expect(provider.PROVIDER_NAME).toBe('anthropic');
      expect(provider.ENV_API_KEY_NAME).toBe('ANTHROPIC_API_KEY');
      expect(provider.API_BASE).toBe('https://api.anthropic.com');
      expect(provider.PROVIDER_DOCUMENTATION_URL).toBe('https://docs.anthropic.com');
    });
    
    it('should report correct capabilities', () => {
      const provider = new AnthropicProvider({ apiKey: 'sk-ant-test-key' });
      
      expect(provider.SUPPORTS_STREAMING).toBe(true);
      expect(provider.SUPPORTS_TOOLS).toBe(true);
      expect(provider.SUPPORTS_VISION).toBe(true);
      expect(provider.SUPPORTS_LIST_MODELS).toBe(false); // Anthropic doesn't have models endpoint
      expect(provider.SUPPORTS_REASONING).toBe(true);
    });
    
    it('should get metadata correctly', () => {
      const provider = new AnthropicProvider({ apiKey: 'sk-ant-test-key' });
      const metadata = provider.getMetadata();
      
      expect(metadata.name).toBe('anthropic');
      expect(metadata.envKey).toBe('ANTHROPIC_API_KEY');
      expect(metadata.streaming).toBe(true);
      expect(metadata.reasoning).toBe(true);
      expect(metadata.completion).toBe(true);
    });
  });
  
  describe('isAvailable', () => {
    it('should validate API key format', async () => {
      const validProvider = new AnthropicProvider({ apiKey: 'sk-ant-api123' });
      expect(await validProvider.isAvailable()).toBe(true);
    });
    
    it('should return false for invalid API key format', async () => {
      const invalidProvider = new AnthropicProvider({ apiKey: 'invalid-key' });
      expect(await invalidProvider.isAvailable()).toBe(false);
    });
    
    it('should return false when no API key is set', async () => {
      const originalEnv = process.env.ANTHROPIC_API_KEY;
      delete process.env.ANTHROPIC_API_KEY;
      
      const provider = new AnthropicProvider({});
      const result = await provider.isAvailable();
      
      if (originalEnv) process.env.ANTHROPIC_API_KEY = originalEnv;
      
      expect(result).toBe(false);
    });
  });
  
  describe('listModels', () => {
    it('should return static model list', async () => {
      const provider = new AnthropicProvider({ apiKey: 'sk-ant-test-key' });
      const models = await provider.listModels();
      
      expect(models.length).toBeGreaterThan(0);
      expect(models.map(m => m.id)).toContain('claude-3-5-sonnet-20241022');
      expect(models.map(m => m.id)).toContain('claude-3-opus-20240229');
      expect(models.every(m => m.provider === 'anthropic')).toBe(true);
      expect(models.every(m => m.supports_tools === true)).toBe(true);
      expect(models.every(m => m.supports_vision === true)).toBe(true);
    });
  });
});
